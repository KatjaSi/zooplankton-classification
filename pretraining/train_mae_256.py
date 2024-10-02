import torch
import torch.nn as nn
import numpy as np
import os
import albumentations as A
import cv2
import random
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.utils.data import WeightedRandomSampler
from common.transforms import ResizeAndPad
from datetime import datetime
import torch.optim as optim
from common.dataset import ZooScanImageFolder
from transformers import AutoImageProcessor, ViTMAEForPreTraining, ViTFeatureExtractor, ViTMAEConfig
import ipdb
from tqdm import tqdm
import yaml
from visualize import visualize
from data_utils import get_dataloader, get_default_train_transform, get_default_val_transform
from data_utils import get_train_transform_with_random_zoom

import util.lr_decay as lrd
from util.optim import get_optimizer 
from util.scheduler import CosineAnnealingWithWarmUpWithLLRD
from util.checkpointing import save_checkpoint, load_checkpoint

import wandb


with open("pretraining/config2.yaml", 'r') as file:
    config = yaml.safe_load(file)
    mean = config['transforms']['normalize']['mean']
    std = config['transforms']['normalize']['std']
    steps = config['eval_every_x_steps']


def main():
    device = torch.device("cuda")
    num_gpus = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
    print(f"Number of available GPUs: {num_gpus}")

    wandb.init(
    project="vit_mae_pretraining",
    entity="katja-sivertsen",
    config={
        "model": "ViT_MAE_base_npl_false",
        "num_gpus": num_gpus,
        "gpu_names": gpu_names
    }
)


    #config = ViTMAEConfig(norm_pix_loss=False,  # corresponding vit-mae-base layers TODO: check
     #                     mask_ratio=0.75,
     #                     hidden_size=768,
     #                     intermediate_size=3072,
     #                     num_attention_heads=12,
     #                     num_hidden_layers=12,
     #                     num_channels=1
     #                     )
    config = ViTMAEConfig(norm_pix_loss = True,  #corresponding vit-mae-large layers
                            mask_ratio = 0.75,
                            hidden_size = 1024,
                            intermediate_size = 4096,
                            num_attention_heads = 16,
                            num_hidden_layers = 24,
                            num_channels = 1
                        )
    model = ViTMAEForPreTraining(config).to(device)

    base_batch_size = 64  # Batch size for a single GPU # TODO: chznge here
    effective_batch_size = 256  # Fixed effective batch size
    batch_size = base_batch_size * num_gpus  
    accumulate_steps = effective_batch_size // batch_size

    max_num_epochs = 10 # after that the swa epochs will be added
    warmup_epochs = 1
    num_workers = 8
    dataset1 = ZooScanImageFolder(root="datasets/ZooScan77/train", transform=get_train_transform_with_random_zoom(mean, std), grayscale=True)
    dataset2 = ZooScanImageFolder(root="datasets/PELGAS", transform=get_train_transform_with_random_zoom(mean, std), grayscale=True)
    dataset3 = ZooScanImageFolder(root="datasets/ZooScan2018", transform=get_train_transform_with_random_zoom(mean, std), grayscale=True)
    combined_dataset = ConcatDataset([dataset1, dataset2, dataset3])
    train_dataloader = DataLoader(dataset1, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = get_dataloader(root="datasets/ZooScan77_small/val", 
                                        transform=get_default_val_transform(mean, std),
                                        batch_size=batch_size,
                                        num_workers=num_workers)
    
    param_group_names, param_group_values = lrd.param_groups_lrd(model,
                        weight_decay=0.05, 
                        no_weight_decay_list=[
                        'vit.embeddings.cls_token', 
                        'decoder.mask_token',
                        'vit.embeddings.patch_embeddings.projection.weight',
                        'vit.embeddings.patch_embeddings.projection.bias'
                        ], 
                        layer_decay=.65)                                       
    optimizer = get_optimizer(lr=1.5e-3, weight_decay=0.0, param_groups=param_group_values)

    scheduler = CosineAnnealingWithWarmUpWithLLRD(warmup_steps=len(train_dataloader)*warmup_epochs // accumulate_steps,
                                            total_steps=len(train_dataloader)*max_num_epochs // accumulate_steps, 
                                            optimizer=optimizer,
                                            base_lr=1.5e-3,
                                            min_lr=1.5e-6)
    
    start_epoch, effective_step_count, best_loss, total_loss = load_checkpoint(model, optimizer, scheduler)
   # model = nn.DataParallel(model)
    best_loss = float('inf')
    step_count = effective_step_count*accumulate_steps 
    num_epochs = max_num_epochs
    save_checkpoint_flag = False
    for epoch in range(start_epoch, num_epochs): 
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        # Determine how many batches to skip if resuming mid-epoch
        if epoch == start_epoch:
            skip_batches = step_count % len(train_dataloader)
        else:
            skip_batches = 0
            total_loss = 0
        for i, (images, _) in enumerate(progress_bar):
            # Skip batches if resuming mid-epoch
            if i < skip_batches:
                continue
            step_count += 1
            images = images.to(device)
            outputs = model(images)
            loss = outputs.loss.mean()
            total_loss += loss.item()

            effective_step_count = step_count // accumulate_steps

            progress_bar.set_description(f"Epoch {epoch + 1}/{num_epochs} | Effective Step: {effective_step_count}")
            progress_bar.set_postfix(train_loss=total_loss / (i + 1))

            loss.backward() # do not need scaling bcs the loss is averaged

            if (effective_step_count % steps == 0 and step_count % accumulate_steps == 0) or (i + 1 == len(train_dataloader)):
                save_checkpoint_flag = True

            if ((i + 1) % accumulate_steps == 0) or (i+1 == len(train_dataloader)):     
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                if save_checkpoint_flag:
                    print(f"Epoch {epoch + 1}, Step: {step_count}, Train Loss: {total_loss / step_count}")
                    model.eval()
                    total_val_loss = 0.0
                    for images, _ in val_dataloader:
                        images = images.to(device)
                        outputs = model(images)
                        loss = outputs.loss.mean()
                        total_val_loss += loss.item()

                    avg_val_loss = total_val_loss / len(val_dataloader)
                    print(f"Effective Step {effective_step_count}, Valid Loss: {avg_val_loss}")

                    # Log to wandb
                    wandb.log({
                        "step": effective_step_count,
                        "valid_loss": avg_val_loss,
                        "train_loss": total_loss / step_count
                    })

                    for idx, param_group in enumerate(optimizer.param_groups):
                        lr = param_group['lr']
                        wandb.log({f"learning_rate_group_{idx}": lr})

                    print("Generating Visualizations")
                    indices = random.sample(range(len(val_dataloader.dataset)), 12)
                    epoch_folder = os.path.join("checkpoints_mae", f"epoch{epoch+1}", f"effective_step_{effective_step_count}")
                    os.makedirs(os.path.join(epoch_folder), exist_ok=True)
                    for idx in indices:
                        save_path = os.path.join(epoch_folder, f"image_{idx}.png")
                        img, _ = val_dataloader.dataset[idx]
                        img = img.unsqueeze(0).to(device)
                        visualize(img, model, save_path, np.array(mean), np.array(std))

                    print("Generating Train Visualizations")
                    indices = random.sample(range(len(train_dataloader.dataset)), 12)
                    epoch_folder = os.path.join("checkpoints_mae", f"epoch{epoch+1} train", f"effective_step_{effective_step_count}")
                    os.makedirs(os.path.join(epoch_folder), exist_ok=True)
                    for idx in indices:
                        save_path = os.path.join(epoch_folder, f"image_{idx}.png")
                        img, _ = train_dataloader.dataset[idx]
                        img = img.unsqueeze(0).to(device)
                        visualize(img, model, save_path, np.array(mean), np.array(std))

                    if avg_val_loss < best_loss:
                        best_loss = avg_val_loss
                        torch.save(model.state_dict(), "best_model_phase1_acc_batch_256.pth")
                        print(f"Best model saved with loss: {best_loss}")

                    save_checkpoint(model, optimizer, scheduler, epoch, effective_step_count, best_loss, total_loss)
                    save_checkpoint_flag = False  
                    model.train()


    wandb.finish()
        


if __name__ == "__main__":
    main()