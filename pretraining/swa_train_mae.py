import torch
import torch.nn as nn
import numpy as np
import os
import random
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
import copy
import pandas as pd
import numpy as np
import yaml

from common.dataset import ZooScanImageFolder
from transformers import AutoImageProcessor, ViTMAEForPreTraining, ViTFeatureExtractor, ViTMAEConfig

from data_utils import get_dataloader, get_default_train_transform, get_default_val_transform
from util.checkpointing import save_checkpoint

import wandb



with open("pretraining/config.yaml", 'r') as file:
    config = yaml.safe_load(file)
    mean = config['transforms']['normalize']['mean']
    std = config['transforms']['normalize']['std']
    steps = config['eval_every_x_steps']


def main():
    train_losses = []
    val_losses = []

    device = torch.device("cuda")
    num_gpus = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
    wandb.init(
    project="vit_mae_pretraining_base_swa",
    entity="katja-sivertsen",
    config={
        "model": "ViT_MAE_base_npl_false",
        "num_gpus": num_gpus,
        "gpu_names": gpu_names
    })
    batch_size = 16*4
    max_num_epochs = 12
    config = ViTMAEConfig(norm_pix_loss = False,  #corresponding vit-mae-large layers
                            mask_ratio = 0.75,
                           # hidden_size = 1024,
                           # intermediate_size = 4096,
                           # num_attention_heads = 16,
                           # num_hidden_layers = 24,
                            num_channels = 1
                        )
    dataset1 = ZooScanImageFolder(root="datasets/ZooScan77/train", transform=get_default_train_transform(mean, std), grayscale=True)
    dataset2 = ZooScanImageFolder(root="datasets/PELGAS", transform=get_default_train_transform(mean, std), grayscale=True)
    dataset3 = ZooScanImageFolder(root="datasets/ZooScan2018", transform=get_default_train_transform(mean, std), grayscale=True)
    combined_dataset = ConcatDataset([dataset1, dataset2, dataset3])
    train_dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=20)
    val_dataloader = get_dataloader(root="datasets/ZooScan77_small/val", 
                                        transform=get_default_val_transform(mean, std),
                                        batch_size=batch_size,
                                        num_workers=20)
    
    checkpoint = torch.load("checkpoints/checkpoints_mae_base/checkpoints/checkpoint_261000.pth")
    model_state_dict = checkpoint['model_state_dict']
    step_count = 0# checkpoint['step']
    best_loss = checkpoint['best_loss']

    model = ViTMAEForPreTraining(config).to(device) 
    model.load_state_dict(model_state_dict)
    model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.000015, weight_decay=0.0, betas=(0.845, 0.985))

    num_epochs = 3

    swa_scheduler = SWALR(optimizer, swa_lr=0.000015)
    
    swa_model = AveragedModel(model.module)
    swa_model.to(device)


    for epoch in range(num_epochs):  # Number of epochs
        model.train()
        step_total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        # Determine how many batches to skip if resuming mid-epoch
        for i, (images, _) in enumerate(progress_bar):
            step_count += 1
            images = images.to(device)
            outputs = model(images)
            loss = outputs.loss.mean()
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            swa_scheduler.step()
            step_total_loss += loss.item()
            progress_bar.set_description(f"Epoch {epoch + 1}/{num_epochs} | Step: {step_count}")
            
            if step_count % steps == 0:
                print(f"Epoch {epoch + 1}, Step: {step_count}, Train Loss: {step_total_loss/steps}")
                step_total_loss = 0
                swa_model.update_parameters(model.module)
                swa_model.eval()
                total_val_loss = 0.0
                for images, _ in val_dataloader:
                    images = images.to(device)
                    outputs = swa_model(images)
                    loss = outputs.loss.mean()
                    total_val_loss += loss.item()

                avg_val_loss = total_val_loss / len(val_dataloader)
                val_losses.append(avg_val_loss)
                print(f"Step {step_count}, Valid Loss: {avg_val_loss}")
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    torch.save(swa_model.state_dict(), f"checkpoints/checkpoints_mae_base/checkpoints/best_swa_model_test_{step_count}.pth")
                    print(f"Best model saved with loss: {best_loss}")
       
                wandb.log({
                    "step": step_count,
                        "valid_loss": avg_val_loss,
                        "train_loss": step_total_loss / steps,
                        'lr': optimizer.param_groups[0]['lr']
                    })
                model.train()
            else:
                progress_bar.set_postfix(train_loss=step_total_loss / (step_count%steps))


    wandb.finish()

    print("Finalizing SWA...")
    update_bn(train_dataloader, swa_model)
    torch.save(swa_model.state_dict(), "checkpoints/checkpoints_mae_base/checkpoints/swa_final_model.pth")
    print("SWA model saved.")

if __name__=="__main__":
    main()