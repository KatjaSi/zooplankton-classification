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

from common.dataset import ImagenetDataset
from transformers import AutoImageProcessor, ViTMAEForPreTraining, ViTFeatureExtractor, ViTMAEConfig

from data_utils import get_dataloader, get_default_train_transform, get_default_val_transform, get_standard_imagenet_transform
from util.checkpointing import save_checkpoint

from datasets import load_dataset
import wandb

from huggingface_hub import login

login('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')

with open("pretraining/config.yaml", 'r') as file:
    config = yaml.safe_load(file)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    steps = config['eval_every_x_steps']


def main():
    train_losses = []
    val_losses = []

    device = torch.device("cuda")
    num_gpus = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
    wandb.init(
    project="vit_mae_pretraining_swa_imagenet_base",
    entity="katja-sivertsen",
    config={
        "model": "ViT_MAE_base_npl_true",
        "num_gpus": num_gpus,
        "gpu_names": gpu_names,
        "dataset": 'imagenet'
    })
    batch_size = 16*4
    max_num_epochs = 12
    config = ViTMAEConfig(norm_pix_loss = True,  #corresponding vit-mae-large layers
                            mask_ratio = 0.75,
                          #  hidden_size = 1024,
                          #  intermediate_size = 4096,
                          #  num_attention_heads = 16,
                          #  num_hidden_layers = 24,
                            num_channels = 1
                        )

    train_transform = get_standard_imagenet_transform(mean=[0.456], std=[0.224])
    val_transform = get_default_val_transform(mean=[0.456], std=[0.224])
    train_dataset = ImagenetDataset(dataset=load_dataset('ILSVRC/imagenet-1k', split='train', trust_remote_code=True),
                                    transform=train_transform)
    val_dataset = ImagenetDataset(dataset=load_dataset('ILSVRC/imagenet-1k', split='validation', trust_remote_code=True),
                                    transform=val_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=20)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=20)
    
    checkpoint = torch.load("checkpoints/checkpoints_mae_base_npl_imagenet/checkpoint_240000.pth")
    model_state_dict = checkpoint['model_state_dict']
    step_count = 0# checkpoint['step']
    best_loss = float('inf') #checkpoint['best_loss']

    model = ViTMAEForPreTraining(config).to(device) # use the same parameters as mael-large
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
                    torch.save(swa_model.state_dict(), f"checkpoints/checkpoints_mae_base_npl_imagenet/best_swa_model_test_{step_count}.pth")
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
    torch.save(swa_model.state_dict(), "checkpoints/checkpoints_mae_base_npl_imagenet/swa_final_model.pth")
    print("SWA model saved.")

if __name__=="__main__":
    main()