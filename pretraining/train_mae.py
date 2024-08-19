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
from common.scheduler import CosineAnnealingWithWarmUp
from transformers import AutoImageProcessor, ViTMAEForPreTraining, ViTFeatureExtractor, ViTMAEConfig
import ipdb
from tqdm import tqdm
import yaml

from visualize import visualize
from data_utils import get_dataloader, get_default_train_transform, get_default_val_transform


from common.dataset import ZooScanImageFolder as dih

with open("pretraining/config.yaml", 'r') as file:
    config = yaml.safe_load(file)
    mean = config['transforms']['normalize']['mean']
    std = config['transforms']['normalize']['std']
    steps = config['eval_every_x_steps']

def main():
    train_losses = []
    val_losses = []

    device = torch.device("cuda")
   
    config = ViTMAEConfig(norm_pix_loss = True,  #corresponding vit-mae-large layers
                            mask_ratio = 0.75,
                            hidden_size = 1024,
                            intermediate_size = 4096,
                            num_attention_heads = 16,
                            num_hidden_layers = 24,
                            num_channels = 1
                        )
    model = ViTMAEForPreTraining(config).to(device) # use the same parameters as mael-large

    batch_size = 16
    max_num_epochs = 30
    dataset1 = ZooScanImageFolder(root="datasets/ZooScan77/train", transform=get_default_train_transform(mean, std), grayscale=True)
    dataset2 = ZooScanImageFolder(root="datasets/PELGAS", transform=get_default_train_transform(mean, std), grayscale=True)
    dataset3 = ZooScanImageFolder(root="datasets/ZooScan2018", transform=get_default_train_transform(mean, std), grayscale=True)
    combined_dataset = ConcatDataset([dataset1, dataset2, dataset3])
    train_dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = get_dataloader(root="datasets/ZooScan77_small/val", 
                                        transform=get_default_val_transform(mean, std),
                                        batch_size=batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.000164746, weight_decay=0.0, betas=(0.845, 0.985)) # lr = 1e-5
    scheduler = CosineAnnealingWithWarmUp(warmup_steps=len(train_dataloader)*1.0,
                                            total_steps=len(train_dataloader)*max_num_epochs, 
                                            optimizer=optimizer,
                                            warmup_fraction=0.05,
                                            eta_min=0.00002)
    

    best_loss = float('inf')

    num_epochs = max_num_epochs
    step_count = 0
    for epoch in range(num_epochs):  # Number of epochs
        model.train()
        total_loss = 0
        step_total_loss = 0
        for images, _ in train_dataloader:
            step_count += 1
            images = images.to(device)
            outputs = model(images)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            step_total_loss += loss.item()

            if step_count % steps == 0:
                print(f"Epoch {epoch + 1}, Step: {step_count}, Train Loss: {step_total_loss/steps}")
                train_losses.append(step_total_loss/steps)
                step_total_loss = 0
                model.eval()
                total_val_loss = 0.0
                for images, _ in val_dataloader:
                    images = images.to(device)
                    outputs = model(images)
                    loss = outputs.loss
                    total_val_loss += loss.item()

                avg_val_loss = total_val_loss / len(val_dataloader)
                val_losses.append(avg_val_loss)
                print(f"Step {step_count}, Valid Loss: {avg_val_loss}")
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    torch.save(model.state_dict(), "best_model_phase1.pth")
                    print(f"Best model saved with loss: {best_loss}")

                model.train()


        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_loss}")

    np.save('train_losses.npy', np.array(train_losses))
    np.save('val_losses.npy', np.array(val_losses))

        


if __name__ == "__main__":
    main()