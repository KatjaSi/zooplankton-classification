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
from common.scheduler import CosineAnnealingWithWarmUp
from datetime import datetime

import torch.optim as optim

from common.dataset import ZooScanImageFolder
from transformers import AutoImageProcessor, ViTMAEForPreTraining, ViTFeatureExtractor, ViTMAEConfig
import ipdb
from tqdm import tqdm
import yaml

from visualize import visualize
from pretraining.data_utils import get_default_train_transform, get_default_val_transform, get_dataloader

from common.scheduler import CosineAnnealingWithWarmUp
from validate import validate

with open("pretraining/config2.yaml", 'r') as file:
    config = yaml.safe_load(file)
    mean = config['transforms']['normalize']['mean']
    std = config['transforms']['normalize']['std']
    steps = config['eval_every_x_steps']

def main():

    device = torch.device("cuda")
    
    train_transform = A.Compose([
        ResizeAndPad(224, fill=255),
        A.OneOf([
                            A.HorizontalFlip(p=0.5),
                            A.RandomRotate90(p=0.5),
                            A.VerticalFlip(p=0.5),
                            A.Rotate(limit=(-90, 90), border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=1)
        ], p=1),
        A.Normalize(mean=mean, std=std),
        ToTensorV2() ])

    val_transform = A.Compose([
        ResizeAndPad(224, fill=255),
        A.Normalize(mean=mean, std=std),
        ToTensorV2() ])

    
    train_transform = get_default_train_transform(mean, std)
    val_transform = get_default_val_transform(mean, std)
   
    config_mae = ViTMAEConfig(
                            norm_pix_loss = True,  #corresponding vit-mae-large layers
                            hidden_size = 1024,
                            intermediate_size = 4096,
                            num_attention_heads = 16,
                            num_hidden_layers = 24,
                            num_channels = 1
                        )

    model = ViTMAEForPreTraining(config_mae)
    state_dict = torch.load("best_model_phase2_2.pth")
    model.load_state_dict(state_dict)
    model.to(device) # use the same parameters as mael-large

    batch_size = 64
    num_epochs = 15

    dataset1 = ZooScanImageFolder(root="datasets/ZooScan77/train", transform=get_default_train_transform(mean, std), grayscale=True)
    dataset2 = ZooScanImageFolder(root="datasets/PELGAS", transform=get_default_train_transform(mean, std), grayscale=True)
    dataset3 = ZooScanImageFolder(root="datasets/ZooScan2018", transform=get_default_train_transform(mean, std), grayscale=True)
    combined_dataset = ConcatDataset([dataset1, dataset2, dataset3])
    train_dataloader = DataLoader(combined_dataset, batch_size=batch_size*2, shuffle=True)


    val_dataloader = get_dataloader(root="datasets/ZooScan77_small/val",
                                        transform=val_transform,
                                        batch_size=batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.000125, weight_decay=0.001) # lr = 1e-5
    

    scheduler = CosineAnnealingWithWarmUp(warmup_steps=len(train_dataloader), # 1 epoch
                                            total_steps=num_epochs*len(train_dataloader),
                                            optimizer=optimizer,
                                            warmup_fraction=0.1,
                                            eta_min=0)

    best_loss = float('inf')

    step_count = 0
    total_loss = 0 # evaluation every x steps
    for epoch in range(num_epochs):  # Number of epochs
        model.train()
        for images, _ in train_dataloader:
            step_count += 1
            images = images.to(device)
            with torch.no_grad():
                model.eval()
                outputs = model(images)
                target = model.patchify(images)
                if model.config.norm_pix_loss:
                    patch_mean = target.mean(dim=-1, keepdim=True)
                    patch_var = target.var(dim=-1, keepdim=True)
                    target = (target - patch_mean) / (patch_var + 1.0e-6) ** 0.5
                l2 = (outputs.logits - target)**2
                l2 = l2.mean(dim=-1)
                loss_per_sample = (l2*outputs.mask).sum(dim=1)/outputs.mask.sum(dim=1)
                hardest_images = images[loss_per_sample.sort(descending=True).indices[:batch_size]]
            model.train()
            outputs = model(hardest_images)
                
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if step_count % steps == 0:
                print(f"Epoch {epoch + 1}, Step: {step_count}, Train Loss: {total_loss/steps}")
                total_loss = 0
                model.eval()
                val_loss = validate(model, val_dataloader, device)
                print(f"Step {step_count}, Valid Loss: {val_loss}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), "best_model_phase2_3.pth")
                    print(f"Best model saved with loss: {best_loss}")

                indices = random.sample(range(len(val_dataloader.dataset)), 4)
                epoch_folder = os.path.join("checkpoints_mae", f"epoch{epoch+1}", f"step_{step_count}")
                os.makedirs(os.path.join(epoch_folder), exist_ok=True)
                for idx in indices:
                    save_path = os.path.join(epoch_folder, f"image_{idx}.png")
                    img, _ = val_dataloader.dataset[idx]
                    img = img.unsqueeze(0).to(device)
                    visualize(img, model, save_path, np.array(mean), np.array(std))

                model.train()

        


if __name__ == "__main__":
    main()