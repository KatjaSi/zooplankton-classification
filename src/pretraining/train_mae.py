import torch
import torch.nn as nn
import numpy as np
import os
import albumentations as A
import cv2
import random
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import WeightedRandomSampler
from transforms import ResizeAndPad
from datetime import datetime

import torch.optim as optim

from dataset import ZooScanImageFolder
from transformers import AutoImageProcessor, ViTMAEForPreTraining, ViTFeatureExtractor, ViTMAEConfig
import ipdb
from tqdm import tqdm
import yaml

from visualize import visualize


with open("src/pretraining/config.yaml", 'r') as file:
    config = yaml.safe_load(file)
    mean = config['transforms']['normalize']['mean']
    std = config['transforms']['normalize']['std']
    steps = config['eval_every_x_steps']

def main():

    device = torch.device("cuda")
    
    #TODO: optimier, sceduler
    
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

    
    train_dataset = ZooScanImageFolder(root=f"datasets/ZooScan77/train", transform=train_transform)
    val_dataset = ZooScanImageFolder(root=f"datasets/ZooScan77_010/val", transform=val_transform)
    
    train_labels = np.array(train_dataset.targets)
    class_counts = np.bincount(train_labels)
    class_weights = 1. / class_counts
    weights = class_weights[train_labels]
    train_sampler = WeightedRandomSampler(weights, len(weights))

    val_labels = np.array(val_dataset.targets)
    class_counts = np.bincount(val_labels)
    class_weights = 1. / class_counts
    weights = class_weights[val_labels]
    val_sampler = WeightedRandomSampler(weights, len(weights))

   
    #model_large = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-large")#.to(device)
    config = ViTMAEConfig() #by default bit-mae-base layers
    #config.hidden_size = 1024
    #config.intermediate_size = 4096
    config.norm_pixel_loss = True
    #config.num_attention_heads = 16
    #config.num_hidden_layers = 24
    #config.torch_dtype": "float16", #TODO: prøv etterpå
    model = ViTMAEForPreTraining(config).to(device) # use the same parameters as mael-large
   # model.load_state_dict(torch.load("best_model.pth"))
   # model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base").to(device)
    model.config.mask_ratio = 0.75
    #image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")

    batch_size = 32
    max_num_epochs = 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.000125, weight_decay=0.01) # lr = 1e-5

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    best_loss = float('inf')

    num_epochs = max_num_epochs
    step_count = 0
    for epoch in range(num_epochs):  # Number of epochs
        model.train()
        total_loss = 0
        step_total_loss = 0
        for images in train_dataloader:
            step_count += 1
            images = images.to(device)
            outputs = model(images)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            step_total_loss += loss.item()

            if step_count % steps == 0:
                print(f"Epoch {epoch + 1}, Step: {step_count}, Train Loss: {step_total_loss/steps}")
                step_total_loss = 0
                model.eval()
                total_val_loss = 0.0
                for images in val_dataloader:
                    images = images.to(device)
                    outputs = model(images)
                    loss = outputs.loss
                    total_val_loss += loss.item()

                avg_val_loss = total_val_loss / len(val_dataloader)
                print(f"Step {step_count}, Valid Loss: {avg_val_loss}")
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    torch.save(model.state_dict(), "best_model.pth")
                    print(f"Best model saved with loss: {best_loss}")

                indices = random.sample(range(len(val_dataloader.dataset)), 4)
                epoch_folder = os.path.join("checkpoints_mae", f"epoch{epoch+1}", f"step_{step_count}")
                os.makedirs(os.path.join(epoch_folder), exist_ok=True)
                for idx in indices:
                    save_path = os.path.join(epoch_folder, f"image_{idx}.png")
                    img = val_dataloader.dataset[idx].unsqueeze(0).to(device)
                    visualize(img, model, save_path, np.array(mean), np.array(std))

                model.train()


        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_loss}")

        


if __name__ == "__main__":
    main()