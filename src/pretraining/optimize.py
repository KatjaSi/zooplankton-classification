import optuna
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

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from dataset import ZooScanImageFolder
from transformers import  ViTMAEForPreTraining, ViTMAEConfig
import yaml
import wandb 
import ipdb
from visualize import visualize


with open("src/pretraining/config.yaml", 'r') as file:
    train_config = yaml.safe_load(file)
    mean = train_config['transforms']['normalize']['mean']
    std = train_config['transforms']['normalize']['std']
    steps = train_config['eval_every_x_steps']

def objective(trial):
    device = torch.device("cuda")

    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'SGD'])
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32, 64])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    warmup_fraction = 0.01
    warmup_epochs = 1
    
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

    
    train_dataset = ZooScanImageFolder(root=f"datasets/ZooScan77_small/train", transform=train_transform)
    val_dataset = ZooScanImageFolder(root=f"datasets/ZooScan77_small/val", transform=val_transform)
    
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
    config = ViTMAEConfig(norm_pixel_loss = True) #by default bit-mae-base layers
    model = ViTMAEForPreTraining(config).to(device) # use the same parameters as mael-large
  
    max_num_epochs = 15
    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)

    num_epochs = max_num_epochs
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    # define warmup and scheduler
    warmup_steps = warmup_epochs*len(train_dataloader)
    warmup_scheduler = LinearLR(optimizer, start_factor=warmup_fraction, end_factor=1.0, total_iters=warmup_steps)

    num_steps = num_epochs*len(train_dataloader)-warmup_steps
    scheduler = CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=0, last_epoch=-1)

    best_loss = float('inf')
    trial_config = dict(trial.params)
    trial_config["trial.number"] = trial.number
    wandb.init(
        project="optuna",
        entity="katja-sivertsen",
        config=trial_config,
        group="VIT_MAE_PRETRAINING",
        reinit=True,
    )
    step_count = 0
    for epoch in range(num_epochs):
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
            if step_count < warmup_steps:
                warmup_scheduler.step()
            else:
                scheduler.step()
            total_loss += loss.item()
            step_total_loss += loss.item()

            if step_count % steps == 0:
                model.eval()
                total_val_loss = 0.0
                for images in val_dataloader:
                    images = images.to(device)
                    outputs = model(images)
                    loss = outputs.loss
                    total_val_loss += loss.item()

                avg_val_loss = total_val_loss / len(val_dataloader)
                print(f"Step {step_count}, Valid Loss: {avg_val_loss}")
                trial.report(avg_val_loss, step_count)
                wandb.log(data={"loss": avg_val_loss}, step=step_count)
                if trial.should_prune():
                    wandb.run.summary["state"] = "pruned"
                    wandb.finish(quiet=True)
                    raise optuna.exceptions.TrialPruned()
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss 
                model.train()
    wandb.run.summary["final loss"] = avg_val_loss
    wandb.run.summary["state"] = "complated"
    wandb.finish(quiet=True)
    return best_loss

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize',
                                study_name="VIT_MAE_PRETRAINING",
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=100)


    # Print the best parameters
    print(study.best_params)