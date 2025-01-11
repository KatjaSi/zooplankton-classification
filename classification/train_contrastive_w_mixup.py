import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import shutil
import os
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import WeightedRandomSampler
from plots import plot_confusion_matrix
from parsers import TrainConfigParser # type: ignore
from common.transforms import apply_clahe, resize_and_pad, ResizeAndPad
from datetime import datetime
from transformers import SwinForImageClassification, SwinConfig
import torch.optim as optim
import csv
import pandas as pd
import copy
import ipdb

from train_utils import one_iter
from common.dataset import ZooScanImageFolder
from tqdm import tqdm

class GenSupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(GenSupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: (2N, d)
        # labels: (2N, C) soft or one-hot labels
        # Normalize label vectors for similarity
        labels_norm = labels / (labels.norm(dim=1, keepdim=True) + 1e-8)
        label_sim_matrix = torch.matmul(labels_norm, labels_norm.T)  # (2N, 2N)

        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        N = features.size(0)
        mask = torch.eye(N, dtype=torch.bool, device=features.device)

        exp_sim = torch.exp(similarity_matrix)
        exp_sim = exp_sim * (~mask)

        # Mask out diagonal in label similarity too
        label_sim_matrix = label_sim_matrix * (~mask)

        # Normalize label similarities row-wise
        row_sums = label_sim_matrix.sum(dim=1, keepdim=True) + 1e-8
        Y = label_sim_matrix / row_sums  # Distribution from labels
        denom = exp_sim.sum(dim=1, keepdim=True) + 1e-8
        P = exp_sim / denom  # Distribution from embeddings

        P_clamped = torch.clamp(P, min=1e-8)
        CE = -(Y * torch.log(P_clamped)).sum(dim=1)
        loss = CE.mean()
        return loss

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=2048, proj_dim=128):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, proj_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class MultiViewDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        path, label = self.base_dataset.samples[idx]
        image = self.base_dataset.loader(path)
        image_np = np.array(image)
        aug1 = self.transform(image=image_np)['image']
        aug2 = self.transform(image=image_np)['image']
        return aug1, aug2, label

def convert_labels_to_one_hot(labels, num_classes, device):
    one_hot = torch.zeros(labels.size(0), num_classes, device=device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return one_hot

def mixup_data(x, y, alpha=1.0):
    # x: (2N, C, H, W), y: (2N, num_classes)
    # Apply mixup on the combined batch of 2N samples
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
    return mixed_x, mixed_y

def train_one_epoch(model, proj_head, criterion, data_loader, optimizer, device, epoch, num_classes, mixup_alpha=0.0):
    model.train()
    proj_head.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}", leave=False)

    for i, batch in enumerate(progress_bar):
        view1, view2, labels = batch
        view1, view2, labels = view1.to(device), view2.to(device), labels.to(device)

        # Combine views into a single batch of size 2N
        # view1: (N, C, H, W), view2: (N, C, H, W)
        # concatenate along batch dimension
        images_combined = torch.cat([view1, view2], dim=0)  # (2N, C, H, W)

        labels_onehot = convert_labels_to_one_hot(labels, num_classes, device)
        labels_all = torch.cat([labels_onehot, labels_onehot], dim=0) # (2N, num_classes)

        # Apply Mixup if alpha > 0
        if mixup_alpha > 0.0:
            images_combined, labels_all = mixup_data(images_combined, labels_all, alpha=mixup_alpha)

        outputs = model(images_combined)
        emb = outputs.logits  # (2N, 1024)
        emb = nn.functional.normalize(emb, dim=1)  # normalize encoder output
        z = proj_head(emb)  # (2N, proj_dim)
        z = nn.functional.normalize(z, dim=1)  # normalize projection output

        loss = criterion(z, labels_all)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (i + 1)
        progress_bar.set_postfix(loss=avg_loss)

    return total_loss / len(data_loader)


def main():
    parser = TrainConfigParser()

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    batch_size = 256#128 # change
    max_num_epochs = 50
    dataset = parser.get_dataset_name()
    mixup_alpha = 1.0  
    mean = parser.get_transforms_normalize_mean()
    std = parser.get_transforms_normalize_std()
    patience = 10

    checkpoint_path = os.path.join('checkpoints/swin_contrastive_3', 'full_set')
    os.makedirs(checkpoint_path, exist_ok=True)
    latest_checkpoint_file = os.path.join(checkpoint_path, "latest_checkpoint.pth")
    best_checkpoint_file = os.path.join(checkpoint_path, "best_checkpoint.pth")
    os.makedirs(checkpoint_path, exist_ok=True)

    train_transform = A.Compose([
        # same transforms as before
        ResizeAndPad(224, fill=255), 
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=(-90, 90), border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=1),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=0,
                            border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=0.5),
        A.MotionBlur(p=0.5),
        A.OpticalDistortion(p=0.5),
        A.GaussNoise(p=0.5),
        A.CoarseDropout(max_holes=16, max_height=20, max_width=20, fill_value=255, p=0.5),
        A.Defocus(radius=(1, 3), alias_blur=(0.1, 0.3), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    
    train_dataset_raw = ZooScanImageFolder(root=f"datasets/{dataset}/train", transform=None, grayscale=False)
    num_classes = len(train_dataset_raw.classes)
    train_dataset = MultiViewDataset(train_dataset_raw, train_transform)

    train_labels = np.array(train_dataset_raw.targets)
    class_counts = np.bincount(train_labels)
    class_weights = 1. / class_counts
    weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(weights, len(weights))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=20, drop_last=True)

    model = SwinForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224",
                                                       num_labels=num_classes,
                                                       ignore_mismatched_sizes=True)
    model.classifier = nn.Identity()
    model = model.to(device)
    model = nn.DataParallel(model)

    proj_head = ProjectionHead(input_dim=1024, hidden_dim=2048, proj_dim=128).to(device)
    proj_head = nn.DataParallel(proj_head)

    criterion = GenSupConLoss(temperature=0.1)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(proj_head.parameters()), lr=1e-5, weight_decay=1e-8)

    best_loss = float('inf')
    best_model_weights = None
    patience_count = patience
    start_epoch = 0

    ## load checkpoint
    if os.path.isfile(latest_checkpoint_file):
        print(f"Loading checkpoint from '{latest_checkpoint_file}'...")
        checkpoint = torch.load(latest_checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        proj_head.load_state_dict(checkpoint['proj_head_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1 
        best_loss = checkpoint['best_loss']
        patience_count = checkpoint['patience_count']
        print(f"Resumed from epoch {start_epoch}, best loss: {best_loss}")

    ### stop oad checkpoint



    for epoch in range(start_epoch, max_num_epochs):
        train_loss = train_one_epoch(
            model, proj_head, criterion, train_loader, optimizer, device, epoch, num_classes, mixup_alpha=mixup_alpha
        )
        print(f"Epoch {epoch+1}, Train GenContrastive Loss: {train_loss:.6f}")

        latest_checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'proj_head_state_dict': proj_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            'patience_count': patience_count,
        }
        torch.save(latest_checkpoint_data, latest_checkpoint_file)

        if train_loss < best_loss:
            best_loss = train_loss
            patience_count = patience
            best_checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'proj_head_state_dict': proj_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
            }
            torch.save(best_checkpoint_data, best_checkpoint_file)
            patience_count = patience
        else:
            patience_count -= 1
            if patience_count == 0:
                print("Early stopping triggered.")
                break


    print("Finished generalized contrastive training with Mixup.")
 

if __name__ == "__main__":
    main()
