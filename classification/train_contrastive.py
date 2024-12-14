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
from tqdm import tqdm  # <-- Added tqdm import

from common.dataset import ZooScanImageFolder

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss from the "Supervised Contrastive Learning" paper.
    """
    def __init__(self, temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        N = features.size(0)
        mask = torch.eye(N, dtype=torch.bool, device=features.device)
       
        labels = labels.unsqueeze(1) 
        positive_mask = (labels == labels.T) & ~mask

        exp_sim = torch.exp(similarity_matrix)
        exp_sim = exp_sim * (~mask)

        numerator = (exp_sim * positive_mask).sum(dim=1)
        denominator = exp_sim.sum(dim=1)

        #numerator = torch.clamp(numerator, min=1e-8)
        denominator = torch.clamp(denominator, min=1e-8)

        loss = -torch.log(numerator / denominator)
        return loss.mean()


class ProjectionHead(nn.Module):
    def __init__(self, embed_dim=1024, proj_dim=128):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(embed_dim, proj_dim)

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


def train_one_epoch(model, proj_head, criterion, data_loader, optimizer, device, epoch):
    model.train()
    proj_head.train()
    total_loss = 0

    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}", leave=False)
    for i, batch in enumerate(progress_bar):
        view1, view2, labels = batch
        view1, view2, labels = view1.to(device), view2.to(device), labels.to(device)

        output1 = model(view1)
        emb1 = output1.logits
        emb1 = nn.functional.normalize(emb1, dim=1)
        z1 = proj_head(emb1)
        z1 = nn.functional.normalize(z1, dim=1)      # normalize projection output
        output2 = model(view2)
        emb2 = output2.logits
        emb2 = nn.functional.normalize(emb2, dim=1)
        z2 = proj_head(emb2)
        z2 = nn.functional.normalize(z2, dim=1)      # normalize projection output
        features = torch.cat([z1, z2], dim=0)
        labels_all = torch.cat([labels, labels], dim=0)
        
        loss = criterion(features, labels_all)

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

    batch_size = 16
    max_num_epochs = parser.get_max_num_epochs()
    dataset = parser.get_dataset_name()
    is_enable_report = parser.is_enable_report()
    report_frequency = parser.get_report_frequency()
    num_workers = parser.get_num_workers()
    is_checkpoint = parser.is_checkpoint()
    early_stopping_metric = parser.get_early_stopping_metric()
    compare_op = parser.get_compare_operator()
    mean = parser.get_transforms_normalize_mean()
    std = parser.get_transforms_normalize_std()

    checkpoint_path = os.path.join('checkpoints', 'swin_contrastive')
    os.makedirs(checkpoint_path, exist_ok=True)

    train_transform = A.Compose([
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
    train_dataset = MultiViewDataset(train_dataset_raw, train_transform)

    train_labels = np.array(train_dataset_raw.targets)
    class_counts = np.bincount(train_labels)
    class_weights = 1. / class_counts
    weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(weights, len(weights))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=20, drop_last=True)

    model = SwinForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224",
                                                       num_labels=77,
                                                       ignore_mismatched_sizes=True)
    model.classifier = nn.Identity()
    model = model.to(device)
    model = nn.DataParallel(model)
    proj_head = ProjectionHead(embed_dim=1024, proj_dim=128).to(device)
    proj_head = nn.DataParallel(proj_head)

    criterion = SupConLoss(temperature=0.07)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(proj_head.parameters()), lr=1e-5, weight_decay=1e-8)

    best_loss = float('inf')
    best_model_weights = None
    patience = parser.get_patience()
    patience_count = patience

    for epoch in range(max_num_epochs):
        train_loss = train_one_epoch(model, proj_head, criterion, train_loader, optimizer, device, epoch)
        print(f"Epoch {epoch+1}, Train Contrastive Loss: {train_loss:.6f}")

        # Simple early stopping on train loss (no validation here)
        if train_loss < best_loss:
            best_loss = train_loss
            best_model_weights = copy.deepcopy((model.state_dict(), proj_head.state_dict()))
            patience_count = patience
        else:
            patience_count -= 1
            if patience_count == 0:
                print("Early stopping triggered.")
                break

    print("Finished contrastive training.")
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights[0])
        proj_head.load_state_dict(best_model_weights[1])

    # Save the model and projection head if needed
    torch.save({
        'backbone': model.state_dict(),
        'projection_head': proj_head.state_dict()
    }, os.path.join(checkpoint_path, "best_model_contrastive.pth"))

if __name__ == "__main__":
    main()
