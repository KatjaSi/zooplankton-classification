import numpy as np
import cv2
import albumentations as A
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from torch.utils.data import WeightedRandomSampler
from common.transforms import ResizeAndPad
from common.dataset import ZooScanImageFolder

def get_default_train_transform(mean, std):
    train_transform = A.Compose([
            ResizeAndPad(224, fill=255),
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.VerticalFlip(p=0.5),
               # A.Rotate(limit=(-90, 90), border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=1)
            ], p=1),
            A.Normalize(mean=mean, std=std),
            ToTensorV2() ])
    return train_transform



def get_train_transform_with_random_zoom(mean, std):
    train_transform = A.Compose([
        A.Resize(224, 224, p=1),
        A.RandomScale(scale_limit=(1, 1.05), p=0.5),
        A.CenterCrop(224, 224, p=1.0),
        
        A.OneOf([ 
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ], p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    return train_transform

def get_default_val_transform(mean, std):
    val_transform = A.Compose([
        A.Resize(224, 224, p=1), #ResizeAndPad(224, fill=255),
        A.Normalize(mean=mean, std=std),
        ToTensorV2() ])
    return val_transform

def get_dataloader(root, transform, batch_size, num_workers):
    dataset = ZooScanImageFolder(root=root, transform=transform, grayscale=True)
    labels = np.array(dataset.targets)
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    sampler = WeightedRandomSampler(weights, len(weights))
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    return dataloader
