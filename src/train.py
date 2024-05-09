import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def apply_clahe(img):
    img_np = img.numpy().squeeze() * 255.0  
    img_np = img_np.astype('uint8')  
    clahe = cv2.createCLAHE()
    img_clahe = clahe.apply(img_np)
    img_clahe = torch.from_numpy(img_clahe).float() / 255.0
    if img.dim() > 2 and img.size(0) == 1:
        img_clahe = img_clahe.unsqueeze(0)

    return img_clahe

def resize_and_pad(img, size=224, fill=0, padding_mode='constant'):
    aspect_ratio = img.width / img.height
    if img.width > img.height:
        new_width = size
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = size
        new_width = int(new_height * aspect_ratio)

    # Resize image
    img = img.resize((new_width, new_height), Image.BILINEAR)

    # Calculate padding to make the image square
    padding_left = (size - new_width) // 2
    padding_right = size - new_width - padding_left
    padding_top = (size - new_height) // 2
    padding_bottom = size - new_height - padding_top

    # Pad the resized image to make it square
    img = F.pad(img, padding=(padding_left, padding_top, padding_right, padding_bottom), fill=fill, padding_mode=padding_mode)
    return img




if __name__ == "__main__":

    device = torch.device("cuda")

    in_channels = 1
    num_classes = 77
    learing_rate = 5e-4
    batch_size = 64
    num_epochs = 100

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda img: resize_and_pad(img)),  
        transforms.ToTensor(),
       # transforms.Lambda(apply_clahe),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    train_dataset = ImageFolder(root='datasets/ZooScan77_small/train', transform=transform)
    val_dataset = ImageFolder(root='datasets/ZooScan77_small/val')
    test_dataset  = ImageFolder(root='datasets/ZooScan77_small/test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    model = torchvision.models.vgg16(weights=True)
    model.classifier[-1] = nn.Linear(in_features=4096, out_features=77)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=learing_rate)
    
    for epoch in range(num_epochs): 
        
        running_loss = 0.0
        count = 0.0
        model.train()
        for data, labels in (train_loader):
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Train loss: {loss.item()}")


    print('Finished Training')