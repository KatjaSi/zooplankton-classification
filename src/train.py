import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image
import sklearn.metrics as metrics


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
    learning_rate = 5e-4
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
    val_dataset = ImageFolder(root='datasets/ZooScan77_small/val', transform=transform)
    test_dataset  = ImageFolder(root='datasets/ZooScan77_small/test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    model = torchvision.models.vgg16(weights=True)
    model.classifier[-1] = nn.Linear(in_features=4096, out_features=77)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):  
        running_loss = 0.0
        count = 0.0
        model.train()

        train_pred = []
        train_true = []
        for data, labels in (train_loader):
            batch_size = len(labels)
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            preds = outputs.max(dim=1)[1]

            count += batch_size
            running_loss += loss.item() * batch_size

            train_true.append(labels.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        accuracy = metrics.accuracy_score(train_true, train_pred)
        balanced_accuracy = metrics.balanced_accuracy_score(train_true, train_pred)
        print(f"Epoch {epoch+1}, \
            Train loss: {running_loss*1.0/count}, \
            train accuracy: {accuracy:.6f},\
            balanced train accuracy: {balanced_accuracy:.6f}")

        
        ### validation ###
        val_loss = 0.0
        count = 0.0
        model.eval()
        val_pred = []
        val_true = []

        for data, labels in (val_loader):
            batch_size = len(labels)
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            preds = outputs.max(dim=1)[1]
            count += batch_size
            val_loss += loss.item() * batch_size
            val_true.append(labels.cpu().numpy())
            val_pred.append(preds.detach().cpu().numpy())

        val_true = np.concatenate(val_true)
        val_pred = np.concatenate(val_pred)
        accuracy = metrics.accuracy_score(val_true, val_pred)
        balanced_accuracy = metrics.balanced_accuracy_score(val_true, val_pred)
        print(f"Epoch {epoch+1}, \
            Valid loss: {val_loss*1.0/count}, \
            valid accuracy: {accuracy:.6f},\
            balanced valid accuracy: {balanced_accuracy:.6f}\n")


    print('Finished Training')