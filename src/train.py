import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import torchvision.transforms.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import shutil
import os
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from PIL import Image


def plot_confusion_matrix(cm, class_names, epoch, save_path):
    plt.figure(figsize=(60, 60))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='coolwarm',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Fraction'})

    plt.title(f'Confusion Matrix at Epoch {epoch + 1}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"{save_path}/epoch_{epoch + 1}_confusion_matrix.png")
    plt.close()

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

def get_model(model_config, num_classes):
    if model_config['name'] == 'vgg16':
        model = torchvision.models.vgg16(weights=model_config['pretrained'])
        model.classifier[-1] = nn.Linear(in_features=4096, out_features=num_classes)
    elif model_config['name'] == 'resnet50':
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT if model_config['pretrained'] else None)
        model.fc = nn.Linear(in_features=2048, out_features=num_classes)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return model

if __name__ == "__main__":

    with open('src/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda")

    graph_path = 'graphs'
    if os.path.exists(graph_path):
        shutil.rmtree(graph_path)
    os.makedirs(graph_path)

    in_channels = 1
    num_classes = config['num_classes']
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    report_config = config['reports']


    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda img: resize_and_pad(img)),  
        transforms.ToTensor(),
       # transforms.Lambda(apply_clahe),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Lambda(lambda img: resize_and_pad(img)),  
        transforms.ToTensor(),
       # transforms.Lambda(apply_clahe),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = ImageFolder(root='datasets/ZooScan77_small/train', transform=train_transform)
    val_dataset = ImageFolder(root='datasets/ZooScan77_small/val', transform=val_transform)
    test_dataset  = ImageFolder(root='datasets/ZooScan77_small/test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    model = get_model(config['model'], num_classes).to(device)

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

        if report_config['enable'] and (epoch + 1) % report_config['frequency'] == 0:
            if report_config['types']['confusion_matrix']:
                cm = metrics.confusion_matrix(val_true, val_pred)
                class_names = val_dataset.classes
                plot_confusion_matrix(cm, class_names, epoch, graph_path)
        
        print(f"Epoch {epoch+1}, \
            Valid loss: {val_loss*1.0/count}, \
            valid accuracy: {accuracy:.6f},\
            balanced valid accuracy: {balanced_accuracy:.6f}\n")


    print('Finished Training')