import torch
import torch.nn as nn
import torch.optim as optim
#import torch.nn.functional as F
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def resize_and_pad(img, size=224, fill=0, padding_mode='constant'):
    # Calculate the aspect ratio and determine the scaling factor
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_channels = 1
    num_classes = 77
    learing_rate = 1e-3
    batch_size = 16
    num_epochs = 1

    transform = transforms.Compose([
        transforms.Lambda(lambda img: resize_and_pad(img)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    train_dataset = ImageFolder(root='datasets/ZooScan77/train', transform=transform)
    val_dataset = ImageFolder(root='datasets/ZooScan77/val')
    test_dataset  = ImageFolder(root='datasets/ZooScan77/test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    model = torchvision.models.vgg16(weights=None)
    model.classifier[-1] = nn.Linear(in_features=4096, out_features=77)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=learing_rate)


    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    image = images[1].detach().numpy()


    
    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0
        for data in train_loader:

            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            print(loss.item())


    print('Finished Training')