import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse

def show_random_images(num_images):
    # Define transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 50x50
        transforms.ToTensor()
    ])

    train_dataset = ImageFolder(root='datasets/ZooScan77/train', transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=num_images, shuffle=True)

    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    
    num_rows = (num_images + 3) // 4 

    # Plotting
    plt.figure(figsize=(10, 3 * num_rows))  
    for idx in range(num_images):
        ax = plt.subplot(num_rows, 4, idx + 1)
        img = images[idx].numpy()
        img = np.transpose(img, (1, 2, 0))
        plt.imshow(img)
        plt.title(f"{train_dataset.classes[labels[idx]]}")
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('num_images', type=int)
    args = parser.parse_args()
    show_random_images(args.num_images)
