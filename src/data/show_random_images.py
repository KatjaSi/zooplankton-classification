import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
from PIL import Image
import torchvision.transforms.functional as F

def apply_clahe(img):
    # Convert PyTorch tensor to numpy array and scale to [0, 255]
    img_np = img.numpy().squeeze() * 255.0  # Squeeze in case there's an extra channel dimension
    img_np = img_np.astype('uint8')  # Convert to unsigned 8-bit integer format

    # Create a CLAHE object (with default parameters here)
    clahe = cv2.createCLAHE()

    # Apply CLAHE
    img_clahe = clahe.apply(img_np)

    # Convert back to float and scale to [0, 1]
    img_clahe = torch.from_numpy(img_clahe).float() / 255.0

    # If the original tensor had more than one channel, you might need to unsqueeze to add a channel dimension back
    if img.dim() > 2 and img.size(0) == 1:
        img_clahe = img_clahe.unsqueeze(0)

    return img_clahe

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

def show_random_images(num_images):
    
    mean_imagenet = [0.485, 0.456, 0.406]
    std_imagenet = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
       # transforms.RandomResizedCrop(size=(224, 224)),
        transforms.Lambda(lambda img: resize_and_pad(img)),  # Resize the image to 50x50
        transforms.Pad(),
       # transforms.Resize((224,224)),
        transforms.Grayscale(num_output_channels=1),  # Ensure it's single-channel
        transforms.ToTensor(),
        transforms.Lambda(apply_clahe),
        transforms.Grayscale(num_output_channels=3),
        transforms.Normalize(mean=[0.5], std=[0.5])
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
    parser.add_argument('--num_images', type=int, default=8)
    args = parser.parse_args()
    show_random_images(args.num_images)
