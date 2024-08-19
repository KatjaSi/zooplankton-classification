import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
import ipdb
from sklearn.decomposition import PCA
from copy import deepcopy

device = torch.device("cuda")
# Load pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])  # Remove the last classification layer
model.eval()  # Set model to evaluation mode
model = model.to(device) 
# Define image transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to preprocess images and extract features using ResNet-18
def load_and_extract_features(folder_path, batch_size=32, output_file='features2.npy'):
    features = []
    filenames = []
    batch_images = []
    batch_filenames = []
    i = 0
    for root, _, files in os.walk(folder_path):
        print(root)
        i += 1
        print(i)
        for filename in files:
            if filename.endswith(('.png', '.jpg')):
                img_path = os.path.join(root, filename)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                img = transform(img)  # Apply transformations
                batch_images.append(img)
                batch_filenames.append(os.path.relpath(img_path, folder_path))
                
                # Process the batch when it reaches the batch size
                if len(batch_images) == batch_size:
                    batch_images_tensor = torch.stack(batch_images).to(device)  # Stack and move to device
                    with torch.no_grad():
                        batch_features = model(batch_images_tensor)  # Extract features
                    features.append(batch_features.cpu().squeeze().numpy())  # Move to CPU and store
                    filenames.extend(batch_filenames)
                    batch_images, batch_filenames = [], []  # Reset the batch

                    # Save incrementally to avoid memory issues
                    save_features(features, filenames, output_file)
                    features = []  # Reset features after saving
                    filenames = []  # Reset filenames after saving
        #if i == 10:
         #   break

    # Process any remaining images that didn't fill a batch
    if batch_images:
        batch_images_tensor = torch.stack(batch_images).to(device)
        with torch.no_grad():
            batch_features = model(batch_images_tensor)
        features.append(batch_features.cpu().squeeze().numpy())
        filenames.extend(batch_filenames)
        
        # Save the final batch
        save_features(features, filenames, output_file)

def save_features(features, filenames, output_file):
    features = np.concatenate(features, axis=0)
    with open(output_file, 'ab') as f:
        np.save(f, features)
    with open(output_file.replace('.npy', '_filenames2.txt'), 'a') as f:
        for filename in filenames:
            f.write(f"{filename}\n")

# Load and preprocess images, then extract features
#folder_path = 'datasets/ZooScan77_good_quality'
folder_path = 'datasets/ZooScan77_small/train'

def load_features(output_file='features2.npy'):
    features = []
    with open(output_file, 'rb') as f:
        while True:
            try:
                features.append(np.load(f, allow_pickle=True))
            except EOFError:
                break
    return np.concatenate(features, axis=0)

def load_filenames(filename_file='features_filenames2.txt'):
    with open(filename_file, 'r') as f:
        filenames = f.read().splitlines()
    return filenames

load_and_extract_features(folder_path, batch_size=32)

#features = load_features('features2.npy')
#filenames = load_filenames('features_filenames2.txt')
