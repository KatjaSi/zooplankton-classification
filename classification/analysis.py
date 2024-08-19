import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from itertools import combinations
import numpy as np
import os

model = models.resnet50(pretrained=True)
model.eval()


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Function to extract features from an image
def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(image)
    return features

# Function to calculate mean feature distance for a list of images
def calculate_mean_feature_distance(folder_path):
    # List all files in the folder
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # Extract features for all images
    feature_vectors = [extract_features(img_path) for img_path in image_paths]

    distances = []
    for f1, f2 in combinations(feature_vectors, 2):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        dist = cos(f1, f2) #torch.nn.functional.pairwise_distance(f1, f2)
        distances.append(dist.item())
    
    # Calculate mean distance
    mean_distance = np.mean(distances)
    
    return mean_distance

folder_path = 'datasets/ZooScan77_small/train'
image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
    
for ip in image_paths:
    mean_feature_distance = calculate_mean_feature_distance(ip)
    print(f"Mean Feature Distance for the  { os.path.basename(ip)}: {mean_feature_distance}")
