import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from PIL import Image
import ipdb


class ZooScanImageFolderTorch(ImageFolder):
    def __init__(self, root, transform=None, grayscale=False):
    
        if grayscale:
            super(ZooScanImageFolderTorch, self).__init__(root, transform=None, loader=grayscale_loader)
        else:
            super(ZooScanImageFolderTorch, self).__init__(root, transform=None)
        
        self.torch_transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        
        if self.torch_transform is not None:
            sample = self.torch_transform(sample)
        
        return sample, target

class ZooScanImageFolder(ImageFolder):
    def __init__(self, root, transform=None, grayscale=False):
        if not grayscale:
            super(ZooScanImageFolder, self).__init__(root, transform=None)
        else:
            super(ZooScanImageFolder, self).__init__(root, transform=None, loader=grayscale_loader)
        self.albumentations_transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = np.array(sample)  # Convert PIL image to numpy array # to avoid negative stride error
        if self.albumentations_transform is not None:
            sample = self.albumentations_transform(image=sample)['image']
        return sample, target


class ImagenetDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset  # Original dataset (e.g., Hugging Face dataset)
        self.transform = transform 

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example['image']
        image = image.convert('L')
        image = np.array(image)  
        if self.transform:
            image = self.transform(image=image)['image'] 
        label = example['label']
        return image, label

def grayscale_loader(path: str):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("L")