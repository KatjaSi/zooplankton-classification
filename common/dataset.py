import numpy as np
from torchvision.datasets import ImageFolder
from PIL import Image
import ipdb

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


def grayscale_loader(path: str):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("L")