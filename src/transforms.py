import cv2
import torch
import torchvision.transforms.functional as F
import albumentations as A
from PIL import Image
import numpy as np

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
        new_height = max(int(new_width / aspect_ratio), 1)
    else:
        new_height = size
        new_width = max(int(new_height * aspect_ratio), 1)

    img = img.resize((new_width, new_height), Image.BILINEAR)
   

    # Calculate padding to make the image square
    padding_left = (size - new_width) // 2
    padding_right = size - new_width - padding_left
    padding_top = (size - new_height) // 2
    padding_bottom = size - new_height - padding_top

    img = F.pad(img, padding=(padding_left, padding_top, padding_right, padding_bottom), fill=fill, padding_mode=padding_mode)
    return img

class ResizeAndPad(A.ImageOnlyTransform):
    def __init__(self, size=224, fill=0, always_apply=False, p=1.0):
        super(ResizeAndPad, self).__init__(always_apply, p)
        self.size = size
        self.fill = fill

    def apply(self, img, **params):
        return self.resize_and_pad(img, self.size, self.fill)

    def resize_and_pad(self, img, size, fill):
        aspect_ratio = img.shape[1] / img.shape[0]
        if img.shape[1] > img.shape[0]:
            new_width = size
            new_height = max(int(new_width / aspect_ratio), 1)
        else:
            new_height = size
            new_width = max(int(new_height * aspect_ratio), 1)

        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        padding_left = (size - new_width) // 2
        padding_right = size - new_width - padding_left
        padding_top = (size - new_height) // 2
        padding_bottom = size - new_height - padding_top

        img = cv2.copyMakeBorder(img, padding_top, padding_bottom, padding_left, padding_right, 
                                 borderType=cv2.BORDER_CONSTANT, value=[fill]*3)
        return img

class EnsurePositiveStrides(A.ImageOnlyTransform):

    def __init__(self, always_apply=False, p=1.0):
        super(EnsurePositiveStrides, self).__init__(always_apply, p)

    def apply(self, img, **params):
        # Make a copy of the image to ensure positive strides
        return img.copy()