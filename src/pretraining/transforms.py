import cv2
import torch
import torchvision.transforms.functional as F
import albumentations as A
from PIL import Image
import numpy as np


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
