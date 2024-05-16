import cv2
import torch
import torchvision.transforms.functional as F

from PIL import Image

def apply_clahe(img):
    img_np = img.numpy().squeeze() * 255.0  
    img_np = img_np.astype('uint8')  
    clahe = cv2.createCLAHE()
    img_clahe = clahe.apply(img_np)
    img_clahe = torch.from_numpy(img_clahe).float() / 255.0
    if img.dim() > 2 and img.size(0) == 1:
        img_clahe = img_clahe.unsqueeze(0)

    return img_clahe

def resize_and_pad(img, size=224, fill=1, padding_mode='constant'):
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