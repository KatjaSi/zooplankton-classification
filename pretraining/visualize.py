import torch
import matplotlib.pyplot as plt
import ipdb
import os
import numpy as np
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import ViTMAEForPreTraining, ViTMAEConfig

import ipdb

def visualize(pixel_values, model, save_path, mean, std):
    outputs = model(pixel_values)
    y = model.unpatchify(outputs.logits)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()
    
    mask = outputs.mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.config.patch_size**2 ) 
    mask = model.unpatchify(mask) 
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    x = torch.einsum('nchw->nhwc', pixel_values)
    
    x = x.detach().cpu()
    im_masked = x * (1 - mask)
    
    im_paste = x * (1 - mask) + y * mask
    for i in range(x.shape[0]):
        original_img = torch.clip((x[i] * std + mean) * 255, 0, 255).int().squeeze(-1)
        masked_img = torch.clip((im_masked[i] * std + mean) * 255, 0, 255).int().squeeze(-1)
        reconstruction_img = torch.clip((y[i] * std + mean) * 255, 0, 255).int().squeeze(-1)
        paste_img = torch.clip((im_paste[i] * std + mean) * 255, 0, 255).int().squeeze(-1)

        os.makedirs(save_path, exist_ok=True)
        plt.imsave(os.path.join(save_path, f'original_{i}.png'), original_img.numpy(), cmap='gray')
        plt.imsave(os.path.join(save_path, f'masked_{i}.png'), masked_img.numpy(), cmap='gray')
        plt.imsave(os.path.join(save_path, f'reconstruction_{i}.png'), reconstruction_img.numpy(), cmap='gray')
        plt.imsave(os.path.join(save_path, f'reconstruction_visible_{i}.png'), paste_img.numpy(), cmap='gray')

    print(f"Saved {x.shape[0]} images to {save_path}")


def visualize_2(pixel_values, model, save_path, mean, std, convert_to_rgb=False):
    outputs = model(pixel_values)
    y = model.unpatchify(outputs.logits)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()
    
    mask = outputs.mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.config.patch_size**2)  # (N, H*W, p*p)
    mask = model.unpatchify(mask)
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    x = torch.einsum('nchw->nhwc', pixel_values).detach().cpu()
    im_masked = x * (1 - mask)

    im_paste = x * (1 - mask) + y * mask
    
    os.makedirs(save_path, exist_ok=True)
    
    for i in range(x.shape[0]):
        original_img = torch.clip((x[i] * std + mean) * 255, 0, 255).squeeze(-1).numpy().astype(np.uint8)
        masked_img = torch.clip((im_masked[i] * std + mean) * 255, 0, 255).squeeze(-1).numpy().astype(np.uint8)
        reconstruction_img = torch.clip((y[i] * std + mean) * 255, 0, 255).squeeze(-1).numpy().astype(np.uint8)
        paste_img = torch.clip((im_paste[i] * std + mean) * 255, 0, 255).squeeze(-1).numpy().astype(np.uint8)

        if convert_to_rgb:
            original_img = np.stack([original_img] * 3, axis=-1)
            masked_img = np.stack([masked_img] * 3, axis=-1)
            reconstruction_img = np.stack([reconstruction_img] * 3, axis=-1)
            paste_img = np.stack([paste_img] * 3, axis=-1)
        

        plt.imsave(os.path.join(save_path, f'original_{i}.png'), original_img, cmap='gray')
        plt.imsave(os.path.join(save_path, f'masked_{i}.png'), masked_img, cmap='gray')
        plt.imsave(os.path.join(save_path, f'reconstruction_{i}.png'), reconstruction_img, cmap='gray')
        plt.imsave(os.path.join(save_path, f'reconstruction_visible_{i}.png'), paste_img, cmap='gray')

    print(f"Saved {x.shape[0]} images to {save_path}")

if __name__ == "__main__":
    config = ViTMAEConfig(norm_pix_loss = False,  
                        mask_ratio = 0.75,
                        num_channels = 1
                    )
    device = torch.device("cpu")
    checkpoint = torch.load("checkpoints/checkpoints_mae_base/checkpoints/checkpoint_130000.pth")
    model_state_dict = checkpoint['model_state_dict']
    model = ViTMAEForPreTraining(config).to(device) # use the same parameters as mael-large
    model.load_state_dict(model_state_dict)
    mean = 0.9044
    std = 0.1485
    save_path = 'temp'
    img_path = 'datasets/ZooScan77_small/val/COP10_Eucalanidae/46963283.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    transform = A.Compose([
        A.Resize(224, 224, p=1), #ResizeAndPad(224, fill=255),
        A.Normalize(mean=mean, std=std),
        ToTensorV2() ])
    transformed = transform(image=img)
    pixel_values = transformed['image'].unsqueeze(0)#.permute(0,2,3,1)
    
    
    visualize_2(pixel_values, model, save_path, mean, std, convert_to_rgb=False)