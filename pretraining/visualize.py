import torch
import matplotlib.pyplot as plt
import ipdb
import os


def visualize(pixel_values, model, save_path, mean, std):
    outputs = model(pixel_values)
    y = model.unpatchify(outputs.logits)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()
    
    mask = outputs.mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.config.patch_size**2 )  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    x = torch.einsum('nchw->nhwc', pixel_values)
    
    x = x.detach().cpu()
    im_masked = x * (1 - mask)
    
    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask
    for i in range(x.shape[0]):
        # Original image
        original_img = torch.clip((x[i] * std + mean) * 255, 0, 255).int().squeeze(-1)
        masked_img = torch.clip((im_masked[i] * std + mean) * 255, 0, 255).int().squeeze(-1)
        reconstruction_img = torch.clip((y[i] * std + mean) * 255, 0, 255).int().squeeze(-1)
        paste_img = torch.clip((im_paste[i] * std + mean) * 255, 0, 255).int().squeeze(-1)

        os.makedirs(save_path, exist_ok=True)
        # Save each image as an individual file
        plt.imsave(os.path.join(save_path, f'original_{i}.png'), original_img.numpy(), cmap='gray')
        plt.imsave(os.path.join(save_path, f'masked_{i}.png'), masked_img.numpy(), cmap='gray')
        plt.imsave(os.path.join(save_path, f'reconstruction_{i}.png'), reconstruction_img.numpy(), cmap='gray')
        plt.imsave(os.path.join(save_path, f'reconstruction_visible_{i}.png'), paste_img.numpy(), cmap='gray')

    print(f"Saved {x.shape[0]} images to {save_path}")