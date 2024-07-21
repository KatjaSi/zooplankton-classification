import torch
import matplotlib.pyplot as plt


def visualize(pixel_values, model, save_path, mean, std):
    # Forward pass
    outputs = model(pixel_values)
    y = model.unpatchify(outputs.logits)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()
    
    # Visualize the mask
    mask = outputs.mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.config.patch_size**2 * 3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    x = torch.einsum('nchw->nhwc', pixel_values)
    
    # Masked image
    x = x.detach().cpu()
    im_masked = x * (1 - mask)
    
    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask
    
    # Make the plt figure larger
    plt.rcParams['figure.figsize'] = [48, 48]
    
    # Create subplots and save the combined image
    _, axs = plt.subplots(1, 4)
    axs[0].imshow(torch.clip((x[0]* std + mean) * 255, 0, 255).int())
    axs[0].set_title("original")
    axs[0].axis('off')
    
    axs[1].imshow(torch.clip((im_masked[0] * std + mean) * 255, 0, 255).int())
    axs[1].set_title("masked")
    axs[1].axis('off')
    
    axs[2].imshow(torch.clip((y[0] * std + mean) * 255, 0, 255).int())
    axs[2].set_title("reconstruction")
    axs[2].axis('off')
    
    axs[3].imshow(torch.clip((im_paste[0] * std + mean) * 255, 0, 255).int())
    axs[3].set_title("reconstruction + visible")
    axs[3].axis('off')
    
    plt.savefig(save_path)
    plt.close()