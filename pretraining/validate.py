import torch

def validate(model, dataloader, device):
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            loss = outputs.loss
            total_val_loss += loss.item()
    return total_val_loss / len(dataloader)