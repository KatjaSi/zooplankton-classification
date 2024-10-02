import os
import torch
import ipdb

def save_checkpoint(model, optimizer, scheduler, epoch, step, best_loss, filepath="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': best_loss
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at step {step} of epoch {epoch + 1}.")

    

def save_checkpoint_old(model, optimizer, scheduler, epoch, effective_step_count, best_loss, total_loss, filename="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'effective_step_count': effective_step_count,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': best_loss,
        'total_loss': total_loss
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at effective step {effective_step_count} of epoch {epoch + 1}.")

def load_checkpoint_old(model, optimizer, scheduler, filename="checkpoint.pth"):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        effective_step_count = checkpoint['effective_step_count']
        best_loss = checkpoint['best_loss']
        total_loss = checkpoint.get('total_loss', 0.0)
        print(f"Checkpoint loaded from effective step {effective_step_count} of epoch {epoch + 1}.")
        return epoch, effective_step_count, best_loss, total_loss
    else:
        print("No checkpoint found. Starting from scratch.")
        return 0, 0, float('inf'), 0.0