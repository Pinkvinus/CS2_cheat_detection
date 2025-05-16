import numpy as np
import torch
import os

def save_checkpoint(model, optimizer, epoch, file_path, train_losses, val_losses, val_accs, recalls, precisions, rocs):
    """
    Save the training state to a checkpoint file.

    Args:
        model: PyTorch model to save.
        optimizer: Optimizer used in training.
        epoch: Current epoch.
        file_path: Path to save the checkpoint.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'recalls': recalls,
        'precisions': precisions,
        'rocs': rocs
    }
    file_path_file = os.path.join(file_path, f"model_1024_epoch_{epoch}.pth")
    torch.save(checkpoint, file_path_file)
    print(f"Checkpoint saved at {file_path}")