import numpy as np
import torch

def save_checkpoint(model, optimizer, epoch, file_path, train_losses, val_losses, val_accs):
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
        'val_accs': val_accs
    }
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved at {file_path}")