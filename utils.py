import torch
import os

def save_checkpoint(model, filepath, save_parallel=False):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if save_parallel:
        torch.save(model.module.state_dict(), filepath)
    else:
        torch.save(model.state_dict(), filepath)

def load_checkpoint(model, filepath, device):
    """Load model checkpoint"""
    state_dict = torch.load(filepath, map_location=device)
    model.load_state_dict(state_dict)
    return model

def print_model_summary(model):
    """Print model parameter summary"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}') 