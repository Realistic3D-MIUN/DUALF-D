from compressai.datasets import ImageFolder
from compressai.losses import RateDistortionLoss
from compressai.optimizers import net_aux_optimizer
from compressai.zoo import image_models
from compressai.entropy_models import EntropyBottleneck
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *
from dataloader import *
from loss import *
from parameters import *
import os
import multiprocessing
import torch
from model import VAE
from dataloader import LightFieldDataset
from loss import dual_hyperprior_loss, calculate_psnr
from parameters import *
import os
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from compressai.datasets import ImageFolder
from compressai.losses import RateDistortionLoss
from compressai.optimizers import net_aux_optimizer
from compressai.zoo import image_models
from compressai.entropy_models import GaussianConditional
import torch
from model import VAE
from dataloader import LightFieldDataset
from loss import dual_hyperprior_loss, calculate_psnr
from parameters import *
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import math
import os
import numpy as np
from utils import save_checkpoint, load_checkpoint, print_model_summary
from loss import DualHyperpriorLoss

def save_checkpoint(model, optimizer, aux_optimizer, epoch, metrics, save_path, is_best=False):
    """Save model checkpoint and metrics"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'aux_optimizer_state_dict': aux_optimizer.state_dict(),
        'metrics': metrics
    }
    
    # Save the epoch-specific checkpoint
    epoch_path = f"{save_path}vae_model_epoch_{epoch+1}.pth"
    torch.save(checkpoint, epoch_path)
    print(f"Saved checkpoint for epoch {epoch+1} at: {epoch_path}")
    
    # Save as latest checkpoint
    latest_path = f"{save_path}vae_model_latest.pth"
    torch.save(checkpoint, latest_path)
    print(f"Saved latest checkpoint at: {latest_path}")
    
    # Save best model if this is the best performance
    if is_best:
        best_path = f"{save_path}vae_model_best.pth"
        torch.save(checkpoint, best_path)
        print(f"Saved best model at: {best_path}")

# Set up parameters and model
torch.manual_seed(42)
writer = SummaryWriter(log_dir='./runs/compression_experiment')
train_dataset = LightFieldDataset(train_dataset_path)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
model = VAE(latent_channels).to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    model_without_ddp = model.module
else:
    model_without_ddp = model  # Use this when accessing model attributes

# Before training loop, modify how we add the model to tensorboard
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out["x_hat"]  # Only return the reconstructed image

# Create wrapped model for visualization
wrapped_model = ModelWrapper(model)
dummy_input = torch.randn(1, 3, param_height, param_width).to(device)
writer.add_graph(wrapped_model, dummy_input)

# Load model state if resuming training
if resume_training:
    # Load the state dict first to see the correct sizes
    state_dict = torch.load(resume_location, map_location=device)
    
    # Get the correct sizes from the saved model
    spatial_cdf_size = state_dict['encoder.spatial_hyperprior.entropy_bottleneck._quantized_cdf'].size(1)
    angular_cdf_size = state_dict['encoder.angular_hyperprior.entropy_bottleneck._quantized_cdf'].size(1)
    
    # Initialize entropy bottleneck parameters with correct sizes
    model.encoder.spatial_hyperprior.entropy_bottleneck._offset = torch.zeros(64, device=device)
    model.encoder.spatial_hyperprior.entropy_bottleneck._quantized_cdf = torch.zeros(64, spatial_cdf_size, device=device)
    model.encoder.spatial_hyperprior.entropy_bottleneck._cdf_length = torch.zeros(64, dtype=torch.int32, device=device)
    
    model.encoder.angular_hyperprior.entropy_bottleneck._offset = torch.zeros(64, device=device)
    model.encoder.angular_hyperprior.entropy_bottleneck._quantized_cdf = torch.zeros(64, angular_cdf_size, device=device)
    model.encoder.angular_hyperprior.entropy_bottleneck._cdf_length = torch.zeros(64, dtype=torch.int32, device=device)
    
    # Now load the state dict
    model.load_state_dict(state_dict)
    print(f"Resumed training from {resume_location}")
else:
    print("Training from scratch")


# Confirm all necessary parameters are trainable
#print("### Checking Trainable Parameters ###")
#for name, param in model.named_parameters():
    #if param.requires_grad:
        #print(f"Trainable: {name}")
    #else:
        #print(f"Non-Trainable: {name}")


# Modify parameter separation
parameters = {
    "main": [],
    "aux": []
}

print("\nSeparating parameters:")
for name, param in model.named_parameters():
    if name.endswith(".quantiles"):  # Only quantiles go to aux optimizer
        parameters["aux"].append(param)
        print(f"Aux parameter: {name}")
    else:
        parameters["main"].append(param)
        print(f"Main parameter: {name}")

print(f"\nMain parameters count: {len(parameters['main'])}")
print(f"Auxiliary parameters count: {len(parameters['aux'])}")
print("\nAuxiliary parameters:")
for param in parameters["aux"]:
    print(f"Shape: {param.shape}, Requires grad: {param.requires_grad}")

# Initialize optimizers with correct parameter groups
optimizer = torch.optim.Adam(parameters["main"], lr=lr)
aux_optimizer = torch.optim.Adam(parameters["aux"], lr=aux_lr)

# Add after optimizer initialization
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=5, 
    verbose=True
)

def compute_metrics(out, x):
    """Compute detailed metrics for monitoring"""
    x_hat = out["x_hat"]
    likelihoods = out["likelihoods"]
    
    # Compute MSE and PSNR with sum reduction
    N, _, H, W = x.size()
    num_pixels = N * H * W
    mse = F.mse_loss(x_hat, x, reduction='sum') / num_pixels
    psnr = -10 * torch.log10(mse)
    
    # Compute bpp for each component
    bpp_y_s = -torch.log2(likelihoods["y_s"]).sum().item() / num_pixels
    bpp_y_a = -torch.log2(likelihoods["y_a"]).sum().item() / num_pixels
    bpp_z_s = -torch.log2(likelihoods["z_s"]).sum().item() / num_pixels
    bpp_z_a = -torch.log2(likelihoods["z_a"]).sum().item() / num_pixels
    
    total_bpp = bpp_y_s + bpp_y_a + bpp_z_s + bpp_z_a
    
    return {
        'mse': mse.item(),
        'psnr': psnr.item(),
        'bpp_total': total_bpp,
        'bpp_y_spatial': bpp_y_s,
        'bpp_y_angular': bpp_y_a,
        'bpp_z_spatial': bpp_z_s,
        'bpp_z_angular': bpp_z_a
    }

# Split dataset into train and validation
def create_train_val_loaders(dataset, batch_size, val_split=0.1):
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def debug_gradients(model, name):
    """Debug gradients of entropy bottleneck parameters"""
    print(f"\n=== Debugging {name} ===")
    for n, p in model.named_parameters():
        if 'entropy_bottleneck' in n:
            if p.grad is not None:
                print(f"{n}:")
                print(f"  grad mean: {p.grad.mean().item():.6f}")
                print(f"  grad std: {p.grad.std().item():.6f}")
                print(f"  param mean: {p.mean().item():.6f}")
                print(f"  param std: {p.std().item():.6f}")
            else:
                print(f"{n}: No gradients!")

def debug_hyperprior(model, name, x):
    """Debug hyperprior network outputs"""
    with torch.no_grad():
        # Get hyperprior outputs
        if hasattr(model, 'module'):
            spatial_hp = model.module.encoder.spatial_hyperprior
            angular_hp = model.module.encoder.angular_hyperprior
        else:
            spatial_hp = model.encoder.spatial_hyperprior
            angular_hp = model.encoder.angular_hyperprior

        # Monitor entropy bottleneck states
        print(f"\n=== {name} Hyperprior States ===")
        print("Spatial Hyperprior:")
        print(f"  entropy bottleneck training: {spatial_hp.entropy_bottleneck.training}")
        print(f"  entropy bottleneck parameters:")
        for name, param in spatial_hp.entropy_bottleneck.named_parameters():
            if param.requires_grad:
                print(f"    {name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}")
        
        print("\nAngular Hyperprior:")
        print(f"  entropy bottleneck training: {angular_hp.entropy_bottleneck.training}")
        print(f"  entropy bottleneck parameters:")
        for name, param in angular_hp.entropy_bottleneck.named_parameters():
            if param.requires_grad:
                print(f"    {name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}")

        # Print entropy bottleneck device
        print(f"\nSpatial Entropy Bottleneck Device: {next(spatial_hp.entropy_bottleneck.parameters()).device}")
        print(f"Angular Entropy Bottleneck Device: {next(angular_hp.entropy_bottleneck.parameters()).device}")

def train_one_epoch(model, train_loader, optimizer, aux_optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    total_mse = 0
    total_bpp = 0
    total_aux_loss = 0
    total_psnr = 0
    total_mi_loss = 0
    num_batches = len(train_loader)

    for batch_idx, batch_patches in enumerate(train_loader):
        batch_patches = batch_patches.to(device)
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out = model(batch_patches)
        loss, mse_loss, bpp_loss, mi_loss = criterion(out, batch_patches)

        # Main loss backward and optimize
        loss.backward()
        optimizer.step()

        # Auxiliary loss for entropy bottleneck
        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        # Calculate PSNR properly
        mse = F.mse_loss(out["x_hat"], batch_patches, reduction='mean')
        psnr = -10 * torch.log10(mse) if mse > 0 else torch.tensor(100.0)

        # Update totals
        total_loss += loss.item()
        total_mse += mse_loss.item()
        total_bpp += bpp_loss.item()
        total_aux_loss += aux_loss.item()
        total_psnr += psnr.item()
        total_mi_loss += mi_loss.item()

        if batch_idx % 10 == 0:
            print(f"Train Batch: [{batch_idx}/{num_batches}]\t"
                  f"Loss: {loss.item():.6f}\t"
                  f"MSE: {mse_loss.item():.6f}\t"
                  f"BPP: {bpp_loss.item():.6f}\t"
                  f"PSNR: {psnr.item():.2f}\t"
                  f"MI Loss: {mi_loss.item():.6f}\t"
                  f"Aux Loss: {aux_loss.item():.6f}")

    # Calculate averages at the end of epoch
    avg_loss = total_loss / num_batches
    avg_mse = total_mse / num_batches
    avg_bpp = total_bpp / num_batches
    avg_psnr = total_psnr / num_batches
    avg_mi_loss = total_mi_loss / num_batches
    avg_aux_loss = total_aux_loss / num_batches

    # Print epoch summary
    print("\n=== Epoch Summary ===")
    print(f"Loss: {avg_loss:.6f}")
    print(f"MSE: {avg_mse:.6f}")
    print(f"BPP: {avg_bpp:.6f}")
    print(f"PSNR: {avg_psnr:.2f} dB")
    print(f"Mutual Information Loss: {avg_mi_loss:.6f}")
    print(f"Auxiliary Loss: {avg_aux_loss:.6f}")
    print("=" * 20)

    return avg_loss, avg_mse, avg_bpp, avg_psnr, avg_mi_loss, avg_aux_loss


def validate(model, val_loader, criterion, device):
    model.eval()
    metrics_sum = {'loss': 0, 'mse': 0, 'psnr': 0, 
                  'bpp_total': 0, 'bpp_y_spatial': 0, 'bpp_y_angular': 0,
                  'bpp_z_spatial': 0, 'bpp_z_angular': 0}
    
    with torch.no_grad():
        for batch_patches in val_loader:
            batch_patches = batch_patches.to(device)
            out = model(batch_patches)
            loss, _, _ = dual_hyperprior_loss(out, batch_patches)
            
            # Compute detailed metrics
            batch_metrics = compute_metrics(out, batch_patches)
            
            # Update metrics sums
            metrics_sum['loss'] += loss.item()
            for k, v in batch_metrics.items():
                metrics_sum[k] += v
    
    # Compute averages
    num_batches = len(val_loader)
    return {k: v / num_batches for k, v in metrics_sum.items()}

# Main training loop
def main():
    # Set up datasets and dataloaders
    train_dataset = LightFieldDataset(train_dataset_path)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if validation_split > 0:
        val_dataset = LightFieldDataset(test_dataset_path)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    else:
        val_loader = None

    print(f"Training samples: {len(train_dataset)}")
    if val_loader:
        print(f"Validation samples: {len(val_dataset)}")
    else:
        print("No validation split - using all data for training")
    
    # Set up model and move to device
    model = VAE(latent_channels).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # Initialize the loss function (criterion)
    criterion = DualHyperpriorLoss(lmbda=lambda_factor).to(device)

    # Load checkpoint if resuming
    start_epoch = 0
    if resume_training:
        print(f"Loading checkpoint from {resume_location}")
        checkpoint = torch.load(resume_location, map_location=device)
        
        # Get sizes from checkpoint
        spatial_cdf_size = checkpoint['encoder.spatial_hyperprior.entropy_bottleneck._quantized_cdf'].size(1)
        angular_cdf_size = checkpoint['encoder.angular_hyperprior.entropy_bottleneck._quantized_cdf'].size(1)
        
        # Initialize with correct sizes
        model.encoder.spatial_hyperprior.entropy_bottleneck._offset = torch.zeros(
            64, device=device)
        model.encoder.spatial_hyperprior.entropy_bottleneck._quantized_cdf = torch.zeros(
            64, spatial_cdf_size, device=device)
        model.encoder.spatial_hyperprior.entropy_bottleneck._cdf_length = torch.zeros(
            64, dtype=torch.int32, device=device)
        
        model.encoder.angular_hyperprior.entropy_bottleneck._offset = torch.zeros(
            64, device=device)
        model.encoder.angular_hyperprior.entropy_bottleneck._quantized_cdf = torch.zeros(
            64, angular_cdf_size, device=device)
        model.encoder.angular_hyperprior.entropy_bottleneck._cdf_length = torch.zeros(
            64, dtype=torch.int32, device=device)
        
        # Now load the state dict
        model.load_state_dict(checkpoint)
        print("Checkpoint loaded successfully")
        start_epoch = int(resume_location.split('_')[-1].split('.')[0])
    
    # Set up parameter groups and optimizers
    parameters = {"main": [], "aux": []}
    for name, param in model_without_ddp.named_parameters():
        if name.endswith(".quantiles"):
            parameters["aux"].append(param)
        else:
            parameters["main"].append(param)
    
    optimizer = torch.optim.Adam(parameters["main"], lr=lr)
    aux_optimizer = torch.optim.Adam(parameters["aux"], lr=aux_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                         factor=0.5, patience=5, verbose=True)
    
    # Training loop
    best_val_loss = float('inf')
    print("Starting training...")
    
    try:
        for epoch in range(start_epoch, epochs):
            train_metrics = train_one_epoch(model, train_loader, optimizer, 
                                          aux_optimizer, criterion, device, epoch)
            
            # Update entropy bottleneck
            if (epoch + 1) % update_entropy_frequency == 0:
                updated = model.update()
                if updated:
                    print("Updated entropy bottleneck parameters")
            
            if val_loader:
                val_metrics = validate(model, val_loader, criterion, device)
            
            # Log metrics
            writer.add_scalar('train/loss', train_metrics[0], epoch)
            writer.add_scalar('train/mse', train_metrics[1], epoch)
            writer.add_scalar('train/bpp', train_metrics[2], epoch)
            writer.add_scalar('train/aux_loss', train_metrics[3], epoch)
            writer.add_scalar('train/psnr', train_metrics[4], epoch)
            if val_loader:
                writer.add_scalar('val/loss', val_metrics['loss'], epoch)
                writer.add_scalar('val/psnr', val_metrics['psnr'], epoch)
            
            # Save model based on training metrics instead of validation
            if (epoch + 1) % save_frequency == 0:
                torch.save(model.state_dict(), 
                          save_path + f"vae_model_epoch_{epoch+1}.pth")
            
            # Update learning rate
            scheduler.step(train_metrics[0])
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        writer.close()
        print("Training finished")

if __name__ == "__main__":
    main()
