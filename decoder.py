from model import *
from parameters import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import math
import pickle
import argparse


def load_model(epoch_n):
    # Load and initialize model
    model = VAE(latent_channels)
    model = model.to(device)
    
    # Load the state dict first to see the correct sizes
    state_dict = torch.load(save_path + f"vae_model_epoch_{epoch_n}.pth", map_location=device)
    
    # Get the correct sizes from the saved model
    spatial_cdf_size = state_dict['encoder.spatial_hyperprior.entropy_bottleneck._quantized_cdf'].size(1)
    angular_cdf_size = state_dict['encoder.angular_hyperprior.entropy_bottleneck._quantized_cdf'].size(1)
    
    # Set the correct CDF sizes
    model.encoder.spatial_hyperprior.entropy_bottleneck._offset = torch.zeros(64, device=device)
    model.encoder.spatial_hyperprior.entropy_bottleneck._quantized_cdf = torch.zeros(64, spatial_cdf_size, device=device)
    model.encoder.spatial_hyperprior.entropy_bottleneck._cdf_length = torch.zeros(64, dtype=torch.int32, device=device)
    
    model.encoder.angular_hyperprior.entropy_bottleneck._offset = torch.zeros(64, device=device)
    model.encoder.angular_hyperprior.entropy_bottleneck._quantized_cdf = torch.zeros(64, angular_cdf_size, device=device)
    model.encoder.angular_hyperprior.entropy_bottleneck._cdf_length = torch.zeros(64, dtype=torch.int32, device=device)
    
    # Now load the state dict
    model.load_state_dict(state_dict)
    
    # Set to eval mode
    model.eval()
    
    return model


def reassemble_image(patches, original_size, patch_size, step_size, device):
    reconstructed = torch.zeros((3, original_size[1], original_size[0]), device=device)
    counts = torch.zeros_like(reconstructed)

    patch_idx = 0
    for y in range(0, original_size[1] - patch_size[0] + 1, step_size[0]):
        for x in range(0, original_size[0] - patch_size[1] + 1, step_size[1]):
            if patch_idx >= len(patches):
                break

            patch = patches[patch_idx].to(device)
            patch_idx += 1

            reconstructed[:, y:y + patch_size[0], x:x + patch_size[1]] += patch
            counts[:, y:y + patch_size[0], x:x + patch_size[1]] += 1

    reconstructed /= counts.clamp(min=1)
    return reconstructed


def decode_image(model, encoded_data_path, output_dir):
    """
    Decode compressed representation back to an image
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load encoded data
    with open(encoded_data_path, 'rb') as f:
        encoded_data = pickle.load(f)
    
    # Extract metadata
    original_size = encoded_data['original_size']  # (width, height)
    patch_size = encoded_data['patch_size']
    step_size = encoded_data['step_size']
    
    # Get base filename without extension
    base_filename = os.path.basename(encoded_data_path).split('.')[0]
    
    # Decode patches
    reconstructed_patches = []
    
    with torch.no_grad():
        for patch_data in encoded_data['latents']:
            # Get latent representation
            y_hat = patch_data['y_hat'].to(device)
            
            # Decode
            reconstructed = model.decoder(y_hat)
            reconstructed_patches.append(reconstructed.squeeze(0))
    
    # Reassemble image
    reconstructed_image = reassemble_image(
        reconstructed_patches, 
        (original_size[0], original_size[1]),  # Convert to (width, height)
        patch_size, 
        step_size, 
        device
    )
    
    # Save reconstructed image
    output_path = f"{output_dir}/{base_filename}_reconstructed.png"
    save_image(reconstructed_image, output_path)
    
    # Load original image for comparison if available
    image_name = base_filename.split('_encoded')[0]
    original_path = f"{os.path.dirname(encoded_data_path)}/original_{image_name}.png"
    
    if os.path.exists(original_path):
        original_tensor = transforms.ToTensor()(Image.open(original_path)).to(device)
        
        # Calculate metrics
        mse = F.mse_loss(reconstructed_image, original_tensor)
        psnr = -10 * torch.log10(mse)  # For [0,1] range images
        
        print(f"Full Image PSNR: {psnr:.2f} dB")
        print(f"Average Bits per pixel: {encoded_data['avg_bpp']:.4f}")
        
        # Save side-by-side comparison
        comparison = torch.cat([original_tensor, reconstructed_image], dim=2)
        comparison_path = f"{output_dir}/{base_filename}_comparison.png"
        save_image(comparison, comparison_path)
        print(f"Comparison image saved to {comparison_path}")
    
    print(f"Reconstructed image saved to {output_path}")
    
    return reconstructed_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Decode a compressed image using the VAE model')
    parser.add_argument('--input', type=str, default="./output/compressed/macropixel_029_encoded_epoch85.pkl", 
                        help='Path to the encoded data file (.pkl)')
    parser.add_argument('--output', type=str, default="./output/reconstructed", 
                        help='Output directory for reconstructed images')
    parser.add_argument('--epoch', type=int, default=85, 
                        help='Epoch number of the model to use')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.epoch)
    
    # Decode image
    reconstructed_image = decode_image(model, args.input, args.output)