from model import *
from parameters import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import math
import pickle
import argparse


def extract_patches(image, patch_size=(216, 312), step_size=(180, 260)):
    patches = []
    img_width, img_height = image.size

    for y in range(0, img_height - patch_size[0] + 1, step_size[0]):
        for x in range(0, img_width - patch_size[1] + 1, step_size[1]):
            box = (x, y, x + patch_size[1], y + patch_size[0])
            if box[2] <= img_width and box[3] <= img_height:
                patch = image.crop(box)
                patches.append(patch)
            if len(patches) == 49:  
                return patches

    return patches


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


def encode_image(model, image_path, output_dir, epoch_n):
    """
    Encode an image into compressed representation and save the latents
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and process image
    image = Image.open(image_path)
    image_name = os.path.basename(image_path).split('.')[0]
    
    # Save original image
    transform = transforms.ToTensor()
    save_image(transform(image), f"{output_dir}/original_{image_name}.png")
    
    # Extract patches
    patches = extract_patches(image, patch_size=(216, 312), step_size=(180, 260))
    patches_tensor = torch.stack([transform(patch) for patch in patches]).to(device)
    
    # Process patches
    encoded_data = {
        'latents': [],
        'likelihoods': [],
        'patch_positions': [],
        'original_size': image.size,  # (width, height)
        'patch_size': (216, 312),
        'step_size': (180, 260)
    }
    
    total_bpp = 0
    total_pixels = 0
    
    with torch.no_grad():
        for i, patch in enumerate(patches_tensor):
            patch = patch.unsqueeze(0)
            
            # Get encoder output
            enc_out = model.encoder(patch)
            
            # Store latent representations
            encoded_data['latents'].append({
                'y_s': enc_out["latents"]["y_s"].cpu(),
                'y_a': enc_out["latents"]["y_a"].cpu(),
                'y_hat': enc_out["y_hat"].cpu()
            })
            
            # Store likelihoods for bpp calculation
            encoded_data['likelihoods'].append({k: v.cpu() for k, v in enc_out['likelihoods'].items()})
            
            # Calculate patch position
            y_pos = (i // 7) * 180
            x_pos = (i % 7) * 260
            encoded_data['patch_positions'].append((x_pos, y_pos))
            
            # Calculate bpp for this patch
            N, _, H, W = patch.size()
            num_pixels = N * H * W
            
            bpp = sum((-torch.log2(likelihoods).sum() / num_pixels)
                     for likelihoods in enc_out["likelihoods"].values())
            
            total_bpp += bpp.item() * num_pixels
            total_pixels += num_pixels
    
    # Calculate average bpp
    avg_bpp = total_bpp / total_pixels
    encoded_data['avg_bpp'] = avg_bpp
    
    # Save encoded data
    output_filename = f"{image_name}_encoded_epoch{epoch_n}"
    
    with open(f"{output_dir}/{output_filename}.pkl", 'wb') as f:
        pickle.dump(encoded_data, f)
    
    print(f"Encoded image saved to {output_dir}/{output_filename}.pkl")
    print(f"Average Bits per pixel: {avg_bpp:.4f}")
    
    return encoded_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encode an image using the VAE model')
    parser.add_argument('--image', type=str, default="./dataset/collected_118_right_1_3x3_macropixels/macropixel_029.png", 
                        help='Path to the input image')
    parser.add_argument('--output', type=str, default="./output/compressed", 
                        help='Output directory for compressed data')
    parser.add_argument('--epoch', type=int, default=85, 
                        help='Epoch number of the model to use')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.epoch)
    
    # Encode image
    encoded_data = encode_image(model, args.image, args.output, args.epoch)
    
    print(f"Encoding complete. Use decoder.py to reconstruct the image.")