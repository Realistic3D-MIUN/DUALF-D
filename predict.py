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


epoch_n = 242

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

# Load and process image
test_loc = "./dataset/collected_118_right_1_3x3_macropixels/macropixel_029.png"
image = Image.open(test_loc)
patches = extract_patches(image, patch_size=(216, 312), step_size=(180, 260))
transform = transforms.ToTensor()
patches_tensor = torch.stack([transform(patch) for patch in patches]).to(device)

# Process patches
reconstructed_patches = []
with torch.no_grad():
    for patch in patches_tensor:
        patch = patch.unsqueeze(0)
        out = model(patch)  # Now returns a dict with 'x_hat' and 'likelihoods'
        reconstructed_patches.append(out['x_hat'].squeeze(0))

# Reassemble and save
original_size = (1872, 1296)  # Width, Height
patch_size = (216, 312)
step_size = (180, 260)

reconstructed_image = reassemble_image(reconstructed_patches, original_size, patch_size, step_size, device)

# Save images
save_image(transform(image), "./output/Input_Image.png")
save_image(reconstructed_image, f"./output/Output_Image_new_iran_32LD_{epoch_n}.png")

# Calculate metrics for the full image
mse = F.mse_loss(reconstructed_image, transform(image).to(device))
psnr = -10 * torch.log10(mse)  # For [0,1] range images

# Calculate metrics for each patch and average
total_bpp = 0
total_pixels = 0

with torch.no_grad():
    for patch in patches_tensor:
        patch = patch.unsqueeze(0)
        out = model(patch)
        N, _, H, W = patch.size()
        num_pixels = N * H * W
        
        bpp = sum((-torch.log2(likelihoods).sum() / num_pixels)
                 for likelihoods in out["likelihoods"].values())
        
        total_bpp += bpp.item() * num_pixels
        total_pixels += num_pixels

avg_bpp = total_bpp / total_pixels
print(f"Average Bits per pixel: {avg_bpp:.4f}")

print(f"Full Image PSNR: {psnr:.2f} dB")

