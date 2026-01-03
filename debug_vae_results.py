#!/usr/bin/env python3
"""
Debug VAE Results
================

Investigate and fix the calculation issues in VAE evaluation.
"""

import torch
import numpy as np
import os
import time
from PIL import Image
from torchvision import transforms
from model import VAE
from parameters import *
from compression_pipeline import CompletePipelineDecoder


def debug_vae_calculations():
    """Debug the calculation issues step by step."""
    print("DEBUG VAE COMPRESSION CALCULATIONS")
    print("="*60)
    
    # Load VAE model
    model_path = "checkpoint/vae_model_epoch_85.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device}")
    
    model = VAE(latent_channels).to(device)
    state_dict = torch.load(model_path, map_location=device)
    
    # Initialize entropy bottlenecks
    spatial_cdf_size = state_dict['encoder.spatial_hyperprior.entropy_bottleneck._quantized_cdf'].size(1)
    angular_cdf_size = state_dict['encoder.angular_hyperprior.entropy_bottleneck._quantized_cdf'].size(1)
    
    model.encoder.spatial_hyperprior.entropy_bottleneck._offset = torch.zeros(64, device=device)
    model.encoder.spatial_hyperprior.entropy_bottleneck._quantized_cdf = torch.zeros(64, spatial_cdf_size, device=device)
    model.encoder.spatial_hyperprior.entropy_bottleneck._cdf_length = torch.zeros(64, dtype=torch.int32, device=device)
    
    model.encoder.angular_hyperprior.entropy_bottleneck._offset = torch.zeros(64, device=device)
    model.encoder.angular_hyperprior.entropy_bottleneck._quantized_cdf = torch.zeros(64, angular_cdf_size, device=device)
    model.encoder.angular_hyperprior.entropy_bottleneck._cdf_length = torch.zeros(64, dtype=torch.int32, device=device)
    
    model.load_state_dict(state_dict)
    model.eval()
    
    # Extract latents from real image
    data_dir = "dataset/collected_118_right_1_3x3_macropixels"
    import glob
    image_paths = glob.glob(os.path.join(data_dir, "*.png"))
    
    if image_paths:
        img = Image.open(image_paths[0]).convert('RGB')
        img = img.resize((param_width, param_height))
        transform = transforms.ToTensor()
        test_image = transform(img).unsqueeze(0).to(device)
        print(f"Loaded real image: {os.path.basename(image_paths[0])}")
        print(f"Image size: {test_image.shape}")
    else:
        test_image = torch.randn(1, 3, param_height, param_width).to(device)
        print("Using synthetic image")
    
    with torch.no_grad():
        encoder_output = model.encoder(test_image)
        spatial_latents = encoder_output["latents"]["y_s"]
        angular_latents = encoder_output["latents"]["y_a"]
    
    # Move to CPU for compression pipeline
    spatial_latents = spatial_latents.cpu()
    angular_latents = angular_latents.cpu()
    
    print(f"\nLatent Analysis:")
    print(f"Spatial latents: {spatial_latents.shape}")
    print(f"Angular latents: {angular_latents.shape}")
    print(f"Spatial range: [{spatial_latents.min():.6f}, {spatial_latents.max():.6f}]")
    print(f"Angular range: [{angular_latents.min():.6f}, {angular_latents.max():.6f}]")
    print(f"Spatial mean: {spatial_latents.mean():.6f}, std: {spatial_latents.std():.6f}")
    print(f"Angular mean: {angular_latents.mean():.6f}, std: {angular_latents.std():.6f}")
    
    # Test simple quantization
    print(f"\n" + "="*60)
    print("TESTING SIMPLE QUANTIZATION")
    print("="*60)
    
    config = {
        'use_reordering': False,
        'use_clipping_sparsification': False,
        'use_non_uniform_quantization': True,
        'use_vector_quantization': False,
        'use_transform_coding': False,
        'use_bit_plane_coding': False,
        'use_bitstream_structuring': False,
        'use_arithmetic_coding': False
    }
    
    component_configs = {
        'quantizer': {'num_levels': 8, 'channel_wise': True}
    }
    
    pipeline = CompletePipelineDecoder(pipeline_config=config)
    pipeline.initialize_components(component_configs)
    
    # Train pipeline
    print("Training quantizer...")
    pipeline.train_pipeline(spatial_latents, angular_latents)
    
    # Encode
    print("Encoding...")
    encoded_data, side_info = pipeline.encode_complete(spatial_latents, angular_latents)
    
    # Debug encoded data
    print(f"\nEncoded Data Analysis:")
    for key, value in encoded_data.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}, dtype: {value.dtype}")
            print(f"  Range: [{value.min():.6f}, {value.max():.6f}]")
            print(f"  Unique values: {len(torch.unique(value))}")
        else:
            print(f"{key}: {type(value)}")
    
    # Decode
    print("\nDecoding...")
    reconstructed_spatial, reconstructed_angular = pipeline.decode_complete(encoded_data, side_info)
    
    print(f"\nReconstruction Analysis:")
    print(f"Original spatial range: [{spatial_latents.min():.6f}, {spatial_latents.max():.6f}]")
    print(f"Reconstructed spatial range: [{reconstructed_spatial.min():.6f}, {reconstructed_spatial.max():.6f}]")
    print(f"Original angular range: [{angular_latents.min():.6f}, {angular_latents.max():.6f}]")
    print(f"Reconstructed angular range: [{reconstructed_angular.min():.6f}, {reconstructed_angular.max():.6f}]")
    
    # Calculate MSE properly
    spatial_mse = torch.mean((spatial_latents - reconstructed_spatial) ** 2).item()
    angular_mse = torch.mean((angular_latents - reconstructed_angular) ** 2).item()
    combined_mse = (spatial_mse + angular_mse) / 2
    
    print(f"\nMSE Calculations:")
    print(f"Spatial MSE: {spatial_mse:.6f}")
    print(f"Angular MSE: {angular_mse:.6f}")
    print(f"Combined MSE: {combined_mse:.6f}")
    
    # Calculate PSNR correctly
    # For latent space, we need to normalize properly
    # Method 1: Using data range
    spatial_range = spatial_latents.max() - spatial_latents.min()
    angular_range = angular_latents.max() - angular_latents.min()
    
    if spatial_mse > 0:
        spatial_psnr = 20 * np.log10(spatial_range.item()) - 10 * np.log10(spatial_mse)
    else:
        spatial_psnr = 100.0
    
    if angular_mse > 0:
        angular_psnr = 20 * np.log10(angular_range.item()) - 10 * np.log10(angular_mse)
    else:
        angular_psnr = 100.0
    
    combined_psnr = (spatial_psnr + angular_psnr) / 2
    
    print(f"\nPSNR Calculations (Method 1 - Data Range):")
    print(f"Spatial PSNR: {spatial_psnr:.2f} dB")
    print(f"Angular PSNR: {angular_psnr:.2f} dB")
    print(f"Combined PSNR: {combined_psnr:.2f} dB")
    
    # Method 2: Using standard deviation as reference
    spatial_std = spatial_latents.std().item()
    angular_std = angular_latents.std().item()
    
    if spatial_mse > 0:
        spatial_psnr_std = 20 * np.log10(spatial_std) - 10 * np.log10(spatial_mse)
    else:
        spatial_psnr_std = 100.0
    
    if angular_mse > 0:
        angular_psnr_std = 20 * np.log10(angular_std) - 10 * np.log10(angular_mse)
    else:
        angular_psnr_std = 100.0
    
    combined_psnr_std = (spatial_psnr_std + angular_psnr_std) / 2
    
    print(f"\nPSNR Calculations (Method 2 - Standard Deviation):")
    print(f"Spatial PSNR: {spatial_psnr_std:.2f} dB")
    print(f"Angular PSNR: {angular_psnr_std:.2f} dB")
    print(f"Combined PSNR: {combined_psnr_std:.2f} dB")
    
    # Calculate compression metrics
    print(f"\n" + "="*60)
    print("COMPRESSION METRICS ANALYSIS")
    print("="*60)
    
    # Original size calculation
    total_elements = spatial_latents.numel() + angular_latents.numel()
    original_size_bytes = total_elements * 4  # float32 = 4 bytes
    original_size_bits = original_size_bytes * 8
    
    print(f"Original data:")
    print(f"  Total elements: {total_elements:,}")
    print(f"  Original size: {original_size_bytes:,} bytes ({original_size_bits:,} bits)")
    
    # Calculate actual compressed size
    performance = pipeline.calculate_end_to_end_performance(
        spatial_latents, angular_latents, encoded_data, side_info
    )
    
    print(f"\nCompression performance:")
    print(f"  Compression ratio: {performance['compression_ratio']:.2f}x")
    print(f"  Size reduction: {performance['size_reduction_percent']:.1f}%")
    print(f"  Pipeline efficiency: {performance['pipeline_efficiency']:.1f}%")
    
    # BPP calculation
    image_pixels = param_height * param_width
    compressed_bits = original_size_bits / performance['compression_ratio']
    bpp = compressed_bits / image_pixels
    
    print(f"\nBPP Calculation:")
    print(f"  Image pixels: {image_pixels:,}")
    print(f"  Compressed bits: {compressed_bits:,.0f}")
    print(f"  BPP: {bpp:.4f}")
    
    # Alternative BPP calculation - using latent domain
    latent_pixels = spatial_latents.shape[-2] * spatial_latents.shape[-1]  # 8 * 12 = 96
    total_latent_pixels = latent_pixels * 2  # spatial + angular
    latent_bpp = compressed_bits / total_latent_pixels
    
    print(f"\nLatent-domain BPP:")
    print(f"  Latent pixels per component: {latent_pixels}")
    print(f"  Total latent pixels: {total_latent_pixels}")
    print(f"  Latent BPP: {latent_bpp:.4f}")
    
    print(f"\n" + "="*60)
    print("SUMMARY OF ISSUES FOUND")
    print("="*60)
    
    issues = []
    
    if combined_mse > 1.0:
        issues.append(f"High MSE ({combined_mse:.3f}) suggests quantization too aggressive")
    
    if combined_psnr < 20:
        issues.append(f"Low PSNR ({combined_psnr:.2f} dB) indicates poor reconstruction quality")
    
    if bpp > 2.0:
        issues.append(f"High BPP ({bpp:.4f}) suggests inefficient compression")
    
    if len(issues) == 0:
        print("No major issues detected!")
    else:
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
    
    print(f"\nRecommendations:")
    print("1. Use more quantization levels for better quality")
    print("2. Implement proper PSNR calculation for latent space")
    print("3. Verify BPP calculation methodology")
    print("4. Consider latent-domain metrics instead of image-domain")


if __name__ == "__main__":
    debug_vae_calculations() 