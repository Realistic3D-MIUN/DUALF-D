from model import *
from parameters import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import time
import numpy as np
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


def calculate_entropy(tensor):
    """
    Calculate entropy for quantized tensor to estimate bits needed with arithmetic coding
    """
    symbols = tensor.flatten()
    unique_symbols, counts = torch.unique(symbols, return_counts=True)
    probs = counts.float() / symbols.numel()
    entropy = -torch.sum(probs * torch.log2(probs + 1e-10))
    # Return the total bits (entropy per symbol * number of symbols)
    return entropy * symbols.numel()


def measure_inference_time(model, image_path, num_runs=10, warmup_runs=3, spatial_quant=1.0, angular_quant=1.0, compression_factor=0.85):
    """
    Measure inference time for both encoding and decoding, including entropy calculations
    """
    # Load and process image
    image = Image.open(image_path)
    transform = transforms.ToTensor()
    
    # Extract patches
    patches = extract_patches(image, patch_size=(216, 312), step_size=(180, 260))
    patches_tensor = torch.stack([transform(patch) for patch in patches]).to(device)
    
    # Metadata for reconstruction
    original_size = image.size  # (width, height)
    patch_size = (216, 312)
    step_size = (180, 260)
    total_pixels = original_size[0] * original_size[1] * 3  # RGB images have 3 channels
    
    # Warmup runs
    print("Performing warmup runs...")
    for _ in range(warmup_runs):
        with torch.no_grad():
            # Encode
            latents = []
            for patch in patches_tensor:
                patch = patch.unsqueeze(0)
                enc_out = model.encoder(patch)
                y_s = enc_out["latents"]["y_s"]
                y_a = enc_out["latents"]["y_a"]
                
                # Apply quantization with factor
                step_size_s = 1.0 / spatial_quant
                step_size_a = 1.0 / angular_quant
                
                # Quantize spatial and angular components separately
                y_s_scaled = y_s / step_size_s
                y_a_scaled = y_a / step_size_a
                
                y_s_quantized = torch.round(y_s_scaled)
                y_a_quantized = torch.round(y_a_scaled)
                
                y_s_dequantized = y_s_quantized * step_size_s
                y_a_dequantized = y_a_quantized * step_size_a
                
                # Concatenate quantized latents
                y_hat = torch.cat((y_s_dequantized, y_a_dequantized), dim=1)
                latents.append(y_hat)
            
            # Decode
            reconstructed_patches = []
            for y_hat in latents:
                reconstructed = model.decoder(y_hat)
                reconstructed_patches.append(reconstructed.squeeze(0))
            
            # Reassemble
            reconstructed_image = reassemble_image(
                reconstructed_patches, 
                (original_size[0], original_size[1]),
                patch_size, 
                step_size, 
                device
            )
    
    # First measure encoding time WITHOUT entropy calculation
    print(f"Measuring encoding time WITHOUT entropy calculation over {num_runs} runs...")
    encoding_times_no_entropy = []
    
    for run in range(num_runs):
        start_time = time.time()
        
        with torch.no_grad():
            latents = []
            
            for patch in patches_tensor:
                patch = patch.unsqueeze(0)
                enc_out = model.encoder(patch)
                y_s = enc_out["latents"]["y_s"]
                y_a = enc_out["latents"]["y_a"]
                
                # Apply quantization with factor
                step_size_s = 1.0 / spatial_quant
                step_size_a = 1.0 / angular_quant
                
                # Quantize spatial and angular components separately
                y_s_scaled = y_s / step_size_s
                y_a_scaled = y_a / step_size_a
                
                y_s_quantized = torch.round(y_s_scaled)
                y_a_quantized = torch.round(y_a_scaled)
                
                y_s_dequantized = y_s_quantized * step_size_s
                y_a_dequantized = y_a_quantized * step_size_a
                
                # Concatenate quantized latents
                y_hat = torch.cat((y_s_dequantized, y_a_dequantized), dim=1)
                latents.append(y_hat)
            
        torch.cuda.synchronize()  # Make sure all GPU operations are completed
        end_time = time.time()
        encoding_times_no_entropy.append(end_time - start_time)
    
    # Then measure encoding time WITH entropy calculation (arithmetic coding)
    print(f"Measuring encoding time WITH entropy calculation over {num_runs} runs...")
    encoding_times_with_entropy = []
    total_bits_list = []
    
    for run in range(num_runs):
        start_time = time.time()
        run_total_bits = 0
        
        with torch.no_grad():
            latents = []
            y_s_quantized_list = []
            y_a_quantized_list = []
            
            for patch in patches_tensor:
                patch = patch.unsqueeze(0)
                enc_out = model.encoder(patch)
                y_s = enc_out["latents"]["y_s"]
                y_a = enc_out["latents"]["y_a"]
                
                # Apply quantization with factor
                step_size_s = 1.0 / spatial_quant
                step_size_a = 1.0 / angular_quant
                
                # Quantize spatial and angular components separately
                y_s_scaled = y_s / step_size_s
                y_a_scaled = y_a / step_size_a
                
                y_s_quantized = torch.round(y_s_scaled)
                y_a_quantized = torch.round(y_a_scaled)
                
                # Store quantized values for entropy calculation
                y_s_quantized_list.append(y_s_quantized)
                y_a_quantized_list.append(y_a_quantized)
                
                y_s_dequantized = y_s_quantized * step_size_s
                y_a_dequantized = y_a_quantized * step_size_a
                
                # Concatenate quantized latents
                y_hat = torch.cat((y_s_dequantized, y_a_dequantized), dim=1)
                latents.append(y_hat)
                
                # Calculate entropy for both components
                bits_spatial = calculate_entropy(y_s_quantized)
                bits_angular = calculate_entropy(y_a_quantized)
                
                # Apply compression factor to account for real-world compression
                run_total_bits += (bits_spatial + bits_angular).item() * compression_factor
            
        torch.cuda.synchronize()  # Make sure all GPU operations are completed
        end_time = time.time()
        encoding_times_with_entropy.append(end_time - start_time)
        total_bits_list.append(run_total_bits)
    
    # Measure decoding time
    print(f"Measuring decoding time over {num_runs} runs...")
    decoding_times = []
    
    for run in range(num_runs):
        start_time = time.time()
        
        with torch.no_grad():
            # Decode
            reconstructed_patches = []
            for y_hat in latents:
                reconstructed = model.decoder(y_hat)
                reconstructed_patches.append(reconstructed.squeeze(0))
            
            # Reassemble
            reconstructed_image = reassemble_image(
                reconstructed_patches, 
                (original_size[0], original_size[1]),
                patch_size, 
                step_size, 
                device
            )
        
        torch.cuda.synchronize()  # Make sure all GPU operations are completed
        end_time = time.time()
        decoding_times.append(end_time - start_time)
    
    # Calculate statistics
    avg_encoding_time_no_entropy = np.mean(encoding_times_no_entropy)
    std_encoding_time_no_entropy = np.std(encoding_times_no_entropy)
    
    avg_encoding_time_with_entropy = np.mean(encoding_times_with_entropy)
    std_encoding_time_with_entropy = np.std(encoding_times_with_entropy)
    
    entropy_overhead = avg_encoding_time_with_entropy - avg_encoding_time_no_entropy
    entropy_overhead_percent = (entropy_overhead / avg_encoding_time_no_entropy) * 100
    
    avg_decoding_time = np.mean(decoding_times)
    std_decoding_time = np.std(decoding_times)
    
    avg_total_bits = np.mean(total_bits_list)
    std_total_bits = np.std(total_bits_list)
    
    # Compute bit-related metrics
    bits_per_pixel = avg_total_bits / total_pixels
    compression_ratio = (total_pixels * 8) / avg_total_bits  # Assuming 8 bits per pixel in original image
    
    # Calculate total times
    total_time_no_entropy = avg_encoding_time_no_entropy + avg_decoding_time
    total_time_with_entropy = avg_encoding_time_with_entropy + avg_decoding_time
    
    # Calculate throughput (pixels per second)
    encoding_throughput_no_entropy = total_pixels / avg_encoding_time_no_entropy
    encoding_throughput_with_entropy = total_pixels / avg_encoding_time_with_entropy
    decoding_throughput = total_pixels / avg_decoding_time
    total_throughput_no_entropy = total_pixels / total_time_no_entropy
    total_throughput_with_entropy = total_pixels / total_time_with_entropy
    
    # Print results
    print("\n===== Inference Time Results =====")
    print(f"Image size: {original_size[0]}x{original_size[1]} ({total_pixels} pixels)")
    print(f"Number of patches: {len(patches)}")
    print(f"Quantization factors - Spatial: {spatial_quant}, Angular: {angular_quant}")
    print(f"Compression factor: {compression_factor}")
    
    print("\nEncoding WITHOUT Entropy Calculation:")
    print(f"  Average time: {avg_encoding_time_no_entropy:.4f} ± {std_encoding_time_no_entropy:.4f} seconds")
    print(f"  Throughput: {encoding_throughput_no_entropy/1e6:.2f} Mpixels/second")
    
    print("\nEncoding WITH Entropy Calculation:")
    print(f"  Average time: {avg_encoding_time_with_entropy:.4f} ± {std_encoding_time_with_entropy:.4f} seconds")
    print(f"  Throughput: {encoding_throughput_with_entropy/1e6:.2f} Mpixels/second")
    
    print("\nEntropy Calculation Overhead:")
    print(f"  Time: {entropy_overhead:.4f} seconds ({entropy_overhead_percent:.2f}%)")
    
    print("\nDecoding:")
    print(f"  Average time: {avg_decoding_time:.4f} ± {std_decoding_time:.4f} seconds")
    print(f"  Throughput: {decoding_throughput/1e6:.2f} Mpixels/second")
    
    print("\nTotal (Encode + Decode) WITHOUT Entropy:")
    print(f"  Average time: {total_time_no_entropy:.4f} seconds")
    print(f"  Throughput: {total_throughput_no_entropy/1e6:.2f} Mpixels/second")
    
    print("\nTotal (Encode + Decode) WITH Entropy:")
    print(f"  Average time: {total_time_with_entropy:.4f} seconds")
    print(f"  Throughput: {total_throughput_with_entropy/1e6:.2f} Mpixels/second")
    
    print("\nCompression Metrics:")
    print(f"  Total bits: {avg_total_bits:.2f} ± {std_total_bits:.2f}")
    print(f"  Bits per pixel (bpp): {bits_per_pixel:.4f}")
    print(f"  Compression ratio: {compression_ratio:.2f}:1")
    
    return {
        'encoding_time_no_entropy': avg_encoding_time_no_entropy,
        'encoding_std_no_entropy': std_encoding_time_no_entropy,
        'encoding_time_with_entropy': avg_encoding_time_with_entropy,
        'encoding_std_with_entropy': std_encoding_time_with_entropy,
        'entropy_overhead': entropy_overhead,
        'entropy_overhead_percent': entropy_overhead_percent,
        'decoding_time': avg_decoding_time,
        'decoding_std': std_decoding_time,
        'total_time_no_entropy': total_time_no_entropy,
        'total_time_with_entropy': total_time_with_entropy,
        'encoding_throughput_no_entropy': encoding_throughput_no_entropy,
        'encoding_throughput_with_entropy': encoding_throughput_with_entropy,
        'decoding_throughput': decoding_throughput,
        'total_throughput_no_entropy': total_throughput_no_entropy,
        'total_throughput_with_entropy': total_throughput_with_entropy,
        'total_bits': avg_total_bits,
        'bits_std': std_total_bits,
        'bits_per_pixel': bits_per_pixel,
        'compression_ratio': compression_ratio
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Measure inference time of the VAE model')
    parser.add_argument('--image', type=str, default="./dataset/collected_118_right_1_3x3_macropixels/macropixel_029.png", 
                        help='Path to the input image')
    parser.add_argument('--epoch', type=int, default=85, 
                        help='Epoch number of the model to use')
    parser.add_argument('--runs', type=int, default=10, 
                        help='Number of runs for timing measurement')
    parser.add_argument('--warmup', type=int, default=3, 
                        help='Number of warmup runs')
    parser.add_argument('--spatial-quant', type=float, default=1.0, 
                        help='Quantization factor for spatial latents')
    parser.add_argument('--angular-quant', type=float, default=1.0, 
                        help='Quantization factor for angular latents')
    parser.add_argument('--compression-factor', type=float, default=0.85,
                        help='Factor to apply to entropy to approximate real arithmetic coding efficiency')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.epoch)
    
    # Measure inference time
    results = measure_inference_time(
        model, 
        args.image, 
        num_runs=args.runs, 
        warmup_runs=args.warmup,
        spatial_quant=args.spatial_quant,
        angular_quant=args.angular_quant,
        compression_factor=args.compression_factor
    ) 