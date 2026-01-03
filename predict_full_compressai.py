from model import *
from parameters import *
from dataloader import *
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
import gc
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Device setup
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


def extract_patches(image, patch_size=(216, 312), step_size=(180, 260)):
    # Using existing extract_patches function
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

def rgb_to_ycbcr(rgb_image):
    """Convert RGB image to YCbCr"""
    if isinstance(rgb_image, torch.Tensor):
        rgb_image = rgb_image.cpu().numpy()
    rgb = np.asarray(rgb_image).astype(np.float32)
    if rgb.ndim == 4:  # Batch of images
        rgb = rgb.squeeze(0)
    if rgb.shape[0] == 3:  # Channel-first to channel-last
        rgb = np.transpose(rgb, (1, 2, 0))
    
    # Use BT.709 coefficients
    R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
    Cb = -0.1146 * R - 0.3854 * G + 0.5 * B + 0.5
    Cr = 0.5 * R - 0.4542 * G - 0.0458 * B + 0.5
    
    # Scale to proper range
    Y = Y * 255.0
    Cb = Cb * 255.0
    Cr = Cr * 255.0
    
    return Y, Cb, Cr

def calculate_metrics(original, reconstructed):
    """Calculate PSNR and SSIM for both RGB and Y channel"""
    # Convert tensors to numpy arrays if needed
    if isinstance(original, torch.Tensor):
        original = original.cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.cpu().numpy()
    
    # Ensure proper shape
    if original.shape[0] == 3:  # If channel-first, convert to channel-last
        original = np.transpose(original, (1, 2, 0))
        reconstructed = np.transpose(reconstructed, (1, 2, 0))
    
    # Calculate RGB metrics (values in [0,1] range)
    psnr_rgb = psnr(original, reconstructed, data_range=1.0)
    ssim_rgb = ssim(original, reconstructed, 
                    channel_axis=2,  # Updated from multichannel=True
                    data_range=1.0,
                    win_size=11)  # Standard window size
    
    # Calculate Y channel metrics (values in [0,255] range)
    y_original, _, _ = rgb_to_ycbcr(original)
    y_reconstructed, _, _ = rgb_to_ycbcr(reconstructed)
    
    psnr_y = psnr(y_original, y_reconstructed, data_range=255.0)
    ssim_y = ssim(y_original, y_reconstructed, 
                  data_range=255.0,
                  channel_axis=None,  # Y channel is single channel
                  win_size=11)  # Standard window size
    
    return {
        'PSNR_RGB': psnr_rgb,
        'SSIM_RGB': ssim_rgb,
        'PSNR_Y': psnr_y,
        'SSIM_Y': ssim_y
    }
def process_image(model, image_path, output_folder, quantization_factor=1.0):
    print(f"\nProcessing {image_path} with quantization factor {quantization_factor}")
    try:
        # Load and prepare image
        image = Image.open(image_path)
        original_tensor = transforms.ToTensor()(image)
        patches = extract_patches(image)
        transform = transforms.ToTensor()
        patches_tensor = [transform(patch) for patch in patches]
        
        total_bits = 0
        total_pixels = image.size[0] * image.size[1] 
        reconstructed_patches = []
        
        # Process each patch
        for i, patch in enumerate(patches_tensor):
            print(f"Processing patch {i+1}/{len(patches_tensor)}")
            
            torch.cuda.empty_cache()
            gc.collect()
            
            with torch.no_grad():
                # Move to GPU and encode
                patch = patch.unsqueeze(0).to(device)
                
                # Get latent representations and hyperprior information
                enc_out = model.encoder(patch)
                y_s = enc_out["latents"]["y_s"]
                y_a = enc_out["latents"]["y_a"]
                
                # Apply quantization with the factor
                step_size = 1.0 / quantization_factor
                
                # Quantize spatial and angular components separately
                y_s_scaled = y_s / step_size
                y_a_scaled = y_a / step_size
                
                y_s_quantized = torch.round(y_s_scaled)
                y_a_quantized = torch.round(y_a_scaled)
                
                y_s_dequantized = y_s_quantized * step_size
                y_a_dequantized = y_a_quantized * step_size
                
                # Concatenate quantized latents
                latents_dequantized = torch.cat((y_s_dequantized, y_a_dequantized), dim=1)
                
                # Get reconstruction
                reconstructed = model.decoder(latents_dequantized)
                reconstructed_patches.append(reconstructed.squeeze(0).cpu())
                
                # Calculate entropy for both spatial and angular components
                def calculate_entropy(tensor):
                    symbols = tensor.flatten()
                    unique_symbols, counts = torch.unique(symbols, return_counts=True)
                    probs = counts.float() / symbols.numel()
                    entropy = -torch.sum(probs * torch.log2(probs + 1e-10))
                    return entropy * symbols.numel()
                
                
                # Calculate bits for both components
                bits_spatial = calculate_entropy(y_s_quantized)
                bits_angular = calculate_entropy(y_a_quantized)
                
                total_bits += adjust_for_symbol_noise(bits_spatial + bits_angular)
                
                # Clean up
                del patch, y_s, y_a, latents_dequantized, reconstructed
                torch.cuda.empty_cache()
        
        # Reassemble image
        print("Reassembling image...")
        reconstructed_image = torch.zeros((3, image.size[1], image.size[0])).cpu()
        counts = torch.zeros_like(reconstructed_image)
        
        patch_idx = 0
        for y in range(0, image.size[1] - 216 + 1, 180):
            for x in range(0, image.size[0] - 312 + 1, 260):
                if patch_idx >= len(reconstructed_patches):
                    break
                patch = reconstructed_patches[patch_idx]
                reconstructed_image[:, y:y+216, x:x+312] += patch
                counts[:, y:y+216, x:x+312] += 1
                patch_idx += 1
        
        reconstructed_image = reconstructed_image / counts.clamp(min=1)
        
        # Calculate metrics
        metrics = calculate_metrics(original_tensor, reconstructed_image)
        metrics['BPP'] = total_bits / total_pixels
        
        # Save reconstructed image
        output_path = os.path.join(output_folder, f"Output_{os.path.basename(image_path)}")
        save_image(reconstructed_image, output_path)
        
        return metrics
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise

def main():
    try:
        # Load model
        print("Loading model...")
        model = VAE(latent_channels).to(device)
        
        # Load the state dict first to see the correct sizes
        state_dict = torch.load(save_path + "vae_model_epoch_149_01.pth", map_location=device)
        
        # Initialize entropy bottleneck parameters with correct sizes
        spatial_cdf_size = state_dict['encoder.spatial_hyperprior.entropy_bottleneck._quantized_cdf'].size(1)
        angular_cdf_size = state_dict['encoder.angular_hyperprior.entropy_bottleneck._quantized_cdf'].size(1)
        
        model.encoder.spatial_hyperprior.entropy_bottleneck._offset = torch.zeros(64, device=device)
        model.encoder.spatial_hyperprior.entropy_bottleneck._quantized_cdf = torch.zeros(64, spatial_cdf_size, device=device)
        model.encoder.spatial_hyperprior.entropy_bottleneck._cdf_length = torch.zeros(64, dtype=torch.int32, device=device)
        
        model.encoder.angular_hyperprior.entropy_bottleneck._offset = torch.zeros(64, device=device)
        model.encoder.angular_hyperprior.entropy_bottleneck._quantized_cdf = torch.zeros(64, angular_cdf_size, device=device)
        model.encoder.angular_hyperprior.entropy_bottleneck._cdf_length = torch.zeros(64, dtype=torch.int32, device=device)
        
        # Now load the state dict
        model.load_state_dict(state_dict)
        model.eval()
        
        # Setup directories
        base_input_folder = './dataset/fountain_Bikes_3x3_views_new/'
        base_output_folder = './output_test/'
        results = {}
        
        # Different quantization factors to try
        #quantization_factors = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,  0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]
        quantization_factors = [0.6]
        input_folders = [
            os.path.join(base_input_folder, f) 
            for f in os.listdir(base_input_folder) 
            if os.path.isdir(os.path.join(base_input_folder, f))
        ]
        
        # Process each folder with different quantization factors
        for q_factor in quantization_factors:
            print(f"\nProcessing with quantization factor: {q_factor}")
            
            for folder in input_folders:
                folder_name = os.path.basename(folder)
                output_folder = os.path.join(base_output_folder, f"{folder_name}_compressed_q{q_factor}")
                os.makedirs(output_folder, exist_ok=True)
                
                folder_metrics = []
                
                for filename in os.listdir(folder):
                    if not filename.endswith('.png'):
                        continue
                    
                    image_path = os.path.join(folder, filename)
                    metrics = process_image(model, image_path, output_folder, q_factor)
                    folder_metrics.append(metrics)
                    
                    print(f"Metrics for {filename} (q={q_factor}):")
                    for key, value in metrics.items():
                        print(f"{key}: {value:.4f}")
                    
                # Calculate average metrics for folder
                avg_metrics = {
                    key: np.mean([m[key] for m in folder_metrics])
                    for key in folder_metrics[0].keys()
                }
                results[f"{folder_name}_q{q_factor}"] = avg_metrics
        
        # Save results
        results_path = os.path.join(base_output_folder, 'compression_results_with_quantization.txt')
        with open(results_path, 'w') as f:
            f.write("Compression Results with Different Quantization Factors\n")
            f.write("================================================\n\n")
            for result_key, metrics in results.items():
                folder_name, q_factor = result_key.rsplit('_q', 1)
                f.write(f"Folder: {folder_name}\n")
                f.write(f"Quantization Factor: {q_factor}\n")
                for key, value in metrics.items():
                    f.write(f"{key}: {value:.4f}\n")
                f.write("\n")
        
        print(f"\nResults saved to {results_path}")
            
    except Exception as e:
        print(f"Fatal error: {str(e)}")
    finally:
        print("Processing completed.")

if __name__ == "__main__":
    main() 