import torch
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy, kurtosis, skew
import matplotlib.pyplot as plt
import seaborn as sns
import os
from model import VAE
from torchvision import transforms
from PIL import Image
from parameters import *

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

def extract_latents(model, patches_tensor, device):
    """Extract spatial and angular latents from the model for all patches"""
    model.eval()
    spatial_latents = []
    angular_latents = []
    
    with torch.no_grad():
        for patch in patches_tensor:
            # Add batch dimension and move to device
            patch = patch.unsqueeze(0).to(device)
            
            # Get latent representations
            enc_out = model.encoder(patch)
            y_s = enc_out["latents"]["y_s"]
            y_a = enc_out["latents"]["y_a"]
            
            # Remove batch dimension and move to CPU
            spatial_latents.append(y_s[0].cpu().numpy())
            angular_latents.append(y_a[0].cpu().numpy())
    
    # Stack all latents
    spatial_latents = np.stack(spatial_latents)  # Shape: (49, 64, 8, 12)
    angular_latents = np.stack(angular_latents)  # Shape: (49, 64, 8, 12)
    
    print(f"Spatial latents shape: {spatial_latents.shape}")
    print(f"Angular latents shape: {angular_latents.shape}")
    
    return spatial_latents, angular_latents

def calculate_all_feature_mutual_information(spatial_latents, angular_latents):
    """
    Calculate mutual information between all features (spatial-spatial, angular-angular, and spatial-angular).
    Input shapes: spatial_latents (49, 64, 8, 12), angular_latents (49, 64, 8, 12)
    Returns: mutual_info_matrix (128, 128)
    """
    # Reshape to (49, 64, 96) where 96 = 8 * 12
    spatial = spatial_latents.reshape(49, 64, -1)
    angular = angular_latents.reshape(49, 64, -1)
    
    # Initialize full mutual information matrix
    mi_matrix = np.zeros((128, 128))
    
    # Calculate MI for all combinations
    for i in range(128):
        for j in range(128):
            # Determine if we're looking at spatial or angular features
            feat1 = spatial[:, i % 64, :].flatten() if i < 64 else angular[:, i % 64, :].flatten()
            feat2 = spatial[:, j % 64, :].flatten() if j < 64 else angular[:, j % 64, :].flatten()
            
            # Discretize the data
            feat1_bins = np.histogram_bin_edges(feat1, bins='auto')
            feat2_bins = np.histogram_bin_edges(feat2, bins='auto')
            
            feat1_digitized = np.digitize(feat1, feat1_bins)
            feat2_digitized = np.digitize(feat2, feat2_bins)
            
            # Calculate mutual information
            mi_matrix[i, j] = mutual_info_score(feat1_digitized, feat2_digitized)
    
    return mi_matrix

def main():
    # Set device (already defined in parameters.py)
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = VAE(latent_channels).to(device)
    
    # Load the state dict
    checkpoint_path = save_path + "vae_model_epoch_85.pth"
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Initialize entropy bottleneck parameters
    spatial_cdf_size = state_dict['encoder.spatial_hyperprior.entropy_bottleneck._quantized_cdf'].size(1)
    angular_cdf_size = state_dict['encoder.angular_hyperprior.entropy_bottleneck._quantized_cdf'].size(1)
    
    model.encoder.spatial_hyperprior.entropy_bottleneck._offset = torch.zeros(64, device=device)
    model.encoder.spatial_hyperprior.entropy_bottleneck._quantized_cdf = torch.zeros(64, spatial_cdf_size, device=device)
    model.encoder.spatial_hyperprior.entropy_bottleneck._cdf_length = torch.zeros(64, dtype=torch.int32, device=device)
    
    model.encoder.angular_hyperprior.entropy_bottleneck._offset = torch.zeros(64, device=device)
    model.encoder.angular_hyperprior.entropy_bottleneck._quantized_cdf = torch.zeros(64, angular_cdf_size, device=device)
    model.encoder.angular_hyperprior.entropy_bottleneck._cdf_length = torch.zeros(64, dtype=torch.int32, device=device)
    
    # Load state dict and set to eval mode
    model.load_state_dict(state_dict)
    model.eval()
    
    # Load and process image
    image_path = './dataset/fountain_Bikes_3x3_views/1/macropixel_257.png'
    if not os.path.exists(image_path):
        raise ValueError(f"Image not found: {os.path.abspath(image_path)}")
    
    # Extract patches
    image = Image.open(image_path)
    patches = extract_patches(image)
    transform = transforms.ToTensor()
    patches_tensor = torch.stack([transform(patch) for patch in patches])
    
    print(f"Number of patches: {len(patches)}")
    print(f"Patches tensor shape: {patches_tensor.shape}")
    
    try:
        # Extract latents for all patches
        spatial_latents, angular_latents = extract_latents(model, patches_tensor, device)
        
        # Calculate feature-wise mutual information for all combinations
        mi_matrix = calculate_all_feature_mutual_information(spatial_latents, angular_latents)
        
        # Create custom tick labels
        tick_positions = np.arange(0, 129, 32)  # Adjust spacing as needed
        tick_labels = [str(i) for i in tick_positions]
        
        # Plot mutual information matrix with quadrant labels
        plt.figure(figsize=(12, 10))
        heatmap = sns.heatmap(mi_matrix, cmap='viridis',
                             xticklabels=tick_positions,
                             yticklabels=tick_positions,
                             cbar_kws={'label': 'Mutual Information'})
        
        # Add quadrant lines
        plt.axhline(y=64, color='red', linestyle='-', linewidth=1)
        plt.axvline(x=64, color='red', linestyle='-', linewidth=1)
        
        # Add quadrant labels
        plt.text(32, -5, 'Spatial Features', ha='center', va='top')
        plt.text(96, -5, 'Angular Features', ha='center', va='top')
        plt.text(-5, 32, 'Spatial Features', ha='center', va='bottom', rotation=90)
        plt.text(-5, 96, 'Angular Features', ha='center', va='bottom', rotation=90)
        
        plt.title('Feature-wise Mutual Information Matrix')
        plt.xlabel('Feature Index')
        plt.ylabel('Feature Index')
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics for each quadrant
        print("\nMutual Information Analysis by Quadrant:")
        
        # Spatial-Spatial (top-left)
        ss_mi = mi_matrix[:64, :64]
        print("\nSpatial-Spatial MI:")
        print(f"Average: {ss_mi.mean():.4f}")
        print(f"Max: {ss_mi.max():.4f}")
        print(f"Min: {ss_mi.min():.4f}")
        
        # Angular-Angular (bottom-right)
        aa_mi = mi_matrix[64:, 64:]
        print("\nAngular-Angular MI:")
        print(f"Average: {aa_mi.mean():.4f}")
        print(f"Max: {aa_mi.max():.4f}")
        print(f"Min: {aa_mi.min():.4f}")
        
        # Spatial-Angular (cross-terms)
        sa_mi = mi_matrix[:64, 64:]
        print("\nSpatial-Angular MI:")
        print(f"Average: {sa_mi.mean():.4f}")
        print(f"Max: {sa_mi.max():.4f}")
        print(f"Min: {sa_mi.min():.4f}")
        
        # Save results
        output_dir = 'latent_analysis_results'
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, 'mutual_information_matrix_full.npy'), mi_matrix)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 