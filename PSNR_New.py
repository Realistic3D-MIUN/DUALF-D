from torchvision import transforms
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio
import numpy as np
import torch

'''


# Function to load an image and convert it to a tensor
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Ensure image is RGB
    transform = transforms.ToTensor() # Convert images to tensors
    return transform(image).unsqueeze(0)  # Add batch dimension

def calculate_psnr(img1, img2):
    # Ensure that img1 and img2 are tensors
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1)
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2)

    # Calculate the mean squared error (MSE) using PyTorch
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    max_pixel = 1
    # Calculate PSNR using PyTorch
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

epoch = 260
# Load images
#image1 = load_image('./output/input_image.png')  # Update path to your first image
#image2 = load_image('./output/output_image_'+str(epoch)+'.png')  # Update path to your second image

image1 = load_image('./output/Org_Input/sub_image_1_1.png')
image2 = load_image('./output/sub_image_1_1.png')

# Initialize the PSNR calculator
psnr_calculator = PeakSignalNoiseRatio()

# Compute PSNR
psnr_value1 = psnr_calculator(image1, image2)
psnr_value2 = calculate_psnr(image1, image2)

# Print the result
print("PSNR value:", psnr_value1)
print("PSNR value:", psnr_value2)

import numpy as np
from PIL import Image

def rgb_to_y(rgb_image):
    """Convert an RGB image to the Y component of the YCbCr color space."""
    if isinstance(rgb_image, Image.Image):
        rgb = np.asarray(rgb_image)
    else:
        rgb = rgb_image
    Y = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return Y

def calculate_psnr(image1, image2):
    """Calculate the PSNR between two images."""#
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def main(image_path1, image_path2):
    # Load images
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)

    # Convert images to Y component
    y1 = rgb_to_y(img1)
    y2 = rgb_to_y(img2)

    # Calculate PSNR
    psnr_value = calculate_psnr(y1, y2)
    print(f"PSNR for the Y component: {psnr_value} dB")

# Replace 'path_to_image1.jpg' and 'path_to_image2.jpg' with the paths to your images
main('./output/Org_Input/sub_image_1_1.png', './output/sub_image_1_1.png')
'''
import numpy as np
from PIL import Image

def rgb_to_y(rgb_image):
    """Convert an RGB image to the Y component of the YCbCr color space."""
    if isinstance(rgb_image, Image.Image):
        rgb = np.asarray(rgb_image)
    else:
        rgb = rgb_image
    rgb = rgb.astype(np.float32)
    R = rgb[:, :, 0]
    G = rgb[:, :, 1]
    B = rgb[:, :, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    return Y

def rgb_to_ycbcr(rgb_image):
    """Convert an RGB image to YCbCr."""
    if isinstance(rgb_image, Image.Image):
        rgb = np.asarray(rgb_image)
    else:
        rgb = rgb_image
    rgb = rgb.astype(np.float32)
    R = rgb[:, :, 0]
    G = rgb[:, :, 1]
    B = rgb[:, :, 2]
    # Conversion formulas
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
    Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128
    return Y, Cb, Cr

def subsample_420(chroma):
    """Subsample the chroma component to 4:2:0."""
    return chroma[::2, ::2]

def calculate_psnr(image1, image2):
    """Calculate the PSNR between two images."""
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_psnr_ycbcr420(image1, image2):
    """Calculate PSNR between two images in YCbCr 4:2:0 format."""
    # Convert images to YCbCr
    Y1, Cb1, Cr1 = rgb_to_ycbcr(image1)
    Y2, Cb2, Cr2 = rgb_to_ycbcr(image2)

    # Subsample Cb and Cr components
    Cb1_420 = subsample_420(Cb1)
    Cb2_420 = subsample_420(Cb2)
    Cr1_420 = subsample_420(Cr1)
    Cr2_420 = subsample_420(Cr2)

    # Compute MSE for Y component
    mse_Y = np.mean((Y1 - Y2) ** 2)
    # Compute MSE for Cb component
    mse_Cb = np.mean((Cb1_420 - Cb2_420) ** 2)
    # Compute MSE for Cr component
    mse_Cr = np.mean((Cr1_420 - Cr2_420) ** 2)

    # Total MSE as weighted sum based on the number of samples
    N_Y = Y1.size
    N_C = Cb1_420.size  # Should be N_Y / 4
    total_mse = (mse_Y * N_Y + mse_Cb * N_C + mse_Cr * N_C) / (N_Y + 2 * N_C)

    # Compute PSNR
    if total_mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / total_mse)
    return psnr

def calculate_psnr_rgb(image1, image2):
    """Calculate PSNR between two RGB images."""
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def main(image_path1, image_path2):
    # Load images
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)

    # Convert images to numpy arrays
    img1_array = np.asarray(img1).astype(np.float32)
    img2_array = np.asarray(img2).astype(np.float32)

    # Calculate PSNR for RGB images
    psnr_rgb = calculate_psnr_rgb(img1_array, img2_array)
    print(f"PSNR for RGB images: {psnr_rgb} dB")

    # Calculate PSNR for Y component only
    y1 = rgb_to_y(img1_array)
    y2 = rgb_to_y(img2_array)
    psnr_y = calculate_psnr(y1, y2)
    print(f"PSNR for the Y component: {psnr_y} dB")

    # Calculate PSNR for YCbCr 4:2:0
    psnr_ycbcr420 = calculate_psnr_ycbcr420(img1_array, img2_array)
    print(f"PSNR for YCbCr 4:2:0: {psnr_ycbcr420} dB")

# Replace 'path_to_image1.png' and 'path_to_image2.png' with the paths to your images
#main('./output/Org_Input/sub_image_1_1.png', './output/sub_image_1_1.png')
main('./output/sub_image_1_1.png', './output/Z/sub_image_1_1.png')
