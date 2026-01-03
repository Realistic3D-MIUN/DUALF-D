import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_error_map(image1, image2):
    # Ensure images are of the same size
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same dimensions.")

    # Convert images to YCbCr color space
    image1_ycbcr = cv2.cvtColor(image1, cv2.COLOR_BGR2YCrCb)
    image2_ycbcr = cv2.cvtColor(image2, cv2.COLOR_BGR2YCrCb)

    # Calculate absolute difference between images in Y channel
    error_map = cv2.absdiff(image1_ycbcr[:, :, 0], image2_ycbcr[:, :, 0])
    return error_map

def calculate_average_error_map(folder1, folder2):
    # Get list of images in each folder
    images1 = sorted(os.listdir(folder1))
    images2 = sorted(os.listdir(folder2))

    # Ensure both folders have the same number of images
    if len(images1) != len(images2):
        raise ValueError("Folders must contain the same number of images.")

    total_error_map = None
    num_images = len(images1)

    for img1_name, img2_name in zip(images1, images2):
        img1_path = os.path.join(folder1, img1_name)
        img2_path = os.path.join(folder2, img2_name)

        # Load images
        image1 = cv2.imread(img1_path)
        image2 = cv2.imread(img2_path)

        # Ensure images are loaded
        if image1 is None or image2 is None:
            raise FileNotFoundError(f"One or both of the images {img1_name}, {img2_name} could not be loaded.")

        # Calculate error map
        error_map = calculate_error_map(image1, image2)

        # Accumulate error maps
        if total_error_map is None:
            total_error_map = np.float32(error_map)
        else:
            total_error_map += np.float32(error_map)

    # Calculate average error map
    average_error_map = total_error_map / num_images
    return np.uint8(average_error_map)

def save_error_map_as_png(error_map, output_path):
    # Amplify the error map for better visualization
    error_map_amplified = cv2.convertScaleAbs(error_map, alpha=10, beta=50)

    # Normalize the error map for better visualization
    error_map_normalized = cv2.normalize(error_map_amplified, None, 0, 255, cv2.NORM_MINMAX)

    # Set up the figure without axes for saving with matplotlib
    fig, ax = plt.subplots()
    ax.imshow(error_map_normalized, cmap='jet')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save the error map using matplotlib for better quality
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)



if __name__ == "__main__":
    # Define folders containing reference and target images
    #folder1 = './3x3_results/Jpeg_Pleno_Results/lambda_5000/Danger/5000/'
    #folder1 = './3x3_results/RLVC_Results/Bikes/BasketballPass_PSNR_256/frames/'
    #folder1 = './3x3_results/HLVC_Results/Bikes/BasketballPass_com_fast_PSNR_256/'
    #folder1 = './3x3_results/ours/'
    folder1 = './3x3_results/ours_new/danger/'
    #folder1 = './3x3_results/JingLei/danger_new_0.12/'

    folder2 = './3x3_results/Original_data/Data/danger/'
    #folder2 = './3x3_results/jinglei_gt/danger_gt/'

    # Calculate the average error map
    average_error_map = calculate_average_error_map(folder1, folder2)

    # Save the average error map
    save_error_map_as_png(average_error_map, "./output/Error_map/Error_map_ours_danger.png")



