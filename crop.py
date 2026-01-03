from PIL import Image
import cv2
import json
import os

def crop_image_fixed_ratio(input_path, output_path, crop_width_ratio, crop_height_ratio, crop_origin_x, crop_origin_y):
    # Open the input image
    with Image.open(input_path) as img:
        # Get image dimensions
        width, height = img.size
        
        # Calculate crop dimensions based on the provided ratios
        crop_width = int(width * crop_width_ratio)
        crop_height = int(height * crop_height_ratio)
        
        # Ensure the crop box is within image bounds
        left = max(0, min(crop_origin_x, width - crop_width))
        upper = max(0, min(crop_origin_y, height - crop_height))
        right = min(width, left + crop_width)
        lower = min(height, upper + crop_height)
        
        # Crop the image
        cropped_img = img.crop((left, upper, right, lower))
        
        # Save the cropped image as PNG with high quality
        cropped_img.save(output_path, format="PNG", quality=100)

def select_crop_area(image_path):
    # Load the image
    img = cv2.imread(image_path)
    
    # Allow the user to select a region of interest (ROI)
    roi = cv2.selectROI("Select Crop Area", img, fromCenter=False, showCrosshair=True)
    
    # Close the selection window
    cv2.destroyAllWindows()
    
    # Return the top-left corner and dimensions of the ROI
    return roi  # roi is (x, y, w, h)

def save_crop_coordinates(coordinates, file_path):
    with open(file_path, 'w') as f:
        json.dump(coordinates, f)

def load_crop_coordinates(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None

# Example usage
if __name__ == "__main__":
    input_image_path = "./output/Error_map/Error_map_ours_danger.png"  # Replace with your image path
    output_image_path = "./output/Error_map/cropped_image_ours_danger.png"  # Replace with desired save path
    coordinates_file_path = "./3x3_results/ours_new/danger/crop_coordinates.json"  # File to save/load crop coordinates
    
    # Check if crop coordinates exist, otherwise select using mouse
    crop_coordinates = load_crop_coordinates(coordinates_file_path)
    if crop_coordinates is None:
        x, y, w, h = select_crop_area(input_image_path)
        crop_coordinates = {'x': x, 'y': y, 'w': w, 'h': h}
        save_crop_coordinates(crop_coordinates, coordinates_file_path)
    else:
        x, y, w, h = crop_coordinates['x'], crop_coordinates['y'], crop_coordinates['w'], crop_coordinates['h']
    
    # Calculate ratios based on saved coordinates
    with Image.open(input_image_path) as img:
        width, height = img.size
        crop_width_ratio = w / width
        crop_height_ratio = h / height
        crop_origin_x = x
        crop_origin_y = y
    
    crop_image_fixed_ratio(input_image_path, output_image_path, crop_width_ratio, crop_height_ratio, crop_origin_x, crop_origin_y)
    print(f"Cropped image saved at {output_image_path}")


