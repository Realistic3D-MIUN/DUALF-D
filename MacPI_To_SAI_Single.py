from PIL import Image
import os
import numpy as np

def convert_to_sub_aperture(macropixel_image_path, output_directory, subgrid_size=(3,3)):
    macropixel_image = Image.open(macropixel_image_path)
    macropixel_image_data = np.array(macropixel_image)

    height, width, _ = macropixel_image_data.shape
    
    sub_height = height // subgrid_size[1]
    sub_width = width // subgrid_size[0]
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for i in range(subgrid_size[1]):
        for j in range(subgrid_size[0]):
            # Create an empty numpy array for the sub-aperture image
            sub_image_data = np.zeros((sub_height, sub_width, 3), dtype=np.uint8)
            
            for y in range(sub_height):
                for x in range(sub_width):
                    macropixel_x = subgrid_size[0] * x + j
                    macropixel_y = subgrid_size[1] * y + i

                    #print(macropixel_x)

                    #print(macropixel_y)
                    
                    sub_image_data[y, x] = macropixel_image_data[macropixel_y, macropixel_x]
            
            sub_image = Image.fromarray(sub_image_data)
            sub_image_name = f"sub_image_{i}_{j}.png"
            sub_image_path = os.path.join(output_directory, sub_image_name)
            sub_image.save(sub_image_path)
macropixel_image_path = './output_new_0001/3_compressed_q3.0/Output_macropixel_203.png'
#macropixel_image_path = './output/Input_Image.png'
output_directory = './3x3_results/ours_new/danger/'

convert_to_sub_aperture(macropixel_image_path, output_directory)





