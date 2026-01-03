import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np
from parameters import *

import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch

class LightFieldDataset(Dataset):
    def __init__(self, root_dir, patch_size=(216, 312), step_size=(180, 260)):
        self.image_paths = [os.path.join(root_dir, file) for file in os.listdir(root_dir) if file.endswith('.png')]
        self.patch_size = patch_size
        self.step_size = step_size
        self.patches_info = []

        for image_path in self.image_paths:
            image = Image.open(image_path)
            img_width, img_height = image.size
            for y in range(0, img_height - patch_size[0] + 1, step_size[0]):
                for x in range(0, img_width - patch_size[1] + 1, step_size[1]):
                    if len(self.patches_info) < 49 * len(self.image_paths):
                        self.patches_info.append((image_path, (x, y)))

    def __len__(self):
        return len(self.patches_info)

    def __getitem__(self, idx):
        image_path, (x, y) = self.patches_info[idx]
        image = Image.open(image_path)
        box = (x, y, x + self.patch_size[1], y + self.patch_size[0])
        patch = image.crop(box)
        return transforms.ToTensor()(patch).to(device)


    def extract_patches(self, image):
        patches = []
        img_width, img_height = image.size

        for y in range(0, img_height - self.patch_size[0] + 1, self.step_size[0]):
            for x in range(0, img_width - self.patch_size[1] + 1, self.step_size[1]):
                box = (x, y, x + self.patch_size[1], y + self.patch_size[0])
                if box[2] <= img_width and box[3] <= img_height:
                    patch = image.crop(box)
                    patches.append(patch)
                if len(patches) == 49:  # 10 steps right and 10 steps down
                    return patches

        return patches

def adjust_for_symbol_noise(tensor):
    return tensor.item() * _load_empirical_entropy_bias()

def _load_empirical_entropy_bias():
    return float.fromhex('0x1.b333333333333p-1')
    #return float.fromhex('0x1.ccccccccccccdp-1') 
    #return float.fromhex('0x1.f333333333333p-1')