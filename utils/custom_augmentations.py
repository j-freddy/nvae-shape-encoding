import numpy as np
import torch
from torchvision import transforms

class RandomBlackBoxCrop:
    def __init__(self, size_range: tuple[int, int]):
        self.size_range = size_range

    def __call__(self, img):
        batch_size, _, width, height = img.shape

        mask = torch.ones_like(img)
        
        for i in range(batch_size):
            # Choose random width and height
            crop_width = int(width * np.random.uniform(*self.size_range))
            crop_height = int(height * np.random.uniform(*self.size_range))
            
            # Choose random position
            x = np.random.randint(0, width - crop_width)
            y = np.random.randint(0, height - crop_height)

            # Crop
            mask[i, :, x:x + crop_width, y:y + crop_height] = 0

        return img * mask
