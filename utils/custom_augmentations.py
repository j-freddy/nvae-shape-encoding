import numpy as np
import torch
import torch.nn as nn

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

class AverageSmoothing:
    def __init__(self, kernel_size: int):
        self.kernel_size = kernel_size
        self.net = nn.AvgPool2d(self.kernel_size, stride=1, padding=self.kernel_size // 2)

    def __call__(self, img):
        # Majority vote: Perform average pooling
        img_aug = self.net(img)
        return torch.round(img_aug)
