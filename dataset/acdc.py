import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ACDCMaskDataset(Dataset):
    def __init__(self, masks: torch.Tensor, augment: bool=False):
        self.masks = masks
        self.augment = augment
        
        self.augmentation_pipeline = transforms.Compose([
            transforms.RandomRotation(degrees=30),
        ])
    
    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        mask = self.masks[idx]

        if self.augment:
            mask = self.augmentation_pipeline(mask)

        return mask
