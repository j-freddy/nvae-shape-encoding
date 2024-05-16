import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ACDCMaskDataset(Dataset):
    def __init__(self, masks: torch.Tensor, augment: bool=False):
        self.masks = masks
        self.num_classes = masks.shape[1]
        self.augment = augment
        
        self.augmentation_pipeline = transforms.Compose([
            transforms.RandomRotation(degrees=30),
        ])
    
    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        # 4x128x128
        mask = self.masks[idx]

        if self.augment:
            # Do not apply augmentation to background class
            augmented_mask_no_bg = self.augmentation_pipeline(mask[1:, :, :])
            
            # Collapse all classes into one
            # 128x128
            mask_collapsed, _ = torch.max(augmented_mask_no_bg, dim=0)
            
            # Get new background class
            # 128x128
            bg = (mask_collapsed == 0).int()
            
            # Combine
            mask = torch.cat([bg.unsqueeze(0), augmented_mask_no_bg], dim=0)

        return mask
