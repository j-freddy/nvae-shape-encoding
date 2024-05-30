import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ACDCMaskDataset(Dataset):
    def __init__(
        self,
        masks: torch.Tensor,
        augment_rotation: bool=False,
        augment_simclr: bool=False,
    ):
        self.masks = masks
        self.num_classes = masks.shape[1]
        self.augment_rotation = augment_rotation
        self.augment_simclr = augment_simclr
        
        self.augmentation_pipeline = transforms.Compose([
            transforms.RandomRotation(degrees=30),
        ])
        
        # Colour jitter and Gaussian blur is not sensible for segmentation masks
        self.simclr_pipeline = transforms.Compose([
            transforms.RandomRotation(degrees=30),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomResizedCrop(
                size=128,
                scale=(0.8, 1.0),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            # transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,)),
        ])
    
    def __len__(self):
        return len(self.masks)

    def _augment_mask(self, mask: torch.Tensor, pipeline: transforms.Compose):
        # Do not apply augmentation to background class
        augmented_mask_no_bg = pipeline(mask[1:, :, :])
        
        # Collapse all classes into one
        # 128x128
        mask_collapsed, _ = torch.max(augmented_mask_no_bg, dim=0)
        
        # Get new background class
        # 128x128
        bg = (mask_collapsed == 0).int()
        
        # Combine
        mask = torch.cat([bg.unsqueeze(0), augmented_mask_no_bg], dim=0)
        
        return mask

    def __getitem__(self, idx):
        # 4x128x128
        mask = self.masks[idx]

        if self.augment_rotation:
            assert not self.augment_simclr
            return self._augment_mask(mask, self.augmentation_pipeline)
        
        if self.augment_simclr:
            assert not self.augment_rotation
            return [self.simclr_pipeline(mask) for _ in range(2)]

        return mask
