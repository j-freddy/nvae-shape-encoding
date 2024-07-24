import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import v2

from utils.const import ACDC
from utils.custom_augmentations import RandomGammaCorrection

class ACDCDataset(Dataset):
    """
    Automated Cardiac Diagnosis Challenge (ACDC) dataset.
    
    See ACDCDataModule docstring.
    """

    def __init__(
        self,
        scans: torch.Tensor,
        masks: torch.Tensor,
        conditions: torch.Tensor,
        eds: torch.Tensor,
        equalise: bool=False,
        augment: bool=False,
    ):
        assert len(scans) == len(masks) == len(conditions)
        
        self.equalise = equalise
        self.augment = augment
        
        self.scans = scans
        self.masks = masks
        self.conditions = conditions
        self.eds = eds
        
        self.equalise_pipeline = v2.RandomEqualize(p=1.0)
        
        self.augmentation_pipeline_scan = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            RandomGammaCorrection(range=(0.5, 1.5)),
        ])
        
        self.augmentation_pipeline_mask = transforms.Compose([
            transforms.RandomHorizontalFlip(),
        ])
    
    def __len__(self) -> int:
        return len(self.scans)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scan = self.scans[idx]
        mask = self.masks[idx]
        condition = self.conditions[idx]
        ed = self.eds[idx]
        
        # Values should be preprocessed as 0, 1 before passing into pipeline
        assert set(mask.unique().tolist()).issubset({0, 1})
        
        if self.equalise:
            scan = self.equalise_pipeline(scan)
        
        if self.augment:
            # Ensure same random augmentation is applied to both scan and mask
            state = torch.get_rng_state()
            scan = self.augmentation_pipeline_scan(scan)
            torch.set_rng_state(state)
            mask = self.augmentation_pipeline_mask(mask)
        
        return scan, mask, condition, ed

class ACDCMaskDataset(Dataset):
    """
    Automated Cardiac Diagnosis Challenge (ACDC) masks dataset.
    
    See ACDCMaskDataModule docstring.
    
    Intensity values of unprocessed masks are [0, 1]. This is rescaled to [-1,
    1] in the SimCLR pipeline only, for stability. In practice, the values will
    be rescaled to [-1, 1] for training VAEs as well, but it is not done in this
    preprocessing step as the [0, 1] intensity values are needed in calculating
    cross-entropy loss. Therefore for VAEs, the rescaling is done in the forward
    passes instead (i.e. x := 2 * x - 1.0).
    """

    def __init__(
        self,
        masks: torch.Tensor,
        augment_rotation: bool=False,
        augment_simclr: bool=False,
        # If True, return the original mask as well as the augmented pair
        return_original: bool=False,
    ):
        self.masks = masks
        self.num_classes = masks.shape[1]
        self.augment_rotation = augment_rotation
        self.augment_simclr = augment_simclr
        self.return_original = return_original
        
        self.augmentation_pipeline = transforms.RandomRotation(degrees=30)

        self.simclr_pipeline = transforms.Compose([
            transforms.RandomRotation(degrees=180),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(
                size=ACDC.WIDTH,
                scale=(0.8, 1.0),
                ratio=(1, 1),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        
        self.normalise_pipeline = transforms.Normalize((0.5,), (0.5,))
    
    def __len__(self) -> int:
        return len(self.masks)

    def _augment_mask(self, mask: torch.Tensor, pipeline: transforms.Compose) -> torch.Tensor:
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

    def __getitem__(self, idx: int) -> torch.Tensor:
        # 4x128x128
        mask = self.masks[idx]
        
        # Values should be preprocessed as 0, 1 before passing into pipeline
        assert set(mask.unique().tolist()).issubset({0, 1})

        if self.augment_rotation:
            assert not self.augment_simclr
            return self._augment_mask(mask, self.augmentation_pipeline)
        
        if self.augment_simclr:
            assert not self.augment_rotation
            pair = [self.simclr_pipeline(mask) for _ in range(2)]
            
            if self.return_original:
                mask = self.normalise_pipeline(mask)
                pair = [mask] + pair
            return pair

        return mask
