import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import v2

class MnMsDataset(Dataset):
    """
    Multi-Centre, Multi-Vendor & Multi-Disease Cardiac Image Segmentation
    Challenge (M&Ms) dataset.
    
    See MnMsDataModule docstring.
    """

    def __init__(
        self,
        scans: torch.Tensor,
        masks: torch.Tensor,
        conditions: torch.Tensor,
        eds: torch.Tensor,
        augment: bool=False,
    ):
        assert len(scans) == len(masks) == len(conditions) == len(eds)

        self.augment = augment
        
        self.scans = scans
        self.masks = masks
        self.conditions = conditions
        self.eds = eds
        
        self.num_channels = scans.shape[1]
        self.num_classes = masks.shape[1]
        
        self.augmentation_pipeline_scan = transforms.Compose([
            transforms.RandomHorizontalFlip(),
        ])
        
        self.augmentation_pipeline_mask = transforms.Compose([
            transforms.RandomHorizontalFlip(),
        ])
    
    def __len__(self) -> int:
        return len(self.scans)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scan = self.scans[idx]
        mask = self.masks[idx]
        condition = self.conditions[idx]
        ed = self.eds[idx]
        
        # Values should be preprocessed as 0, 1 before passing into pipeline
        assert set(mask.unique().tolist()).issubset({0, 1})
        
        if self.augment:
            # Ensure same random augmentation is applied to both scan and mask
            state = torch.get_rng_state()
            scan = self.augmentation_pipeline_scan(scan)
            torch.set_rng_state(state)
            mask = self.augmentation_pipeline_mask(mask)
        
        return scan, mask, condition, ed
