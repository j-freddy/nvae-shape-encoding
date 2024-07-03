import numpy as np
from skimage import measure
import torch

from const import ACDC
from utils.utils import acdc_class_id_to_idx

# TODO Write some tests for this
# ./test/test_anatomical_validity.ipynb
# Maybe in a notebook: Take a test image, create a hole, etc.

class AnatomicalValidity:
    """
    Heavily adapted from Segmentation2DMetrics in Vital. See repo[1] and
    paper[2].
    
    [1]: https://github.com/vitalab/vital [2]: Painchaud N, Skandarani Y, Judge
    T, Bernard O, Lalande A, Jodoin PM. Cardiac segmentation with strong
    anatomical guarantees. IEEE transactions on medical imaging. 2020 Jun
    17;39(11):3703-13.
    """

    def __init__(self, mask: torch.Tensor):
        # TODO I try 1 mask at a time for now
        num_classes, _, _ = mask.shape
        assert num_classes == ACDC.NUM_CLASSES
        
        self.mask = mask
    
    def _count_holes(self, struct_mask: np.ndarray) -> int:
        # Extract continuous regions of 1
        props = measure.regionprops(measure.label(struct_mask, connectivity=2))

        num_holes = 0

        for prop in props:
            # Skip the region open by the side (the one that includes padding)
            if prop.bbox[0] != 0:
                num_holes += 1

        return num_holes
    
    def count_holes(self, class_ids_to_merge: list[ACDC.ClassLabel]) -> int:
        struct_mask = torch.zeros_like(self.mask[0, :, :])
        
        # Mask for the specified aggregate class
        for class_id in class_ids_to_merge:
            struct_mask += self.mask[acdc_class_id_to_idx(class_id), :, :]

        # Invert and pad mask
        struct_mask = np.pad(1 - struct_mask, (1, 1), "constant", constant_values=1)
        
        return self._count_holes(struct_mask)

    def perform_all(self):
        # Check for any presence of holes: in LV, RV, MYO, between LV and MYO,
        # between RV and MYO
        num_holes = self.count_holes([
            ACDC.ClassLabel.RV,
            ACDC.ClassLabel.MYO,
            ACDC.ClassLabel.LV,
        ])
        
        print(f"Number of holes: {num_holes}")
