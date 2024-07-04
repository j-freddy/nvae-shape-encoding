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
    Some of the code is adapted from Segmentation2DMetrics in Vital. See repo[1]
    and paper[2].
    
    [1]: https://github.com/vitalab/vital [2]: Painchaud N, Skandarani Y, Judge
    T, Bernard O, Lalande A, Jodoin PM. Cardiac segmentation with strong
    anatomical guarantees. IEEE transactions on medical imaging. 2020 Jun
    17;39(11):3703-13.
    """

    def __init__(self, mask: torch.Tensor):
        # Taking 1 mask at a time instead of a batch
        num_classes, _, _ = mask.shape
        assert num_classes == ACDC.NUM_CLASSES
        
        self.mask = mask
    
    def _merge_classes(self, merged_class_ids: list[ACDC.ClassLabel]) -> np.ndarray:
        struct_mask = torch.zeros_like(self.mask[0, :, :])
        
        # Mask for the specified aggregate class
        for class_id in merged_class_ids:
            struct_mask += self.mask[acdc_class_id_to_idx(class_id), :, :]
        
        return struct_mask

    def _invert_and_pad(self, struct_mask: np.ndarray) -> np.ndarray:
        return np.pad(1 - struct_mask, (1, 1), "constant", constant_values=1)
    
    def _count_holes(self, struct_mask: np.ndarray) -> int:
        # Extract continuous regions of 1
        props = measure.regionprops(measure.label(struct_mask, connectivity=2))

        num_holes = 0

        for prop in props:
            # Skip the region open by the side (the one that includes padding)
            if prop.bbox[0] != 0:
                num_holes += 1

        return num_holes
    
    def count_holes(self, merged_class_ids: list[ACDC.ClassLabel]) -> int:
        struct_mask = self._merge_classes(merged_class_ids)
        struct_mask = self._invert_and_pad(struct_mask)
        return self._count_holes(struct_mask)

    def count_num_components(self, merged_class_ids: list[ACDC.ClassLabel]) -> int:
        struct_mask = self._merge_classes(merged_class_ids)
        return measure.label(struct_mask, connectivity=2).max()

    def perform_all(self):
        # Check for any presence of holes: in LV, RV, MYO, between LV and MYO,
        # between RV and MYO
        num_holes = self.count_holes([
            ACDC.ClassLabel.RV,
            ACDC.ClassLabel.MYO,
            ACDC.ClassLabel.LV,
        ])
        
        # Check for presence of more than 1 LV, RV or MYO
        num_rv = self.count_num_components([ACDC.ClassLabel.RV])
        num_myo = self.count_num_components([ACDC.ClassLabel.MYO])
        num_lv = self.count_num_components([ACDC.ClassLabel.LV])
        
        # Assuming only 1 RV and 1 MYO, check if RV is disconnected from MYO
        num_rv_myo = self.count_num_components([
            ACDC.ClassLabel.RV,
            ACDC.ClassLabel.MYO,
        ])
        
        # Valid: 0
        print(f"Number of holes: {num_holes}")
        # Valid: 1
        print(f"Number of RV: {num_rv}")
        # Valid: 1
        print(f"Number of MYO: {num_myo}")
        # Valid: 1
        print(f"Number of LV: {num_lv}")
        # Valid: 1
        print(f"Number of RV and MYO combined together: {num_rv_myo}")
