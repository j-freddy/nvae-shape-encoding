import numpy as np
from skimage import measure, morphology
from skimage.measure._regionprops import RegionProperties
import torch

from const import ACDC
from utils.utils import acdc_class_id_to_idx

class AnatomicalValidityChecker:
    """
    The code for counting holes is adapted from Segmentation2DMetrics in Vital.
    See repo[1] and paper[2].
    
    [1]: https://github.com/vitalab/vital [2]: Painchaud N, Skandarani Y, Judge
    T, Bernard O, Lalande A, Jodoin PM. Cardiac segmentation with strong
    anatomical guarantees. IEEE transactions on medical imaging. 2020 Jun
    17;39(11):3703-13.
    """

    def __init__(self, mask: torch.Tensor):
        # Using Skimage so move to CPU
        mask = mask.cpu()
        
        num_classes, _, _ = mask.shape
        assert num_classes == ACDC.NUM_CLASSES
        assert set(mask.unique().tolist()).issubset({0, 1})
        self.mask = mask.int()

    def _get_struct_mask(self, class_id: ACDC.ClassLabel) -> torch.Tensor:
        return self.mask[acdc_class_id_to_idx(class_id), :, :]
    
    def _merge_classes(self, merged_class_ids: list[ACDC.ClassLabel]) -> np.ndarray:
        struct_mask = torch.zeros_like(self.mask[0, :, :])
        
        # Mask for the specified aggregate class
        for class_id in merged_class_ids:
            struct_mask += self._get_struct_mask(class_id)
        
        return struct_mask

    def _invert_and_pad(self, struct_mask: np.ndarray) -> np.ndarray:
        return np.pad(1 - struct_mask, (1, 1), "constant", constant_values=1)
    
    def _count_holes(self, struct_mask: np.ndarray) -> int:
        # Extract continuous regions of 1
        region_props: list[RegionProperties] = measure.regionprops(
            measure.label(struct_mask, connectivity=2),
        )

        num_holes = 0

        for region in region_props:
            # Skip the region open by the side (the one that includes padding)
            if region.bbox[0] != 0:
                num_holes += 1

        return num_holes
    
    def count_holes(self, merged_class_ids: list[ACDC.ClassLabel]) -> int:
        struct_mask = self._merge_classes(merged_class_ids)
        struct_mask = self._invert_and_pad(struct_mask)
        return self._count_holes(struct_mask)

    def count_num_components(self, merged_class_ids: list[ACDC.ClassLabel]) -> int:
        struct_mask = self._merge_classes(merged_class_ids)
        return measure.label(struct_mask, connectivity=2).max()

    def are_classes_disconnected(self, class_id1: ACDC.ClassLabel, class_id2: ACDC.ClassLabel) -> bool:
        mask1 = self._get_struct_mask(class_id1)
        mask2 = self._get_struct_mask(class_id2).numpy()
        
        # Exception: class 2 is empty
        if not np.any(mask2):
            return False
        
        labeled_mask1 = measure.label(mask1, connectivity=2)
        region_props: list[RegionProperties] = measure.regionprops(labeled_mask1)
    
        for region in region_props:
            component1 = (labeled_mask1 == region.label).astype(np.uint8)
            
            # Pad 1px around region
            component1_dilated = morphology.binary_dilation(component1).astype(np.uint8)
            
            # If padded mask does not overlap, a region in class 1 is
            # disconnected
            if not np.any(component1_dilated & mask2):
                return True
        
        return False

    def do_classes_touch(self, class_id1: ACDC.ClassLabel, class_id2: ACDC.ClassLabel) -> bool:
        mask1 = self._get_struct_mask(class_id1)
        mask2 = self._get_struct_mask(class_id2).numpy()
        # Pad 1px around one mask
        mask1_dilated = morphology.binary_dilation(mask1).astype(np.uint8)
        
        # If classes originally touch, the padded masks will overlap
        return np.any(mask1_dilated & mask2)

    def summarise_conditions(self) -> int:
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
        
        # Check if RV is disconnected from MYO
        rv_disconnected_from_myo = self.are_classes_disconnected(
            ACDC.ClassLabel.RV,
            ACDC.ClassLabel.MYO,
        )
        
        # Check if LV touches RV or background
        touch_lv_rv = self.do_classes_touch(ACDC.ClassLabel.LV, ACDC.ClassLabel.RV)
        touch_lv_bg = self.do_classes_touch(ACDC.ClassLabel.LV, ACDC.ClassLabel.BG)
        
        return {
            "num_holes": num_holes,
            "num_rv": num_rv,
            "num_myo": num_myo,
            "num_lv": num_lv,
            "rv_disconnected_from_myo": rv_disconnected_from_myo,
            "lv_touches_rv": touch_lv_rv,
            "lv_touches_bg": touch_lv_bg,
        }

    def count_violations(self) -> int:
        conditions = self.summarise_conditions()
        violations = [
            conditions["num_holes"] > 0,
            conditions["num_rv"] > 1,
            conditions["num_myo"] > 1,
            conditions["num_lv"] > 1,
            conditions["rv_disconnected_from_myo"],
            conditions["lv_touches_rv"],
            conditions["lv_touches_bg"],
        ]
        return sum(violations)
