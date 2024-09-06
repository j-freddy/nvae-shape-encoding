import torch
import torchio as tio

from utils.const import CARDIAC_WIDTH

def preprocess(subject: tio.Subject) -> tio.Subject:
    mask = subject.mask.data[0, :, :, :]
    og_width, og_height, num_slices = mask.shape

    # :2 to ignore slice index
    nonzero_coords = torch.nonzero(mask)[:, :2]

    # Edge case: No non-zero coordinates
    if nonzero_coords.shape[0] == 0:
        width = max(og_width, og_height)
    else:
        # Get bounding box
        min_x = torch.min(nonzero_coords[:, 1]).item()
        max_x = torch.max(nonzero_coords[:, 1]).item()
        min_y = torch.min(nonzero_coords[:, 0]).item()
        max_y = torch.max(nonzero_coords[:, 0]).item()
        
        width = max(max_x - min_x, max_y - min_y)
    
    padding = 4

    transform = tio.transforms.Compose([
        # Crop to dimensions centred around the mask to minimise background
        tio.CropOrPad(
            (width + padding, width + padding, num_slices),
            mask_name="mask",
        ),
        tio.Resize((CARDIAC_WIDTH, CARDIAC_WIDTH, num_slices)),
        tio.RescaleIntensity((0, 1), percentiles=(1, 99)),
    ])
    
    return transform(subject)
