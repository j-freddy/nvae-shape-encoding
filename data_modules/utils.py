import torch
import torchio as tio

from utils.const import CARDIAC_WIDTH

def preprocess(subject: tio.Subject) -> tio.Subject:
    mask = subject.mask.data[0, :, :, :]
    _, _, num_slices = mask.shape

    # :2 to ignore slice index
    nonzero_coords = torch.nonzero(mask)[:, :2]

    # Get bounding box
    min_x = torch.min(nonzero_coords[:, 1]).item()
    max_x = torch.max(nonzero_coords[:, 1]).item()
    min_y = torch.min(nonzero_coords[:, 0]).item()
    max_y = torch.max(nonzero_coords[:, 0]).item()
    
    width = max(max_x - min_x, max_y - min_y)
    
    # With rotation augmentation, padding is required to prevent cropping

    # The exact padding can be calculated by drawing a square inscribed within a
    # circle inscribed within a larger square, since rotating a square traces
    # out a circle
    
    # Let x be the original size (i.e. width of square)
    # Then the radius of circle is x / sqrt(2)
    # Then the width of larger square is 2x / sqrt(2) = x * sqrt(2)
    
    # Absolute padding after resizing to 128x128 is 128 - (64 * sqrt(2)) = 37.5
    # padding = math.ceil(width * math.sqrt(2))
    
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
