import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from utils.const import MASK_NUM_CLASSES, SEED, MaskClassLabel

def setup_device() -> torch.device:
    """
    Set up Torch device and set seed. Enforce all operations to be
    deterministic.

    Returns:
        torch.device: Device used for Torch operations.
    """
    # Use GPU if available
    device = torch.device("cpu")
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.mps.manual_seed(SEED)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        # Enforce all operations to be deterministic on GPU for reproducibility
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    return device

def listdir(dir: str) -> list[str]:
    ids = os.listdir(dir)

    # Exclude .DS_Store on macOS
    if ".DS_Store" in ids:
        ids.remove(".DS_Store")

    return ids

def clamp(x: float, low: float, high: float) -> float:
    return max(low, min(x, high))

def mask_class_id_to_idx(class_id: MaskClassLabel) -> int:
    return class_id.value

def show_samples(
    images: torch.Tensor,
    overlay_mask: torch.Tensor=None,
    rgb: bool=True,
    ncol: int=8,
    figsize: tuple[int, int]=(6,6),
    save_path: str=None,
    display: bool=True,
):
    images = images.cpu().float()
    images = make_grid(images, nrow=ncol, padding=2, normalize=True)
    
    if not rgb:
        # Remove channel dimension so imshow uses cmap
        images = images[0]
    else:
        images = np.transpose(images.numpy(), (1, 2, 0))

    if overlay_mask is not None:
        assert torch.min(overlay_mask) >= 0
        
        # Binarise mask: set all values > 0 to 1
        overlay_mask = overlay_mask > 0
        
        overlay_mask = overlay_mask.cpu().float()
        overlay_mask = make_grid(overlay_mask, nrow=ncol, padding=2, normalize=True)
        overlay_mask = overlay_mask[0]
    
        overlay_rgba = torch.zeros(overlay_mask.shape + (4,))
        # Red overlay
        overlay_rgba[:, :, 0] = 1
        # Make everything else transparent
        overlay_rgba[:, :, 3] = overlay_mask
    
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(images)
    if overlay_mask is not None:
        plt.imshow(overlay_rgba)
    plt.tight_layout()
    
    if save_path:
        save_dir, _ = os.path.split(save_path)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        plt.savefig(save_path)

    if display:
        plt.show()

def show_scans_and_masks(
    scans: torch.Tensor,
    masks: torch.Tensor,
    ncol: int=6,
    figsize: tuple[int, int]=(6, 4),
    save_path: str=None,
    display: bool=True,
):
    scans = scans.cpu().float()
    masks = masks.cpu().float()
    
    scans = make_grid(scans, nrow=ncol, padding=2, normalize=True)
    masks = make_grid(masks, nrow=ncol, padding=2, normalize=True)
    
    scans = np.transpose(scans.numpy(), (1, 2, 0))
    masks = masks[0]
    
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(scans)
    plt.imshow(masks, alpha=0.64)
    plt.tight_layout()
    
    if save_path:
        save_dir, _ = os.path.split(save_path)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        plt.savefig(save_path)

    if display:
        plt.show()

def discretise(x_hat: torch.Tensor) -> torch.Tensor:
    """
    Given a probablistic segmentation map, round each pixel to the nearest
    class and return the non-probablistic map.
    """
    x_hat_argmax = torch.argmax(x_hat, dim=1)
    x_hat_hard = F.one_hot(
        x_hat_argmax.long(),
        num_classes=MASK_NUM_CLASSES,
    ).permute(0, 3, 1, 2)
    
    return x_hat_hard

def one_hot(masks: torch.Tensor) -> torch.Tensor:
    """
    Given a segmentation map with shape [B, 1, H, W], return the one-hot encoded
    representation with shape [B, num_classes, H, W].
    """
    masks = torch.squeeze(masks, dim=1)
    masks_onehot = F.one_hot(
        masks.long(),
        num_classes=MASK_NUM_CLASSES,
    ).permute(0, 3, 1, 2)
    
    return masks_onehot

def one_hot_to_image(x: torch.Tensor) -> torch.Tensor:
    """
    Given a one-hot encoded segmentation map, return the image representation.
    The map can either be probabilistic or discrete.
    """
    return x[:, 1:, :, :].float()

def soft_clamp(x: torch.Tensor, factor: float=5.0) -> torch.Tensor:
    return torch.tanh(x.div(factor)) * factor

def get_data(
    loader: torch.utils.data.DataLoader,
    is_tuple: bool=False,
    scan_idx: int=0,
    mask_idx: int=1,
) -> torch.Tensor:
    """
    Get all data from a DataLoader.
    
    Each data point is either a mask (e.g. ACDCMaskDataModule), or a tuple (e.g.
    ACDCDataModule). An example of a tuple is (scan, mask, other_data). If using
    a data loader that returns tuples, set is_tuple=True and specify scan and
    mask index positions if needed.
    """
    if not is_tuple:
        data = []
        
        for x in loader:
            data.append(x)
        
        return torch.cat(data, dim=0)

    # Tuple
    scans = []
    masks = []
    
    for x in loader:
        scans.append(x[scan_idx])
        masks.append(x[mask_idx])
    
    return torch.cat(scans, dim=0), torch.cat(masks, dim=0)
