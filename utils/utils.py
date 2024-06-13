import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from const import SEED

def setup_device() -> torch.device:
    """
    Set up Torch device and set seed. Enforce all operations to be
    deterministic.

    Returns:
        torch.device: Device used for Torch scripts.
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

def show_samples(
    images: torch.Tensor,
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
    
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(images)
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
        num_classes=len(x_hat_argmax.unique())
    ).permute(0, 3, 1, 2)
    
    return x_hat_hard

def one_hot_to_image(x: torch.Tensor) -> torch.Tensor:
    """
    Given a one-hot encoded segmentation map, return the image representation.
    The map can either be probabilistic or discrete.
    """
    return x[:, 1:, :, :].float()

def soft_clamp(x: torch.Tensor, factor: float=5.0) -> torch.Tensor:
    return torch.tanh(x.div(factor)) * factor
