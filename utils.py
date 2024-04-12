import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

from const import SEED

def setup_device():
    """
    Set up Torch device and set seed. Enforce all operations to be
    deterministic.

    Returns:
        torch.device: Device used for Torch scripts.
    """
    # Use GPU if available
    device = torch.device("cuda") if torch.cuda.is_available()\
        else torch.device("cpu")

    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        # Enforce all operations to be deterministic on GPU for reproducibility
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    return device

def show_samples(
    images: torch.Tensor,
    rgb: bool=True,
    nrow: int=8,
    figsize: tuple[int, int]=(6,6),
):  
    images = make_grid(images, nrow=nrow, padding=2, normalize=True)
    
    if not rgb:
        # Remove channel dimension so imshow uses cmap
        images = images[0]
    else:
        images = np.transpose(images.numpy(), (1, 2, 0))
    
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(images)
    plt.tight_layout()
    plt.show()
