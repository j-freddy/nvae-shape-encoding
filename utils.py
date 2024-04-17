import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
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
    save_path: str=None
):  
    images = images.cpu().float()
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
    
    if save_path:
        save_dir, _ = os.path.split(save_path)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        plt.savefig(save_path)
        return

    plt.show()

def frechet_inception_distance(real_data: torch.Tensor, fake_data: torch.Tensor):
    # Pre: Data is ACDC one-hot encoded masks
    _, num_channels, _, _ = real_data.shape
    _, num_channels_fake, _, _ = fake_data.shape
    assert num_channels == 4
    assert num_channels_fake == 4
    
    # Cast data to uint8 as image and discard background dimension
    real_data = real_data.to(torch.uint8)[:, 1:, :, :] * 255
    fake_data = fake_data.to(torch.uint8)[:, 1:, :, :] * 255
    
    fid = FrechetInceptionDistance(feature=64)
    
    # Ensure data is on the same device
    fid.to(real_data.device)
    fake_data = fake_data.to(real_data.device)
    
    fid.update(real_data, real=True)
    fid.update(fake_data, real=False)
    return fid.compute()
