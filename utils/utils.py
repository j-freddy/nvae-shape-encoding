import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import sqrtm
from scipy.linalg import sqrtm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from torchvision.models import inception_v3
from torchvision.utils import make_grid

from arch.simclr.simclr import SimCLR
from arch.simclr.utils import load_simclr_backbone
from const import SEED

def setup_device() -> torch.device:
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
    return x[:, 1:, :, :].float() * 255

def fid_torchmetrics(real_data: torch.Tensor, fake_data: torch.Tensor) -> torch.Tensor:
    """
    Deprecated. Use fid_manual instead.
    
    Reason: This pipeline to evaluate generation quality does not align with
    empirical analysis as well as fid_manual.
    """
    # Pre: Data is ACDC one-hot, discretised encoded masks
    _, num_channels, _, _ = real_data.shape
    _, num_channels_fake, _, _ = fake_data.shape
    assert num_channels == 4
    assert num_channels_fake == 4
    
    assert len(real_data.unique()) == 2
    assert len(fake_data.unique()) == 2
    
    # Cast data to uint8 as image and discard background dimension
    real_data = real_data.to(torch.uint8)[:, 1:, :, :] * 255
    fake_data = fake_data.to(torch.uint8)[:, 1:, :, :] * 255
    
    fid = FrechetInceptionDistance(feature=2048)
    
    # Ensure data is on the same device
    fid.to(real_data.device)
    fake_data = fake_data.to(real_data.device)
    
    fid.update(real_data, real=True)
    fid.update(fake_data, real=False)
    return fid.compute()

# TODO Cite this function
# TODO Delete if resnet works better
def fid_manual(
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
    device: torch.device,
):
    def get_feats(x, model, device):
        with torch.no_grad():
            x = x.to(device)
            # Discard background dimension
            x = x[:, 1:, :, :].float()

            transform = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                nn.Upsample(size=(299, 299), mode="bilinear", align_corners=True),
            ])

            feats = model(transform(x))

        return feats.detach().cpu()
    
    # Load inception model
    inception_model = inception_v3(weights="DEFAULT", transform_input=False).to(device)
    
    inception_model.fc = nn.Identity()
    inception_model.eval()

    # Extract features for real images
    real_feats = []
    
    batch_size = 2
    real_data_split = torch.split(real_data, batch_size, dim=0)

    for real_data_batch in real_data_split:
        real_data_batch = real_data_batch * 2 - 1
        feats = get_feats(real_data_batch, inception_model, device)
        real_feats.append(feats)
        
    real_feats = torch.cat(real_feats, 0)

    # Extract features for generated images
    fake_feats = []
    
    batch_size = 2
    fake_data_split = torch.split(fake_data, batch_size, dim=0)

    for batch in fake_data_split:
        batch = batch * 2 - 1
        fake_feats.append(get_feats(batch, inception_model, device))
    
    fake_feats = torch.cat(fake_feats, 0)

    # Calculate mean and covariance
    mu_real = real_feats.mean(0)
    sigma_real = np.cov(real_feats.numpy(), rowvar=False)

    mu_fake = fake_feats.mean(0)
    sigma_fake = np.cov(fake_feats.numpy(), rowvar=False)

    # Compute FID score
    sum_sq_diff = torch.sum((mu_real - mu_fake) ** 2).item()
    covm_real_fake = sqrtm(sigma_real.dot(sigma_fake))
    
    # Check and correct for imaginary numbers from sqrt
    if np.iscomplexobj(covm_real_fake):
        covm_real_fake = covm_real_fake.real

    return sum_sq_diff + np.trace(sigma_real + sigma_fake - 2.0 * covm_real_fake)

def encode_embeddings(x: torch.Tensor, model: nn.Module, device: torch.device) -> torch.Tensor:
    def encode(x: torch.Tensor, model: nn.Module, device: torch.device):
        with torch.no_grad():
            x = x.to(device)
            x = one_hot_to_image(x)
            feats = model(x)

        return feats.detach().cpu()

    embeddings = []
    
    batch_size = 2
    x_split = torch.split(x, batch_size, dim=0)

    for x_batch in x_split:
        x_batch = x_batch * 2 - 1
        embeddings.append(encode(x_batch, model, device))
        
    return torch.cat(embeddings, dim=0)
    

def fid_resnet(
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
    device: torch.device,
):
    # TODO Do not hardcode
    path = "logs-simclr/simclr_acdc/resnet-18/checkpoints/epoch=18-step=133.ckpt"
    
    # Load pretrained SimCLR model
    resnet_model = load_simclr_backbone(path)
    resnet_model = resnet_model.to(device)

    # Extract features for real images
    real_feats = encode_embeddings(real_data, resnet_model, device)

    # Extract features for generated images
    fake_feats = encode_embeddings(fake_data, resnet_model, device)

    # Calculate mean and covariance
    mu_real = real_feats.mean(0)
    sigma_real = np.cov(real_feats.numpy(), rowvar=False)

    mu_fake = fake_feats.mean(0)
    sigma_fake = np.cov(fake_feats.numpy(), rowvar=False)

    # Compute FID score
    sum_sq_diff = torch.sum((mu_real - mu_fake) ** 2).item()
    covm_real_fake = sqrtm(sigma_real.dot(sigma_fake))
    
    # Check and correct for imaginary numbers from sqrt
    if np.iscomplexobj(covm_real_fake):
        covm_real_fake = covm_real_fake.real

    return sum_sq_diff + np.trace(sigma_real + sigma_fake - 2.0 * covm_real_fake)

def soft_clamp(x: torch.Tensor, factor: float=5.0) -> torch.Tensor:
    return torch.tanh(x.div(factor)) * factor
