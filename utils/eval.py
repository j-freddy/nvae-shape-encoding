import numpy as np
from scipy.linalg import sqrtm
import torch
import torch.nn as nn
from torchmetrics.image.fid import FrechetInceptionDistance

from arch.simclr.utils import load_simclr_backbone
from utils.utils import one_hot_to_image

def fid_torchmetrics(real_data: torch.Tensor, fake_data: torch.Tensor) -> torch.Tensor:
    """
    Deprecated. Use fid_resnet instead.
    
    Reason: This pipeline to evaluate generation quality does not align with
    empirical analysis as well as fid_resnet.
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
