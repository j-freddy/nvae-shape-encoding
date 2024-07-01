import numpy as np
from scipy.linalg import sqrtm
import torch
import torch.nn as nn
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from torchvision.models import inception_v3

from arch.simclr.utils import load_simclr_backbone
from utils.utils import one_hot_to_image

def compute_fid(real_data: torch.Tensor, fake_data: torch.Tensor) -> torch.Tensor:
    """
    Deprecated. Use compute_fid_manual instead.
    
    Reason: This pipeline to evaluate generation quality does not align with
    empirical analysis as well as compute_fid_manual.
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

def encode_embeddings(
    x: torch.Tensor,
    model: nn.Module,
    device: torch.device,
    resnet: bool=True,
    is_x_onehot: bool=True,
) -> torch.Tensor:
    def encode_inception(
        x: torch.Tensor,
        model: nn.Module,
        device: torch.device,
        is_x_onehot: bool,
    ) -> torch.Tensor:
        with torch.no_grad():
            x = x.to(device)
            if is_x_onehot:
                # Discard background dimension
                x = x[:, 1:, :, :].float()
            x = x * 2 - 1

            transform = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                nn.Upsample(size=(299, 299), mode="bilinear", align_corners=True),
            ])

            feats = model(transform(x))

        return feats.detach().cpu()
    
    def encode_resnet(
        x: torch.Tensor,
        model: nn.Module,
        device: torch.device,
        is_x_onehot: bool,
    ) -> torch.Tensor:
        with torch.no_grad():
            x = x.to(device)
            if is_x_onehot:
                x = one_hot_to_image(x)
            x = x * 2 - 1
            # Values should be preprocessed as 0, 1 so after scaling they should
            # be -1, 1
            assert set(x.unique().tolist()).issubset({-1, 1})
            feats = model(x)

        return feats.detach().cpu()

    embeddings = []
    
    batch_size = 2
    x_split = torch.split(x, batch_size, dim=0)
    
    f = encode_resnet if resnet else encode_inception

    for x_batch in x_split:
        embeddings.append(f(x_batch, model, device, is_x_onehot))
        
    return torch.cat(embeddings, dim=0)

def compute_frechet_distance(real_feats: torch.Tensor, fake_feats: torch.Tensor) -> torch.Tensor:
    # Calculate mean and covariance
    mu_real = real_feats.mean(0)
    sigma_real = np.cov(real_feats.numpy(), rowvar=False)

    mu_fake = fake_feats.mean(0)
    sigma_fake = np.cov(fake_feats.numpy(), rowvar=False)

    # Compute Frechet distance
    sum_sq_diff = torch.sum((mu_real - mu_fake) ** 2).item()
    covm_real_fake = sqrtm(sigma_real.dot(sigma_fake))
    
    # Check and correct for imaginary numbers from sqrt
    if np.iscomplexobj(covm_real_fake):
        covm_real_fake = covm_real_fake.real

    return sum_sq_diff + np.trace(sigma_real + sigma_fake - 2.0 * covm_real_fake)

def compute_fid_manual(
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
    device: torch.device,
    is_data_onehot: bool=True,
):
    """
    Code adapted from Priya Jain. Original code[1] has been modularised to
    compute_frechet_distance and encode_inception functions, and modified for
    code efficiency.

    [1]: https://github.com/pj2222/Deep_Learning_To_Match_Cardiac_Shapes_With_Images
    """
    # Load inception model
    inception_model = inception_v3(weights="DEFAULT", transform_input=False).to(device)
    
    inception_model.fc = nn.Identity()
    inception_model.eval()

    # Extract features for real images
    real_feats = encode_embeddings(real_data, inception_model, device, resnet=False, is_x_onehot=is_data_onehot)

    # Extract features for generated images
    fake_feats = encode_embeddings(fake_data, inception_model, device, resnet=False, is_x_onehot=is_data_onehot)

    return compute_frechet_distance(real_feats, fake_feats)

def compute_frds(
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
    resnet_path: str,
    device: torch.device,
    is_data_onehot: bool=True,
):  
    # Load pretrained SimCLR model
    resnet_model = load_simclr_backbone(resnet_path)
    resnet_model = resnet_model.to(device)

    # Extract features for real images
    real_feats = encode_embeddings(real_data, resnet_model, device, is_x_onehot=is_data_onehot)

    # Extract features for generated images
    fake_feats = encode_embeddings(fake_data, resnet_model, device, is_x_onehot=is_data_onehot)

    return compute_frechet_distance(real_feats, fake_feats)

def get_samples_and_reconstructions(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    reconstructions = torch.argmax(x_hat, dim=1).unsqueeze(1)
    samples = torch.argmax(x, dim=1).unsqueeze(1)

    # Interleave samples and reconstructions
    batch_size, num_channels, width, height = samples.shape
    assert width == height
    samples_and_reconstructions = torch.empty(batch_size * 2, num_channels, width, height)
    
    for i in range(samples.shape[0]):
        samples_and_reconstructions[i * 2] = samples[i]
        samples_and_reconstructions[i * 2 + 1] = reconstructions[i]
    
    return samples_and_reconstructions

def get_samples_and_reconstructions_pixel_diff(
    x: torch.Tensor,
    x_hat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    reconstructions = torch.argmax(x_hat, dim=1).unsqueeze(1)
    samples = torch.argmax(x, dim=1).unsqueeze(1)
    diff = torch.abs(samples - reconstructions)
    
    return samples, diff
