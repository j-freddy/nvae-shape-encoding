import argparse
import lightning as L
import torch

from arch.vae.vae import VAE
from const import SEED
from data_modules.acdc import ACDCMaskDataModule
from utils import setup_device, show_samples

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to model checkpoint.",
        required=True,
    )

    return parser.parse_args()

def view_reconstructions(model: VAE, samples: torch.Tensor, device: torch.device):
    with torch.no_grad():
        model.eval()
        model.to(device)
        _, _, x_hat = model(samples)

        reconstructions = torch.argmax(x_hat, dim=1).unsqueeze(1)

    samples = torch.argmax(samples, dim=1).unsqueeze(1)
    
    # Interleave samples and reconstructions
    batch_size, num_channels, width, height = samples.shape
    assert width == height
    samples_and_reconstructions = torch.empty(batch_size * 2, num_channels, width, height)
    
    for i in range(samples.shape[0]):
        samples_and_reconstructions[i * 2] = samples[i]
        samples_and_reconstructions[i * 2 + 1] = reconstructions[i]

    show_samples(samples_and_reconstructions, rgb=False, nrow=10, figsize=(10, 4))

def view_generations(model: VAE, device: torch.device):
    with torch.no_grad():
        model.eval()
        model.to(device)
        
        # Sample from latent space
        z = torch.randn(40, model.hparams.latent_dim).to(device)
        
        # Generate segmentation maps from latent variables
        x_hat = model.decoder(z)
        
        generations = torch.argmax(x_hat, dim=1).unsqueeze(1)
    
    show_samples(generations, rgb=False, nrow=10, figsize=(10, 4))

def view_lerp(model: VAE, samples: torch.Tensor, device: torch.device):
    """
    Linearly interpolate between the latent representations of two samples, then
    visualise the reconstructions.
    """
    with torch.no_grad():
        model.eval()
        model.to(device)
        _, _, z = model.get_latent(samples)
    
    # Hand pick 2 masks that look different
    z1, z2 = z[1], z[19]
    
    # Linear interpolation between z1 and z2
    z_lerps = []

    for i in range(10):
        z_lerps.append(torch.lerp(z1, z2, i / 9))
    
    z_lerps = torch.stack(z_lerps)
    
    # Pass through decoder
    with torch.no_grad():
        x_hat = model.decoder(z_lerps)
    
    reconstructions = torch.argmax(x_hat, dim=1).unsqueeze(1)
    
    show_samples(reconstructions, rgb=False, nrow=10, figsize=(10, 4))

def main(flags: argparse.Namespace):
    # Setup device
    device = setup_device()
    print(f"Device: {device}")
    
    # Seed
    L.seed_everything(SEED)
    
    # Load data
    data_module = ACDCMaskDataModule(batch_size=20)
    
    # Reseed after preprocessing data
    L.seed_everything(SEED)
    
    # Load model
    model = VAE.load_from_checkpoint(flags.model_path)
    
    loader_test = data_module.test_dataloader()
    samples: torch.Tensor = next(iter(loader_test))
    
    # View reconstructions
    # view_reconstructions(model, samples, device)
    
    # View generations
    # view_generations(model, device)
    
    # View linear interpolation
    view_lerp(model, samples, device)
    
if __name__ == "__main__":
    flags = parse_args()
    main(flags)
