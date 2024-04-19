import argparse
import os
import lightning as L
import torch

from arch.vae.vae import VAE
from const import ACDC, OUT_PATH, SEED
from data_modules.acdc import ACDCMaskDataModule
from utils import frechet_inception_distance, setup_device, show_samples

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to model checkpoint.",
        required=True,
    )
    
    parser.add_argument(
        "--save_fig",
        action=argparse.BooleanOptionalAction,
        help="If set, save figures locally.",
        default=False,
    )

    return parser.parse_args()

def view_reconstructions(model: VAE, samples: torch.Tensor, device: torch.device, save_path: str=None):
    with torch.no_grad():
        model.eval()
        model.to(device)
        
        samples = samples.to(device)
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

    show_samples(samples_and_reconstructions, rgb=False, nrow=10, figsize=(10, 4), save_path=save_path)

def view_generations(model: VAE, test_data: torch.Tensor, device: torch.device, model_name: str=None, save_path: str=None):
    num_samples, _, _, _ = test_data.shape
    
    with torch.no_grad():
        model.eval()
        model.to(device)
        
        # Sample from latent space
        z = torch.randn(num_samples, model.hparams.latent_dim).to(device)
        
        # Generate segmentation maps from latent variables
        fake_data: torch.Tensor = model.decoder(z)

    # View generations
    generations = torch.argmax(fake_data[:40], dim=1).unsqueeze(1)
    show_samples(generations, rgb=False, nrow=10, figsize=(10, 4), save_path=save_path)
    
    fid_value = frechet_inception_distance(test_data, model.discretise(fake_data))
    print(f"Frechet Inception Distance: {fid_value}")
    
    # Save FID to csv file
    if model_name:
        save_dir, _ = os.path.split(save_path)
        
        with open(os.path.join(save_dir, f"fid.csv"), "a") as f:
            f.write(f"{model_name},{fid_value}\n")

def view_lerp(model: VAE, samples: torch.Tensor, device: torch.device, save_path: str=None):
    """
    Linearly interpolate between the latent representations of two samples, then
    visualise the reconstructions.
    """
    with torch.no_grad():
        model.eval()
        model.to(device)
        
        samples = samples.to(device)
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
        x_hat: torch.Tensor = model.decoder(z_lerps)
    
    reconstructions = torch.argmax(x_hat, dim=1).unsqueeze(1)
    
    show_samples(reconstructions, rgb=False, nrow=10, figsize=(10, 4), save_path=save_path)

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
    
    if flags.save_fig:
        save_dir = os.path.join(OUT_PATH, ACDC.DIR.VAE)
        # TODO noqa
        model_name = flags.model_path.split("/")[2]

    # View reconstructions
    view_reconstructions(
        model,
        samples,
        device,
        save_path=os.path.join(save_dir, f"{model_name}-reconstructions.png") if flags.save_fig else None,
    )
    
    # View generations
    view_generations(
        model,
        data_module.data_test,
        device,
        model_name=model_name if flags.save_fig else None,
        save_path=os.path.join(save_dir, f"{model_name}-generations.png") if flags.save_fig else None,
    )
    
    # View linear interpolation
    view_lerp(
        model,
        samples,
        device,
        save_path=os.path.join(save_dir, f"{model_name}-lerp.png") if flags.save_fig else None,
    )
    
if __name__ == "__main__":
    flags = parse_args()
    main(flags)
