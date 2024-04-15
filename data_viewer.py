import argparse
import lightning as L
import torch

from const import SEED
from data_modules.acdc import ACDCDataModule, ACDCMaskDataModule
from data_modules.cifar10 import CIFAR10DataModule
from utils import setup_device, show_samples

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--dataset",
        type=str,
        help="Which dataset to use.",
        choices=["cifar10", "acdc"],
        required=True,
    )
    
    return parser.parse_args()

def view_cifar10():
    data_module = CIFAR10DataModule(preprocess=False)
    
    # Seed
    L.seed_everything(SEED)
            
    # View samples
    loader_test = data_module.test_dataloader()

    samples: tuple[torch.Tensor, torch.Tensor] = next(iter(loader_test))
    images, _ = samples
    show_samples(images, nrow=8, figsize=(8, 4))

def view_acdc():
    data_module = ACDCMaskDataModule(batch_size=40, filter_empty=True, one_hot=False)

    print(f"Number of train samples: {len(data_module.data_train)}")
    print(f"Number of test samples: {len(data_module.data_test)}")
    
    # Seed
    L.seed_everything(SEED)
            
    # View samples
    loader_test = data_module.test_dataloader()
    
    samples: torch.Tensor = next(iter(loader_test))
    
    # If one-hot encoded, uncomment this line to view each channel separately
    # samples = samples[:, 0, :, :].unsqueeze(1)
    # samples = samples[:, 1, :, :].unsqueeze(1)
    # samples = samples[:, 2, :, :].unsqueeze(1)
    # samples = samples[:, 3, :, :].unsqueeze(1)
    
    # Or recombine the channels
    # samples = torch.argmax(samples, dim=1).unsqueeze(1)
    
    show_samples(samples, rgb=False, nrow=10, figsize=(10, 4))

def main(flags: argparse.Namespace):
    # Setup device
    device = setup_device()
    print(f"Device: {device}")
    
    # Load data
    match flags.dataset:
        case "cifar10":
            view_cifar10()

        case "acdc":
            view_acdc()

        case _:
            raise ValueError(f"Unknown dataset: {flags.dataset}")

if __name__ == "__main__":
    flags = parse_args()
    main(flags)
