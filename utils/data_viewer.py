import argparse
import lightning as L
import torch

from const import SEED
from data_modules.acdc import ACDCMaskDataModule
from data_modules.cifar10 import CIFAR10DataModule
from utils.utils import setup_device, show_samples

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--dataset",
        type=str,
        help="Which dataset to use.",
        choices=["cifar10", "acdc"],
        default="acdc",
    )
    
    parser.add_argument(
        "--filter_empty",
        action=argparse.BooleanOptionalAction,
        help="If set, filter out empty masks.",
        default=False,
    )
    
    parser.add_argument(
        "--register_alignment",
        action=argparse.BooleanOptionalAction,
        help="If set, use masks that have been rotated such that the right ventricle points upwards.",
        default=False,
    )
    
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        help="If set, augment training data with small random rotation.",
        default=False,
    )
    
    parser.add_argument(
        "--augment_simclr",
        action=argparse.BooleanOptionalAction,
        help="If set, augment training data with SimCLR pipeline.",
        default=False,
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
    show_samples(images, ncol=8, figsize=(8, 2))

def view_acdc():
    # Seed
    L.seed_everything(SEED)
    
    data_module = ACDCMaskDataModule(
        batch_size=10 if flags.augment_simclr else 20,
        filter_empty=flags.filter_empty,
        register_alignment=flags.register_alignment,
        as_image=flags.augment_simclr,
        augment_rotation_test=flags.augment,
        augment_simclr_test=flags.augment_simclr,
    )

    print(f"Number of train samples: {len(data_module.data_train)}")
    print(f"Number of test samples: {len(data_module.data_test)}")
    
    # Reseed
    L.seed_everything(SEED)
            
    # View samples
    loader_test = data_module.test_dataloader()
    
    samples: list[torch.Tensor] = next(iter(loader_test))
    
    if flags.augment_simclr:
        samples: torch.Tensor = torch.cat(samples, dim=0)
    else:  
        # Uncomment this line to view each channel separately
        # samples = samples[:, 0, :, :].unsqueeze(1)
        # samples = samples[:, 1, :, :].unsqueeze(1)
        # samples = samples[:, 2, :, :].unsqueeze(1)
        # samples = samples[:, 3, :, :].unsqueeze(1)
        
        # Or recombine the channels
        samples: torch.Tensor = torch.argmax(samples, dim=1).unsqueeze(1)
    
    show_samples(samples, rgb=flags.augment_simclr, ncol=10, figsize=(10, 2))

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
