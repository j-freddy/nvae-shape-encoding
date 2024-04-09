import argparse
import lightning as L
import torch

from const import SEED
from data_modules.acdc import ACDCDataModule
from data_modules.cifar10 import CIFAR10DataModule
from utils import setup_device, show_samples

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--dataset",
        type=str,
        help="Which dataset to use.",
        choices=["cifar10", "acdc"],
        required=True,
    )
    
    return parser.parse_args()

def main(flags):
    # Seed
    L.seed_everything(SEED)

    # Setup device
    device = setup_device()
    print(f"Device: {device}")
    
    # Load data
    match flags.dataset:
        case "cifar10":
            data_module = CIFAR10DataModule(preprocess=False)
        case "acdc":
            data_module = ACDCDataModule()
        case _:
            raise ValueError(f"Unknown dataset: {flags.dataset}")
    
    # View samples
    loader_test = data_module.test_dataloader()
    
    samples: tuple[torch.Tensor, torch.Tensor] = next(iter(loader_test))
    show_samples(*samples)

if __name__ == "__main__":
    flags = parse_args()
    main(flags)
