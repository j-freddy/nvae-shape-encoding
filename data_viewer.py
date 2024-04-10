import argparse
import lightning as L
from matplotlib import pyplot as plt
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

def view_cifar10():
    data_module = CIFAR10DataModule(preprocess=False)
            
    # View samples
    loader_test = data_module.test_dataloader()

    samples: tuple[torch.Tensor, torch.Tensor] = next(iter(loader_test))
    show_samples(*samples)

def view_acdc():
    data_module = ACDCDataModule()
            
    # View samples
    loader_test = data_module.test_dataloader()
    
    samples: dict = next(iter(loader_test))
    
    _, _, _, _, num_slice = samples["ed_image"]["data"].shape
    
    # View each slice
    for i in range(num_slice):
        plt.figure()
        # Take 1st image in batch
        plt.imshow(samples["ed_image"]["data"][0][:, :, :, i].numpy().squeeze(), cmap="gray")
        plt.imshow(samples["ed_mask"]["data"][0][:, :, :, i].numpy().squeeze(), alpha=0.5)
        plt.show()

def main(flags):
    # Seed
    L.seed_everything(SEED)

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
