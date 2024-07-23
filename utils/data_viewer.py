import argparse
import lightning as L
from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

from utils.const import ACDC, SEED
from data_modules.acdc import ACDCDataModule, ACDCMaskDataModule
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
        "--show_scans",
        action=argparse.BooleanOptionalAction,
        help="If set, also show the MRI scans along with the masks.",
        default=False,
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

def show_scans_and_masks(
    scans: torch.Tensor,
    masks: torch.Tensor,
    ncol: int=6,
    figsize: tuple[int, int]=(6, 4),
):
    scans = scans.cpu().float()
    masks = masks.cpu().float()
    
    scans = make_grid(scans, nrow=ncol, padding=2, normalize=True)
    masks = make_grid(masks, nrow=ncol, padding=2, normalize=True)
    
    scans = np.transpose(scans.numpy(), (1, 2, 0))
    masks = masks[0]
    
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(scans)
    plt.imshow(masks, alpha=0.5)
    plt.tight_layout()
    
    plt.show()

def view_acdc_scans():
    # Seed
    L.seed_everything(SEED)
    
    data_module = ACDCDataModule(
        batch_size=24,
        filter_empty=flags.filter_empty,
        register_alignment=flags.register_alignment,
    )
    
    print(f"Number of train samples: {len(data_module.data_train)}")
    print(f"Number of test samples: {len(data_module.data_test)}")
    
    # Reseed
    L.seed_everything(SEED)
    
    # View samples
    loader_test = data_module.test_dataloader(shuffle=True)
    
    samples: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = next(iter(loader_test))
    scans, masks, _ = samples
    
    masks = torch.argmax(masks, dim=1).unsqueeze(1)
        
    show_scans_and_masks(scans, masks, ncol=6, figsize=(6, 4))

def view_acdc_masks():
    # Seed
    L.seed_everything(SEED)
    
    data_module = ACDCMaskDataModule(
        batch_size=12 if flags.augment_simclr else 24,
        filter_empty=flags.filter_empty,
        register_alignment=flags.register_alignment,
        as_image=flags.augment_simclr,
        augment_rotation_test=flags.augment,
        augment_simclr_test=flags.augment_simclr,
        return_original=flags.augment_simclr,
    )

    print(f"Number of train samples: {len(data_module.data_train)}")
    print(f"Number of test samples: {len(data_module.data_test)}")
    
    # Reseed
    L.seed_everything(SEED)
    
    # View samples
    loader_test = data_module.test_dataloader(shuffle=True)
    
    if not flags.augment_simclr:
        samples: torch.Tensor = next(iter(loader_test))

        # Uncomment this to view each channel separately
        # class_idx = acdc_class_to_idx(ACDC.CardiacClass.RV)
        # samples = samples[:, class_idx, :, :].unsqueeze(1)
        
        # Or recombine the channels
        samples = torch.argmax(samples, dim=1).unsqueeze(1)
        
        show_samples(samples, rgb=flags.augment_simclr, ncol=6, figsize=(6, 4))
    else:
        samples_pair: list[torch.Tensor] = next(iter(loader_test))
        samples = torch.cat(samples_pair, dim=0)
        
        show_samples(samples, rgb=flags.augment_simclr, ncol=12, figsize=(12, 3))
    

def main(flags: argparse.Namespace):
    # Setup device
    device = setup_device()
    print(f"Device: {device}")
    
    # Load data
    match flags.dataset:
        case "cifar10":
            view_cifar10()

        case "acdc":
            if flags.show_scans:
                view_acdc_scans()
            else:
                view_acdc_masks()

        case _:
            raise ValueError(f"Unknown dataset: {flags.dataset}")

if __name__ == "__main__":
    flags = parse_args()
    main(flags)
