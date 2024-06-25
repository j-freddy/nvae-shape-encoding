import argparse
import lightning as L
import torch

from const import SEED
from data_modules.acdc import ACDCMaskDataModule

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
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
    
    return parser.parse_args()

def main(flags: argparse.Namespace):
    # Seed
    L.seed_everything(SEED)

    data_module = ACDCMaskDataModule(
        batch_size=20,
        filter_empty=flags.filter_empty,
        register_alignment=flags.register_alignment,
        augment_rotation=flags.augment,
        augment_rotation_test=flags.augment,
    )

    data_train = data_module.data_train.masks
    data_test = data_module.data_test.masks
    data = torch.cat([data_train, data_test], dim=0)
    
    data = torch.argmax(data, dim=1).unsqueeze(1)
    
    # Compute % background
    num_samples, _, width, height = data.shape
    
    n_background = torch.sum(data == 0)
    n_pixels = num_samples * width * height
    
    print(f"% Background: {n_background / n_pixels}")

if __name__ == "__main__":
    flags = parse_args()
    main(flags)
