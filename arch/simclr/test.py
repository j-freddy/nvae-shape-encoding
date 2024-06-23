import argparse
import lightning as L
import torch

from const import LOGS_PATH, SEED
from data_modules.acdc import ACDCMaskDataModule
from utils.custom_augmentations import AverageSmoothing
from utils.eval import compute_frds
from utils.utils import setup_device

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to model checkpoint.",
        required=True,
    )
    
    parser.add_argument(
        "--logs",
        type=str,
        help="Root save directory for logs.",
        default=LOGS_PATH,
    )
    
    return parser.parse_args()

def test(
    datasets: tuple[torch.Tensor, torch.Tensor],
    model_path: str,
    device: torch.device,
):
    """
    Evaluate how robust the FRDS metric is for the pretrained model, by
    computing it between the test set and a modified test set. The modified
    test set is subject to different types and levels of noise, such as
    Gaussian noise, smoothing and cropping.
    """
    # Evaluation protocol for FRDS
    disturbances = [
        AverageSmoothing(kernel_size=3),
        AverageSmoothing(kernel_size=5),
        AverageSmoothing(kernel_size=7),
        AverageSmoothing(kernel_size=9),
        # ElasticTransform(alpha=300.0, sigma=8.0, interpolation=InterpolationMode.NEAREST),
        # ElasticTransform(alpha=300.0, sigma=6.0, interpolation=InterpolationMode.NEAREST),
        # ElasticTransform(alpha=300.0, sigma=4.0, interpolation=InterpolationMode.NEAREST),
        # ElasticTransform(alpha=300.0, sigma=2.0, interpolation=InterpolationMode.NEAREST),
        # RandomBlackBoxCrop(size_range=(0.1, 0.3)),
        # RandomBlackBoxCrop(size_range=(0.2, 0.5)),
        # RandomBlackBoxCrop(size_range=(0.3, 0.7)),
        # RandomBlackBoxCrop(size_range=(0.4, 0.9)),
    ]

    x, y = datasets

    for disturbance in disturbances:
        x_aug = disturbance(x)

        # show_samples(
        #     torch.cat([x[:10], x_aug[:10]], dim=0),
        #     ncol=10,
        #     figsize=(10, 2),
        # )
        
        frds_value = compute_frds(
            y,
            x_aug,
            resnet_path=model_path,
            device=device,
            is_data_onehot=False,
        )
        
        print(f"FRDS: {frds_value}")

def main(flags: argparse.Namespace):
    # Setup device
    device = setup_device()
    print(f"Device: {device}")
    
    # Seed
    L.seed_everything(SEED)
    
    # Load data
    data_module = ACDCMaskDataModule(batch_size=20, as_image=True)
    
    # Reseed after preprocessing data
    L.seed_everything(SEED)

    data_train = data_module.data_train.masks
    data_test = data_module.data_test.masks
    
    num_samples = min(data_train.shape[0], data_test.shape[0])
    
    # Shuffle and select subset
    data_train = data_train[torch.randperm(data_train.shape[0])]
    data_test = data_test[torch.randperm(data_test.shape[0])]
    data_train = data_train[:num_samples]
    data_test = data_test[:num_samples]
    
    # Perform test
    test(
        (data_train, data_test),
        flags.model_path,
        device,
    )

if __name__ == "__main__":
    flags = parse_args()
    main(flags)
