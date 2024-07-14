import argparse
import os
import lightning as L
import torch
from torchvision.transforms import ElasticTransform, InterpolationMode

from const import FRDS_MODEL_PATH, LOGS_PATH, OUT_PATH, SEED
from data_modules.acdc import ACDCMaskDataModule
from utils.custom_augmentations import AverageSmoothing, RandomBlackBoxCrop, RandomPepperNoise
from utils.eval import compute_fid_manual, compute_frds
from utils.utils import get_data, setup_device, show_samples

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to model checkpoint.",
        default=FRDS_MODEL_PATH,
    )
    
    parser.add_argument(
        "--use_inception",
        action=argparse.BooleanOptionalAction,
        help="If set, use Inception-v3 and compute FID.",
        default=False,
    )
    
    parser.add_argument(
        "--logs",
        type=str,
        help="Root save directory for logs.",
        default=LOGS_PATH,
    )
    
    parser.add_argument(
        "--show_preview",
        action=argparse.BooleanOptionalAction,
        help="If set, show effect of the various disturbances only and do not run tests.",
        default=False,
    )
    
    return parser.parse_args()

# Evaluation protocol for FRDS
disturbances = {
    "Smoothing": {
        "3x3": AverageSmoothing(kernel_size=3),
        "5x5": AverageSmoothing(kernel_size=5),
        "7x7": AverageSmoothing(kernel_size=7),
        "9x9": AverageSmoothing(kernel_size=9),
    },
    "Black Box Crop": {
        "10-30%": RandomBlackBoxCrop(size_range=(0.1, 0.3)),
        "20-50%": RandomBlackBoxCrop(size_range=(0.2, 0.5)),
        "30-70%": RandomBlackBoxCrop(size_range=(0.3, 0.7)),
        "40-90%": RandomBlackBoxCrop(size_range=(0.4, 0.9)),
    },
    "Elastic Deformation": {
        "sigma=8": ElasticTransform(alpha=300.0, sigma=8.0, interpolation=InterpolationMode.NEAREST),
        "sigma=6": ElasticTransform(alpha=300.0, sigma=6.0, interpolation=InterpolationMode.NEAREST),
        "sigma=4": ElasticTransform(alpha=300.0, sigma=4.0, interpolation=InterpolationMode.NEAREST),
        "sigma=2": ElasticTransform(alpha=300.0, sigma=2.0, interpolation=InterpolationMode.NEAREST),
    },
    "Pepper Noise": {
        "p=0.0005": RandomPepperNoise(p=0.0005),
        "p=0.005": RandomPepperNoise(p=0.005),
        "p=0.05": RandomPepperNoise(p=0.05),
        "p=0.5": RandomPepperNoise(p=0.5),
    },
}

def show_preview(data: torch.Tensor):
    x = data[1:2]
    
    for disturbance_type, specific_disturbances in disturbances.items():
        x_augs = []
        
        for _, disturbance in specific_disturbances.items():
            x_augs.append(disturbance(x))
            
        fig_quality = 3
            
        show_samples(
            torch.cat([x] + x_augs, dim=0),
            ncol=len(x_augs) + 1,
            figsize=(fig_quality * (len(x_augs) + 1), fig_quality),
            save_path=os.path.join(OUT_PATH, f"{disturbance_type}.png"),
            display=False,
        )
        
        print(f"Saved preview for {disturbance_type} to {OUT_PATH}")

def test(
    datasets: tuple[torch.Tensor, torch.Tensor],
    model_path: str,
    use_inception: bool,
    device: torch.device,
):
    """
    Evaluate how robust the FRDS metric is for the pretrained model, by
    computing it between the test set and a modified test set. The modified
    test set is subject to different types and levels of noise, such as
    Gaussian noise, smoothing and cropping.
    """
    x, y = datasets
    
    metric_label = "FID" if use_inception else "FRDS"
    
    # Ideal FRDS/FID: No disturbance
    if use_inception:
        frechet_value = compute_fid_manual(
            y, x, device=device, is_data_onehot=False
        )
    else:
        frechet_value = compute_frds(
            y, x, resnet_path=model_path, device=device, is_data_onehot=False
        )
    
    print(f"Ideal {metric_label} (no disturbance): {frechet_value}")
    
    # Disturbances
    for disturbance_type, specific_disturbances in disturbances.items():
        for intensity, disturbance in specific_disturbances.items():
            x_aug = disturbance(x)
            
            if use_inception:
                frechet_value = compute_fid_manual(
                    y, x_aug, device=device, is_data_onehot=False
                )
            else:
                frechet_value = compute_frds(
                    y, x_aug, resnet_path=model_path, device=device, is_data_onehot=False
                )
            
            print(f"{metric_label} for {disturbance_type} ({intensity}): {frechet_value}")

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
    
    loader_train = data_module.train_dataloader(shuffle=True)
    loader_test = data_module.test_dataloader(shuffle=True)
    
    data_train = get_data(loader_train)
    data_test = get_data(loader_test)
    
    num_samples = min(data_train.shape[0], data_test.shape[0])
    
    data_train = data_train[:num_samples]
    data_test = data_test[:num_samples]
    
    if flags.show_preview:
        show_preview(data_test)
    else:
        # Perform test
        test(
            (data_train, data_test),
            flags.model_path,
            flags.use_inception,
            device,
        )

if __name__ == "__main__":
    flags = parse_args()
    main(flags)
