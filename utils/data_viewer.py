import argparse
import os
import lightning as L
import torch

from data_modules.mnms import MnMsDataModule
from utils.anatomical_validity_checker import AnatomicalValidityChecker
from utils.const import ACDC, OUT_PATH, SEED, MaskClassLabel, MnMs
from data_modules.acdc import ACDCDataModule, ACDCMaskDataModule
from data_modules.cifar10 import CIFAR10DataModule
from utils.utils import mask_class_id_to_idx, setup_device, show_samples, show_scans_and_masks

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--dataset",
        type=str,
        help="Which dataset to use.",
        choices=["cifar10", "acdc", "mnms"],
        default="acdc",
    )
    
    parser.add_argument(
        "--centre",
        type=int,
        help="If using M&Ms and set, only show scans from the specified centre.",
        choices=MnMs.centres,
        default=None,
    )
    
    parser.add_argument(
        "--num_subjects",
        type=int,
        help="Few-shot learning: Number of subjects to use. If -1, use all subjects.",
        default=-1,
    )
    
    parser.add_argument(
        "--sort_by_validity",
        action=argparse.BooleanOptionalAction,
        help="Few-shot learning: If set, use subjects with highest % anatomical validity.",
        default=False,
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
        help="If set, augment training data.",
        default=False,
    )
    
    parser.add_argument(
        "--augment_simclr",
        action=argparse.BooleanOptionalAction,
        help="If set, augment training data with SimCLR pipeline.",
        default=False,
    )
    
    parser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        help="If set, save high-quality figures in out/ instead of showing them.",
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

def view_acdc_scans():
    # Seed
    L.seed_everything(SEED)
    
    data_module = ACDCDataModule(
        batch_size=24,
        filter_empty=flags.filter_empty,
        register_alignment=flags.register_alignment,
        augment_test=flags.augment,
    )
    
    print(f"Number of train samples: {len(data_module.data_train)}")
    print(f"Number of test samples: {len(data_module.data_test)}")
    
    # Reseed
    L.seed_everything(SEED)
    
    # View samples
    loader_test = data_module.test_dataloader(shuffle=True)
    
    samples: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = next(iter(loader_test))
    scans, masks, _, _ = samples
    
    masks = torch.argmax(masks, dim=1).unsqueeze(1)
    
    save_path = os.path.join(OUT_PATH, "acdc_data.png") if flags.save else None
    
    show_scans_and_masks(
        scans,
        masks,
        ncol=6,
        figsize=(24, 16) if flags.save else (6, 4),
        save_path=save_path,
        display=not flags.save,
    )

def view_acdc_masks(save: bool):
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
    
    save_path = os.path.join(OUT_PATH, "acdc_data.png") if flags.save else None
    
    if not flags.augment_simclr:
        samples: torch.Tensor = next(iter(loader_test))

        # Uncomment this to view each channel separately
        # class_idx = acdc_class_to_idx(ACDC.CardiacClass.RV)
        # samples = samples[:, class_idx, :, :].unsqueeze(1)
        
        # Or recombine the channels
        samples = torch.argmax(samples, dim=1).unsqueeze(1)

        show_samples(
            samples,
            rgb=flags.augment_simclr,
            ncol=6,
            figsize=(18, 12) if flags.save else (6, 4),
            save_path=save_path,
            display=not flags.save,
        )
    else:
        samples_pair: list[torch.Tensor] = next(iter(loader_test))
        samples = torch.cat(samples_pair, dim=0)
        
        show_samples(
            samples,
            rgb=flags.augment_simclr,
            ncol=12,
            figsize=(48, 12) if flags.save else (12, 3),
            save_path=save_path,
            display=not flags.save,
        )
        
def view_mnms_scans():
    # Seed
    L.seed_everything(SEED)
    
    data_module = MnMsDataModule(
        batch_size=24,
        filter_empty=flags.filter_empty,
        from_centre=flags.centre,
        num_subjects=flags.num_subjects,
        sort_by_validity=flags.sort_by_validity,
        augment_test=flags.augment,
    )
    
    print(f"Number of train samples: {len(data_module.data_train)}")
    print(f"Number of test samples: {len(data_module.data_test)}")
    
    # Reseed
    L.seed_everything(SEED)
    
    # View samples
    loader = data_module.test_dataloader(shuffle=True) \
        if flags.num_subjects == -1 \
        else data_module.train_dataloader(shuffle=True)
    
    samples: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = next(iter(loader))
    scans, masks, _, _ = samples
    
    # Uncomment this to view each channel separately
    # class_idx = mask_class_id_to_idx(MaskClassLabel.LV)
    # masks = masks[:, class_idx, :, :].unsqueeze(1)
    
    # Or recombine the channels
    masks = torch.argmax(masks, dim=1).unsqueeze(1)
    
    save_path = os.path.join(OUT_PATH, "mnms_data.png") if flags.save else None
    
    show_scans_and_masks(
        scans,
        masks,
        ncol=6,
        figsize=(24, 16) if flags.save else (6, 4),
        save_path=save_path,
        display=not flags.save,
    )

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
        
        case "mnms":
            view_mnms_scans()

        case _:
            raise ValueError(f"Unknown dataset: {flags.dataset}")

if __name__ == "__main__":
    flags = parse_args()
    main(flags)
