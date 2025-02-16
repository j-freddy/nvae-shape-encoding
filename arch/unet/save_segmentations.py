import argparse
import os
import lightning as L
import torch

from arch.unet.segmentation_base import SegmentationBase
from arch.unet.utils import ID_TO_MODEL
from utils.const import DATA_PATH, SEED
from data_modules.acdc import ACDC3DDataModule, ACDCDataModule
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
        "--split",
        type=str,
        help="Data split to use.",
        choices=["train", "val", "test"],
        required=True,
    )
    
    parser.add_argument(
        "--model_type",
        type=str,
        help="Model type.",
        choices=ID_TO_MODEL.keys(),
        required=True,
    )
    
    return parser.parse_args()

def main(flags: argparse.Namespace):
    data_path = os.path.join(
        DATA_PATH,
        f"acdc_processed_with_predicted_segmentation_{flags.model_type}_{flags.split}.pt",
    )
    
    # Check if data path already exists
    if os.path.exists(data_path):
        raise FileExistsError(f"Data path already exists: {data_path}")
    
    # Setup device
    device = setup_device()
    print(f"Device: {device}")
    
    # Seed
    L.seed_everything(SEED)
    
    # Load data
    match flags.split:
        case "train":
            data_module = ACDCDataModule(batch_size=32)
            data_loader = data_module.train_dataloader()
        case "val":
            data_module = ACDCDataModule(batch_size=32)
            data_loader = data_module.val_dataloader()
        case "test":
            # Use 3D dataloader for test set
            data_module = ACDC3DDataModule()
            data_loader = data_module.test_dataloader()
        case _:
            raise ValueError(f"Invalid split: {flags.split}")
    
    # Reseed after preprocessing data
    L.seed_everything(SEED)
    
    # Load model
    checkpoint = torch.load(flags.model_path, map_location=device)
    Model: L.LightningModule = ID_TO_MODEL[checkpoint["hyper_parameters"]["model_type"]]
    del checkpoint
    model: SegmentationBase = Model.load_from_checkpoint(flags.model_path)
    
    assert model.hparams.model_type == flags.model_type, f"Model type mismatch. Flag: {flags.model_type}, Model: {model.hparams.model_type}."

    # Save segmentations
    model.save_segmentations(
        data_loader,
        data_path,
        test_data=flags.split == "test",
    )

if __name__ == "__main__":
    flags = parse_args()
    main(flags)
