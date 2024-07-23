import argparse
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
import torch

from arch.vae.train import ID_TO_MODEL
from utils.const import ACDC, LOGS_PATH, SEED
from data_modules.acdc import ACDCMaskDataModule
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
    # Setup device
    device = setup_device()
    print(f"Device: {device}")
    
    # Seed
    L.seed_everything(SEED)
    
    # Load data
    data_module = ACDCMaskDataModule(
        batch_size=16,
        register_alignment=flags.register_alignment,
        augment_rotation_test=flags.augment,
    )
    
    # Reseed after preprocessing data
    L.seed_everything(SEED)
    
    # Load model
    # Inefficient: Need to load the hparams first to determine class type,
    # then use load_from_checkpoint. So, this loads the checkpoint twice.
    checkpoint = torch.load(flags.model_path, map_location=device)
    Model: L.LightningModule = ID_TO_MODEL[checkpoint["hyper_parameters"]["loss_reg"]]
    del checkpoint
    model = Model.load_from_checkpoint(flags.model_path)

    # noqa
    model_name = flags.model_path.split("/")[2]

    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        logger=TensorBoardLogger(
            save_dir=flags.logs,
            name=ACDC.DIR.VAE,
            version=model_name,
            default_hp_metric=False,
        ),
    )

    trainer.test(model, data_module)
    
if __name__ == "__main__":
    flags = parse_args()
    main(flags)
