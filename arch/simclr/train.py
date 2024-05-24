import argparse
import os
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from arch.simclr.simclr import SimCLR
from const import ACDC, LOGS_PATH, SEED
from data_modules.acdc import ACDCMaskDataModule
from utils.utils import setup_device

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--epochs",
        type=int,
        help="Max number of epochs.",
        default=100,
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        help="Directory name of saved model checkpoints and metadata.",
    )
    
    parser.add_argument(
        "--logs",
        type=str,
        help="Root save directory for logs.",
        default=LOGS_PATH,
    )
    
    return parser.parse_args()

def main(flags: argparse.Namespace):
    if flags.model_name:
        # Check if model name already exists
        model_dir = os.path.join(flags.logs, ACDC.DIR.VAE, flags.model_name)
        
        if os.path.exists(model_dir):
            raise ValueError(f"Model {flags.model_name} already exists.")
    
    # Setup device
    device = setup_device()
    print(f"Device: {device}")
    
    # Seed
    L.seed_everything(SEED)
    
    # Load data
    data_module = ACDCMaskDataModule(batch_size=128, augment_simclr=True)
    
    # Reseed after preprocessing data
    L.seed_everything(SEED)
    
    # Train
    model = SimCLR()
    
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=flags.epochs,
        logger=TensorBoardLogger(
            save_dir=flags.logs,
            name=ACDC.DIR.SIMCLR,
            version=flags.model_name if flags.model_name else None,
            default_hp_metric=False,
        ),
        callbacks=[
            ModelCheckpoint(monitor="val_loss", mode="min"),
            LearningRateMonitor("epoch"),
        ]
    )
    
    trainer.fit(model, data_module)

if __name__ == "__main__":
    flags = parse_args()
    main(flags)
