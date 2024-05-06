import argparse
import os
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from const import CIFAR10, LOGS_PATH, SEED
from data_modules.cifar10 import CIFAR10DataModule
from arch.nvae.nvae import NVAE
from utils import setup_device

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--epochs",
        type=int,
        help="Max number of epochs.",
        default=100,
    )
    
    return parser.parse_args()

def main(flags: argparse.Namespace):
    if flags.model_name:
        # Check if model name already exists
        model_dir = os.path.join(LOGS_PATH, CIFAR10.DIR.NVAE, flags.model_name)
        
        if os.path.exists(model_dir):
            raise ValueError(f"Model {flags.model_name} already exists.")
    
    # Setup device
    device = setup_device()
    print(f"Device: {device}")
    
    # Seed
    L.seed_everything(SEED)
    
    # Load data
    data_module = CIFAR10DataModule()
    
    # Reseed after preprocessing data
    L.seed_everything(SEED)

    # Train
    model = NVAE()
    
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=flags.epochs,
        logger=TensorBoardLogger(
            save_dir=LOGS_PATH,
            name=CIFAR10.DIR.NVAE,
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
