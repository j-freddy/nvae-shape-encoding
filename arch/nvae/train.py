import argparse
import os
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from const import ACDC, LOGS_PATH, SEED
from data_modules.acdc import ACDCMaskDataModule
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
    
    parser.add_argument(
        "--projected_channels",
        type=int,
        help="Number of channels in the immediate space projected through the stem (and conditional coder).",
        default=64,
    )
    
    parser.add_argument(
        "--warmup_steps",
        type=int,
        help="Number of steps for KL divergence linear deterministic warmup.",
        # Each epoch has 214 steps with batch size of 8 and ACDC dataset
        # (with empty masks preserved)
        default=500,
    )
    
    parser.add_argument(
        "--beta0",
        type=float,
        help="Beta value for KL divergence corresponding to layer 0 (shallowest layer).",
        default=1.0,
    )
    
    parser.add_argument(
        "--beta1",
        type=float,
        help="Beta value for KL divergence corresponding to layer 1.",
        default=1.0,
    )
    
    parser.add_argument(
        "--beta2",
        type=float,
        help="Beta value for KL divergence corresponding to layer 2 (topmost layer).",
        default=1.0,
    )
    
    parser.add_argument(
        "--filter_empty",
        action=argparse.BooleanOptionalAction,
        help="If set, filter out empty masks.",
        default=False,
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
        model_dir = os.path.join(flags.logs, ACDC.DIR.NVAE, flags.model_name)
        
        if os.path.exists(model_dir):
            raise ValueError(f"Model {flags.model_name} already exists.")
    
    # Setup device
    device = setup_device()
    print(f"Device: {device}")
    
    # Seed
    L.seed_everything(SEED)
    
    # Load data
    data_module = ACDCMaskDataModule(batch_size=8, filter_empty=flags.filter_empty)
    
    # Reseed after preprocessing data
    L.seed_everything(SEED)
    
    _, num_classes, _, _ = data_module.data_test.shape

    # Train
    model = NVAE(
        in_channels=num_classes,
        initial_channels=flags.projected_channels,
        max_epochs=flags.epochs,
        beta_per_scale=[flags.beta0, flags.beta1, flags.beta2],
        kl_warmup_steps=flags.warmup_steps,
    )
    
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=flags.epochs,
        logger=TensorBoardLogger(
            save_dir=flags.logs,
            name=ACDC.DIR.NVAE,
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
