import argparse
import os
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from arch.vae_corrector.vae_corrector import VAECorrector
from utils.const import ACDC, LOGS_PATH, SEED
from data_modules.acdc import ACDCWithPredictedMaskDataModule
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
        "--latent_dim",
        type=int,
        help="Dimension of latent space.",
        default=8,
    )
    
    parser.add_argument(
        "--beta",
        type=float,
        help="Beta value for KL divergence.",
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
        model_dir = os.path.join(flags.logs, ACDC.DIR.VAE_CORRECTOR, flags.model_name)
        
        if os.path.exists(model_dir):
            raise FileExistsError(f"Model {flags.model_name} already exists.")
    
    # Setup device
    device = setup_device()
    print(f"Device: {device}")
    
    # Seed
    L.seed_everything(SEED)
    
    # Load data
    data_module = ACDCWithPredictedMaskDataModule(batch_size=16)
    
    # Reseed after preprocessing data
    L.seed_everything(SEED)

    # Train

    model = VAECorrector(
        in_channels=data_module.data_train.num_classes,
        latent_dim=flags.latent_dim,
        beta=flags.beta,
    )
    
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=flags.epochs,
        logger=TensorBoardLogger(
            save_dir=flags.logs,
            name=ACDC.DIR.VAE_CORRECTOR,
            version=flags.model_name if flags.model_name else None,
            default_hp_metric=False,
        ),
        callbacks=[
            ModelCheckpoint(monitor="loss/val", mode="min"),
            LearningRateMonitor("epoch"),
        ]
    )
    
    trainer.fit(model, data_module)

if __name__ == "__main__":    
    flags = parse_args()
    main(flags)
