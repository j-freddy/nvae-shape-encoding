import argparse
import os
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from arch.vae.utils import ID_TO_MODEL
from utils.const import ACDC, LOGS_PATH, SEED
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
        "--latent_dim",
        type=int,
        help="Dimension of latent space.",
        default=8,
    )
    
    parser.add_argument(
        "--loss_reg",
        type=str,
        help="Regulariser technique.",
        choices=ID_TO_MODEL.keys(),
        default="beta_vae",
    )
    
    parser.add_argument(
        "--beta",
        type=float,
        help="Beta value for KL divergence.",
        default=1.0,
    )
    
    parser.add_argument(
        "--gamma",
        type=float,
        help="Gamma value for divergence between q(z) and p(z).",
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
    if flags.model_name:
        # Check if model name already exists
        model_dir = os.path.join(flags.logs, ACDC.DIR.VAE, flags.model_name)
        
        if os.path.exists(model_dir):
            raise FileExistsError(f"Model {flags.model_name} already exists.")
    
    # Setup device
    device = setup_device()
    print(f"Device: {device}")
    
    # It is important to seed before preprocessing data as shuffling the data
    # before partitioning the train and val set happens when the data module is
    # created.
    
    # Then, seed again after creating the data module. This is because the
    # preprocessing stage is skipped if the preprocessed data files are found.
    
    # Seed
    L.seed_everything(SEED)
    
    # Load data
    data_module = ACDCMaskDataModule(
        batch_size=16,
        filter_empty=flags.filter_empty,
        register_alignment=flags.register_alignment,
        augment_rotation=flags.augment,
    )
    
    # Reseed after preprocessing data
    L.seed_everything(SEED)

    # Train
    Model: L.LightningModule = ID_TO_MODEL[flags.loss_reg]
    
    model = Model(
        in_channels=data_module.data_test.num_classes,
        latent_dim=flags.latent_dim,
        loss_reg=flags.loss_reg,
        beta=flags.beta,
        gamma=flags.gamma,
    )
    
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=flags.epochs,
        logger=TensorBoardLogger(
            save_dir=flags.logs,
            name=ACDC.DIR.VAE,
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
