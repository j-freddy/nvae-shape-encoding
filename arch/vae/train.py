import argparse
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from arch.vae.factor_vae import FactorVAE
from arch.vae.tcvae import TCVAE
from arch.vae.vae import VAE
from const import ACDC, LOGS_PATH, SEED
from data_modules.acdc import ACDCMaskDataModule
from utils import setup_device

ID_TO_MODEL = {
    "beta_vae": VAE,
    "beta_tcvae": TCVAE,
    "factor_vae": FactorVAE,
}

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
    
    return parser.parse_args()

def main(flags: argparse.Namespace):
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
    data_module = ACDCMaskDataModule(batch_size=16, filter_empty=flags.filter_empty)
    
    # Reseed after preprocessing data
    L.seed_everything(SEED)
    
    _, num_classes, _, _ = data_module.data_test.shape

    # Train
    Model = ID_TO_MODEL[flags.loss_reg]
    
    model = Model(
        in_channels=num_classes,
        latent_dim=flags.latent_dim,
        beta=flags.beta,
    )
    
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=flags.epochs,
        logger=TensorBoardLogger(
            save_dir=LOGS_PATH,
            name=ACDC.DIR.VAE,
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
