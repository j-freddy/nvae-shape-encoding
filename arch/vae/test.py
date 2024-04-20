import argparse
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from arch.vae.vae import VAE
from const import ACDC, LOGS_PATH, SEED
from data_modules.acdc import ACDCMaskDataModule
from utils import setup_device

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to model checkpoint.",
        required=True,
    )

    return parser.parse_args()

def main(flags: argparse.Namespace):
    # Setup device
    device = setup_device()
    print(f"Device: {device}")
    
    # Seed
    L.seed_everything(SEED)
    
    # Load data
    data_module = ACDCMaskDataModule(batch_size=20)
    # TODO Bit hacky but I want to use 1 batch only to calculate FID
    data_module.batch_size = data_module.data_test.shape[0]
    
    # Reseed after preprocessing data
    L.seed_everything(SEED)
    
    # Load model
    model = VAE.load_from_checkpoint(flags.model_path)

    # TODO noqa
    model_name = flags.model_path.split("/")[2]

    trainer = L.Trainer(
        # Using CPU for testing as FID calculation with 1 large batch can give
        # OOM error (script runs ~1 min on CPU)
        accelerator="cpu",
        devices="auto",
        logger=TensorBoardLogger(
            save_dir=LOGS_PATH,
            name=ACDC.DIR.VAE,
            version=model_name,
            default_hp_metric=False,
        ),
    )

    trainer.test(model, data_module)
    
if __name__ == "__main__":
    flags = parse_args()
    main(flags)
