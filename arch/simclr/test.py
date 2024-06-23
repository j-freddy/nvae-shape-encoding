import argparse
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from arch.simclr.simclr import SimCLR
from const import ACDC, LOGS_PATH, SEED
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
    
    return parser.parse_args()

def main(flags: argparse.Namespace):
    # Setup device
    device = setup_device()
    print(f"Device: {device}")
    
    # Seed
    L.seed_everything(SEED)
    
    # Load data
    data_module = ACDCMaskDataModule(batch_size=20, as_image=True)
    # TODO Bit hacky but I want to use 1 batch only to calculate FRDS
    data_module.batch_size = len(data_module.data_test)
    
    # Reseed after preprocessing data
    L.seed_everything(SEED)
    
    # Load model
    model = SimCLR.load_from_checkpoint(flags.model_path)

    # TODO noqa
    model_name = flags.model_path.split("/")[2]

    trainer = L.Trainer(
        # Using CPU for testing as FRDS calculation with 1 large batch can give
        # OOM error
        accelerator="cpu",
        devices="auto",
        logger=TensorBoardLogger(
            save_dir=flags.logs,
            name=ACDC.DIR.SIMCLR,
            version=model_name,
            default_hp_metric=False,
        ),
    )

    trainer.test(model, data_module)

if __name__ == "__main__":
    flags = parse_args()
    main(flags)
