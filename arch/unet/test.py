import argparse
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from arch.unet.unet import UNet
from const import ACDC, LOGS_PATH, SEED
from data_modules.acdc import ACDCDataModule
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
    data_module = ACDCDataModule(batch_size=32)
    
    # Reseed after preprocessing data
    L.seed_everything(SEED)
    
    # Load model
    model = UNet.load_from_checkpoint(flags.model_path)

    # noqa
    model_name = flags.model_path.split("/")[2]

    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        logger=TensorBoardLogger(
            save_dir=flags.logs,
            name=ACDC.DIR.UNET,
            version=model_name,
            default_hp_metric=False,
        ),
    )

    trainer.test(model, data_module)

if __name__ == "__main__":
    flags = parse_args()
    main(flags)
