import argparse
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from arch.nvae_corrector.nvae_corrector import NVAECorrector
from utils.const import ACDC, LOGS_PATH, SEED
from data_modules.acdc import ACDC3DWithPredictedMaskDataModule
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
    data_module = ACDC3DWithPredictedMaskDataModule()
    
    # Reseed after preprocessing data
    L.seed_everything(SEED)
    
    # Load model
    model = NVAECorrector.load_from_checkpoint(flags.model_path)

    # noqa
    model_name = flags.model_path.split("/")[2]

    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        logger=TensorBoardLogger(
            save_dir=flags.logs,
            name=ACDC.DIR.NVAE_CORRECTOR,
            version=model_name,
            default_hp_metric=False,
        ),
    )

    trainer.test(model, data_module)
    
if __name__ == "__main__":
    flags = parse_args()
    main(flags)
