import argparse
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
import torch

from arch.unet.utils import ID_TO_MODEL
from data_modules.mnms import MnMs3DDataModule
from utils.const import ACDC, LOGS_PATH, SEED, MnMs
from data_modules.acdc import ACDC3DDataModule
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
        "--dataset",
        type=str,
        help="Which dataset to use.",
        choices=["acdc", "mnms"],
        default="acdc",
    )
    
    parser.add_argument(
        "--centre",
        type=int,
        help="If using M&Ms and set, only use scans from the specified centre.",
        choices=MnMs.centres,
        default=None,
    )
    
    parser.add_argument(
        "--logs",
        type=str,
        help="Root save directory for logs.",
        default=LOGS_PATH,
    )

    return parser.parse_args()

def main(flags: argparse.Namespace):
    match flags.dataset:
        case "acdc":
            dataset_dir = ACDC.DIR.UNET
        case "mnms":
            dataset_dir = MnMs.DIR.UNET
        case _:
            raise ValueError(f"Unknown dataset: {flags.dataset}")
    
    # Setup device
    device = setup_device()
    print(f"Device: {device}")
    
    # Seed
    L.seed_everything(SEED)
    
    # Load data
    match flags.dataset:
        case "acdc":
            data_module = ACDC3DDataModule()
        case "mnms":
            data_module = MnMs3DDataModule(from_centre=flags.centre)
    
    # Reseed after preprocessing data
    L.seed_everything(SEED)
    
    # Load model
    # Inefficient: Need to load the hparams first to determine class type,
    # then use load_from_checkpoint. So, this loads the checkpoint twice.
    checkpoint = torch.load(flags.model_path, map_location=device)
    Model: L.LightningModule = ID_TO_MODEL[checkpoint["hyper_parameters"]["loss_reg"]]
    del checkpoint
    model = Model.load_from_checkpoint(flags.model_path)

    # noqa
    model_name = flags.model_path.split("/")[2]

    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        logger=TensorBoardLogger(
            save_dir=flags.logs,
            name=dataset_dir,
            version=model_name,
            default_hp_metric=False,
        ),
    )

    trainer.test(model, data_module)

if __name__ == "__main__":
    flags = parse_args()
    main(flags)
