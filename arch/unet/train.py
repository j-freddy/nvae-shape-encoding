import argparse
import os
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from arch.unet.utils import ID_TO_MODEL
from utils.const import ACDC, LOGS_PATH, SEED
from data_modules.acdc import ACDCDataModule
from utils.utils import setup_device

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--epochs",
        type=int,
        help="Max number of epochs.",
        default=50,
    )
    
    parser.add_argument(
        "--loss_reg",
        type=str,
        help="Regulariser technique.",
        choices=ID_TO_MODEL.keys(),
        default="shape_prior",
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        help="If using shape prior loss, the weight of cross entropy loss.",
        default=1.0,
    )
    
    parser.add_argument(
        "--filter_empty",
        action=argparse.BooleanOptionalAction,
        help="If set, filter out empty masks.",
        default=False,
    )
    
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        help="If set, augment training data with random flips.",
        default=False,
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for train reproducibility. This only affects training, not data split.",
        default=SEED,
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
        model_dir = os.path.join(flags.logs, ACDC.DIR.UNET, flags.model_name)
        
        if os.path.exists(model_dir):
            raise ValueError(f"Model {flags.model_name} already exists.")
    
    # Setup device
    device = setup_device()
    print(f"Device: {device}")
    
    # Seed
    L.seed_everything(SEED)
    
    # Load data
    data_module = ACDCDataModule(
        batch_size=32,
        filter_empty=flags.filter_empty,
        augment=flags.augment,
    )
    
    # Reseed after preprocessing data
    # Accept a custom seed for training, but ensure data split is consistent
    L.seed_everything(flags.seed)
    
    # Train
    Model: L.LightningModule = ID_TO_MODEL[flags.loss_reg]
    
    model = Model(
        in_channels=data_module.data_test.num_channels,
        out_channels=data_module.data_test.num_classes,
        loss_reg=flags.loss_reg,
        # alpha=flags.alpha,
    )
    
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=flags.epochs,
        logger=TensorBoardLogger(
            save_dir=flags.logs,
            name=ACDC.DIR.UNET,
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
