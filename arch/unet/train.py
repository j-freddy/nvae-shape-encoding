import argparse
import os
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from arch.unet.acunet import ACUNet
from arch.unet.acunet_vae import ACVAEUNet
from arch.unet.attentionunet import AttentionUNet
from arch.unet.densenet import DenseNet
from arch.unet.resunet import ResUNet
from arch.unet.swinunet import SwinUNet
from arch.unet.unet import UNet
from arch.unet.utils import ID_TO_MODEL
from data_modules.mnms import MnMsDataModule
from utils.const import ACDC, CARDIAC_WIDTH, LOGS_PATH, SEED, MnMs
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
        "--model_type",
        type=str,
        help="Model architecture.",
        choices=ID_TO_MODEL.keys(),
        default="unet",
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        help="If using shape prior loss, the weight of cross entropy loss.",
        default=1.0,
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
        "--num_subjects",
        type=int,
        help="Few-shot learning for M&Ms: Number of subjects to use. If -1, use all subjects.",
        default=-1,
    )
    
    parser.add_argument(
        "--sort_by_validity",
        action=argparse.BooleanOptionalAction,
        help="Few-shot learning for M&Ms: If set, use subjects with highest anatomical validity.",
        default=False,
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
    
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        help="If set, load a pretrained model from this path and continue training.",
        default=None,
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
    
    if flags.model_name:
        # Check if model name already exists
        model_dir = os.path.join(flags.logs, dataset_dir, flags.model_name)
        
        if os.path.exists(model_dir):
            raise FileExistsError(f"Model {flags.model_name} already exists.")
    
    # Setup device
    device = setup_device()
    print(f"Device: {device}")
    
    # Seed
    L.seed_everything(SEED)
    
    # Load data
    match flags.dataset:
        case "acdc":
            data_module = ACDCDataModule(
                batch_size=32,
                filter_empty=flags.filter_empty,
                augment=flags.augment,
            )
        
        case "mnms":
            data_module = MnMsDataModule(
                # Use a smaller batch size due to fewewer samples
                batch_size=16,
                filter_empty=flags.filter_empty,
                from_centre=flags.centre,
                num_subjects=flags.num_subjects,
                sort_by_validity=flags.sort_by_validity,
                augment=flags.augment,
            )
    
    # Reseed after preprocessing data
    # Accept a custom seed for training, but ensure data split is consistent
    L.seed_everything(flags.seed)
    
    # Train
    if flags.pretrained_model_path:
        Model: L.LightningModule = ID_TO_MODEL[flags.model_type]
        model = Model.load_from_checkpoint(flags.pretrained_model_path)
    else:
        match flags.model_type:
            case "unet":
                model = UNet(
                    in_channels=data_module.data_test.num_channels,
                    out_channels=data_module.data_test.num_classes,
                )
            
            case "acunet":
                model = ACUNet(
                    in_channels=data_module.data_test.num_channels,
                    out_channels=data_module.data_test.num_classes,
                    alpha=flags.alpha,
                )
            
            case "acunet_vae":
                model = ACVAEUNet(
                    in_channels=data_module.data_test.num_channels,
                    out_channels=data_module.data_test.num_classes,
                    alpha=flags.alpha,
                )
            
            case "swinunet":
                model = SwinUNet(
                    img_size=(CARDIAC_WIDTH, CARDIAC_WIDTH),
                    in_channels=data_module.data_test.num_channels,
                    out_channels=data_module.data_test.num_classes,
                )
            
            case "attentionunet":
                model = AttentionUNet(
                    in_channels=data_module.data_test.num_channels,
                    out_channels=data_module.data_test.num_classes,
                )
            
            case "densenet":
                model = DenseNet(
                    in_channels=data_module.data_test.num_channels,
                    out_channels=data_module.data_test.num_classes,
                )
            
            case "resunet":
                model = ResUNet(
                    in_channels=data_module.data_test.num_channels,
                    out_channels=data_module.data_test.num_classes,
                )
            
            case _:
                raise ValueError(f"Unknown model type: {flags.model_type}")
    
    # By default, Lightning logs every 50 steps.
    log_every_n_steps = 50
    
    # Few-shot learning: log more frequently (but slower training)
    if flags.num_subjects != -1 and flags.num_subjects < 20:
        log_every_n_steps = 6
    
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=flags.epochs,
        logger=TensorBoardLogger(
            save_dir=flags.logs,
            name=dataset_dir,
            version=flags.model_name if flags.model_name else None,
            default_hp_metric=False,
        ),
        callbacks=[
            ModelCheckpoint(monitor="loss/val", mode="min"),
            LearningRateMonitor("epoch"),
        ],
        log_every_n_steps=log_every_n_steps,
    )
    
    trainer.fit(model, data_module)

if __name__ == "__main__":
    flags = parse_args()
    main(flags)
