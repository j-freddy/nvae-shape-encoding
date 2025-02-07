import argparse
import os
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from arch.nvae.utils import ID_TO_ARCH
from utils.const import ACDC, LOGS_PATH, SEED
from data_modules.acdc import ACDCWithPredictedMaskDataModule
from arch.nvae.nvae import NVAE
from utils.utils import setup_device

def parse_args() -> argparse.Namespace:
    def list_float(arg: str) -> list[float]:
        return list(map(float, arg.split(",")))
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--epochs",
        type=int,
        help="Max number of epochs.",
        default=100,
    )
    
    parser.add_argument(
        "--arch",
        type=str,
        help="Architecture configuration. See ID_TO_ARCH in arch/nvae/utils.py.",
        choices=ID_TO_ARCH.keys(),
        default="default",
    )
    
    parser.add_argument(
        "--projected_channels",
        type=int,
        help="Number of channels in the immediate space projected through the stem (and conditional coder).",
        default=4,
    )
    
    parser.add_argument(
        "--min_channels",
        type=int,
        help="If set, set a lower clamp to the number of channels anywhere in the model.",
        default=0,
    )
    
    parser.add_argument(
        "--z_channels",
        type=int,
        help="Number of channels in the latent space at each layer.",
        default=20,
    )
    
    parser.add_argument(
        "--warmup_steps",
        type=int,
        help="Number of steps for KL divergence linear deterministic warmup.",
        default=500,
    )
    
    parser.add_argument(
        "--betas",
        type=list_float,
        help="Beta values for KL divergence for each layer. First layer is shallowest; last entry is topmost layer.",
        default=[1.0, 1.0, 1.0],
    )
    
    parser.add_argument(
        "--sr",
        action=argparse.BooleanOptionalAction,
        help="If set, use spectral regularisation in the loss.",
        default=False,
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
        model_dir = os.path.join(flags.logs, ACDC.DIR.NVAE_CORRECTOR, flags.model_name)
        
        if os.path.exists(model_dir):
            raise FileExistsError(f"Model {flags.model_name} already exists.")
    
    # Setup device
    device = setup_device()
    print(f"Device: {device}")
    
    # Seed
    L.seed_everything(SEED)
    
    # Load data
    data_module = ACDCWithPredictedMaskDataModule(batch_size=8)
    
    # Reseed after preprocessing data
    L.seed_everything(SEED)
    
    num_classes = data_module.data_test.num_classes
    arch_config = ID_TO_ARCH[flags.arch]
    
    print("Working so far!")
    
    import sys
    sys.exit()

    # Train
    model = NVAE(
        in_channels=num_classes,
        initial_channels=flags.projected_channels,
        min_channels=flags.min_channels,
        z_channels=flags.z_channels,
        num_groups_per_layer=arch_config["num_groups_per_layer"],
        is_layer_shared=arch_config["is_layer_shared"],
        initial_downsample_factor=arch_config["initial_downsample_factor"],
        max_epochs=flags.epochs,
        beta_per_layer=flags.betas,
        kl_warmup_steps=flags.warmup_steps,
        use_sr=flags.sr,
    )
    
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=flags.epochs,
        logger=TensorBoardLogger(
            save_dir=flags.logs,
            name=ACDC.DIR.NVAE,
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
