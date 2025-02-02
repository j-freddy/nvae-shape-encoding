import argparse
import os
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import torch.nn as nn

from arch.cnvae.cnvae import CNVAE
from arch.nvae.utils import ID_TO_ARCH
from utils.const import ACDC, LOGS_PATH, SEED
from data_modules.acdc import ACDC3DDataModule, ACDCDataModule
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
        "--cbetas",
        type=list_float,
        help="Beta values for KL divergence between approximate posterior and conditional prior for each layer. First layer is shallowest; last entry is topmost layer.",
        default=[1.0, 1.0, 1.0],
    )
    
    parser.add_argument(
        "--betas",
        type=list_float,
        help="Beta values for KL divergence between approximate posterior and unconditional prior for each layer.",
        default=[1.0, 1.0, 1.0],
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
        "--pretrained_nvaeseg_model_path",
        type=str,
        help="If set, load a pretrained NVAE-Seg model from this path and use its weights.",
    )
    
    parser.add_argument(
        "--freeze_decoder",
        action=argparse.BooleanOptionalAction,
        help="If set, freeze the decoder and conditional coder weights.",
        default=False,
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for train reproducibility. This only affects training, not data split.",
        default=SEED,
    )
    
    return parser.parse_args()

def load_pretrained_module(
    component: nn.Module,
    component_id: str,
    pretrained_state_dict: dict,
    exact_match: bool = True,
):
    filtered_state_dict = {}
        
    for key in list(pretrained_state_dict.keys()):
        if key.startswith(component_id):
            # Remove prefix
            shortened_key = key[len(component_id) + 1:]
            filtered_state_dict[shortened_key] = pretrained_state_dict[key]
    
    return component.load_state_dict(filtered_state_dict, strict=exact_match)

def main(flags: argparse.Namespace):
    if flags.model_name:
        # Check if model name already exists
        model_dir = os.path.join(flags.logs, ACDC.DIR.CNVAE, flags.model_name)
        
        if os.path.exists(model_dir):
            raise ValueError(f"Model {flags.model_name} already exists.")
    
    # Setup device
    device = setup_device()
    print(f"Device: {device}")
    
    # Seed
    L.seed_everything(SEED)
    
    # Load data
    data_module = ACDCDataModule(
        batch_size=8,
        filter_empty=flags.filter_empty,
        augment=flags.augment,
    )
    
    # Reseed after preprocessing data
    # Accept a custom seed for training, but ensure data split is consistent
    L.seed_everything(flags.seed)

    arch_config = ID_TO_ARCH[flags.arch]

    # Train
    model = CNVAE(
        in_channels=data_module.data_test.num_channels,
        out_channels=data_module.data_test.num_classes,
        initial_channels=flags.projected_channels,
        min_channels=flags.min_channels,
        z_channels=flags.z_channels,
        num_groups_per_layer=arch_config["num_groups_per_layer"],
        is_layer_shared=arch_config["is_layer_shared"],
        initial_downsample_factor=arch_config["initial_downsample_factor"],
        max_epochs=flags.epochs,
        cbeta_per_layer=flags.cbetas,
        beta_per_layer=flags.betas,
        kl_warmup_steps=flags.warmup_steps,
        freeze_decoder=flags.freeze_decoder,
    )
    
    if flags.pretrained_nvaeseg_model_path:
        state_dict = torch.load(
            flags.pretrained_nvaeseg_model_path,
            map_location=device,
            weights_only=True,
        )["state_dict"]
        
        components_list = [
            (model.bottom_up["image"]["stem"], "stem", True),
            (model.bottom_up["image"]["encoder"], "encoder", True),
            (model.decoder, "decoder", False),
            (model.conditional_coder, "conditional_coder", True),
        ]
        
        for component, component_id, exact_match in components_list:
            incompatible_keys = load_pretrained_module(
                component,
                component_id,
                state_dict,
                exact_match=exact_match,
            )
            
            # Should be top_combiner_cell and top_sampler only for decoder
            print(incompatible_keys)
        
        print(f"Loaded pretrained NVAESeg model from {flags.pretrained_nvaeseg_model_path}.")
    
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=flags.epochs,
        logger=TensorBoardLogger(
            save_dir=flags.logs,
            name=ACDC.DIR.CNVAE,
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
