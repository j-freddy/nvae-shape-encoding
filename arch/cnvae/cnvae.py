from collections import defaultdict
import lightning as L
import math
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from arch.nvae.decoder import Decoder
from arch.nvae.distribution import Normal
from arch.nvae.encoder import Encoder
from utils.const import CARDIAC_WIDTH, FRDS_MODEL_PATH, MASK_CLASSES
from utils.anatomical_validity_checker import AnatomicalValidityChecker
from utils.eval import compute_dice_score, compute_frds, get_samples_and_reconstructions_pixel_diff
from utils.utils import clamp, discretise, show_samples

class CNVAE(L.LightningModule):
    """
    Conditional Nouveau VAE.
    """
    
    def __init__(
        self,
        in_channels: int=1,
        initial_channels: int=64,
        out_channels: int=4,
        min_channels: int=0,
        z_channels: int=20,
        num_groups_per_layer: list[int]=[4, 2, 1],
        is_layer_shared: list[bool]=[True, True, True],
        initial_downsample_factor: int=8,
        max_epochs: int=50,
        beta_per_layer: list[float]=[1.0, 1.0, 1.0],
        kl_warmup_steps: int=500,
        use_sr: bool=False,
        freeze_decoder: bool=False,
    ):
        """
        Create an instance of the Conditional NVAE model. All constructor
        arguments are saved in the checkpoint as hyperparameters.
        
        Args:
            in_channels (int): Number of input channels. Corresponds to number
                of colour channels in the input image. Default: 1.
            initial_channels (int): Number of channels @in_channels gets
                projected to. This is done immediately in the stem, then
                reverted at the very end of the pass in the conditional coder.
                Default: 64.
            out_channels (int): Number of output channels. Corresponds to number
                of classes in segmentation mask. Default: 4.
            min_channels (int): Minimum number of channels anywhere. By default,
                the number of channels is doubled at each deeper layer. For
                example, if @initial_channels is 16 and there are 3 layers, it
                goes 16 -> 32 -> 64 -> 128. But if @initial_channels is 4 and
                @min_channels is 16, it goes 16 -> 16 -> 16 -> 32 instead of 4
                -> 8 -> 16 -> 32. The idea is to not have an initial bottleneck,
                which may be caused by limited capacity from the small number of
                channels. Default: 16.
            z_channels (int): Number of channels in the latent space. For
                example, if the latent space is 4x4 for a particular layer, the
                latent variable is @z_channelsx4x4. Default: 20.
            num_groups_per_layer (list[int]): Number of groups in each layer.
                This is traversed sequentially, so more groups allow for more
                corrections and refinement within a single layer. Order: from
                shallowest to topmost layer. Default: [4, 2, 1].
            is_layer_shared (list[bool]): Whether the latent space in each layer
                is shared with the decoder. If a layer is not shared, it only
                consists of residual cells and does not have a combiner nor
                sampler. Order corresponds to @num_groups_per_layer. Default:
                [True, True, True].
            initial_downsample_factor (int): Downsample factor in the
                preprocess stage of the encoder tower. This corresponds to the
                upsample factor in the postprocess stage of the decoder tower.
                For example, if @initial_downsample_factor is 8 and the input is
                128x128, the preprocess stage downsamples to 16x16. Default: 8.
            max_epochs (int): Maximum number of epochs for training. Default:
                50.
            beta_per_layer (list[float]): Beta coefficient for each shared
                layer. Order corresponds to the shared layers of
                @is_layer_shared. Default: [1.0, 1.0, 1.0].
            kl_warmup_steps (int): Number of steps to perform KL annealing.
                Each epoch has 214 steps. Default: 500.
            use_sr (bool): If True, use spectral regularisation. Default: False.
            freeze_decoder (bool): If True, freeze the decoder and conditional 
                coder weights. Default: False.
        """
        
        super().__init__()
        
        assert len(num_groups_per_layer) == len(is_layer_shared)
        assert sum(is_layer_shared) == len(beta_per_layer)
        
        self.save_hyperparameters()
        
        self.img_width = CARDIAC_WIDTH
        self.num_layers = len(self.hparams.num_groups_per_layer)
        self.num_latent_layers = len(self.hparams.beta_per_layer)
        
        self.layer_idx_to_latent_idx = self._get_layer_idx_to_latent_idx_map()
        
        # Conditional NVAE has 2 encoders that take in the image and the mask
        # respectively
        
        self.bottom_up = nn.ModuleDict({
            "image": nn.ModuleDict(),
            "mask": nn.ModuleDict(),
        })
        
        for key in self.bottom_up.keys():
            # The mask encoder has the same number of channels as the output
            in_channels = self.hparams.in_channels if key == "image" else self.hparams.out_channels
            
            # Table 6: # initial channels in enc. (NVAE paper)
            self.bottom_up[key]["stem"] = nn.Conv2d(
                in_channels,
                max(self.hparams.initial_channels, self.hparams.min_channels),
                kernel_size=3,
                padding=1,
                bias=True,
            )
            
            self.bottom_up[key]["encoder"] = Encoder(
                num_groups_per_layer=self.hparams.num_groups_per_layer,
                is_layer_shared=self.hparams.is_layer_shared,
                initial_channels=self.hparams.initial_channels,
                min_channels=self.hparams.min_channels,
                z_channels=self.hparams.z_channels,
                initial_downsample_factor=self.hparams.initial_downsample_factor,
            )
        
        top_latent_dim = self._get_latent_dim(self.num_layers - 1)
        
        # TODO The Decoder will be different, so create a new Decoder class
    
    def configure_optimizers(self):
        optimiser = torch.optim.Adamax(
            self.parameters(),
            lr=1e-2,
            eps=1e-3,
            weight_decay=3e-4,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser,
            T_max=self.hparams.max_epochs,
            eta_min=1e-4,
        )
        
        return [optimiser], [lr_scheduler]
    
    def _get_layer_idx_to_latent_idx_map(self) -> dict[int, int]:
        latent_idx_to_layer_idx = [
            i for i, is_latent in enumerate(self.hparams.is_layer_shared)
            if is_latent
        ]
        
        mapping = {
            layer_idx: latent_idx for latent_idx, layer_idx in enumerate(latent_idx_to_layer_idx)
        }
        
        return mapping

    def _get_latent_dim(self, layer: int) -> int:
        # Layer 0 is the shallowest layer
        # Layer @(num_layers - 1) is the deepest (topmost) layer
        return (self.img_width // self.hparams.initial_downsample_factor) // (2 ** layer)
    
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        scans, feats, _, _ = batch
        
        print(scans.shape)
        print(feats.shape)
        
        import sys
        sys.exit()
        
        loss = 0
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        pass
    
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        pass
