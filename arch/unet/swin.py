from monai.networks.nets.swin_unetr import SwinUNETR as SwinUNetRModel
import torch

from arch.unet.segmentation_base import SegmentationBase
from arch.unet.swin_network.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

class SwinUNet(SegmentationBase):
    def __init__(
        self, 
        img_size: int=128,
        in_channels: int=1,
        out_channels: int=4,
        optim_name: str="adam",
        lr: int=1e-3,
        weight_decay: int=0,
    ):
        super().__init__(
            in_channels,
            out_channels,
            optim_name,
            lr,
            weight_decay,
            model_type="swin",
        )

        self.hparams.update({"img_size": img_size})

        self.model = SwinTransformerSys(
            img_size=224,
            in_chans=self.hparams.in_channels,
            num_classes=self.hparams.out_channels,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # We must rescale from img_size to 224, then rescale back to img_size
        x = torch.nn.functional.interpolate(x, size=224, mode="bilinear")
        y_hat = self.model(x)
        y_hat = torch.nn.functional.interpolate(y_hat, size=self.hparams.img_size, mode="bilinear")
        return y_hat
