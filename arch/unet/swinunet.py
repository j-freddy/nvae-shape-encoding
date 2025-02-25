from monai.networks.nets.swin_unetr import SwinUNETR as SwinUNetRModel

from arch.unet.segmentation_base import SegmentationBase

class SwinUNetR(SegmentationBase):
    def __init__(
        self, 
        img_size: tuple=(128,128),
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
            model_type="swinunet",
        )

        self.hparams.update({"img_size": img_size})

        self.model = SwinUNetRModel(
            img_size=self.hparams.img_size,
            in_channels=self.hparams.in_channels,
            out_channels=self.hparams.out_channels,
            spatial_dims=2,
        )
