import monai

from arch.unet.segmentation_base import SegmentationBase

class DenseNet(SegmentationBase):
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
            model_type="densenet",
        )

        self.hparams.update({"img_size": img_size})

        self.model = monai.networks.nets.densenet.DenseNet(
            spatial_dims=2,
            in_channels=self.hparams.in_channels,
            out_channels=self.hparams.out_channels,
        )
