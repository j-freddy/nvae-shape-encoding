from monai.networks.nets.highresnet import HighResNet as HighResnetModel

from arch.unet.segmentation_base import SegmentationBase

class HighResnet(SegmentationBase):
    def __init__(
        self,
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
            model_type="highresnet",
        )

        self.model = HighResnetModel(
            in_channels=self.hparams.in_channels,
            out_channels=self.hparams.out_channels,
            spatial_dims=2,
        )
