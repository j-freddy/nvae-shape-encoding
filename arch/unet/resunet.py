from monai.networks.nets.unet import Unet as ResUNetModel

from arch.unet.segmentation_base import SegmentationBase

class ResUNet(SegmentationBase):
    def __init__(
        self,
        in_channels: int=1,
        out_channels: int=4,
        optim_name: str="adam",
        lr: int=1e-3,
        weight_decay: int=0,
        res_units: int=2,
    ):
        super().__init__(
            in_channels,
            out_channels,
            optim_name,
            lr,
            weight_decay,
            model_type="resunet",
        )

        self.hparams.update({"res_units": res_units})

        self.model = ResUNetModel(
            in_channels=self.hparams.in_channels,
            out_channels=self.hparams.out_channels,
            spatial_dims=2,
            channels = [64, 128, 256, 512, 1024],
            strides = [2, 2, 2, 2],
            num_res_units=self.hparams.res_units,
        )
