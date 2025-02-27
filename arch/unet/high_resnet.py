from monai.networks.nets.highresnet import HighResNet as HighResnetModel

from arch.unet.segmentation_base import SegmentationBase

DEFAULT_LAYER_PARAMS_2D = (
    # initial conv layer
    {"name": "conv_0", "n_features": 64, "kernel_size": 3},
    # residual blocks
    {"name": "res_1", "n_features": 64, "kernels": (3, 3), "repeat": 3},
    {"name": "res_2", "n_features": 128, "kernels": (3, 3), "repeat": 3},
    {"name": "res_3", "n_features": 256, "kernels": (3, 3), "repeat": 3},
    # final conv layers
    {"name": "conv_1", "n_features": 512, "kernel_size": 1},
    {"name": "conv_2", "kernel_size": 1},
)

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
            layer_params=DEFAULT_LAYER_PARAMS_2D,
        )
