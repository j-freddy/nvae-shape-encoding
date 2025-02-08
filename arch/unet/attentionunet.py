from monai.networks.nets.attentionunet import AttentionUnet as AttentionUNetModel

from arch.unet.segmentation_base import SegmentationBase

class AttentionUNet(SegmentationBase) :
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
            model_type="attentionunet",
        )
        
        self.model = AttentionUNetModel(
            in_channels = self.hparams.in_channels, 
            out_channels = self.hparams.out_channels,
            spatial_dims=2,
            channels = [64, 128, 256, 512, 1024],
            strides = [2, 2, 2, 2],
        )
