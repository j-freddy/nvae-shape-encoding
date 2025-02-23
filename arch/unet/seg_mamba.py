from monai.networks.nets.densenet import DenseNet as DenseNetModel

from arch.seg_mamba.model_segmamba.segmamba import SegMamba
from arch.unet.segmentation_base import SegmentationBase

class SegMambaLightningModule(SegmentationBase):
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
            model_type="seg_mamba",
        )
        
        self.model = SegMamba(
            in_chans=in_channels,
            out_chans=out_channels,
            depths=[2,2,2,2],
            feat_size=[48, 96, 192, 384],
        )
