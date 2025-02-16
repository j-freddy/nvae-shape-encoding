from arch.unet.acunet import ACUNet
from arch.unet.attentionunet import AttentionUNet
from arch.unet.densenet import DenseNet
from arch.unet.resunet import ResUNet
from arch.unet.swinunet import SwinUNet
from arch.unet.unet import UNet

ID_TO_MODEL = {
    "unet": UNet,
    "acunet": ACUNet,
    "swinunet": SwinUNet,
    "attentionunet": AttentionUNet,
    "densenet": DenseNet,
    "resunet": ResUNet,
}

MODEL_TYPES = set(ID_TO_MODEL.keys())
