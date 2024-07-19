from arch.unet.acunet import ACUNet
from arch.unet.unet import UNet

ID_TO_MODEL = {
    "cross_entropy": UNet,
    "shape_prior": ACUNet,
}
