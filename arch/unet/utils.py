from arch.unet.acunet import ACUNet
from arch.unet.swinunet import SwinUNet
from arch.unet.unet import UNet

ID_TO_MODEL = {
    "unet": UNet,
    "acunet": ACUNet,
    "swinunet": SwinUNet,
}
