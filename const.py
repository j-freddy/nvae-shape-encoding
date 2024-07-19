from dataclasses import dataclass
from enum import Enum
import os

DATA_PATH = "data"
LOGS_PATH = "logs"
OUT_PATH = "out"
SCRIPTS_PATH = "scripts"
SEED = 1969

FRDS_MODEL_PATH = "logs/simclr_acdc/resnet-18-v2-no-elastic/checkpoints/epoch=143-step=1008.ckpt"
NVAE_MODEL_PATH = "logs/nvae_acdc/default/pc-4-ws-6420-b-10/checkpoints/epoch=97-step=20972.ckpt"

@dataclass
class CIFAR10:
    """
    Deprecated. For initial experimentation / early debugging. Main dataset is
    ACDC.
    
    CIFAR-10 constants.
    """
    class DIR:
        VAE = "vae_cifar10"
        NVAE = "nvae_cifar10"
        SIMCLR = "simclr_cifar10"
        UNET = "unet_cifar10"

@dataclass
class ACDC:
    """
    ACDC constants.
    """
    WIDTH = 128
    NUM_CLASSES = 4
    
    conditions = ["NOR", "MINF", "DCM", "HCM", "RV"]
    condition_to_idx = {condition: i for i, condition in enumerate(conditions)}

    class ClassLabel(Enum):
        BG = 0
        RV = 1
        MYO = 2
        LV = 3

    class DIR:
        VAE = "vae_acdc"
        NVAE = "nvae_acdc"
        SIMCLR = "simclr_acdc"
        UNET = "unet_acdc"

    @dataclass
    class RAW:
        TRAIN_PATH = os.path.join(DATA_PATH, "ACDC", "database", "training")
        TEST_PATH = os.path.join(DATA_PATH, "ACDC", "database", "testing")
    
    TRAIN_PATH = os.path.join(DATA_PATH, "acdc_processed_train.pt")
    TEST_PATH = os.path.join(DATA_PATH, "acdc_processed_test.pt")
    
    @dataclass
    class ALIGNED:
        TRAIN_PATH = os.path.join(DATA_PATH, "acdc_aligned_mask_train.pt")
        TEST_PATH = os.path.join(DATA_PATH, "acdc_aligned_mask_test.pt")
