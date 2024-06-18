from dataclasses import dataclass
import os

DATA_PATH = "data"
LOGS_PATH = "logs"
OUT_PATH = "out"
SCRIPTS_PATH = "scripts"
SEED = 1969

# TODO Remove once FRDS is finalised
FRDS_MODEL_PATH_V2 = "logs/simclr_acdc/resnet-18-v2/checkpoints/epoch=199-step=1400.ckpt"
FRDS_MODEL_PATH_V2_NO_ELASTIC = "logs/simclr_acdc/resnet-18-v2-no-elastic/checkpoints/epoch=143-step=1008.ckpt"
FRDS_MODEL_PATH_V3 = "logs/simclr_acdc/resnet-18-v3/checkpoints/epoch=29-step=210.ckpt"

FRDS_MODEL_PATH = FRDS_MODEL_PATH_V2_NO_ELASTIC

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

@dataclass
class ACDC:
    """
    ACDC constants.
    """
    WIDTH = 128
    class DIR:
        VAE = "vae_acdc"
        NVAE = "nvae_acdc"
        SIMCLR = "simclr_acdc"
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
