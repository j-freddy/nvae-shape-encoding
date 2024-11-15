from dataclasses import dataclass
from enum import Enum
import os

DATA_PATH = "data"
LOGS_PATH = "logs"
OUT_PATH = "out"
SCRIPTS_PATH = "scripts"
SEED = 1969

# See Quick Start section in README.md for instructions on how to download these
# models. [Most] scripts will not work without these 2 models installed. FRDS
# model is used to compute FRDS while NVAE model is used to compute shape loss
# in ACU-Net.
FRDS_MODEL_PATH = "logs/simclr_acdc/frds-resnet-18/checkpoints/epoch=143-step=1008.ckpt"
NVAE_MODEL_PATH = "logs/nvae_acdc/default/checkpoints/epoch=97-step=20972.ckpt"
NVAE_LS_MODEL_PATH = "logs/nvae_acdc/latent-skip/checkpoints/epoch=98-step=21186.ckpt"

CARDIAC_WIDTH = 128
MASK_NUM_CLASSES = 4
MASK_CLASSES = ["BG", "RV", "MYO", "LV"]

class MaskClassLabel(Enum):
    BG = 0
    RV = 1
    MYO = 2
    LV = 3
@dataclass
class ACDC:
    """
    ACDC constants.
    """

    conditions = ["NOR", "MINF", "DCM", "HCM", "RV"]
    condition_to_idx = {condition: i for i, condition in enumerate(conditions)}

    @dataclass
    class DIR:
        VAE = "vae_acdc"
        NVAE = "nvae_acdc"
        NVAESEG = "nvae_seg_acdc"
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

@dataclass
class MnMs:
    """
    MnMs constants.
    """

    conditions = ["DCM", "HCM", "HHD", "NOR", "Other", "ARV", "AHS", "IHD", "LVNC"]
    condition_to_idx = {condition: i for i, condition in enumerate(conditions)}
    
    vendors = ["A", "B", "C", "D"]
    centres = [1, 2, 3, 4, 5]

    @dataclass
    class DIR:
        UNET = "unet_mnms"

    @dataclass
    class RAW:
        INFO_FILE = os.path.join(DATA_PATH, "MnMs", "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv")
        TRAIN_PATH_LABELLED = os.path.join(DATA_PATH, "MnMs", "Training", "Labeled")
        TRAIN_PATH_UNLABELLED = os.path.join(DATA_PATH, "MnMs", "Training", "Unlabeled")
        VAL_PATH = os.path.join(DATA_PATH, "MnMs", "Validation")
        TEST_PATH = os.path.join(DATA_PATH, "MnMs", "Testing")
    
    TRAIN_PATH = os.path.join(DATA_PATH, "mnms_processed_train.pt")
    VAL_PATH = os.path.join(DATA_PATH, "mnms_processed_val.pt")
    TEST_PATH = os.path.join(DATA_PATH, "mnms_processed_test.pt")
