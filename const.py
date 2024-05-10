from dataclasses import dataclass
import os

DATA_PATH = "data"
LOGS_PATH = "logs"
OUT_PATH = "out"
SCRIPTS_PATH = "scripts"
SEED = 1969

@dataclass
class CIFAR10:
    class DIR:
        VAE = "vae_cifar10"
        NVAE = "nvae_cifar10"

@dataclass
class ACDC:
    class DIR:
        VAE = "vae_acdc"
        NVAE = "nvae_acdc"
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
