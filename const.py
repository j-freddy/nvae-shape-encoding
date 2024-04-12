from dataclasses import dataclass
import os


DATA_PATH = "data"
LOGS_PATH = "logs"
SCRIPTS_PATH = "scripts"
SEED = 1969

@dataclass
class ACDC:
    @dataclass
    class RAW:
        TRAIN_PATH = os.path.join(DATA_PATH, "ACDC", "database", "training")
        TEST_PATH = os.path.join(DATA_PATH, "ACDC", "database", "testing")
    
    TRAIN_PATH = os.path.join(DATA_PATH, "acdc_processed_train.pt")
    TEST_PATH = os.path.join(DATA_PATH, "acdc_processed_test.pt")
