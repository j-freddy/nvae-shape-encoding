import pytorch_lightning as pl

from const import SEED
from utils import setup_device

if __name__ == "__main__":
    # Seed
    pl.seed_everything(SEED)

    # Setup device
    device = setup_device()
    print(f"Device: {device}")
