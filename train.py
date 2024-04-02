import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from const import SEED
from utils import load_data, setup_device, show_samples

if __name__ == "__main__":
    # Seed
    pl.seed_everything(SEED)

    # Setup device
    device = setup_device()
    print(f"Device: {device}")
    
    # Load data
    data_train, data_test = load_data()
    
    loader_train = DataLoader(data_train, batch_size=64)
    loader_test = DataLoader(data_test, batch_size=64)
    
    samples: torch.Tensor = next(iter(loader_test))[0]
    show_samples(samples)
