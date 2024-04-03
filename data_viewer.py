import lightning as L
import torch

from const import SEED
from data_modules.cifar10 import CIFAR10DataModule
from utils import setup_device, show_samples

if __name__ == "__main__":
    # Seed
    L.seed_everything(SEED)

    # Setup device
    device = setup_device()
    print(f"Device: {device}")
    
    # Load data
    data_module = CIFAR10DataModule(preprocess=False)
    
    # View samples
    loader_test = data_module.test_dataloader()
    
    samples: tuple[torch.Tensor, torch.Tensor] = next(iter(loader_test))
    show_samples(*samples)
