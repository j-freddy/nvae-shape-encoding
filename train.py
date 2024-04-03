import lightning as L

from const import SEED
from data_modules.cifar10 import CIFAR10DataModule
from nvae import NVAE
from utils import setup_device

if __name__ == "__main__":
    # Seed
    L.seed_everything(SEED)

    # Setup device
    device = setup_device()
    print(f"Device: {device}")
    
    # Load data
    data_module = CIFAR10DataModule()

    # Train
    model = NVAE()
    
    # TODO Configure trainer parameters
    trainer = L.Trainer()
    trainer.fit(model, data_module)
