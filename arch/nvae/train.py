import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from const import LOGS_PATH, SEED
from data_modules.cifar10 import CIFAR10DataModule
from arch.nvae.nvae import NVAE
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
    
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=100,
        logger=TensorBoardLogger(save_dir=LOGS_PATH, name="nvae_cifar10"),
        callbacks=[
            ModelCheckpoint(monitor="val_loss", mode="min"),
            LearningRateMonitor("epoch"),
        ]
    )
    
    trainer.fit(model, data_module)
