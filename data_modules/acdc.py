import os
from lightning import LightningDataModule
import subprocess
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from const import DATA_PATH, SCRIPTS_PATH

class ACDCDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str=DATA_PATH,
        batch_size: int=32,
    ):
        super().__init__()
        
        if not os.path.exists(os.path.join(data_dir, "ACDC")):
            subprocess.run(["sh", os.path.join(SCRIPTS_PATH, "download-acdc.sh")], check=True)
        
        print("Done.")
        
        import sys
        sys.exit()

        self.data_train = NotImplemented
        self.data_test = NotImplemented
    
    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)
