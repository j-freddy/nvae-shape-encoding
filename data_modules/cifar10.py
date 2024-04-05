from lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from const import DATA_PATH

class CIFAR10DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str=DATA_PATH,
        batch_size: int=32,
        preprocess: bool=True,
    ):
        super().__init__()
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.preprocess = preprocess
    
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.Tensor([0.5, 0.5, 0.5]),
                std=torch.Tensor([0.5, 0.5, 0.5]),
            )
        ])

        self.data_train = datasets.CIFAR10(
            self.data_dir,
            train=True,
            download=True,
            transform=transform if self.preprocess else transforms.ToTensor(),
        )
        self.data_test = datasets.CIFAR10(
            self.data_dir,
            train=False,
            download=True,
            transform=transform if self.preprocess else transforms.ToTensor(),
        )
    
    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)
