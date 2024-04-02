import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.utils import make_grid

from const import DATA_PATH, SEED

def setup_device():
    """
    Set up Torch device and set seed. Enforce all operations to be
    deterministic.

    Returns:
        torch.device: Device used for Torch scripts.
    """
    # Use GPU if available
    device = torch.device("cuda") if torch.cuda.is_available()\
        else torch.device("cpu")

    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        # Enforce all operations to be deterministic on GPU for reproducibility
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    return device

def load_data() -> tuple[Dataset, Dataset]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.Tensor([0.5, 0.5, 0.5]),
            std=torch.Tensor([0.5, 0.5, 0.5]),
        )
    ])

    data_train = datasets.CIFAR10(
        DATA_PATH,
        train=True,
        download=True,
        transform=transform,
    )
    data_test = datasets.CIFAR10(
        DATA_PATH,
        train=False,
        download=True,
        transform=transform,
    )
    
    return data_train, data_test

def show_samples(samples: torch.Tensor):
    # TODO Denormalise samples
    samples = make_grid(samples, nrow=8, padding=2)
    
    plt.figure(figsize = (6,6))
    plt.axis("off")
    plt.imshow(np.transpose(samples.cpu().numpy(), (1, 2, 0)))
    plt.show()
