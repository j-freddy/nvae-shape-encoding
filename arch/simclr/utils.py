import torch.nn as nn

from arch.simclr.simclr import SimCLR

def load_simclr_backbone(path: str):
    # Load pretrained SimCLR model net
    model = SimCLR.load_from_checkpoint(path).net
    # Remove projection head
    model.fc = nn.Identity()
    model.eval()
    
    return model
