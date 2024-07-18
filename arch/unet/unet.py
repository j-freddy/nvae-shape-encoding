import lightning as L
from monai.losses.dice import DiceLoss
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import discretise

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x: torch.Tensor, x_res: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, x_res], dim=1)
        return self.conv(x)

class UNet(L.LightningModule):
    def __init__(
        self,
        in_channels: int=1,
        out_channels: int=4,
    ):
        super().__init__()
        
        self.contracting = nn.ModuleList([
            DoubleConv(in_channels, 64),
            Down(64, 128),
            Down(128, 256),
            Down(256, 512),
            Down(512, 1024),
        ])
        
        self.expansive = nn.ModuleList([
            Up(1024, 512),
            Up(512, 256),
            Up(256, 128),
            Up(128, 64),
        ])
        
        self.conditional_coder = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=0)
    
    def reconstruction_loss(self, y: torch.Tensor, y_hat_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute the reconstruction loss using cross-entropy.
        
        Args:
            y (torch.Tensor): One-hot encoded GT segmentations.
            y_hat_logits (torch.Tensor): Logits of output.
        
        Returns:
            recon_loss (torch.Tensor): Reconstruction loss.
        """
        batch_size = y.size(0)
        return F.cross_entropy(y_hat_logits, y, reduction="sum") / batch_size
    
    def loss(
        self,
        y: torch.Tensor,
        y_hat_logits: torch.Tensor,
    ) -> torch.Tensor:
        recon_loss = self.reconstruction_loss(y, y_hat_logits)
        return recon_loss
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        
        for layer in self.contracting:
            x = layer(x)
            skips.append(x)
        
        # Last layer is not used for residual connection
        skips = skips[:-1]
        # Reverse residual buffer
        skips = skips[::-1]
        
        for layer, skip in zip(self.expansive, skips):
            x = layer(x, skip)
        
        return self.conditional_coder(x)

    def _compute_dice(self, y: torch.Tensor, y_hat_logits: torch.Tensor) -> torch.Tensor:
        y_hat = torch.softmax(y_hat_logits, dim=1)
        y_hat_onehot = discretise(y_hat)
        dl = DiceLoss(reduction="mean", include_background=False)
        dice_score = 1 - dl(input=y_hat_onehot, target=y)
        return dice_score
    
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y, _ = batch
        
        y_hat_logits = self(x)
        
        # Compute loss
        loss = self.loss(y, y_hat_logits)
        self.log("loss/train", loss)
        print(f"Train loss: {loss}")
        
        if torch.isnan(loss):
            raise ValueError("NaN loss")
    
        dice_score = self._compute_dice(y, y_hat_logits)
        self.log("dsc/train", dice_score)
        print(f"Train DSC: {dice_score}")
        
        return loss
    
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y, _ = batch
        
        y_hat_logits = self(x)
        
        # Compute loss
        loss = self.loss(y, y_hat_logits)
        self.log("loss/val", loss)
        print(f"Val loss: {loss}")
        
        dice_score = self._compute_dice(y, y_hat_logits)
        self.log("dsc/val", dice_score)
        print(f"Val DSC: {dice_score}")
    
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y, _ = batch
        
        y_hat_logits = self(x)

        dice_score = self._compute_dice(y, y_hat_logits)
        self.log("dsc/test", dice_score)
