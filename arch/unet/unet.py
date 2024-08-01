import lightning as L
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.anatomical_validity_checker import AnatomicalValidityChecker
from utils.const import ACDC
from utils.eval import compute_dice_score, get_samples_and_reconstructions_pixel_diff
from utils.utils import discretise, show_samples

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
    """
    TODO Write Docstring
    
    In this class, x denotes the scans and y denotes the segmentation masks.
    """
    def __init__(
        self,
        in_channels: int=1,
        out_channels: int=4,
        loss_reg: str="cross_entropy",
        alpha: float=1.0,
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.contracting = nn.ModuleList([
            DoubleConv(self.hparams.in_channels, 64),
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
        
        self.conditional_coder = nn.Conv2d(64, self.hparams.out_channels, kernel_size=1)
        
        # To keep track of test set during test time, to later generate figures
        self.y_buffer: list[torch.Tensor] = []
        self.y_hat_logits_buffer: list[torch.Tensor] = []
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
    
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
        log_components: bool=True,
    ) -> torch.Tensor:
        return self.reconstruction_loss(y, y_hat_logits)
    
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
    
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y, _, _ = batch
        
        y_hat_logits = self(x)
        
        # Compute loss
        loss = self.loss(y, y_hat_logits)
        self.log("loss/train", loss)
        print(f"Train loss: {loss}")
        
        if torch.isnan(loss):
            raise ValueError("NaN loss")
    
        # Compute Dice score
        y_hat = torch.softmax(y_hat_logits, dim=1)
        y_hat_onehot = discretise(y_hat)

        dice_score: torch.Tensor = compute_dice_score(y, y_hat_onehot, self.device)
    
        self.log("dsc/train", dice_score)
        print(f"Train DSC: {dice_score}")
        
        return loss
    
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        x, y, _, _ = batch
        
        y_hat_logits = self(x)
        
        # Compute loss
        loss = self.loss(y, y_hat_logits, log_components=False)
        self.log("loss/val", loss)
        print(f"Val loss: {loss}")
        
        # Compute Dice score
        y_hat = torch.softmax(y_hat_logits, dim=1)
        y_hat_onehot = discretise(y_hat)

        dice_score: torch.Tensor = compute_dice_score(y, y_hat_onehot, self.device)

        self.log("dsc/val", dice_score)
        print(f"Val DSC: {dice_score}")
    
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        """
        Testing uses ACDC3DDataModule instead of ACDCDataModule to compute 3D
        Dice scores.
        """
        x, y, _, _ = batch
        
        # 3D data module ensures 1 batch only, but each data point is 4D of
        # shape (S, C, W, H) where S is the number of slices.
        x = x.squeeze(0)
        y = y.squeeze(0)
        
        num_samples, _, _, _ = x.shape
        
        y_hat_logits = self(x)

        # Compute 3D Dice score
        y_hat = torch.softmax(y_hat_logits, dim=1)
        y_hat_onehot = discretise(y_hat)

        dice_score, dice_score_per_class = compute_dice_score(
            y,
            y_hat_onehot,
            self.device,
            is_3d=True,
            dice_per_class=True,
        )

        self.log("dsc/test", dice_score)
        
        for i, dice_score in enumerate(dice_score_per_class):
            # i + 1 as excluding background class
            class_label = ACDC.mask_classes[i + 1]
            self.log(f"dsc/test_{class_label}", dice_score)
        
        # Compute anatomical validity
        num_valid = 0
        
        for discretised_y_hat in discretise(y_hat_logits):
            AV = AnatomicalValidityChecker(discretised_y_hat)
            if AV.count_violations() == 0:
                num_valid += 1
        
        self.log("gen/anatomically_valid", num_valid / num_samples)
        
        self.y_buffer.append(y)
        self.y_hat_logits_buffer.append(y_hat_logits)
    
    def log_reconstruction_visualisation(
        self,
        y: torch.Tensor,
        y_hat_logits: torch.Tensor,
    ):
        num_data = y.shape[0]
        samples_idx = torch.randperm(num_data)[:40]
        y = y[samples_idx]
        y_hat_logits = y_hat_logits[samples_idx]
        
        samples, reconstruction_pixel_error = get_samples_and_reconstructions_pixel_diff(y, y_hat_logits)
        show_samples(samples, reconstruction_pixel_error, rgb=False, ncol=10, figsize=(10, 4), display=False)
        self.logger.experiment.add_figure("img/reconstructions", plt.gcf())
    
    def on_test_end(self):
        y = torch.cat(self.y_buffer, dim=0)
        y_hat_logits = torch.cat(self.y_hat_logits_buffer, dim=0)
        
        # Visualise samples and reconstructions
        self.log_reconstruction_visualisation(y, y_hat_logits)
