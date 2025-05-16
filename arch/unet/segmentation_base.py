import os
import lightning as L
from matplotlib import pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.anatomical_validity_checker import AnatomicalValidityChecker
from utils.const import MASK_CLASSES
from utils.eval import compute_dice_score, get_samples_and_reconstructions_pixel_diff
from utils.utils import discretise, show_samples

class SegmentationBase(L.LightningModule):
    """
    Adapted from UNet (unet.py) to act as a base class for segmentation models 
    imported from MONAI.
    
    TODO Also adapted from other student's code. Add reference here.
    """
    
    def __init__(
        self,
        in_channels: int=1,
        out_channels: int=4,
        optim_name: str="adam",
        lr: int=1e-3,
        weight_decay: int=0,
        model_type: str="segmentation-base",
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.model = NotImplemented
        
        # To keep track of test set during test time, to later generate figures
        self.y_buffer: list[torch.Tensor] = []
        self.y_hat_logits_buffer: list[torch.Tensor] = []
    
    def setup(self, stage: str):
        self.test_dsc = []
        self.test_av = []
    
    def configure_optimizers(self):
        Optimiser = torch.optim.Adam \
            if self.hparams.optim_name == "adam" \
            else torch.optim.AdamW
            
        return Optimiser(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
    
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
        return self.reconstruction_loss(y, y_hat_logits)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
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
        loss = self.loss(y, y_hat_logits)
        self.log("loss/val", loss)
        print(f"Val loss: {loss}")
        
        if torch.isnan(loss):
            raise ValueError("NaN loss")
        
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
        x, y, condition, ed = batch

        condition_label = f"condition_{int(condition)}"
        phase_label = "ed" if ed else "es"
        
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
        self.log(f"dsc/test_{phase_label}", dice_score)
        self.log(f"dsc/test_{condition_label}", dice_score)
        
        for i, dice_score in enumerate(dice_score_per_class):
            # i + 1 as excluding background class
            class_label = MASK_CLASSES[i + 1]
            self.log(f"dsc/test_{class_label}", dice_score)
            self.log(f"dsc/test_{phase_label}_{class_label}", dice_score)
            self.log(f"dsc/test_{condition_label}_{class_label}", dice_score)
        
        # Compute anatomical validity
        num_valid = 0
        
        for discretised_y_hat in discretise(y_hat_logits):
            AV = AnatomicalValidityChecker(discretised_y_hat)
            if AV.count_violations() == 0:
                num_valid += 1
        
        self.log("gen/anatomically_valid", num_valid / num_samples)
        self.log(f"gen/anatomically_valid_{phase_label}", num_valid / num_samples)
        self.log(f"gen/anatomically_valid_{condition_label}", num_valid / num_samples)
        
        self.y_buffer.append(y)
        self.y_hat_logits_buffer.append(y_hat_logits)
        
        self.test_dsc.append(dice_score.item())
        self.test_av.append(num_valid / num_samples)
    
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
        
        # Save individual dice and anatomical validity scores
        df = pd.DataFrame({
            "dice_score": self.test_dsc,
            "anatomical_validity": self.test_av,
        })
        
        df.to_csv("logs-zenodo/test.csv", index=False)
        print(f"Saved test.csv to {self.logger.log_dir}")

    def save_segmentations(
        self,
        data_loader: DataLoader,
        save_dir: str,
        test_data: bool=False,
    ):
        buffer_x = []
        buffer_y = []
        buffer_y_hat = []
        buffer_condition = []
        buffer_ed = []
        
        self.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                x, y, condition, ed = batch
                
                # Move to device
                x = x.to(self.device)
                y = y.to(self.device)

                if test_data:
                    # 3D data module ensures 1 batch only, but each data point
                    # is 4D of shape (S, C, W, H) where S is the number of
                    # slices.
                    x = x.squeeze(0)
                    y = y.squeeze(0)
                
                y_hat_logits = self(x)
                y_hat = torch.softmax(y_hat_logits, dim=1)
                y_hat_onehot = discretise(y_hat)
                
                buffer_x.append(x)
                buffer_y.append(y)
                buffer_y_hat.append(y_hat_onehot)
                buffer_condition.append(condition)
                buffer_ed.append(ed)
                
                print(f"Batch {batch_idx}")
        
        if not test_data:
            buffer_x = torch.cat(buffer_x, dim=0)
            buffer_y = torch.cat(buffer_y, dim=0)
            buffer_y_hat = torch.cat(buffer_y_hat, dim=0)
            buffer_condition = torch.cat(buffer_condition, dim=0)
            buffer_ed = torch.cat(buffer_ed, dim=0)
        
            assert buffer_x.shape[0] == buffer_y.shape[0] == buffer_y_hat.shape[0] == buffer_condition.shape[0] == buffer_ed.shape[0]
        
            print(f"Shape of buffer_x: {buffer_x.shape}")
            print(f"Shape of buffer_y: {buffer_y.shape}")
            print(f"Shape of buffer_y_hat: {buffer_y_hat.shape}")
            print(f"Shape of buffer_condition: {buffer_condition.shape}")
            print(f"Shape of buffer_ed: {buffer_ed.shape}")
        
        buffer = (buffer_x, buffer_y, buffer_y_hat, buffer_condition, buffer_ed)
        
        torch.save(buffer, save_dir)

        print(f"Saved segmentations to {save_dir}")
