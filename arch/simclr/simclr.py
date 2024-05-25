import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

class SimCLR(L.LightningModule):
    """
    A Simple Framework for Contrastive Learning of Visual Representations
    (SimCLR)[1] is a self-supervised contrastive learning framework that uses a
    convolutional neural network (ResNet) to learn representations of images. It
    achieves this by learning to recognise similarities between pairs of
    augmented data points stemming from the same original image (positive pairs)
    and dissimilarities between all other pairs (negative pairs).
    
    This class is used to learn representations of the ACDC dataset, which is
    then used to evaluate the quality of synthetic cardiac segmentation maps. It
    is meant to be an improvement over the FID metric, which does not have good
    performance for evaluating segmentation maps.
    
    Code adapted from:
    - https://github.com/j-freddy/simclr-medical-imaging
    
    [1]: Chen T, Kornblith S, Norouzi M, Hinton G. A simple framework for
    contrastive learning of visual representations. InInternational conference
    on machine learning 2020 Nov 21 (pp. 1597-1607). PMLR.
    """
    
    def __init__(self, latent_dim: int=512, max_epochs: int=100):
        super().__init__()
        
        self.save_hyperparameters()
        
        # InfoNCE temperature
        self.temperature = 0.07
        
        self.net = torchvision.models.resnet18(
            weights=None,
            num_classes=self.hparams.latent_dim,
        )

        self.net.fc = nn.Sequential(
            self.net.fc,
            # Attach projection head
            nn.ReLU(inplace=True),
            nn.Linear(self.hparams.latent_dim, self.hparams.latent_dim // 4),
        )
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=5e-4,
            weight_decay=1e-4,
        )
        
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=1e-5,
        )
        
        return [optimizer], [lr_scheduler]

    def loss(self, z: torch.Tensor, log_rank_metrics: bool=False) -> torch.Tensor:
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(z[:, None, :], z[None, :, :], dim=-1)

        # Mask out cosine similarity to itself
        self_mask = torch.eye(
            cos_sim.shape[0],
            dtype=torch.bool,
            device=cos_sim.device
        )
        cos_sim.masked_fill_(self_mask, -9e15)

        # Find positive example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)

        cos_sim /= self.temperature

        # InfoNCE loss
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        loss = nll.mean()
        
        if log_rank_metrics:
            # Get ranking position of positive example
            comb_sim = torch.cat(
                [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],
                # First position positive example
                dim=-1,
            )
            
            sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)

            # Logging ranking metrics
            self.log("acc_top1", (sim_argsort == 0).float().mean())
            self.log("acc_top5", (sim_argsort < 5).float().mean())
            self.log("acc_mean_pos", 1 + sim_argsort.float().mean())
        
        return loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch: list[torch.Tensor]) -> torch.Tensor:
        batch = torch.cat(batch, dim=0)

        z = self(batch)
        
        # Compute loss
        loss = self.loss(z, log_rank_metrics=True)
        self.log("train_loss", loss)

        print(f"Train loss: {loss}")

        if torch.isnan(loss):
            raise ValueError("NaN loss")

    def validation_step(self, batch: list[torch.Tensor]) -> torch.Tensor:
        batch = torch.cat(batch, dim=0)

        z = self(batch)
        
        # Compute loss
        loss = self.loss(z, log_rank_metrics=True)
        self.log("val_loss", loss)

        print(f"Val loss: {loss}")
