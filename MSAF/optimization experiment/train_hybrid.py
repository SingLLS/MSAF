# -*- coding: utf-8 -*-

"""
Hybrid FT-Transformer + RaftGAT
Final Stable Version (Numerically Robust)

Run:
python train_hybrid.py \
  --dataset ./data/HFBTP.csv \
  --max_epochs 200 \
  --use_gpu
"""


import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from raft_gat import RaftGAT
from MSFTTmodel import FTTransformerMulti
from pytorch_lightning.callbacks import ModelCheckpoint


# =========================================================
# Reproducibility
# =========================================================
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
pl.seed_everything(seed, workers=True)
torch.set_float32_matmul_precision("medium")


# =========================================================
# Dataset
# =========================================================
class BlockChainDataset(Dataset):
    def __init__(self, x_norm, x_raw, topo, y_norm):
        self.x_norm = torch.from_numpy(x_norm).float()
        self.x_raw = torch.from_numpy(x_raw).float()
        self.topo = torch.from_numpy(topo).float()
        self.y = torch.from_numpy(y_norm).float()

    def __len__(self):
        return len(self.x_norm)

    def __getitem__(self, idx):
        return self.x_norm[idx], self.x_raw[idx], self.topo[idx], self.y[idx]


# =========================================================
# Hybrid Model 
# =========================================================
class HybridFTTRaftGAT(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.ftt = FTTransformerMulti(
            input_dim=num_features,
            embed_dim=64,
            num_heads=4,
            num_layers=3
        )

        self.raft = RaftGAT(
            hidden=32,
            out_dim=64,
            layers=2,
            max_orderers=9
        )
        self.gate = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(64)

        self.head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x_norm, x_raw, topo):
        z_num, _ = self.ftt(x_norm, return_feat=True)
        z_raft = self.raft(x_raw, topo)

        alpha = self.gate(x_raw) * 0.3
        z = self.norm(z_num + alpha * z_raft)

        out = self.head(z)

        # ===== SAFE latency transform  =====
        throughput = out[:, 0:1]
        latency_raw = out[:, 1:2]
        latency = torch.exp(torch.clamp(latency_raw, -5.0, 5.0))

        return torch.cat([throughput, latency], dim=1)


# =========================================================
# Multi-task Uncertainty Loss
# =========================================================
class MultiTaskUncertaintyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_sigma_t = nn.Parameter(torch.zeros(1))
        self.log_sigma_l = nn.Parameter(torch.zeros(1))
        self.mse = nn.MSELoss()
        self.min_log_sigma = -1.5

    def forward(self, pred, target, use_uncertainty=True, return_details=False):
        pred_t, pred_l = pred[:, 0], pred[:, 1]
        tgt_t, tgt_l = target[:, 0], target[:, 1]

        mse_t = self.mse(pred_t, tgt_t)
        mse_l = self.mse(
            torch.log(pred_l + 1e-6),
            torch.log(tgt_l + 1e-6)
        )

        if not use_uncertainty:
            loss = mse_t + mse_l
        else:
            log_sigma_t = torch.clamp(self.log_sigma_t, min=self.min_log_sigma)
            log_sigma_l = torch.clamp(self.log_sigma_l, min=self.min_log_sigma)

            loss = (
                torch.exp(-log_sigma_t) * mse_t +
                torch.exp(-log_sigma_l) * mse_l +
                (log_sigma_t + log_sigma_l)
            )

        if return_details:
            return {
                "loss": loss,
                "mse_throughput": mse_t.detach(),
                "mse_latency": mse_l.detach(),
            }

        return loss


# =========================================================
# Lightning Module
# =========================================================
class LitHybrid(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.loss_fn = MultiTaskUncertaintyLoss()
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x_norm, x_raw, topo, y = batch
        pred = self.model(x_norm, x_raw, topo)

        use_uncertainty = self.current_epoch >= 20
        out = self.loss_fn(
            pred, y,
            use_uncertainty=use_uncertainty,
            return_details=True
        )

        # Smooth arrival constraint
        violation = pred[:, 0] - x_raw[:, 0]
        constraint_loss = torch.mean(torch.log1p(torch.exp(violation)))
        lambda_c = min(0.5, self.current_epoch / 50 * 0.5)

        loss = out["loss"] + lambda_c * constraint_loss

        self.log("train_loss", loss, prog_bar=True)
        self.log("mse_throughput", out["mse_throughput"], prog_bar=True)
        self.log("mse_latency", out["mse_latency"], prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


# =========================================================
# Main
# =========================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="./data/HFBTP.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)

    arrival = np.log1p(df["Actual Transaction Arrival Rate"].values.astype(float))
    orderers = df["Orderers"].values.astype(float)
    block = df["Block Size"].values.astype(float)

    X_raw = np.stack([arrival, orderers, block], axis=1)
    topo = orderers.reshape(-1, 1)

    Y = df[["Throughput", "Avg Latency"]].values.astype(float)
    Y = np.nan_to_num(Y, nan=0.0)
    Y[:, 1] = np.clip(Y[:, 1],
                      np.percentile(Y[:, 1], 1),
                      np.percentile(Y[:, 1], 99))

    sx = MinMaxScaler().fit(X_raw)
    sy = MinMaxScaler().fit(Y)

    dataset = BlockChainDataset(
        sx.transform(X_raw),
        X_raw,
        topo,
        sy.transform(Y)
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    model = HybridFTTRaftGAT(num_features=3)
    lit = LitHybrid(model, args.lr)

    # Model checkpoint callback - saves best model based on train_loss
    checkpoint_callback = ModelCheckpoint(
        dirpath="modelH",
        filename="Hybrid_FTT_RaftGAT_best",
        monitor="train_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
        save_weights_only=True
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="cuda" if args.use_gpu and torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=1.0,
        deterministic=True,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(lit, loader)

    # Load the best model weights and clean up keys
    if checkpoint_callback.best_model_path:
        best_checkpoint = torch.load(checkpoint_callback.best_model_path)
        # Clean up state_dict keys
        state_dict = {}
        for key, value in best_checkpoint['state_dict'].items():
            # Remove 'model.' prefix if present
            if key.startswith('model.'):
                key = key[6:]
            # Skip loss_fn parameters
            if key.startswith('loss_fn.'):
                continue
            state_dict[key] = value
        model.load_state_dict(state_dict)

    os.makedirs("modelH", exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "scaler_X": sx, "scaler_Y": sy},
        "modelH/Hybrid_FTT_RaftGAT.pth"
    )

    print("✔ Training finished (autograd-safe).")


if __name__ == "__main__":
    main()