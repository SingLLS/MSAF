# -*- coding: utf-8 -*-

"""
Generative Diffusion Model (GDM) for Blockchain Performance Prediction
Based on BGI Framework (Fu et al., IEEE Wireless Communications 2026)

Run:
python train_gdm.py --dataset ./data/HFBTP.csv --stop_metric both --use_gpu
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pytorch_lightning.callbacks import EarlyStopping


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
# Sinusoidal Time Embedding 
# =========================================================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        """
        Args:
            t: timestep tensor [B, 1] normalized to [0, 1]
        Returns:
            embedding: [B, dim]
        """
        device = t.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


# =========================================================
# Condition Encoding Network (encodes blockchain features)
# =========================================================
class ConditionEncoder(nn.Module):
    """Encodes blockchain configuration parameters as conditioning"""
    def __init__(self, input_dim=3, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.encoder(x)


# =========================================================
# Noise Prediction Network 
# =========================================================
class DenoisingNetwork(nn.Module):
    """
    U-Net style denoising network that predicts noise at each timestep.
    Uses conditioning from blockchain features.
    """
    def __init__(self, output_dim=2, cond_dim=64, time_embed_dim=64, hidden_dim=128):
        super().__init__()
        
        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        
        # Condition projection
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Main denoising network
        # Input: noisy_y [B, output_dim] + time_embed [B, time_embed_dim] + condition [B, hidden_dim]
        input_dim = output_dim + time_embed_dim + hidden_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, noisy_y, t, condition):
        """
        Args:
            noisy_y: noisy target values [B, output_dim]
            t: timestep [B, 1] normalized to [0, 1]
            condition: encoded blockchain features [B, cond_dim]
        Returns:
            predicted_noise: [B, output_dim]
        """
        t_emb = self.time_embed(t)  # [B, time_embed_dim]
        cond_emb = self.cond_proj(condition)  # [B, hidden_dim]
        
        # Concatenate all inputs
        h = torch.cat([noisy_y, t_emb, cond_emb], dim=1)
        
        return self.net(h)


# =========================================================
# GDM Model 
# =========================================================
class BlockchainGDM(nn.Module):
    """
    Generative Diffusion Model for Blockchain Performance Prediction.
    
    Based on the BGI framework, this model:
    1. Encodes blockchain configuration parameters as conditioning
    2. Uses a denoising network to learn the reverse diffusion process
    3. Can predict performance metrics by denoising from noise or direct encoding
    """
    def __init__(self, input_dim=3, output_dim=2, num_timesteps=100,
                 beta_start=1e-4, beta_end=0.02, cond_dim=64):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_timesteps = num_timesteps
        
        # Noise schedule (using register_buffer to auto-handle device)
        beta = torch.linspace(beta_start, beta_end, num_timesteps)
        self.register_buffer('beta', beta)
        self.register_buffer('alpha', 1 - beta)
        self.register_buffer('alpha_bar', torch.cumprod(self.alpha, dim=0))
        
        # Condition encoder
        self.cond_encoder = ConditionEncoder(input_dim, cond_dim)
        
        # Denoising network
        self.denoiser = DenoisingNetwork(
            output_dim=output_dim,
            cond_dim=cond_dim,
            time_embed_dim=64,
            hidden_dim=128
        )
        
        # Direct prediction head 
        self.direct_predictor = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x_norm, x_raw, topo):
        """Training forward pass - direct prediction with diffusion regularization"""
        batch_size = x_norm.shape[0]
        device = x_norm.device
        
        # Encode condition
        condition = self.cond_encoder(x_norm)
        
        # Direct prediction
        pred = self.direct_predictor(condition)
        
        # Add diffusion noise to prediction and try to denoise
        t = torch.randint(0, self.num_timesteps, (batch_size, 1), device=device)
        t_normalized = t.float() / self.num_timesteps
        
        # Get alpha_bar for selected timesteps
        alpha_bar_t = self.alpha_bar[t.squeeze().long()]
        alpha_bar_t = alpha_bar_t.unsqueeze(1)
        
        # Add noise to prediction
        noise = torch.randn_like(pred)
        noisy_pred = torch.sqrt(alpha_bar_t) * pred + torch.sqrt(1 - alpha_bar_t) * noise
        
        # Predict the noise
        predicted_noise = self.denoiser(noisy_pred, t_normalized, condition)
        
        # Output predictions
        throughput = torch.sigmoid(pred[:, 0:1])
        latency = torch.nn.functional.softplus(pred[:, 1:2])
        
        return torch.cat([throughput, latency], dim=1)
    
    def predict(self, x_norm, x_raw, topo):
        """Fast prediction without diffusion sampling"""
        condition = self.cond_encoder(x_norm)
        pred = self.direct_predictor(condition)
        
        throughput = torch.sigmoid(pred[:, 0:1])
        latency = torch.nn.functional.softplus(pred[:, 1:2])
        
        return torch.cat([throughput, latency], dim=1)


# =========================================================
# Loss 
# =========================================================
class MultiTaskUncertaintyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_sigma_t = nn.Parameter(torch.zeros(1))
        self.log_sigma_l = nn.Parameter(torch.zeros(1))
        self.mse = nn.MSELoss()
        self.min_log_sigma = -1.5

    def forward(self, pred, target):
        pred_t, pred_l = pred[:, 0], pred[:, 1]
        tgt_t, tgt_l = target[:, 0], target[:, 1]

        mse_t = self.mse(pred_t, tgt_t)
        mse_l = self.mse(pred_l, tgt_l)

        log_sigma_t = torch.clamp(self.log_sigma_t, min=self.min_log_sigma)
        log_sigma_l = torch.clamp(self.log_sigma_l, min=self.min_log_sigma)

        loss = (
            torch.exp(-log_sigma_t) * mse_t +
            torch.exp(-log_sigma_l) * mse_l +
            (log_sigma_t + log_sigma_l)
        )

        return loss


# =========================================================
# Lightning Module
# =========================================================
class LitGDM(pl.LightningModule):
    def __init__(self, model, lr, scaler_y, patience=25, stop_metric="both"):
        super().__init__()
        self.model = model
        self.loss_fn = MultiTaskUncertaintyLoss()
        self.lr = lr
        self.scaler_y = scaler_y
        self.patience = patience
        self.stop_metric = stop_metric

        self.val_preds = []
        self.val_targets = []
        
        # Best model tracking
        self.best_val_rmse_t = float('inf')
        self.best_epoch_t = -1
        self.best_model_state_t = None
        self.best_metrics_t = None
        
        self.best_val_rmse_l = float('inf')
        self.best_epoch_l = -1
        self.best_model_state_l = None
        self.best_metrics_l = None
        
        self.patience_counter_t = 0
        self.patience_counter_l = 0

    def training_step(self, batch, batch_idx):
        x_norm, x_raw, topo, y = batch
        pred = self.model(x_norm, x_raw, topo)
        loss = self.loss_fn(pred, y)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_norm, x_raw, topo, y = batch
        pred = self.model.predict(x_norm, x_raw, topo)

        self.val_preds.append(pred.detach().cpu())
        self.val_targets.append(y.detach().cpu())

    def on_validation_epoch_end(self):
        if not self.val_preds:
            return
            
        preds = torch.cat(self.val_preds).numpy()
        targets = torch.cat(self.val_targets).numpy()

        preds = self.scaler_y.inverse_transform(preds)
        targets = self.scaler_y.inverse_transform(targets)

        # Throughput metrics
        mae_t = mean_absolute_error(targets[:, 0], preds[:, 0])
        rmse_t = np.sqrt(mean_squared_error(targets[:, 0], preds[:, 0]))
        r2_t = r2_score(targets[:, 0], preds[:, 0])
        mape_t = np.mean(np.abs((targets[:, 0] - preds[:, 0]) / (targets[:, 0] + 1e-8))) * 100

        # Latency metrics
        mae_l = mean_absolute_error(targets[:, 1], preds[:, 1])
        rmse_l = np.sqrt(mean_squared_error(targets[:, 1], preds[:, 1]))
        r2_l = r2_score(targets[:, 1], preds[:, 1])
        mape_l = np.mean(np.abs((targets[:, 1] - preds[:, 1]) / (targets[:, 1] + 1e-8))) * 100

        self.log("val_rmse_t", rmse_t, prog_bar=True)
        self.log("val_rmse_l", rmse_l, prog_bar=False)
        self.log("val_mae_t", mae_t, prog_bar=False)
        self.log("val_mape_t", mape_t, prog_bar=False)
        self.log("val_r2_t", r2_t, prog_bar=False)
        self.log("val_mae_l", mae_l, prog_bar=False)
        self.log("val_mape_l", mape_l, prog_bar=False)
        self.log("val_r2_l", r2_l, prog_bar=False)

        # Track best model for throughput
        if rmse_t < self.best_val_rmse_t - 1e-6:
            self.best_val_rmse_t = rmse_t
            self.best_epoch_t = self.current_epoch
            self.best_model_state_t = {k: v.clone() for k, v in self.model.state_dict().items()}
            self.best_metrics_t = {
                "RMSE_T": rmse_t, "MAE_T": mae_t, "MAPE_T": mape_t, "R2_T": r2_t,
                "epoch": self.current_epoch  
            }
            self.patience_counter_t = 0
            print(f"✓ Best throughput model at epoch {self.current_epoch} (RMSE_t: {rmse_t:.4f})")
        else:
            self.patience_counter_t += 1

        # Track best model for latency
        if rmse_l < self.best_val_rmse_l - 1e-6:
            self.best_val_rmse_l = rmse_l
            self.best_epoch_l = self.current_epoch
            self.best_model_state_l = {k: v.clone() for k, v in self.model.state_dict().items()}
            self.best_metrics_l = {
                "RMSE_L": rmse_l, "MAE_L": mae_l, "MAPE_L": mape_l, "R2_L": r2_l,
                "epoch": self.current_epoch  
            }
            self.patience_counter_l = 0
            print(f"✓ Best latency model at epoch {self.current_epoch} (RMSE_l: {rmse_l:.4f})")
        else:
            self.patience_counter_l += 1

        self.val_preds.clear()
        self.val_targets.clear()
    
    def should_stop(self):
        if self.stop_metric == "throughput":
            return self.patience_counter_t >= self.patience and self.current_epoch > 20
        elif self.stop_metric == "latency":
            return self.patience_counter_l >= self.patience and self.current_epoch > 20
        else:  # both
            return (self.patience_counter_t >= self.patience and 
                    self.patience_counter_l >= self.patience and 
                    self.current_epoch > 20)
    
    def get_best_model_t(self):
        if self.best_model_state_t is not None:
            self.model.load_state_dict(self.best_model_state_t)
        return self.model
    
    def get_best_model_l(self):
        if self.best_model_state_l is not None:
            self.model.load_state_dict(self.best_model_state_l)
        return self.model
    
    def get_best_metrics_t(self):
        return self.best_metrics_t
    
    def get_best_metrics_l(self):
        return self.best_metrics_l

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return [optimizer], [scheduler]


# =========================================================
# Custom Early Stopping
# =========================================================
class DualEarlyStopping(EarlyStopping):
    def __init__(self, monitor_t="val_rmse_t", monitor_l="val_rmse_l", 
                 mode="min", patience=25, verbose=True, stop_metric="both"):
        super().__init__(monitor=monitor_t, mode=mode, patience=patience, verbose=verbose)
        self.stop_metric = stop_metric
        
    def _should_stop(self, trainer):
        return trainer.lightning_module.should_stop()


# =========================================================
# Train one fold
# =========================================================
def run_fold(train_idx, val_idx, dataset, scaler_y, args, fold_id):
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = BlockchainGDM(
        input_dim=3,
        output_dim=2,
        num_timesteps=args.num_timesteps,
        cond_dim=64
    )
    
    lit = LitGDM(
        model, args.lr, scaler_y,
        patience=args.patience,
        stop_metric=args.stop_metric
    )

    if args.stop_metric == "both":
        early_stop = DualEarlyStopping(
            monitor_t="val_rmse_t", monitor_l="val_rmse_l",
            mode="min", patience=args.patience, verbose=True, stop_metric=args.stop_metric
        )
    else:
        monitor_metric = "val_rmse_t" if args.stop_metric == "throughput" else "val_rmse_l"
        early_stop = EarlyStopping(monitor=monitor_metric, mode="min", patience=args.patience, verbose=True)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="cuda" if args.use_gpu and torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[early_stop],
        enable_checkpointing=False,
        deterministic=True
    )

    trainer.fit(lit, train_loader, val_loader)
    
    os.makedirs("best_models_gdm", exist_ok=True)
    
    best_model_t = lit.get_best_model_t()
    best_model_l = lit.get_best_model_l()
    best_metrics_t = lit.get_best_metrics_t()
    best_metrics_l = lit.get_best_metrics_l()
    
    if best_metrics_t is not None:
        torch.save(best_model_t.state_dict(), f"best_models_gdm/best_model_t_fold{fold_id}.pth")
    if best_metrics_l is not None:
        torch.save(best_model_l.state_dict(), f"best_models_gdm/best_model_l_fold{fold_id}.pth")
    
    # 安全地获取 epoch 值
    epoch_t = best_metrics_t.get("epoch", -1) if best_metrics_t else -1
    epoch_l = best_metrics_l.get("epoch", -1) if best_metrics_l else -1
    
    combined_metrics = {
        "RMSE_T": best_metrics_t["RMSE_T"] if best_metrics_t else float('nan'),
        "MAE_T": best_metrics_t["MAE_T"] if best_metrics_t else float('nan'),
        "MAPE_T": best_metrics_t["MAPE_T"] if best_metrics_t else float('nan'),
        "R2_T": best_metrics_t["R2_T"] if best_metrics_t else float('nan'),
        "RMSE_L": best_metrics_l["RMSE_L"] if best_metrics_l else float('nan'),
        "MAE_L": best_metrics_l["MAE_L"] if best_metrics_l else float('nan'),
        "MAPE_L": best_metrics_l["MAPE_L"] if best_metrics_l else float('nan'),
        "R2_L": best_metrics_l["R2_L"] if best_metrics_l else float('nan'),
        "best_epoch_t": epoch_t,
        "best_epoch_l": epoch_l,
    }
    
    return combined_metrics


# =========================================================
# Main
# =========================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="./data/HFBTP.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num_timesteps", type=int, default=100,
                        help="Number of diffusion timesteps")
    parser.add_argument("--stop_metric", type=str, default="both",
                        choices=["throughput", "latency", "both"])
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    print("="*60)
    print("GENERATIVE DIFFUSION MODEL (GDM) for Blockchain")
    print("Based on BGI Framework")
    print("="*60)
    print(f"max_epochs={args.max_epochs}, patience={args.patience}, lr={args.lr}")
    print(f"num_timesteps={args.num_timesteps}, stop_metric={args.stop_metric}")
    print("="*60)

    df = pd.read_csv(args.dataset)

    arrival = np.log1p(df["Actual Transaction Arrival Rate"].values.astype(float))
    orderers = df["Orderers"].values.astype(float)
    block = df["Block Size"].values.astype(float)

    X_raw = np.stack([arrival, orderers, block], axis=1)
    topo = orderers.reshape(-1, 1)

    Y = df[["Throughput", "Avg Latency"]].values.astype(float)
    Y = np.nan_to_num(Y, nan=0.0)

    sx = MinMaxScaler().fit(X_raw)
    sy = MinMaxScaler().fit(Y)

    dataset = BlockChainDataset(
        sx.transform(X_raw),
        X_raw,
        topo,
        sy.transform(Y)
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n{'='*40}")
        print(f"Fold {fold+1}/5")
        print(f"{'='*40}")
        print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")

        metrics = run_fold(train_idx, val_idx, dataset, sy, args, fold)
        results.append(metrics)

        print(f"\nFold {fold+1} Results:")
        print(f"  Throughput (epoch {metrics['best_epoch_t']}): RMSE={metrics['RMSE_T']:.4f}, MAE={metrics['MAE_T']:.4f}, MAPE={metrics['MAPE_T']:.2f}%, R2={metrics['R2_T']:.4f}")
        print(f"  Latency (epoch {metrics['best_epoch_l']}):     RMSE={metrics['RMSE_L']:.4f}, MAE={metrics['MAE_L']:.4f}, MAPE={metrics['MAPE_L']:.2f}%, R2={metrics['R2_L']:.4f}")

    df_res = pd.DataFrame(results)
    df_res_stats = df_res.drop(columns=['best_epoch_t', 'best_epoch_l'])

    print("\n" + "="*60)
    print("GDM - 5-FOLD CROSS VALIDATION RESULTS")
    print("="*60)
    print("\n--- Mean ± Std ---")
    for col in df_res_stats.columns:
        mean_val = df_res_stats[col].mean()
        std_val = df_res_stats[col].std()
        print(f"{col}: {mean_val:.4f} ± {std_val:.4f}")

    os.makedirs("results_gdm", exist_ok=True)
    df_res.to_csv("results_gdm/gdm_5fold_results.csv", index=False)

    print("\n✓ Results saved to results_gdm/")
    print("✓ Best models saved to best_models_gdm/")


if __name__ == "__main__":
    main()