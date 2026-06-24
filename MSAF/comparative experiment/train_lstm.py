# -*- coding: utf-8 -*-

"""
LSTM Baseline for Blockchain Performance Prediction
5-Fold Cross Validation (Dual Best Model Tracking)

LSTM (Long Short-Term Memory) is used as a baseline model to demonstrate
the superiority of the proposed FTT+GAT architecture.

Metrics:
- Throughput: MAE / RMSE / MAPE / R2
- Latency:    MAE / RMSE / MAPE / R2

Run:
python train_lstm.py --dataset ./data/HFBTP.csv --stop_metric both --use_gpu
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
# LSTM Model
# =========================================================
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        
    def forward(self, x_norm, x_raw, topo):
        # x_norm: [B, F] -> [B, 1, F] for LSTM sequence
        # We treat each sample as a sequence of length 1
        x = x_norm.unsqueeze(1)  # [B, 1, input_dim]
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: [B, seq_len, hidden_dim]
        
        # Take the last time step output
        out = out[:, -1, :]  # [B, hidden_dim]
        out = self.dropout(out)
        
        # Output head
        out = self.head(out)
        
        # Output activations
        throughput = torch.sigmoid(out[:, 0:1])
        latency = torch.nn.functional.softplus(out[:, 1:2])
        
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
# Lightning Module (Dual Best Model Tracking)
# =========================================================
class LitLSTM(pl.LightningModule):
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
        
        # Best model tracking for throughput
        self.best_val_rmse_t = float('inf')
        self.best_epoch_t = -1
        self.best_model_state_t = None
        self.best_metrics_t = None
        
        # Best model tracking for latency
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
        pred = self.model(x_norm, x_raw, topo)

        self.val_preds.append(pred.detach().cpu())
        self.val_targets.append(y.detach().cpu())

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_preds).numpy()
        targets = torch.cat(self.val_targets).numpy()

        preds = self.scaler_y.inverse_transform(preds)
        targets = self.scaler_y.inverse_transform(targets)

        y_p = preds
        y_t = targets

        # Throughput metrics
        yt_p = y_p[:, 0]
        yt_t = y_t[:, 0]

        mae_t = mean_absolute_error(yt_t, yt_p)
        rmse_t = np.sqrt(mean_squared_error(yt_t, yt_p))
        r2_t = r2_score(yt_t, yt_p)
        mape_t = np.mean(np.abs((yt_t - yt_p) / (yt_t + 1e-8))) * 100

        # Latency metrics
        yl_p = y_p[:, 1]
        yl_t = y_t[:, 1]

        mae_l = mean_absolute_error(yl_t, yl_p)
        rmse_l = np.sqrt(mean_squared_error(yl_t, yl_p))
        r2_l = r2_score(yl_t, yl_p)
        mape_l = np.mean(np.abs((yl_t - yl_p) / (yl_t + 1e-8))) * 100

        # Log current epoch metrics
        self.log("val_mae_t", mae_t, prog_bar=False)
        self.log("val_rmse_t", rmse_t, prog_bar=True)
        self.log("val_mape_t", mape_t, prog_bar=False)
        self.log("val_r2_t", r2_t, prog_bar=False)

        self.log("val_mae_l", mae_l, prog_bar=False)
        self.log("val_rmse_l", rmse_l, prog_bar=False)
        self.log("val_mape_l", mape_l, prog_bar=False)
        self.log("val_r2_l", r2_l, prog_bar=False)

        # ========== Check best for Throughput ==========
        if rmse_t < self.best_val_rmse_t - 1e-6:
            self.best_val_rmse_t = rmse_t
            self.best_epoch_t = self.current_epoch
            self.best_model_state_t = {
                k: v.clone() for k, v in self.model.state_dict().items()
            }
            self.best_metrics_t = {
                "RMSE_T": rmse_t,
                "MAE_T": mae_t,
                "MAPE_T": mape_t,
                "R2_T": r2_t,
                "epoch": self.current_epoch
            }
            self.patience_counter_t = 0
            print(f"✓ Best throughput model at epoch {self.current_epoch} (RMSE_t: {rmse_t:.4f})")
        else:
            self.patience_counter_t += 1

        # ========== Check best for Latency ==========
        if rmse_l < self.best_val_rmse_l - 1e-6:
            self.best_val_rmse_l = rmse_l
            self.best_epoch_l = self.current_epoch
            self.best_model_state_l = {
                k: v.clone() for k, v in self.model.state_dict().items()
            }
            self.best_metrics_l = {
                "RMSE_L": rmse_l,
                "MAE_L": mae_l,
                "MAPE_L": mape_l,
                "R2_L": r2_l,
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
# Custom Early Stopping Callback
# =========================================================
class DualEarlyStopping(EarlyStopping):
    def __init__(self, monitor_t="val_rmse_t", monitor_l="val_rmse_l", 
                 mode="min", patience=25, verbose=True, stop_metric="both"):
        super().__init__(monitor=monitor_t, mode=mode, patience=patience, verbose=verbose)
        self.stop_metric = stop_metric
        
    def _should_stop(self, trainer):
        lit_module = trainer.lightning_module
        return lit_module.should_stop()


# =========================================================
# Train one fold
# =========================================================
def run_fold(train_idx, val_idx, dataset, scaler_y, args, fold_id):

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = LSTMPredictor(
        input_dim=3,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    lit = LitLSTM(
        model, 
        args.lr, 
        scaler_y,
        patience=args.patience,
        stop_metric=args.stop_metric
    )

    # Choose early stopping
    if args.stop_metric == "both":
        early_stop = DualEarlyStopping(
            monitor_t="val_rmse_t",
            monitor_l="val_rmse_l",
            mode="min",
            patience=args.patience,
            verbose=True,
            stop_metric=args.stop_metric
        )
    else:
        monitor_metric = "val_rmse_t" if args.stop_metric == "throughput" else "val_rmse_l"
        early_stop = EarlyStopping(
            monitor=monitor_metric,
            mode="min",
            patience=args.patience,
            verbose=True
        )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="cuda" if args.use_gpu and torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[early_stop],
        enable_checkpointing=False,
        deterministic=True
    )

    trainer.fit(lit, train_loader, val_loader)
    
    # Save best models
    os.makedirs("best_models_lstm", exist_ok=True)
    
    best_model_t = lit.get_best_model_t()
    best_model_l = lit.get_best_model_l()
    best_metrics_t = lit.get_best_metrics_t()
    best_metrics_l = lit.get_best_metrics_l()
    
    if best_metrics_t is not None:
        torch.save(best_model_t.state_dict(), f"best_models_lstm/best_lstm_t_fold{fold_id}.pth")
    if best_metrics_l is not None:
        torch.save(best_model_l.state_dict(), f"best_models_lstm/best_lstm_l_fold{fold_id}.pth")
    
    # Combine metrics
    combined_metrics = {
        "RMSE_T": best_metrics_t["RMSE_T"] if best_metrics_t else float('nan'),
        "MAE_T": best_metrics_t["MAE_T"] if best_metrics_t else float('nan'),
        "MAPE_T": best_metrics_t["MAPE_T"] if best_metrics_t else float('nan'),
        "R2_T": best_metrics_t["R2_T"] if best_metrics_t else float('nan'),
        "RMSE_L": best_metrics_l["RMSE_L"] if best_metrics_l else float('nan'),
        "MAE_L": best_metrics_l["MAE_L"] if best_metrics_l else float('nan'),
        "MAPE_L": best_metrics_l["MAPE_L"] if best_metrics_l else float('nan'),
        "R2_L": best_metrics_l["R2_L"] if best_metrics_l else float('nan'),
        "best_epoch_t": best_metrics_t["epoch"] if best_metrics_t else -1,
        "best_epoch_l": best_metrics_l["epoch"] if best_metrics_l else -1,
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
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="LSTM hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--stop_metric", type=str, default="both",
                        choices=["throughput", "latency", "both"])
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    print("="*60)
    print("LSTM Baseline - 5-Fold Cross Validation")
    print(f"hidden_dim={args.hidden_dim}, num_layers={args.num_layers}")
    print(f"dropout={args.dropout}, lr={args.lr}")
    print(f"stop_metric={args.stop_metric}")
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

        if metrics['RMSE_T'] is not None:
            print(f"\nFold {fold+1} Results:")
            print(f"  Throughput (best epoch {metrics['best_epoch_t']}): RMSE={metrics['RMSE_T']:.4f}, MAE={metrics['MAE_T']:.4f}, MAPE={metrics['MAPE_T']:.2f}%, R2={metrics['R2_T']:.4f}")
            print(f"  Latency (best epoch {metrics['best_epoch_l']}):    RMSE={metrics['RMSE_L']:.4f}, MAE={metrics['MAE_L']:.4f}, MAPE={metrics['MAPE_L']:.2f}%, R2={metrics['R2_L']:.4f}")

    df_res = pd.DataFrame(results)
    df_res_stats = df_res.drop(columns=['best_epoch_t', 'best_epoch_l'])

    print("\n" + "="*60)
    print("LSTM - 5-FOLD CROSS VALIDATION RESULTS")
    print("="*60)
    print("\n--- Mean ± Std ---")
    for col in df_res_stats.columns:
        mean_val = df_res_stats[col].mean()
        std_val = df_res_stats[col].std()
        print(f"{col}: {mean_val:.4f} ± {std_val:.4f}")

    os.makedirs("results_lstm", exist_ok=True)
    df_res.to_csv("results_lstm/lstm_5fold_results.csv", index=False)

    print("\n✓ Results saved to results_lstm/")
    print("✓ Best models saved to best_models_lstm/")


if __name__ == "__main__":
    main()