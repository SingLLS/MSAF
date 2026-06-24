# -*- coding: utf-8 -*-

"""
Run:
  # Full constraints
  python train_rule_fold.py --dataset ./data/HFBTP.csv --use_gpu

  # Only Lcon
  python train_rule_fold.py --no_lsca --use_gpu

  # Only Lsca
  python train_rule_fold.py --no_lcon --use_gpu

  # No constraints (baseline)
  python train_rule_fold.py --no_lcon --no_lsca --use_gpu
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from pytorch_lightning.callbacks import EarlyStopping

from MSFTTmodel import FTTransformerMulti
from raft_gat import RaftGAT

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
    def __init__(self, x_norm, x_raw, topo, y_norm, t_ref_raw):

        self.x_norm = torch.from_numpy(x_norm).float()
        self.x_raw = torch.from_numpy(x_raw).float()
        self.topo = torch.from_numpy(topo).float()
        self.y = torch.from_numpy(y_norm).float()
        self.t_ref = torch.from_numpy(t_ref_raw).float().reshape(-1, 1)

    def __len__(self):
        return len(self.x_norm)

    def __getitem__(self, idx):
        return self.x_norm[idx], self.x_raw[idx], self.topo[idx], self.y[idx], self.t_ref[idx]


# =========================================================
# Model (unchanged)
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
        throughput = torch.sigmoid(out[:, 0:1])          # [0,1] normalized
        latency = F.softplus(out[:, 1:2])                # positive
        return torch.cat([throughput, latency], dim=1)


# =========================================================
# Multi-Task Uncertainty Loss (unchanged)
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
        loss = (torch.exp(-log_sigma_t) * mse_t +
                torch.exp(-log_sigma_l) * mse_l +
                log_sigma_t + log_sigma_l)
        return loss


# =========================================================
# Lightning Module with Constraints (Lcon & Lsca as upper bounds)
# and PVR/MVR computation
# =========================================================
class LitHybrid(pl.LightningModule):
    def __init__(
        self,
        model,
        lr,
        scaler_y,
        patience=25,
        stop_metric="both",
        use_lcon=True,
        use_lsca=True,
        lambda_lcon=0.01,
        lambda_lsca=0.01,
        warmup_epochs=50,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = MultiTaskUncertaintyLoss()
        self.lr = lr
        self.scaler_y = scaler_y
        self.patience = patience
        self.stop_metric = stop_metric

        self.use_lcon = use_lcon
        self.use_lsca = use_lsca
        self.lambda_lcon = lambda_lcon
        self.lambda_lsca = lambda_lsca
        self.warmup_epochs = warmup_epochs

        # For inverse transform
        self.y_min = scaler_y.min_[0]
        self.y_scale = scaler_y.data_max_[0] - scaler_y.data_min_[0]

        self.val_preds = []
        self.val_targets = []

        # Counters for violation rates (reset each validation epoch)
        self.viol_con_count = 0.0
        self.viol_sca_count = 0.0
        self.total_samples = 0.0

        # Best model tracking (separate for throughput and latency)
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

    def _get_reg_weight(self, epoch):
        """Linear warmup for constraint weights."""
        if epoch < self.warmup_epochs:
            return epoch / max(self.warmup_epochs, 1)
        else:
            return 1.0

    def training_step(self, batch, batch_idx):
        x_norm, x_raw, topo, y, t_ref = batch
        pred = self.model(x_norm, x_raw, topo)

        # Multi-task loss (on normalized targets)
        loss_mt = self.loss_fn(pred, y)

        # Inverse transform throughput to original scale
        pred_t_norm = pred[:, 0:1]
        pred_t_raw = pred_t_norm * self.y_scale + self.y_min
        pred_t_raw = torch.clamp(pred_t_raw, min=0.0)

        # Arrival rate (original scale) from x_raw[:,0] which is log1p(arrival)
        arrival_raw = torch.expm1(x_raw[:, 0:1])

        loss_con = torch.tensor(0.0, device=pred.device)
        loss_sca = torch.tensor(0.0, device=pred.device)

        if self.use_lcon:
            # Lcon: penalize throughput > arrival (upper bound)
            violation_con = F.relu(pred_t_raw - arrival_raw)
            loss_con = violation_con.mean()

        if self.use_lsca:
            # Lsca: penalize throughput > T_ref (upper bound)
            violation_sca = F.relu(pred_t_raw - t_ref)
            loss_sca = violation_sca.mean()

        warmup = self._get_reg_weight(self.current_epoch)
        lambda_con_eff = self.lambda_lcon * warmup if self.use_lcon else 0.0
        lambda_sca_eff = self.lambda_lsca * warmup if self.use_lsca else 0.0

        total_loss = loss_mt + lambda_con_eff * loss_con + lambda_sca_eff * loss_sca

        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_loss_mt", loss_mt)
        self.log("train_loss_con", loss_con)
        self.log("train_loss_sca", loss_sca)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x_norm, x_raw, topo, y, t_ref = batch
        pred = self.model(x_norm, x_raw, topo)
        self.val_preds.append(pred.detach().cpu())
        self.val_targets.append(y.detach().cpu())

        # ---- Accumulate violation counts ----
        batch_size = pred.size(0)
        # Inverse transform throughput to original scale
        pred_t_norm = pred[:, 0:1]
        pred_t_raw = pred_t_norm * self.y_scale + self.y_min
        pred_t_raw = torch.clamp(pred_t_raw, min=0.0)
        arrival_raw = torch.expm1(x_raw[:, 0:1])   # original scale

        # Lcon violation: pred_t > arrival
        viol_con = (pred_t_raw > arrival_raw).float().sum().item()
        # Lsca violation: pred_t > t_ref
        viol_sca = (pred_t_raw > t_ref).float().sum().item()

        self.viol_con_count += viol_con
        self.viol_sca_count += viol_sca
        self.total_samples += batch_size

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_preds).numpy()
        targets = torch.cat(self.val_targets).numpy()

        # Inverse transform to original scale
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

        # Compute violation rates
        pvr = self.viol_con_count / (self.total_samples + 1e-8)
        mvr = self.viol_sca_count / (self.total_samples + 1e-8)

        self.log("val_rmse_t", rmse_t, prog_bar=True)
        self.log("val_mae_t", mae_t)
        self.log("val_mape_t", mape_t)
        self.log("val_r2_t", r2_t)
        self.log("val_rmse_l", rmse_l)
        self.log("val_mae_l", mae_l)
        self.log("val_mape_l", mape_l)
        self.log("val_r2_l", r2_l)
        self.log("val_pvr", pvr, prog_bar=True)
        self.log("val_mvr", mvr, prog_bar=True)

        # Reset counters
        self.viol_con_count = 0.0
        self.viol_sca_count = 0.0
        self.total_samples = 0.0

        # Best model tracking (based on RMSE)
        if rmse_t < self.best_val_rmse_t - 1e-6:
            self.best_val_rmse_t = rmse_t
            self.best_epoch_t = self.current_epoch
            self.best_model_state_t = {k: v.clone() for k, v in self.model.state_dict().items()}
            self.best_metrics_t = {
                "RMSE_T": rmse_t, "MAE_T": mae_t, "MAPE_T": mape_t, "R2_T": r2_t,
                "RMSE_L": rmse_l, "MAE_L": mae_l, "MAPE_L": mape_l, "R2_L": r2_l,
                "PVR": pvr,
                "MVR": mvr,
                "epoch": self.current_epoch
            }
            self.patience_counter_t = 0
            print(f"✓ Best throughput model updated at epoch {self.current_epoch} (RMSE_t: {rmse_t:.4f}, PVR={pvr:.4f}, MVR={mvr:.4f})")
        else:
            self.patience_counter_t += 1

        if rmse_l < self.best_val_rmse_l - 1e-6:
            self.best_val_rmse_l = rmse_l
            self.best_epoch_l = self.current_epoch
            self.best_model_state_l = {k: v.clone() for k, v in self.model.state_dict().items()}
            self.best_metrics_l = {
                "RMSE_T": rmse_t, "MAE_T": mae_t, "MAPE_T": mape_t, "R2_T": r2_t,
                "RMSE_L": rmse_l, "MAE_L": mae_l, "MAPE_L": mape_l, "R2_L": r2_l,
                "PVR": pvr,
                "MVR": mvr,
                "epoch": self.current_epoch
            }
            self.patience_counter_l = 0
            print(f"✓ Best latency model updated at epoch {self.current_epoch} (RMSE_l: {rmse_l:.4f})")
        else:
            self.patience_counter_l += 1

        self.log("patience_t", self.patience_counter_t)
        self.log("patience_l", self.patience_counter_l)

        self.val_preds.clear()
        self.val_targets.clear()

    def should_stop(self):
        if self.stop_metric == "throughput":
            return self.patience_counter_t >= self.patience and self.current_epoch > 20
        elif self.stop_metric == "latency":
            return self.patience_counter_l >= self.patience and self.current_epoch > 20
        elif self.stop_metric == "both":
            return (self.patience_counter_t >= self.patience and
                    self.patience_counter_l >= self.patience and
                    self.current_epoch > 20)
        else:
            return self.patience_counter_t >= self.patience and self.current_epoch > 20

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
# Dual Early Stopping (custom)
# =========================================================
class DualEarlyStopping(EarlyStopping):
    def __init__(self, monitor_t="val_rmse_t", monitor_l="val_rmse_l",
                 mode="min", patience=25, verbose=True, stop_metric="both"):
        super().__init__(monitor=monitor_t, mode=mode, patience=patience, verbose=verbose)
        self.monitor_t = monitor_t
        self.monitor_l = monitor_l
        self.patience = patience
        self.verbose = verbose
        self.stop_metric = stop_metric

    def _should_stop(self, trainer):
        lit_module = trainer.lightning_module
        return lit_module.should_stop()


# =========================================================
# Train one fold (with per-fold T_ref computed via linear regression)
# =========================================================
def run_fold(train_idx, val_idx, df, args, fold_id):
    # Split DataFrame
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    # ---- Compute T_ref based ONLY on training set using linear regression ----
    # Features (original scale): Orderers, Arrival, BlockSize, and interaction Orderers*Arrival
    O_train = train_df['Orderers'].values.reshape(-1, 1)
    A_train = train_df['Actual Transaction Arrival Rate'].values.reshape(-1, 1)
    B_train = train_df['Block Size'].values.reshape(-1, 1)
    OA_train = (O_train * A_train).reshape(-1, 1)

    X_train_reg = np.concatenate([O_train, A_train, B_train, OA_train], axis=1)
    y_train_reg = train_df['Throughput'].values

    # Fit regression (positive coefficients to maintain physical monotonicity)
    reg = LinearRegression(positive=True)
    reg.fit(X_train_reg, y_train_reg)

    # Function to predict T_ref for any dataframe
    def compute_t_ref(df_sub):
        O = df_sub['Orderers'].values.reshape(-1, 1)
        A = df_sub['Actual Transaction Arrival Rate'].values.reshape(-1, 1)
        B = df_sub['Block Size'].values.reshape(-1, 1)
        OA = (O * A).reshape(-1, 1)
        X_sub = np.concatenate([O, A, B, OA], axis=1)
        t_ref_pred = reg.predict(X_sub)
        # Clip to a reasonable range (at least 0, and not too low)
        t_ref_pred = np.maximum(t_ref_pred, train_df['Throughput'].min() * 0.9)
        return t_ref_pred

    train_df['T_ref'] = compute_t_ref(train_df)
    val_df['T_ref']   = compute_t_ref(val_df)

    # ---- Build features and targets ----
    def prepare_data(df_sub):
        arrival = np.log1p(df_sub["Actual Transaction Arrival Rate"].values.astype(float))
        orderers = df_sub["Orderers"].values.astype(float)
        block = df_sub["Block Size"].values.astype(float)
        X_raw = np.stack([arrival, orderers, block], axis=1)
        topo = orderers.reshape(-1, 1)
        Y = df_sub[["Throughput", "Avg Latency"]].values.astype(float)
        Y = np.nan_to_num(Y, nan=0.0)
        t_ref = df_sub["T_ref"].values.astype(float).reshape(-1, 1)
        return X_raw, topo, Y, t_ref

    X_train_raw, topo_train, Y_train, t_ref_train = prepare_data(train_df)
    X_val_raw, topo_val, Y_val, t_ref_val = prepare_data(val_df)

    # Fit scalers on training set only
    sx = MinMaxScaler().fit(X_train_raw)
    sy = MinMaxScaler().fit(Y_train)

    # Transform
    X_train_norm = sx.transform(X_train_raw)
    X_val_norm = sx.transform(X_val_raw)
    Y_train_norm = sy.transform(Y_train)
    Y_val_norm = sy.transform(Y_val)

    # Datasets
    train_set = BlockChainDataset(X_train_norm, X_train_raw, topo_train, Y_train_norm, t_ref_train)
    val_set = BlockChainDataset(X_val_norm, X_val_raw, topo_val, Y_val_norm, t_ref_val)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # Model & Lightning module
    model = HybridFTTRaftGAT(num_features=3)
    lit = LitHybrid(
        model,
        args.lr,
        sy,  # scaler_y
        patience=args.patience,
        stop_metric=args.stop_metric,
        use_lcon=args.use_lcon,
        use_lsca=args.use_lsca,
        lambda_lcon=args.lambda_lcon,
        lambda_lsca=args.lambda_lsca,
        warmup_epochs=args.warmup_epochs,
    )

    # Early stopping
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
        deterministic=True,
        gradient_clip_val=1.0,
    )

    trainer.fit(lit, train_loader, val_loader)

    # Retrieve best models
    best_model_t = lit.get_best_model_t()
    best_model_l = lit.get_best_model_l()
    best_metrics_t = lit.get_best_metrics_t()
    best_metrics_l = lit.get_best_metrics_l()

    # Save with tag
    tag = ""
    if args.use_lcon and args.use_lsca:
        tag = "both"
    elif args.use_lcon:
        tag = "lcon"
    elif args.use_lsca:
        tag = "lsca"
    else:
        tag = "none"

    os.makedirs("best_models", exist_ok=True)
    torch.save(best_model_t.state_dict(), f"best_models/best_model_t_{tag}_fold{fold_id}.pth")
    torch.save(best_model_l.state_dict(), f"best_models/best_model_l_{tag}_fold{fold_id}.pth")

    combined = {
        "RMSE_T": best_metrics_t["RMSE_T"],
        "MAE_T": best_metrics_t["MAE_T"],
        "MAPE_T": best_metrics_t["MAPE_T"],
        "R2_T": best_metrics_t["R2_T"],
        "RMSE_L": best_metrics_l["RMSE_L"],
        "MAE_L": best_metrics_l["MAE_L"],
        "MAPE_L": best_metrics_l["MAPE_L"],
        "R2_L": best_metrics_l["R2_L"],
        "PVR": best_metrics_t["PVR"],
        "MVR": best_metrics_t["MVR"],
        "best_epoch_t": best_metrics_t["epoch"],
        "best_epoch_l": best_metrics_l["epoch"],
    }
    return combined


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
    parser.add_argument("--stop_metric", type=str, default="both",
                        choices=["throughput", "latency", "both"])
    parser.add_argument("--use_gpu", action="store_true")

    # Constraint switches
    parser.add_argument("--use_lcon", action="store_true", default=True,
                        help="Enable Lcon constraint (throughput <= arrival)")
    parser.add_argument("--no_lcon", dest="use_lcon", action="store_false",
                        help="Disable Lcon")
    parser.add_argument("--use_lsca", action="store_true", default=True,
                        help="Enable Lsca constraint (throughput <= T_ref)")
    parser.add_argument("--no_lsca", dest="use_lsca", action="store_false",
                        help="Disable Lsca")

    # Constraint weights
    parser.add_argument("--lambda_lcon", type=float, default=0.01,
                        help="Weight for Lcon loss")
    parser.add_argument("--lambda_lsca", type=float, default=0.01,
                        help="Weight for Lsca loss")
    parser.add_argument("--warmup_epochs", type=int, default=50,
                        help="Epochs for linear warmup of constraint weights")

    args = parser.parse_args()

    tag = ""
    if args.use_lcon and args.use_lsca:
        tag = "both"
    elif args.use_lcon:
        tag = "lcon"
    elif args.use_lsca:
        tag = "lsca"
    else:
        tag = "none"

    print("="*60)
    print(f"Hybrid FTT + RaftGAT - Training with Constraints ({tag})")
    print(f"use_lcon={args.use_lcon}, use_lsca={args.use_lsca}")
    print(f"lambda_lcon={args.lambda_lcon}, lambda_lsca={args.lambda_lsca}")
    print(f"warmup_epochs={args.warmup_epochs}")
    print("="*60)

    df = pd.read_csv(args.dataset)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"\n{'='*40}")
        print(f"Fold {fold+1}/5")
        print(f"{'='*40}")

        metrics = run_fold(train_idx, val_idx, df, args, fold)
        results.append(metrics)

        print(f"\nFold {fold+1} Results:")
        print(f"  Throughput (best epoch {metrics['best_epoch_t']}): RMSE={metrics['RMSE_T']:.4f}, MAE={metrics['MAE_T']:.4f}, MAPE={metrics['MAPE_T']:.2f}%, R2={metrics['R2_T']:.4f}, PVR={metrics['PVR']:.4f}, MVR={metrics['MVR']:.4f}")
        print(f"  Latency (best epoch {metrics['best_epoch_l']}):    RMSE={metrics['RMSE_L']:.4f}, MAE={metrics['MAE_L']:.4f}, MAPE={metrics['MAPE_L']:.2f}%, R2={metrics['R2_L']:.4f}")

    df_res = pd.DataFrame(results)
    df_res_stats = df_res.drop(columns=['best_epoch_t', 'best_epoch_l'])

    print("\n" + "="*60)
    print(f"5-FOLD CROSS VALIDATION RESULTS (tag={tag})")
    print("="*60)
    print("\n--- Per Fold Results ---")
    print(df_res.round(4).to_string())

    print("\n--- Mean ± Std ---")
    for col in df_res_stats.columns:
        mean_val = df_res_stats[col].mean()
        std_val = df_res_stats[col].std()
        print(f"{col}: {mean_val:.4f} ± {std_val:.4f}")

    os.makedirs("results", exist_ok=True)
    df_res.to_csv(f"results/train_constraints_{tag}_5fold_results.csv", index=False)

    summary = []
    for col in df_res_stats.columns:
        summary.append({
            "Metric": col,
            "Mean": df_res_stats[col].mean(),
            "Std": df_res_stats[col].std(),
            "Min": df_res_stats[col].min(),
            "Max": df_res_stats[col].max()
        })
    pd.DataFrame(summary).to_csv(f"results/train_constraints_{tag}_summary.csv", index=False)

    print(f"\n✓ Results saved to results/ (tag={tag})")
    print("✓ Best models saved to best_models/")


if __name__ == "__main__":
    main()