# -*- coding: utf-8 -*-

"""
Run:
python train_csam.py --dataset ./data/HFBTP.csv --compare --use_gpu
python train_csam.py --dataset ./data/HFBTP.csv  --use_gpu
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger

# Assume these modules exist
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
# Model (with optional gate)
# =========================================================
class HybridFTTRaftGAT(nn.Module):
    def __init__(self, num_features, use_gate=True):
        super().__init__()
        self.use_gate = use_gate

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

        if self.use_gate:
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

        if self.use_gate:
            alpha = self.gate(x_raw) * 0.5
            z = self.norm(z_num + alpha * z_raft)
        else:
            z = self.norm(z_num + z_raft)

        out = self.head(z)

        throughput = torch.sigmoid(out[:, 0:1])
        latency = torch.nn.functional.softplus(out[:, 1:2])

        return torch.cat([throughput, latency], dim=1)


# =========================================================
# Loss (unchanged)
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
# Lightning Module (unchanged except model init)
# =========================================================
class LitHybrid(pl.LightningModule):
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
        self.val_alphas = []                     # 新增：存储每个 batch 的 alpha

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
        self.best_combined = float('inf')

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

        # ========== 记录 alpha（仅当模型使用 gate） ==========
        if self.model.use_gate:
            alpha = self.model.gate(x_raw) * 0.5      # 与 forward 中计算一致
            self.val_alphas.append(alpha.detach().cpu())

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

        self.log("val_mae_t", mae_t)
        self.log("val_rmse_t", rmse_t, prog_bar=True)
        self.log("val_mape_t", mape_t)
        self.log("val_r2_t", r2_t)

        self.log("val_mae_l", mae_l)
        self.log("val_rmse_l", rmse_l)
        self.log("val_mape_l", mape_l)
        self.log("val_r2_l", r2_l)

        # ========== 记录 alpha 统计量 ==========
        if self.model.use_gate and self.val_alphas:
            alphas = torch.cat(self.val_alphas).numpy().flatten()
            alpha_mean = np.mean(alphas)
            alpha_std = np.std(alphas)
            self.log("val_alpha_mean", alpha_mean)
            self.log("val_alpha_std", alpha_std)
            self.val_alphas.clear()      # 清空，准备下一个 epoch

        # Best throughput tracking
        if rmse_t < self.best_val_rmse_t - 1e-6:
            self.best_val_rmse_t = rmse_t
            self.best_epoch_t = self.current_epoch
            self.best_model_state_t = {k: v.clone() for k, v in self.model.state_dict().items()}
            self.best_metrics_t = {
                "RMSE_T": rmse_t, "MAE_T": mae_t, "MAPE_T": mape_t, "R2_T": r2_t,
                "RMSE_L": rmse_l, "MAE_L": mae_l, "MAPE_L": mape_l, "R2_L": r2_l,
                "epoch": self.current_epoch
            }
            self.patience_counter_t = 0
            print(f"✓ Best throughput model at epoch {self.current_epoch} (RMSE_t: {rmse_t:.4f})")
        else:
            self.patience_counter_t += 1

        # Best latency tracking
        if rmse_l < self.best_val_rmse_l - 1e-6:
            self.best_val_rmse_l = rmse_l
            self.best_epoch_l = self.current_epoch
            self.best_model_state_l = {k: v.clone() for k, v in self.model.state_dict().items()}
            self.best_metrics_l = {
                "RMSE_T": rmse_t, "MAE_T": mae_t, "MAPE_T": mape_t, "R2_T": r2_t,
                "RMSE_L": rmse_l, "MAE_L": mae_l, "MAPE_L": mape_l, "R2_L": r2_l,
                "epoch": self.current_epoch
            }
            self.patience_counter_l = 0
            print(f"✓ Best latency model at epoch {self.current_epoch} (RMSE_l: {rmse_l:.4f})")
        else:
            self.patience_counter_l += 1

        self.log("patience_t", self.patience_counter_t)
        self.log("patience_l", self.patience_counter_l)

        self.val_preds.clear()
        self.val_targets.clear()

    # should_stop, get_best_model_t, get_best_model_l, get_best_metrics_t, 
    # get_best_metrics_l, configure_optimizers 保持不变
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [scheduler]

# =========================================================
# Early Stopping (unchanged)
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
# Training function (can choose gate usage)
# =========================================================
def train_model(train_idx, val_idx, dataset, scaler_y, args, use_gate, fold_id=0, log_subdir=None):
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = HybridFTTRaftGAT(num_features=3, use_gate=use_gate)
    lit = LitHybrid(
        model,
        args.lr,
        scaler_y,
        patience=args.patience,
        stop_metric=args.stop_metric
    )

    # if args.stop_metric == "both":
    #     early_stop = DualEarlyStopping(
    #         monitor_t="val_rmse_t",
    #         monitor_l="val_rmse_l",
    #         mode="min",
    #         patience=args.patience,
    #         verbose=True,
    #         stop_metric=args.stop_metric
    #     )
    # else:
    #     monitor_metric = "val_rmse_t" if args.stop_metric == "throughput" else "val_rmse_l"
    #     early_stop = EarlyStopping(
    #         monitor=monitor_metric,
    #         mode="min",
    #         patience=args.patience,
    #         verbose=True
    #     )

    # Create logger with specific subdirectory
    logger = CSVLogger(save_dir="./logs", name=log_subdir) if log_subdir else None

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="cuda" if args.use_gpu and torch.cuda.is_available() else "cpu",
        devices=1,
        # callbacks=[early_stop],
        logger=logger,
        enable_checkpointing=False,
        deterministic=True
    )

    trainer.fit(lit, train_loader, val_loader)

    # Save best models
    os.makedirs("best_models", exist_ok=True)
    suffix = "gate" if use_gate else "nogate"
    best_model_t = lit.get_best_model_t()
    best_model_l = lit.get_best_model_l()
    torch.save(best_model_t.state_dict(), f"best_models/best_model_{suffix}_t_fold{fold_id}.pth")
    torch.save(best_model_l.state_dict(), f"best_models/best_model_{suffix}_l_fold{fold_id}.pth")

    best_metrics_t = lit.get_best_metrics_t()
    best_metrics_l = lit.get_best_metrics_l()
    combined_metrics = {
        "RMSE_T": best_metrics_t["RMSE_T"],
        "MAE_T": best_metrics_t["MAE_T"],
        "MAPE_T": best_metrics_t["MAPE_T"],
        "R2_T": best_metrics_t["R2_T"],
        "RMSE_L": best_metrics_l["RMSE_L"],
        "MAE_L": best_metrics_l["MAE_L"],
        "MAPE_L": best_metrics_l["MAPE_L"],
        "R2_L": best_metrics_l["R2_L"],
        "best_epoch_t": best_metrics_t["epoch"],
        "best_epoch_l": best_metrics_l["epoch"],
    }
    return combined_metrics, logger.log_dir if logger else None


# =========================================================
# Comparison Plotting (PDF format)
# =========================================================
def compare_convergence(log_dir_gate, log_dir_nogate, output_pdf="results/convergence_comparison.pdf"):
    """Read metrics.csv from two log directories and plot RMSE curves side by side."""
    def load_rmse(log_dir):
        metrics_csv = os.path.join(log_dir, "metrics.csv")
        if not os.path.exists(metrics_csv):
            return None, None, None
        df = pd.read_csv(metrics_csv)
        df_valid = df[df["val_rmse_t"].notna()].copy()
        if df_valid.empty:
            return None, None, None
        epochs = df_valid["epoch"].values
        rmse_t = df_valid["val_rmse_t"].values
        rmse_l = df_valid["val_rmse_l"].values
        return epochs, rmse_t, rmse_l

    epochs_g, rmse_t_g, rmse_l_g = load_rmse(log_dir_gate)
    epochs_ng, rmse_t_ng, rmse_l_ng = load_rmse(log_dir_nogate)

    if epochs_g is None or epochs_ng is None:
        print("Error: missing log data for comparison.")
        return

    # Align lengths (take minimum)
    min_len = min(len(epochs_g), len(epochs_ng))
    epochs_g = epochs_g[:min_len]
    rmse_t_g = rmse_t_g[:min_len]
    rmse_l_g = rmse_l_g[:min_len]
    epochs_ng = epochs_ng[:min_len]
    rmse_t_ng = rmse_t_ng[:min_len]
    rmse_l_ng = rmse_l_ng[:min_len]

    plt.figure(figsize=(12, 5))

    # Throughput subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_g, rmse_t_g, 'b-o', markersize=3, linewidth=1.5, label="With Gate")
    plt.plot(epochs_ng, rmse_t_ng, 'r-s', markersize=3, linewidth=1.5, label="Without Gate")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Throughput Convergence Comparison")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Latency subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_g, rmse_l_g, 'b-o', markersize=3, linewidth=1.5, label="With Gate")
    plt.plot(epochs_ng, rmse_l_ng, 'r-s', markersize=3, linewidth=1.5, label="Without Gate")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Latency Convergence Comparison")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    plt.savefig(output_pdf, format='pdf', dpi=150)
    plt.show()
    print(f"Comparison plot saved to {output_pdf}")

    # Save RMSE data to CSV (aligned)
    df_comp = pd.DataFrame({
        "epoch": epochs_g,
        "rmse_t_gate": rmse_t_g,
        "rmse_l_gate": rmse_l_g,
        "rmse_t_nogate": rmse_t_ng,
        "rmse_l_nogate": rmse_l_ng
    })
    df_comp.to_csv("results/rmse_comparison.csv", index=False)
    print("RMSE comparison data saved to results/rmse_comparison.csv")


# =========================================================
# Main
# =========================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="./data/HFBTP.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--stop_metric", type=str, default="both",
                        choices=["throughput", "latency", "both"])
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--compare", action="store_true", help="Train and compare gate vs no-gate")
    args = parser.parse_args()

    print("="*60)
    print("Hybrid FTT + RaftGAT - Compare Gate vs No-Gate Fusion")
    print(f"max_epochs={args.max_epochs}, patience={args.patience}, lr={args.lr}")
    print(f"stop_metric={args.stop_metric}")
    print(f"compare={args.compare}")
    print("="*60)

    # Load data once
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

    # Fixed 80/20 split
    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)
    print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}")

    if args.compare:
        # Train with gate
        print("\n>>> Training with Gate (alpha * z_raft)")
        metrics_gate, log_dir_gate = train_model(train_idx, val_idx, dataset, sy, args,
                                                  use_gate=True, fold_id=0, log_subdir="with_gate")
        print(f"Gate model best throughput RMSE: {metrics_gate['RMSE_T']:.4f}")
        print(f"Gate model best latency RMSE:    {metrics_gate['RMSE_L']:.4f}")

        # Train without gate
        print("\n>>> Training without Gate (z_num + z_raft)")
        metrics_nogate, log_dir_nogate = train_model(train_idx, val_idx, dataset, sy, args,
                                                      use_gate=False, fold_id=0, log_subdir="without_gate")
        print(f"No-gate model best throughput RMSE: {metrics_nogate['RMSE_T']:.4f}")
        print(f"No-gate model best latency RMSE:    {metrics_nogate['RMSE_L']:.4f}")

        # Compare and plot
        if log_dir_gate and log_dir_nogate:
            compare_convergence(log_dir_gate, log_dir_nogate, output_pdf="results/convergence_comparison.pdf")
        else:
            print("Could not retrieve log directories for comparison.")

        # Save both metrics
        pd.DataFrame([metrics_gate, metrics_nogate], index=["with_gate", "without_gate"]).to_csv(
            "results/best_metrics_comparison.csv")
    else:
        # Default: just train with gate (original behavior)
        metrics, _ = train_model(train_idx, val_idx, dataset, sy, args,
                                 use_gate=True, fold_id=0, log_subdir="single_run")
        print("\nTraining completed (with gate)")
        print(f"Throughput RMSE: {metrics['RMSE_T']:.4f}, Latency RMSE: {metrics['RMSE_L']:.4f}")
        pd.DataFrame([metrics]).to_csv("results/single_run_metrics.csv", index=False)

    print("\n✓ All results saved to 'results/' directory")
    print("✓ Comparison plot saved as PDF in results/convergence_comparison.pdf (if --compare used)")


if __name__ == "__main__":
    main()