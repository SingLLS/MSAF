# -*- coding: utf-8 -*-
"""
Final ε-constraint Block Size Recommendation
Hybrid FTT + RaftGAT
(Physical feasibility + throughput ε-constraint
 + Adaptive beta + Hard throughput cap)

Usage:
python recommend_block.py \
  --model modelH/Hybrid_FTT_RaftGAT.pth \
  --arrival 90\
  --orderers 9 \
  --block_min 10 \
  --block_max 800 \
  --epsilon 0.05 \
  --delta 0.05 \
  --L_max 1.0 \
  --beta 0.5
"""

import argparse
import numpy as np
import pandas as pd
import torch

from train_hybrid import HybridFTTRaftGAT


# =========================================================
# Safe checkpoint loading (PyTorch >= 2.6)
# =========================================================
def load_checkpoint(path, device="cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--arrival", type=float, required=True)
    parser.add_argument("--orderers", type=float, required=True)
    parser.add_argument("--block_min", type=int, default=1)
    parser.add_argument("--block_max", type=int, default=800)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--L_max", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--lambda_ref", type=float, default=100.0,
                        help="Reference arrival for adaptive beta")
    args = parser.parse_args()

    device = "cpu"

    # ===== Load model =====
    ckpt = load_checkpoint(args.model, device)

    model = HybridFTTRaftGAT(num_features=3)
    model.load_state_dict(ckpt["model"])
    model.eval()

    sx = ckpt["scaler_X"]
    sy = ckpt["scaler_Y"]

    # ===== Adaptive beta (Scheme 1) =====
    beta_eff = args.beta * (args.arrival / args.lambda_ref)

    # ===== Search space =====
    blocks = np.arange(args.block_min, args.block_max + 1)
    arrival_log = np.log1p(args.arrival)

    records = []

    with torch.no_grad():
        for b in blocks:
            x_raw = np.array([[arrival_log, args.orderers, b]], dtype=np.float32)
            x_norm = sx.transform(x_raw)

            pred_norm = model(
                torch.tensor(x_norm),
                torch.tensor(x_raw),
                torch.tensor([[args.orderers]])
            )

            pred = sy.inverse_transform(pred_norm.numpy())[0]
            T_pred = float(pred[0])
            L_pred = float(pred[1])

            # ===== Scheme 2: Hard physical throughput cap =====
            T_final = min(T_pred, args.arrival)

            records.append([b, T_final, L_pred])

    df = pd.DataFrame(
        records,
        columns=["block_size", "throughput", "latency"]
    )

    # =====================================================
    # Step 1: Physical feasibility (arrival tolerance)
    # =====================================================
    df = df[df["throughput"] <= args.arrival * (1.0 + args.delta)]

    # =====================================================
    # Step 2: Latency hard constraint (Scheme 1)
    # =====================================================
    df = df[df["latency"] <= args.L_max]

    if df.empty:
        raise RuntimeError("No feasible configuration after constraints.")

    # =====================================================
    # Step 3: ε-constraint on throughput
    # =====================================================
    T_max = df["throughput"].max()
    T_eps = (1.0 - args.epsilon) * T_max
    feasible = df[df["throughput"] >= T_eps]

    if feasible.empty:
        raise RuntimeError("No feasible config after ε-constraint.")

    # =====================================================
    # Step 4: Final score (block regularized)
    # =====================================================
    feasible["score"] = (
        feasible["throughput"]
        - beta_eff / feasible["block_size"]
    )

    best = feasible.sort_values("score", ascending=False).iloc[0]

    # =====================================================
    # Output
    # =====================================================
    print("\n===== Recommendation Result (ε-constraint + Scheme 1 & 2) =====")
    print(f"Arrival Rate   : {args.arrival}")
    print(f"Orderers       : {args.orderers}")
    print(f"Epsilon ε      : {args.epsilon}")
    print(f"Delta δ        : {args.delta}")
    print(f"L_max          : {args.L_max}")
    print(f"Beta β (eff)   : {beta_eff:.4f}")
    print("---------------------------------------------------------------")
    print(f"Optimal Block  : {int(best.block_size)}")
    print(f"Throughput     : {best.throughput:.3f}")
    print(f"Latency        : {best.latency:.6f}")
    print("===============================================================\n")


if __name__ == "__main__":
    main()
