# -*- coding: utf-8 -*-

"""
SVR Baseline for Blockchain Performance Prediction
5-Fold Cross Validation (NO DATA LEAKAGE)

SVR (Support Vector Regression) is used as a baseline model.

Run:
python train_svr.py --dataset ./data/HFBTP.csv
python train_svr.py --dataset ./data/HFBTP.csv --linear
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR, LinearSVR
from sklearn.multioutput import MultiOutputRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')


# =========================================================
# Reproducibility
# =========================================================
seed = 42
np.random.seed(seed)


# =========================================================
# Metrics
# =========================================================
def calculate_metrics(y_true, y_pred):
    """Calculate MAE, RMSE, MAPE, R2 for each output"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    epsilon = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    return mae, rmse, mape, r2


def evaluate_model(model, X, y, scaler_y):
    """Evaluate model and return metrics in original scale"""
    y_pred_norm = model.predict(X)
    y_pred = scaler_y.inverse_transform(y_pred_norm)
    y_true = scaler_y.inverse_transform(y)
    
    mae_t, rmse_t, mape_t, r2_t = calculate_metrics(y_true[:, 0], y_pred[:, 0])
    mae_l, rmse_l, mape_l, r2_l = calculate_metrics(y_true[:, 1], y_pred[:, 1])
    
    return {
        "MAE_T": mae_t, "RMSE_T": rmse_t, "MAPE_T": mape_t, "R2_T": r2_t,
        "MAE_L": mae_l, "RMSE_L": rmse_l, "MAPE_L": mape_l, "R2_L": r2_l,
    }


# =========================================================
# Train one fold
# =========================================================
def train_fold(X_train_raw, y_train_raw, X_val_raw, y_val_raw, fold_id, args):
    """Train SVR on one fold with per-fold normalization"""
    
    sx = StandardScaler().fit(X_train_raw)
    sy = StandardScaler().fit(y_train_raw)
    
    # Transform both training and validation
    X_train = sx.transform(X_train_raw)
    X_val = sx.transform(X_val_raw)
    y_train = sy.transform(y_train_raw)
    y_val = sy.transform(y_val_raw)
    
    print(f"  Training on {len(X_train)} samples...")
    
    # Choose SVR type
    if args.linear:
        base_svr = LinearSVR(
            C=args.C,
            epsilon=args.epsilon_svr,
            max_iter=args.max_iter,
            random_state=seed,
            dual='auto',
            tol=args.tol
        )
    else:
        base_svr = SVR(
            kernel=args.kernel,
            C=args.C,
            epsilon=args.epsilon_svr,
            gamma=args.gamma,
            cache_size=args.cache_size,
            max_iter=args.max_iter,
            tol=args.tol
        )
    
    model = MultiOutputRegressor(base_svr, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    metrics = evaluate_model(model, X_val, y_val, sy)
    
    return model, metrics, sx, sy


# =========================================================
# Main (5-Fold CV)
# =========================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='SVR Baseline - No Data Leakage')
    parser.add_argument("--dataset", default="./data/HFBTP.csv")
    parser.add_argument("--kernel", type=str, default="rbf",
                        choices=["linear", "poly", "rbf", "sigmoid"])
    parser.add_argument("--linear", action="store_true")
    parser.add_argument("--C", type=float, default=10.0)
    parser.add_argument("--epsilon_svr", type=float, default=0.05)
    parser.add_argument("--gamma", type=str, default="scale")
    parser.add_argument("--max_iter", type=int, default=50000)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--cache_size", type=int, default=1000)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    print("="*60)
    print("SVR Baseline - 5-Fold CV (NO DATA LEAKAGE)")
    if args.linear:
        print("Using LinearSVR")
    else:
        print(f"Using SVR with kernel={args.kernel}")
    print(f"C={args.C}, epsilon={args.epsilon_svr}, max_iter={args.max_iter}")
    print("="*60)

    # ===== Load raw data =====
    df = pd.read_csv(args.dataset)
    
    arrival = np.log1p(df["Actual Transaction Arrival Rate"].values.astype(float))
    orderers = df["Orderers"].values.astype(float)
    block = df["Block Size"].values.astype(float)
    
    X_raw = np.stack([arrival, orderers, block], axis=1)
    Y_raw = df[["Throughput", "Avg Latency"]].values.astype(float)
    Y_raw = np.nan_to_num(Y_raw, nan=0.0)
    
    print(f"Dataset: {X_raw.shape[0]} samples, {X_raw.shape[1]} features")

    # ===== K-Fold CV with per-fold normalization =====
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=seed)
    
    results = []
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_raw)):
        print(f"\n{'='*40}")
        print(f"Fold {fold+1}/{args.folds}")
        print(f"{'='*40}")
        
        X_train_raw, X_val_raw = X_raw[train_idx], X_raw[val_idx]
        y_train_raw, y_val_raw = Y_raw[train_idx], Y_raw[val_idx]
        
        print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
        
        # Train model (per-fold normalization inside)
        model, metrics, _, _ = train_fold(
            X_train_raw, y_train_raw, 
            X_val_raw, y_val_raw, 
            fold, args
        )
        
        results.append(metrics)
        models.append(model)
        
        print(f"\nFold {fold+1} Results:")
        print(f"  Throughput: RMSE={metrics['RMSE_T']:.4f}, MAE={metrics['MAE_T']:.4f}, "
              f"MAPE={metrics['MAPE_T']:.2f}%, R2={metrics['R2_T']:.4f}")
        print(f"  Latency:    RMSE={metrics['RMSE_L']:.4f}, MAE={metrics['MAE_L']:.4f}, "
              f"MAPE={metrics['MAPE_L']:.2f}%, R2={metrics['R2_L']:.4f}")
    
    # ===== Aggregate results =====
    df_res = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("SVR - 5-FOLD CV RESULTS (NO LEAKAGE)")
    print("="*60)
    print("\n--- Mean ± Std ---")
    for col in df_res.columns:
        mean_val = df_res[col].mean()
        std_val = df_res[col].std()
        print(f"{col}: {mean_val:.4f} ± {std_val:.4f}")
    
    # ===== Save results =====
    os.makedirs("results_svr", exist_ok=True)
    os.makedirs("best_models_svr", exist_ok=True)
    
    df_res.to_csv("results_svr/svr_5fold_results.csv", index=False)
    
    # Save summary
    summary = []
    for col in df_res.columns:
        summary.append({
            "Model": "SVR",
            "Metric": col,
            "Mean": df_res[col].mean(),
            "Std": df_res[col].std(),
            "Min": df_res[col].min(),
            "Max": df_res[col].max()
        })
    pd.DataFrame(summary).to_csv("results_svr/svr_summary.csv", index=False)
    
    # Save best model
    best_fold = np.argmin([r['RMSE_T'] for r in results])
    joblib.dump(models[best_fold], f"best_models_svr/best_svr_model_fold{best_fold}.pkl")
    
    print("\n✓ Results saved to results_svr/")
    print("✓ Best models saved to best_models_svr/")


if __name__ == "__main__":
    main()