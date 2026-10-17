"""
06_cross_validation.py

5-fold cross-validation of the Random Forest and the PyTorch feedforward
network on identical folds. Purpose: check whether the single-split gap
(R2 0.722 vs 0.710) is larger than the fold-to-fold variation.

Input : bioactivity_data_preprocessed.csv (written by 02_feature_engineering.py)
Output: cv_results.csv (per-fold scores) + a printed mean +/- SD summary
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

INPUT_CSV = "bioactivity_data_preprocessed.csv"
SEED = 42
N_SPLITS = 5
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3


class QSARNet(nn.Module):
    """Same architecture as 05_pytorch_comparison.py:
    2048 -> 512 -> 128 -> 1, ReLU, dropout 0.3."""

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_nn(X_train, y_train, fold):
    torch.manual_seed(SEED + fold)
    model = QSARNet(X_train.shape[1])
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()
    return model


def scores(y_true, y_pred):
    return r2_score(y_true, y_pred), float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    print("Loading preprocessed data...")
    data = pd.read_csv(INPUT_CSV)
    X = data.drop(["molecule_chembl_id", "pIC50"], axis=1).values.astype(np.float32)
    y = data["pIC50"].values.astype(np.float32)
    print(f"Features: {X.shape}, target: {y.shape}")

    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    rows = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X), start=1):
        print(f"\n--- Fold {fold}/{N_SPLITS} ---")

        rf = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
        rf.fit(X[train_idx], y[train_idx])
        r2_rf, rmse_rf = scores(y[test_idx], rf.predict(X[test_idx]))

        nn_model = train_nn(X[train_idx], y[train_idx], fold)
        nn_model.eval()
        with torch.no_grad():
            pred_nn = nn_model(torch.from_numpy(X[test_idx])).numpy()
        r2_nn, rmse_nn = scores(y[test_idx], pred_nn)

        print(f"Random Forest : R2 = {r2_rf:.3f}, RMSE = {rmse_rf:.3f}")
        print(f"PyTorch NN    : R2 = {r2_nn:.3f}, RMSE = {rmse_nn:.3f}")
        rows.append({
            "fold": fold,
            "rf_r2": r2_rf, "rf_rmse": rmse_rf,
            "nn_r2": r2_nn, "nn_rmse": rmse_nn,
        })

    results = pd.DataFrame(rows)
    results.to_csv("cv_results.csv", index=False)

    print("\n" + "=" * 50)
    print(f"{N_SPLITS}-fold cross-validation, mean +/- SD")
    print("=" * 50)
    for label, prefix in [("Random Forest", "rf"), ("PyTorch NN", "nn")]:
        r2 = results[f"{prefix}_r2"]
        rmse = results[f"{prefix}_rmse"]
        print(f"{label:<14} R2 = {r2.mean():.3f} +/- {r2.std():.3f}   "
              f"RMSE = {rmse.mean():.3f} +/- {rmse.std():.3f}")
    rf_wins = int((results["rf_r2"] > results["nn_r2"]).sum())
    print(f"\nRandom Forest had the higher R2 in {rf_wins} of {N_SPLITS} folds.")
    print("Saved per-fold scores to cv_results.csv")


if __name__ == "__main__":
    main()