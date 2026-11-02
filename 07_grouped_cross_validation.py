"""
07_grouped_cross_validation.py

Leakage check for the QSAR benchmark. 02_feature_engineering.py keeps one row per
IC50 measurement, so the same molecule can appear in several rows. A random row-level
split (03, 05, 06) can then put the same molecule in both training and test data.

This script re-runs the Random Forest and the PyTorch network with 5-fold splits that
keep related rows together:
  * Molecule-grouped : all rows of one molecule_chembl_id stay in the same fold
                       (only run when the data contain repeated molecules)
  * Scaffold-grouped : all molecules sharing a Bemis-Murcko scaffold stay in the same
                       fold, so every test fold contains only unseen scaffolds
                       (acyclic molecules, which have no scaffold, are grouped by their
                       own molecule ID)

Inputs : bioactivity_data_raw.csv, bioactivity_data_preprocessed.csv, and optionally
         cv_results.csv (random-fold results from 06_cross_validation.py)
Output : grouped_cv_results.csv + a printed comparison table
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset

RAW_CSV = "bioactivity_data_raw.csv"
FEATURES_CSV = "bioactivity_data_preprocessed.csv"
RANDOM_CV_CSV = "cv_results.csv"
OUT_CSV = "grouped_cv_results.csv"
SEED = 42
N_SPLITS = 5
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3


class QSARNet(nn.Module):
    """Same architecture as 05_pytorch_comparison.py: 2048 -> 512 -> 128 -> 1, dropout 0.3."""

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_nn(X_train, y_train, seed):
    torch.manual_seed(seed)
    model = QSARNet(X_train.shape[1])
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE, shuffle=True,
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


def to_pic50(value):
    """Identical to the conversion in 02_feature_engineering.py."""
    try:
        molar = value * (10 ** -9)
        return -np.log10(molar) if molar > 0 else None
    except Exception:
        return None


def scaffold_of(mol):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return ""


def rebuild_metadata(pre):
    """Re-create the row order of 02_feature_engineering.py to get one SMILES per feature row."""
    raw = pd.read_csv(RAW_CSV)
    raw["pIC50"] = raw["standard_value"].apply(to_pic50)
    raw = raw.dropna(subset=["pIC50"])
    keep, mols = [], []
    for idx, smiles in enumerate(raw["canonical_smiles"]):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            keep.append(idx)
            mols.append(mol)
    meta = raw.iloc[keep].reset_index(drop=True)
    if len(meta) != len(pre) or not (meta["molecule_chembl_id"].values == pre["molecule_chembl_id"].values).all():
        raise SystemExit(
            "Row alignment failed: the raw file does not reproduce the preprocessed file's rows. "
            "Re-run 01_data_collection.py and 02_feature_engineering.py, then try again."
        )
    return meta, mols


def main():
    print("Loading data...")
    pre = pd.read_csv(FEATURES_CSV)
    meta, mols = rebuild_metadata(pre)
    ids = pre["molecule_chembl_id"].astype(str).values
    X = pre.drop(["molecule_chembl_id", "pIC50"], axis=1).values.astype(np.float32)
    y = pre["pIC50"].values.astype(np.float32)

    scaffolds = [scaffold_of(m) for m in mols]
    scaffold_groups = np.array([s if s else f"ACYCLIC_{i}" for s, i in zip(scaffolds, ids)])

    n_rows, n_ids = len(pre), len(set(ids))
    n_scaf = len(set(s for s in scaffolds if s))
    n_acyclic = sum(1 for s in scaffolds if not s)
    biggest = pd.Series(scaffold_groups).value_counts()
    print("\n=== Dataset diagnostics ===")
    print(f"Rows (IC50 records)           : {n_rows}")
    print(f"Unique molecule_chembl_id     : {n_ids}")
    print(f"Rows sharing a molecule ID    : {n_rows - n_ids} extra rows")
    print(f"Unique Murcko scaffolds       : {n_scaf} (+ {n_acyclic} acyclic rows)")
    print(f"Largest scaffold group        : {biggest.iloc[0]} rows ({100 * biggest.iloc[0] / n_rows:.1f}%)")

    splits = {}
    if n_ids < n_rows:
        splits["Molecule-grouped"] = ids
    else:
        print("No repeated molecule IDs: skipping the molecule-grouped split.")
    splits["Scaffold-grouped"] = scaffold_groups

    rows = []
    for name, groups in splits.items():
        gkf = GroupKFold(n_splits=N_SPLITS)
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
            assert set(groups[train_idx]).isdisjoint(groups[test_idx]), "group leaked across folds"
            print(f"\n--- {name}, fold {fold}/{N_SPLITS} (train {len(train_idx)}, test {len(test_idx)}) ---")

            rf = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
            rf.fit(X[train_idx], y[train_idx])
            r2_rf, rmse_rf = scores(y[test_idx], rf.predict(X[test_idx]))

            model = train_nn(X[train_idx], y[train_idx], SEED + fold)
            model.eval()
            with torch.no_grad():
                pred_nn = model(torch.from_numpy(X[test_idx])).numpy()
            r2_nn, rmse_nn = scores(y[test_idx], pred_nn)

            print(f"Random Forest : R2 = {r2_rf:.3f}, RMSE = {rmse_rf:.3f}")
            print(f"PyTorch NN    : R2 = {r2_nn:.3f}, RMSE = {rmse_nn:.3f}")
            rows.append({"split": name, "fold": fold, "n_test": len(test_idx),
                         "rf_r2": r2_rf, "rf_rmse": rmse_rf, "nn_r2": r2_nn, "nn_rmse": rmse_nn})
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)   # saved after every fold

    results = pd.DataFrame(rows)
    table = []
    if os.path.exists(RANDOM_CV_CSV):
        rnd = pd.read_csv(RANDOM_CV_CSV)
        rnd = rnd.assign(split="Random rows (06)")
        table.append(rnd)
    table.append(results)
    allres = pd.concat(table, ignore_index=True)

    print("\n" + "=" * 78)
    print("5-fold cross-validation, mean +/- SD (R2 and RMSE on held-out folds)")
    print("=" * 78)
    for split in allres["split"].unique():
        part = allres[allres["split"] == split]
        for label, p in (("Random Forest", "rf"), ("PyTorch NN", "nn")):
            r2, rmse = part[f"{p}_r2"], part[f"{p}_rmse"]
            print(f"{split:<18} {label:<14} R2 = {r2.mean():.3f} +/- {r2.std():.3f}   "
                  f"RMSE = {rmse.mean():.3f} +/- {rmse.std():.3f}")
        wins = int((part["rf_r2"] > part["nn_r2"]).sum())
        print(f"{'':<18} Random Forest higher in {wins} of {len(part)} folds")
    print(f"\nSaved per-fold scores to {OUT_CSV}")


if __name__ == "__main__":
    main()