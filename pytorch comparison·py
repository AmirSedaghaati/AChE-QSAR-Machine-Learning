import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

print("Loading preprocessed data (same file used by the Random Forest model)...")
dataset = pd.read_csv('bioactivity_data_preprocessed.csv')

X = dataset.drop(['molecule_chembl_id', 'pIC50'], axis=1)
Y = dataset['pIC50']

# Same split, same random_state, as 03_model_training.py - this is what makes
# the comparison against the Random Forest baseline fair (same train/test rows).
print("Splitting data into Train and Test sets (identical split to the Random Forest model)...")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
Y_train_t = torch.tensor(Y_train.values, dtype=torch.float32).view(-1, 1)
X_test_t = torch.tensor(X_test.values, dtype=torch.float32)
Y_test_t = torch.tensor(Y_test.values, dtype=torch.float32).view(-1, 1)

train_ds = TensorDataset(X_train_t, Y_train_t)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)


class QSARNet(nn.Module):
    """Simple feedforward network: 2048-bit fingerprint -> single pIC50 value."""

    def __init__(self, input_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


torch.manual_seed(42)
model = QSARNet(input_dim=X_train_t.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

n_epochs = 100
train_losses = []
val_losses = []

print(f"Training for {n_epochs} epochs...")
for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    epoch_loss /= len(train_ds)
    train_losses.append(epoch_loss)

    model.eval()
    with torch.no_grad():
        val_pred = model(X_test_t)
        val_loss = criterion(val_pred, Y_test_t).item()
    val_losses.append(val_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{n_epochs} - train loss: {epoch_loss:.4f} - val loss: {val_loss:.4f}")

# --- Evaluate on the same held-out test set used for the Random Forest model ---
model.eval()
with torch.no_grad():
    Y_pred_nn = model(X_test_t).numpy().flatten()

r2_nn = r2_score(Y_test, Y_pred_nn)
rmse_nn = np.sqrt(mean_squared_error(Y_test, Y_pred_nn))

print("-" * 30)
print("PyTorch Neural Network Performance:")
print(f"R-squared (R2): {r2_nn:.3f}")
print(f"RMSE (Error): {rmse_nn:.3f}")
print("-" * 30)

# --- Direct comparison against the existing Random Forest result ---
# Load the Random Forest predictions saved by 03_model_training.py so the
# comparison is against the actual saved run, not a hardcoded number.
rf_results = pd.read_csv('model_predictions.csv')
r2_rf = r2_score(rf_results['Experimental_pIC50'], rf_results['Predicted_pIC50'])
rmse_rf = np.sqrt(mean_squared_error(rf_results['Experimental_pIC50'], rf_results['Predicted_pIC50']))

print("\nComparison: PyTorch NN vs. Random Forest (same test set)")
print(f"{'Model':<20}{'R2':>10}{'RMSE':>10}")
print(f"{'Random Forest':<20}{r2_rf:>10.3f}{rmse_rf:>10.3f}")
print(f"{'PyTorch NN':<20}{r2_nn:>10.3f}{rmse_nn:>10.3f}")

if r2_nn > r2_rf:
    print("\nThe neural network outperformed the Random Forest baseline on this test set.")
else:
    print(
        "\nThe Random Forest baseline outperformed (or matched) the neural network here. "
        "This is a legitimate, expected result, not a failure: neural nets typically need "
        "more than ~8,800 training compounds to reliably beat tree-based models on tabular "
        "fingerprint data, and it's an honest, defensible finding to report either way."
    )

# --- Visualizations ---
plt.figure(figsize=(6, 6))
plt.scatter(Y_test, Y_pred_nn, alpha=0.5, s=15)
lims = [min(Y_test.min(), Y_pred_nn.min()), max(Y_test.max(), Y_pred_nn.max())]
plt.plot(lims, lims, 'k--', linewidth=1)
plt.xlabel('Experimental pIC50')
plt.ylabel('Predicted pIC50 (PyTorch NN)')
plt.title(f'PyTorch NN: Predicted vs. Experimental pIC50 (R2={r2_nn:.3f})')
plt.tight_layout()
plt.savefig('pytorch_scatter_plot.png', dpi=150)
plt.close()

plt.figure(figsize=(7, 5))
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Validation Loss Curve')
plt.legend()
plt.tight_layout()
plt.savefig('pytorch_loss_curve.png', dpi=150)
plt.close()

print("\nSaved pytorch_scatter_plot.png and pytorch_loss_curve.png")

# Save NN predictions in the same format as the Random Forest results, for reuse.
nn_results_df = pd.DataFrame({'Experimental_pIC50': Y_test.values, 'Predicted_pIC50': Y_pred_nn})
nn_results_df.to_csv('pytorch_predictions.csv', index=False)
print("Saved pytorch_predictions.csv")
