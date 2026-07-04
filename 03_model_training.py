import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

print("Loading preprocessed data...")
# Read preprocessed data from the previous phase
dataset = pd.read_csv('bioactivity_data_preprocessed.csv')

# Separate features (X) and target values (Y)
# X: All fingerprint columns (chemical features)
X = dataset.drop(['molecule_chembl_id', 'pIC50'], axis=1)
# Y: Target value to predict (bioactivity)
Y = dataset['pIC50']

print(f"Features shape (X): {X.shape}")
print(f"Target shape (Y): {Y.shape}")

# Split data into Train (80%) and Test (20%) sets
print("Splitting data into Train and Test sets...")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Instantiate and train the Random Forest model
print("Training the Random Forest model (this might take a few minutes)...")
# Use 100 decision trees to achieve high accuracy
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

# Predict bioactivity on the unseen test set
print("Making predictions on the Test set...")
Y_pred = model.predict(X_test)

# Evaluate model performance and accuracy
r2 = r2_score(Y_test, Y_pred)
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))

print("-" * 30)
print("Model Performance:")
print(f"R-squared (R2): {r2:.3f}")
print(f"RMSE (Error): {rmse:.3f}")
print("-" * 30)

# Save experimental vs predicted results for visualization
results_df = pd.DataFrame({'Experimental_pIC50': Y_test, 'Predicted_pIC50': Y_pred})
results_df.to_csv('model_predictions.csv', index=False)
print("Predictions saved as 'model_predictions.csv'")
