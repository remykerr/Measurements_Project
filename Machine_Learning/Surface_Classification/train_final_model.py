"""
Train the final road-surface classifier on all available measurements.

This script is separate from test_algorithms.py: metrics and robustness checks
belong there, while this file fits the final model to deploy/use later.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn import neighbors
from sklearn.preprocessing import StandardScaler

try:
    from .Dataset_construction import build_surface_dataset
except ImportError:
    from Dataset_construction import build_surface_dataset


OUTPUT_DIR = Path(__file__).resolve().parent / "final_model"
OUTPUT_DIR.mkdir(exist_ok=True)


dataset_parts = []
for test_measurement in range(1, 8):
    train_df, test_df, _ = build_surface_dataset(test_measurements=(test_measurement,))
    dataset_parts.extend([train_df, test_df])

full_df = (
    pd.concat(dataset_parts, ignore_index=True)
    .drop_duplicates()
    .reset_index(drop=True)
)

if full_df.empty:
    raise ValueError("No data available for final training.")

X = full_df.drop(columns=["srf"])
y = full_df["srf"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = neighbors.KNeighborsClassifier(n_neighbors=5, weights="uniform")
model.fit(X_scaled, y)

joblib.dump(model, OUTPUT_DIR / "knn_surface_classifier.joblib")
joblib.dump(scaler, OUTPUT_DIR / "surface_scaler.joblib")
joblib.dump(list(X.columns), OUTPUT_DIR / "feature_columns.joblib")

print("Final KNN surface classifier trained.")
print(f"Training samples: {len(full_df)}")
print(f"Classes: {sorted(y.unique())}")
print(f"Model artifacts saved to: {OUTPUT_DIR}")
