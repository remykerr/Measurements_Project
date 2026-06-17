"""
Train and evaluate classifiers on the prepared road-surface dataset.

The script builds the train/test DataFrames, separates features from labels,
scales the features, and calls the shared evaluation utility for each selected
machine-learning model.
"""

from pathlib import Path
import sys

from imblearn import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from numpy import percentile
from keras.models import load_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Machine_Learning.Defect_identification.Dataset_construction import build_surface_dataset
from Machine_Learning.Defect_identification.Dataset_construction_anomaly import build_surface_dataset_anomaly
from Machine_Learning.Defect_identification.AE_utils import (
    evaluate_autoencoder,
    model_training,
    plot_prediction_histogram,
    plot_reconstruction_error_over_time,
    plot_signal_with_detected_anomalies,
    select_defect_windows,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "model_results_AE"
METADATA_COLUMNS = ["lat", "long", "v", "surface", "measurement_id", "start_time"]

training = False  # Set to True to train the model, False to load and evaluate the saved model

normal_dataset = build_surface_dataset(surface_types=["Smooth_asphalt"])
anomaly_dataset = build_surface_dataset_anomaly()

# removing non-feature columns
X_normal = normal_dataset.drop(columns=METADATA_COLUMNS)
X_anomaly = anomaly_dataset.drop(columns=METADATA_COLUMNS)

if list(X_normal.columns) != list(X_anomaly.columns):
    raise ValueError("Normal and anomaly datasets have different feature columns.")

# sanity check of dimensions
print("normal dataset shape:", normal_dataset.shape)
print("anomaly dataset shape:", anomaly_dataset.shape)

# ==================================
# SCALING INPUT FEATURES
# ==================================
X_train, X_val = train_test_split(
    X_normal,
    test_size=0.2,
    shuffle=True,
    random_state=42,
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_anomaly_scaled = scaler.transform(X_anomaly)

# ==================================
# AutoEncoder TRAINING AND EVALUATION
# ==================================

# Training
if training:
    AE = model_training(X_train_scaled, X_val_scaled)

    MODEL_PATH = OUTPUT_DIR / "autoencoder_model.keras"
    OUTPUT_DIR.mkdir(exist_ok=True)
    AE.save(MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

# Evaluation
if not training:
    AE = load_model(OUTPUT_DIR / "autoencoder_model.keras")

normal_errors, anomaly_errors = evaluate_autoencoder(AE, X_val_scaled, X_anomaly_scaled)

#threshold = np.percentile(normal_errors, 98)
threshold = normal_errors.mean() + 5 * normal_errors.std()
anomaly_results = anomaly_dataset[["surface", "measurement_id", "start_time"]].copy()
anomaly_results["reconstruction_error"] = anomaly_errors
anomaly_results["is_anomaly"] = anomaly_results["reconstruction_error"] > threshold
anomaly_results = select_defect_windows(anomaly_results)

print(f"Anomaly threshold: {threshold:.6f}")
print("Top reconstruction errors:")
print(anomaly_results.sort_values("reconstruction_error", ascending=False).head(10))
print("Detected anomalous windows:")
print(anomaly_results[anomaly_results["is_anomaly"]])
print(f"Total detected anomalies: {anomaly_results['is_anomaly'].sum()} out of {len(anomaly_results)}")
print("Selected defect windows:")
print(anomaly_results[anomaly_results["is_selected_defect"]])
print(f"Total selected defect events: {anomaly_results['is_selected_defect'].sum()}")

plot_prediction_histogram(normal_errors, anomaly_errors, threshold)
plot_reconstruction_error_over_time(anomaly_results, threshold)
plot_signal_with_detected_anomalies(anomaly_dataset, anomaly_results)


OUTPUT_DIR.mkdir(exist_ok=True)
for figure_number in plt.get_fignums():
    figure = plt.figure(figure_number)
    figure.savefig(
        OUTPUT_DIR / f"Results_{figure_number}.png",
        dpi=150,
        bbox_inches="tight",
    )

print(f"\nResults figures saved to: {OUTPUT_DIR}")
