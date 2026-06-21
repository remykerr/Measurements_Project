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


OUTPUT_DIR = Path(__file__).resolve().parent / "final_models"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_VARIANTS = {
    "noMedium": ("Rough_asphalt", "Smooth_asphalt", "cobble", "Grass", "Unpaved"),
    "noUnpaved": ("Rough_asphalt", "Smooth_asphalt", "Medium_asphalt", "cobble", "Grass"),
    "noMediumnoUnpaved": ("Rough_asphalt", "Smooth_asphalt", "cobble", "Grass"),
    "3classes": ("Rough_asphalt", "Smooth_asphalt", "Medium_asphalt"),
}


def build_final_training_frame(surface_types):
    dataset_parts = []
    for test_measurement in range(1, 8):
        train_df, test_df, _ = build_surface_dataset(
            surface_types=surface_types,
            test_measurements=(test_measurement,),
        )
        dataset_parts.extend([train_df, test_df])

    full_df = (
        pd.concat(dataset_parts, ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )
    if full_df.empty:
        raise ValueError(f"No data available for surfaces: {surface_types}")
    return full_df


def train_variant(variant_name, surface_types):
    variant_dir = OUTPUT_DIR / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    full_df = build_final_training_frame(surface_types)
    x = full_df.drop(columns=["srf"])
    y = full_df["srf"]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    model = neighbors.KNeighborsClassifier(n_neighbors=5, weights="uniform")
    model.fit(x_scaled, y)

    joblib.dump(model, variant_dir / "knn_surface_classifier.joblib")
    joblib.dump(scaler, variant_dir / "surface_scaler.joblib")
    joblib.dump(list(x.columns), variant_dir / "feature_columns.joblib")
    joblib.dump(
        {
            "variant": variant_name,
            "surface_types": tuple(surface_types),
            "classes": tuple(sorted(y.unique())),
            "training_samples": len(full_df),
        },
        variant_dir / "model_metadata.joblib",
    )

    print(f"\nFinal KNN surface classifier trained: {variant_name}")
    print(f"Training samples: {len(full_df)}")
    print(f"Classes: {sorted(y.unique())}")
    print(f"Model artifacts saved to: {variant_dir}")


for variant_name, surface_types in MODEL_VARIANTS.items():
    train_variant(variant_name, surface_types)
