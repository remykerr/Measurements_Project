"""
Compute stationarity-style statistical checks on 1 s surface windows.

The script builds the full window-level dataset, checks each measurement inside
its own surface class, prints a compact summary, and writes the summary to CSV.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from Dataset_construction_statisticalChecks import build_surface_dataset_2


FEATURES_TO_CHECK = ("az_avg", "az_std", "az_rms", "spec_energy")
MEAN_DEV_THRESHOLD = 2.0
ACF_ERROR_THRESHOLD = 1.0
OUTPUT_CSV = Path(__file__).resolve().parent / "stationarity_summary.csv"


def normalized_deviation(values):
    values = pd.Series(values, dtype="float64")
    sigma = values.std()

    if sigma == 0 or np.isnan(sigma):
        return pd.Series(np.zeros(len(values)), index=values.index)

    return (values - values.mean()).abs() / sigma


def summarize_measurement(surface, measurement_id, measurement_df):
    measurement_df = measurement_df.sort_values("t")
    summary = {
        "surface": surface,
        "measurement_id": measurement_id,
        "n_windows": len(measurement_df),
        "acf_stable_percentage": (
            measurement_df["acf_error"] < ACF_ERROR_THRESHOLD
        ).mean() * 100,
        "acf_error_mean": measurement_df["acf_error"].mean(),
    }

    for feature in FEATURES_TO_CHECK:
        deviation = normalized_deviation(measurement_df[feature])
        summary[f"{feature}_mean"] = measurement_df[feature].mean()
        summary[f"{feature}_std"] = measurement_df[feature].std()
        summary[f"{feature}_mean_deviation"] = deviation.mean()
        summary[f"{feature}_stable_percentage"] = (
            deviation < MEAN_DEV_THRESHOLD
        ).mean() * 100

    return summary


def build_stationarity_summary(full_window_dataset):
    rows = []

    grouped = full_window_dataset.groupby(["srf", "measurement_id"], sort=True)
    for (surface, measurement_id), measurement_df in grouped:
        if measurement_df.empty:
            continue
        rows.append(summarize_measurement(surface, measurement_id, measurement_df))

    return pd.DataFrame(rows)


def main():
    full_window_dataset = build_surface_dataset_2()

    print("Dataset shape:", full_window_dataset.shape)
    print("Dataset columns:", list(full_window_dataset.columns))

    summary = build_stationarity_summary(full_window_dataset)
    summary.to_csv(OUTPUT_CSV, index=False)

    print("\nStationarity summary:")
    print(summary.to_string(index=False))
    print(f"\nSummary saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
