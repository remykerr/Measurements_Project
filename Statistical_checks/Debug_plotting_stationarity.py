"""
Debug plots for approximate weak-stationarity checks.

Set SURFACE_TO_PLOT at the beginning of the file. Optionally set
MEASUREMENT_TO_PLOT to inspect a single measurement, for example "Unpaved_1".
If MEASUREMENT_TO_PLOT is None, all measurements of the selected surface are
included in the summary plots and the first one is used for detailed plots.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Dataset_construction_statisticalChecks import build_surface_dataset_2


# =========================
# USER SETTINGS
# =========================

SURFACE_TO_PLOT = "Cobble"
MEASUREMENT_TO_PLOT = None  # Example: "Unpaved_1"; use None for first available

FEATURES_TO_PLOT = ["az_avg", "az_rms", "az_std", "spec_energy"]
MEAN_DEV_THRESHOLD = 2.0
ACF_ERROR_THRESHOLD = 1.0

SAVE_FIGURES = True
SHOW_FIGURES = False

OUTPUT_DIR = Path(__file__).resolve().parent / "debug_stationarity_plots"


def time_seconds(data):
    if pd.api.types.is_timedelta64_dtype(data):
        return data.dt.total_seconds()
    return data


def normalized_deviation(values):
    values = pd.Series(values).astype(float)
    sigma = values.std()
    if sigma == 0 or np.isnan(sigma):
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - values.mean()).abs() / sigma


def save_or_show(fig, filename):
    fig.tight_layout()

    if SAVE_FIGURES:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_DIR / filename, dpi=150)

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def build_stationarity_summary(surface_df):
    rows = []

    for measurement_id, group in surface_df.groupby("measurement_id", sort=False):
        group = group.sort_values("t")
        row = {
            "measurement_id": measurement_id,
            "n_windows": len(group),
            "acf_stable_percentage": (group["acf_error"] < ACF_ERROR_THRESHOLD).mean() * 100,
        }

        for feature in FEATURES_TO_PLOT:
            dev = normalized_deviation(group[feature])
            row[f"{feature}_stable_percentage"] = (
                dev < MEAN_DEV_THRESHOLD
            ).mean() * 100

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary["measurement_number"] = (
        summary["measurement_id"]
        .str.extract(r"_(\d+)$", expand=False)
        .astype(int)
    )
    return summary.sort_values("measurement_number").drop(columns="measurement_number")


def plot_feature_timeseries(measurement_df, measurement_id):
    t = time_seconds(measurement_df["t"])

    for feature in FEATURES_TO_PLOT:
        values = measurement_df[feature]
        mu = values.mean()
        sigma = values.std()

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, values, marker="o", linewidth=1.2, label=feature)
        ax.axhline(mu, color="tab:red", linestyle="--", label="global mean")

        if sigma > 0:
            ax.axhline(mu + 2 * sigma, color="tab:gray", linestyle=":", label="+/- 2 std")
            ax.axhline(mu - 2 * sigma, color="tab:gray", linestyle=":")

        ax.set_title(f"{measurement_id} - {feature} over 1 s windows")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(feature)
        ax.grid(True)
        ax.legend()
        save_or_show(fig, f"{measurement_id}_{feature}_timeseries.png")


def plot_normalized_deviation(measurement_df, measurement_id):
    t = time_seconds(measurement_df["t"])

    fig, ax = plt.subplots(figsize=(10, 4))

    for feature in FEATURES_TO_PLOT:
        dev = normalized_deviation(measurement_df[feature])
        ax.plot(t, dev, marker="o", linewidth=1.1, label=feature)

    ax.axhline(
        MEAN_DEV_THRESHOLD,
        color="tab:red",
        linestyle="--",
        label=f"threshold = {MEAN_DEV_THRESHOLD:g}",
    )
    ax.set_title(f"{measurement_id} - normalized deviation from global statistics")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("|feature_i - mean| / std")
    ax.grid(True)
    ax.legend()
    save_or_show(fig, f"{measurement_id}_normalized_deviation.png")


def plot_acf_error(measurement_df, measurement_id):
    t = time_seconds(measurement_df["t"])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, measurement_df["acf_error"], marker="o", linewidth=1.2)
    ax.axhline(
        ACF_ERROR_THRESHOLD,
        color="tab:red",
        linestyle="--",
        label=f"ACF threshold = {ACF_ERROR_THRESHOLD:g}",
    )
    ax.set_title(f"{measurement_id} - autocorrelation stability over 1 s windows")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("ACF error")
    ax.grid(True)
    ax.legend()
    save_or_show(fig, f"{measurement_id}_acf_error.png")


def plot_surface_summary(summary, surface):
    labels = summary["measurement_id"]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(
        x - width / 2,
        summary["az_avg_stable_percentage"],
        width,
        label="mean stable windows",
    )
    ax.bar(
        x + width / 2,
        summary["acf_stable_percentage"],
        width,
        label="ACF stable windows",
    )
    ax.axhline(85, color="tab:gray", linestyle=":", label="85% reference")
    ax.set_title(f"{surface} - stationarity summary by measurement")
    ax.set_xlabel("Measurement")
    ax.set_ylabel("Stable windows [%]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y")
    ax.legend()
    save_or_show(fig, f"{surface}_stationarity_summary.png")


def main():
    full_window_dataset = build_surface_dataset_2()
    surface_df = full_window_dataset[
        full_window_dataset["srf"].str.lower() == SURFACE_TO_PLOT.lower()
    ].copy()

    if surface_df.empty:
        available = ", ".join(full_window_dataset["srf"].drop_duplicates())
        raise ValueError(
            f"Surface {SURFACE_TO_PLOT!r} not found. Available surfaces: {available}"
        )

    summary = build_stationarity_summary(surface_df)
    print("\nStationarity summary:")
    print(summary)

    plot_surface_summary(summary, SURFACE_TO_PLOT)

    measurement_id = MEASUREMENT_TO_PLOT or summary.iloc[0]["measurement_id"]
    measurement_df = surface_df[surface_df["measurement_id"] == measurement_id].copy()

    if measurement_df.empty:
        available = ", ".join(summary["measurement_id"])
        raise ValueError(
            f"Measurement {measurement_id!r} not found for {SURFACE_TO_PLOT}. "
            f"Available measurements: {available}"
        )

    measurement_df = measurement_df.sort_values("t")
    plot_feature_timeseries(measurement_df, measurement_id)
    plot_normalized_deviation(measurement_df, measurement_id)
    plot_acf_error(measurement_df, measurement_id)

    if SAVE_FIGURES:
        print(f"\nFigures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
