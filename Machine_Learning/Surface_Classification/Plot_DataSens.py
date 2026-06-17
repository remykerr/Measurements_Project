from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent / "model_results_KNN_datasense"
RESULTS_FILE = OUTPUT_DIR / "model_results_KNN.csv"


results = pd.read_csv(RESULTS_FILE)
metric_columns = ["f1_score", "accuracy", "recall", "precision"]

print(results)
print("\nMetric stability summary:")
print(results[metric_columns].agg(["mean", "std", "min", "max"]))


plt.figure(figsize=(9, 5))
for metric in metric_columns:
    plt.plot(
        results["test_id"],
        results[metric],
        marker="o",
        linewidth=1.8,
        label=metric,
    )
    plt.axhline(
        results[metric].mean(),
        linestyle="--",
        linewidth=1,
        alpha=0.35,
    )

plt.xlabel("Held-out measurement id")
plt.ylabel("Score")
plt.title("Classification stability across held-out measurements")
plt.xticks(results["test_id"])
plt.ylim(0.0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "metric_stability_lines.png", dpi=150)


plt.figure(figsize=(9, 5))
bar_width = 0.35
x_positions = results["test_id"].to_numpy()

plt.bar(
    x_positions - bar_width / 2,
    results["f1_score"],
    width=bar_width,
    label="f1_score",
)
plt.bar(
    x_positions + bar_width / 2,
    results["accuracy"],
    width=bar_width,
    label="accuracy",
)
plt.axhline(
    results["f1_score"].mean(),
    color="tab:blue",
    linestyle="--",
    linewidth=1.5,
    alpha=0.6,
    label="mean f1_score",
)
plt.axhline(
    results["accuracy"].mean(),
    color="tab:orange",
    linestyle="--",
    linewidth=1.5,
    alpha=0.6,
    label="mean accuracy",
)

plt.xlabel("Held-out measurement id")
plt.ylabel("Score")
plt.title("F1 and accuracy by held-out measurement")
plt.xticks(results["test_id"])
plt.ylim(0.0, 1.05)
plt.grid(True, axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "metric_stability_bars.png", dpi=150)

print(f"\nStability plots saved to: {OUTPUT_DIR}")
