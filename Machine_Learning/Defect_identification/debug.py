from pathlib import Path
import os
import sys

MPL_CONFIG_DIR = Path(__file__).resolve().parent / ".matplotlib_cache"
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_CONFIG_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Machine_Learning.Defect_identification.Dataset_construction import build_surface_dataset

DEBUG_SURFACE = "Smooth_asphalt"
DEBUG_MEASUREMENT = 6
DEBUG_WINDOW_INDEX = [30,31,32]

window_df, debug_data = build_surface_dataset(
    plot_debug=True,
    return_debug=True,
    plot_debug_acc = True,
    debug_surface=DEBUG_SURFACE,
    debug_measurement=DEBUG_MEASUREMENT,
    debug_window_index=DEBUG_WINDOW_INDEX,
)

debug_key = f"{DEBUG_SURFACE}_{DEBUG_MEASUREMENT}"
debug_entry = debug_data[debug_key]

print("Window dataset shape:", window_df.shape)
print("Window dataset columns:", list(window_df.columns))
print("Debug keys:", list(debug_data.keys()))
print("Debug window dataset shape:", debug_entry["window_dataset"].shape)
print("Debug window indices:", debug_entry["debug_window_indices"])
print(
    "Debug segment shapes:",
    {
        index: segment.shape
        for index, segment in debug_entry["debug_segments"].items()
    },
)

OUTPUT_DIR = Path(__file__).resolve().parent / "debug_results"
OUTPUT_DIR.mkdir(exist_ok=True)
for figure_number in plt.get_fignums():
    figure = plt.figure(figure_number)
    figure.savefig(
        OUTPUT_DIR / f"debug_plot_{figure_number}.png",
        dpi=150,
        bbox_inches="tight",
    )

print(f"Debug plots saved to: {OUTPUT_DIR}")
