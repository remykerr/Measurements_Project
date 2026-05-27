"""
Script to extract and plot debug data.

Parameters:
- debug_surface: surface to inspect
- debug_measurement: measurement id to inspect
- debug_window_index: FFT/PSD window to inspect
- plot_debug: if True, plots gravity correction, DFT and PSD
"""

from Dataset_construction import build_surface_dataset


DEBUG_SURFACE = "Grass"
DEBUG_MEASUREMENT = 4
DEBUG_WINDOW_INDEX = 1

train_df, test_df, debug_data = build_surface_dataset(
    plot_debug=True,
    return_debug=True,
    debug_surface=DEBUG_SURFACE,
    debug_measurement=DEBUG_MEASUREMENT,
    debug_window_index=DEBUG_WINDOW_INDEX,
)

debug_key = f"{DEBUG_SURFACE}_{DEBUG_MEASUREMENT}"
d = debug_data[debug_key]

print("Sampling frequency:", d["sampling_frequency"])
print("Window size:", d["window_size"])
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

raw = d["raw_acceleration"]
vertical = d["vertical_no_g"]
filtered = d["vertical_no_g_highpass"]

freq = d["frequency"]
dft = d["dft"]
psd = d["psd"]

before_filter = d["merged_before_speed_filter"]
after_filter = d["merged_after_speed_filter"]

print("Rows before speed filter:", len(before_filter))
print("Rows after speed filter:", len(after_filter))