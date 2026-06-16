"""
Construction of the dataset for defect identification.

Each row of the returned dataset is a 1-second window of corrected vertical
acceleration. The GPS position closest to the window start time is attached to
the same row.
"""

from pathlib import Path
import re
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Machine_Learning.Surface_Classification.gravity_correction import correct_gravity


BASE_DIR = Path(__file__).resolve().parents[1]


def discover_measurement_ids(clean_measurements_dir, surface):
    pattern = re.compile(rf"^{re.escape(surface)}_(\d+)$", re.IGNORECASE)
    measurement_ids = []
    if not clean_measurements_dir.exists():
        return measurement_ids

    for child in clean_measurements_dir.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if match:
            measurement_ids.append(int(match.group(1)))
    return sorted(measurement_ids)


def find_column(data, prefix):
    prefix = prefix.lower()
    for column in data.columns:
        if column.strip().lower().startswith(prefix):
            return column
    available = ", ".join(data.columns)
    raise KeyError(
        f"Expected a column starting with {prefix}. Available columns: {available}"
    )


def estimate_sampling_frequency(time_values):
    time_values = np.asarray(time_values, dtype=float)
    time_steps = np.diff(time_values)
    time_steps = time_steps[time_steps > 0]
    if len(time_steps) == 0:
        raise ValueError("Cannot estimate sampling frequency from non-increasing timestamps.")
    return 1.0 / np.median(time_steps)


def hp_filter(signal, fs, fc=0.5, order=4):
    b, a = butter(order, fc / (fs / 2.0), btype="high")
    return filtfilt(b, a, signal)


def show_plot():
    if "agg" not in plt.get_backend().lower():
        plt.show()


def normalize_debug_window_indices(debug_window_index, n_windows):
    if isinstance(debug_window_index, (list, tuple, set, np.ndarray)):
        requested_indices = list(debug_window_index)
    else:
        requested_indices = [debug_window_index]

    safe_indices = []
    for index in requested_indices:
        safe_index = int(np.clip(index, 0, n_windows - 1))
        if safe_index not in safe_indices:
            safe_indices.append(safe_index)
    return safe_indices


def build_surface_dataset(
    base_dir=BASE_DIR,
    surface_types=("Smooth_asphalt", "Rough_asphalt"),
    measurement_ids=None,
    fsamp=None,
    stationary_seconds=3.0,
    min_speed=1.0,
    plot_debug_acc=False,
    plot_debug=False,
    return_debug=False,
    debug_surface=None,
    debug_measurement=None,
    debug_window_index=0,
):
    """
    Build a windowed dataset for defect identification.

    Returns:
        full_window_dataset

    If return_debug=True:
        full_window_dataset, debug_data

    The returned DataFrame has this structure:
        surface, measurement_id, start_time, az_0 ... az_N, lat, long, v
    """
    base_dir = Path(base_dir)
    clean_measurements_dir = base_dir / "Clean_measurements_ML"

    full_window_dataset = pd.DataFrame()
    debug_data = {}

    requested_measurement_ids = (
        None if measurement_ids is None else set(measurement_ids)
    )

    for surface in surface_types:
        available_ids = discover_measurement_ids(clean_measurements_dir, surface)
        ids_to_process = (
            available_ids
            if requested_measurement_ids is None
            else sorted(requested_measurement_ids)
        )

        for measurement_id in ids_to_process:
            collect_debug = (
                plot_debug
                or return_debug
                or debug_surface is not None
                or debug_measurement is not None
            )
            collect_debug = collect_debug and (
                debug_surface is None or surface.lower() == debug_surface.lower()
            )
            collect_debug = collect_debug and (
                debug_measurement is None or measurement_id == debug_measurement
            )

            measure_dir = clean_measurements_dir / f"{surface}_{measurement_id}"
            accel_file = measure_dir / "Accelerometer.csv"
            gps_file = measure_dir / "Location.csv"

            if not accel_file.exists() or not gps_file.exists():
                warnings.warn(f"Skipping missing measurement: {measure_dir}")
                continue

            accel_data = pd.read_csv(accel_file)
            gps_data = pd.read_csv(gps_file)

            if plot_debug_acc and collect_debug:
                plt.figure()
                plt.plot(
                    accel_data["Time (s)"],
                    accel_data["Z (m/s^2)"],
                    label="Vertical Acceleration (g included)",
                )
                plt.xlabel("Time [s]")
                plt.ylabel("Acc [m/s^2]")
                plt.title(f"{surface}_{measurement_id} raw vertical acceleration")
                plt.legend()
                show_plot()

            accel_corrected, _ = correct_gravity(
                accel_data,
                stationary_seconds=stationary_seconds,
                expected_g=9.81,
            )
            vertical_no_g = accel_corrected["a_vertical_no_g"]
            vertical_no_g_raw = vertical_no_g.copy()

            current_fsamp = (
                estimate_sampling_frequency(accel_data["Time (s)"])
                if fsamp is None
                else fsamp
            )
            if collect_debug:
                print(f"{surface}_{measurement_id}: sampling frequency = {current_fsamp:.2f} Hz")

            vertical_filtered = hp_filter(vertical_no_g, fs=current_fsamp)

            if plot_debug and collect_debug:
                plt.figure()
                plt.plot(
                    accel_data["Time (s)"],
                    vertical_no_g_raw,
                    label="Vertical Acceleration (g removed)",
                )
                plt.plot(
                    accel_data["Time (s)"],
                    accel_data["Z (m/s^2)"],
                    label="Vertical Acceleration (g included)",
                )
                plt.xlabel("Time [s]")
                plt.ylabel("Acc [m/s^2]")
                plt.title(f"{surface}_{measurement_id} vertical acceleration comparison")
                plt.legend()
                plt.grid(True)
                show_plot()

                plt.figure()
                plt.plot(
                    accel_data["Time (s)"],
                    vertical_filtered,
                    label="Vertical Acceleration (g removed + high-pass)",
                )
                plt.xlabel("Time [s]")
                plt.ylabel("Acc [m/s^2]")
                plt.title(f"{surface}_{measurement_id} vertical acceleration filtered")
                plt.legend()
                plt.grid(True)
                show_plot()

            accel_z = pd.DataFrame(
                {
                    "t": pd.to_timedelta(accel_data["Time (s)"], unit="s"),
                    "az": vertical_filtered,
                }
            ).set_index("t")

            latitude_col = find_column(gps_data, "latitude")
            longitude_col = find_column(gps_data, "longitude")
            gps = pd.DataFrame(
                {
                    "t": pd.to_timedelta(gps_data["Time (s)"], unit="s"),
                    "lat": gps_data[latitude_col],
                    "long": gps_data[longitude_col],
                    "v": gps_data["Velocity (m/s)"],
                }
            )

            window_size = int(round(current_fsamp))
            signal = accel_z["az"].to_numpy()

            segments = []
            segment_times = []
            for start in range(0, len(signal) - window_size + 1, window_size):
                segments.append(signal[start : start + window_size])
                segment_times.append(accel_z.index[start])

            if len(segments) == 0:
                warnings.warn(f"Skipping measurement with no full windows: {measure_dir}")
                continue

            segments = np.asarray(segments)
            window_df = pd.DataFrame(
                segments,
                columns=[
                    f"az_{sample_index}"
                    for sample_index in range(segments.shape[1])
                ],
            )
            window_df.insert(0, "start_time", segment_times)
            window_df.insert(0, "measurement_id", measurement_id)
            window_df.insert(0, "surface", surface)

            window_df = pd.merge_asof(
                window_df.sort_values("start_time"),
                gps.sort_values("t"),
                left_on="start_time",
                right_on="t",
                direction="nearest",
            )
            window_df = window_df.drop(columns=["t"])
            window_df = window_df[window_df["v"] > min_speed].reset_index(drop=True)

            full_window_dataset = pd.concat(
                [full_window_dataset, window_df],
                ignore_index=True,
            )

            if collect_debug:
                debug_window_indices_safe = normalize_debug_window_indices(
                    debug_window_index,
                    len(segments),
                )
                debug_key = f"{surface}_{measurement_id}"
                debug_data[debug_key] = {
                    "surface": surface,
                    "measurement_id": measurement_id,
                    "sampling_frequency": current_fsamp,
                    "window_size": window_size,
                    "raw_acceleration": accel_data.copy(),
                    "vertical_no_g": pd.Series(
                        vertical_no_g_raw,
                        name="a_vertical_no_g",
                    ),
                    "vertical_no_g_highpass": pd.Series(
                        vertical_filtered,
                        name="a_vertical_no_g_highpass",
                    ),
                    "window_dataset": window_df.copy(),
                    "debug_window_indices": debug_window_indices_safe,
                    "debug_segments": {
                        index: segments[index].copy()
                        for index in debug_window_indices_safe
                    },
                }

            if plot_debug and collect_debug:
                print(
                    f"{surface}_{measurement_id}: created {len(window_df)} windows "
                    f"of size {window_size} samples "
                    f"({window_size / current_fsamp:.2f} seconds)"
                )

                plt.figure()
                for index in debug_data[f"{surface}_{measurement_id}"]["debug_window_indices"]:
                    debug_segment = segments[index]
                    debug_time = np.arange(len(debug_segment)) / current_fsamp
                    plt.plot(
                        debug_time,
                        debug_segment,
                        label=f"Window {index}",
                    )
                    plt.scatter(
                        debug_time,
                        debug_segment,
                        s=12,
                    )
                plt.xlabel("Time [s]")
                plt.ylabel("Acc [m/s^2]")
                plt.title(f"{surface}_{measurement_id} selected windows: vertical acceleration")
                plt.legend()
                plt.grid(True)
                show_plot()

    if return_debug:
        return full_window_dataset, debug_data

    return full_window_dataset
