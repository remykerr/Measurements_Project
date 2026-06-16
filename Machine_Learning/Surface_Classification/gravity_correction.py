"""
Gravity correction utilities for accelerometer measurements.

Use this when the accelerometer signal includes the constant gravity component.
The functions estimate the gravity vector from an initial stationary interval
and remove it from the full signal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


COL_T = "Time (s)"
RAW_COL_X = "Acceleration x (m/s^2)"
RAW_COL_Y = "Acceleration y (m/s^2)"
RAW_COL_Z = "Acceleration z (m/s^2)"
STD_COL_X = "X (m/s^2)"
STD_COL_Y = "Y (m/s^2)"
STD_COL_Z = "Z (m/s^2)"


def find_accelerometer_columns(data: pd.DataFrame) -> tuple[str, str, str]:
    raw_columns = (RAW_COL_X, RAW_COL_Y, RAW_COL_Z)
    standard_columns = (STD_COL_X, STD_COL_Y, STD_COL_Z)

    if all(column in data.columns for column in raw_columns):
        return raw_columns
    if all(column in data.columns for column in standard_columns):
        return standard_columns

    raise KeyError(
        "Expected accelerometer columns either "
        f"{raw_columns} or {standard_columns}. "
        f"Available columns: {list(data.columns)}"
    )


def estimate_fs(data: pd.DataFrame, time_col: str = COL_T) -> float:
    t = data[time_col].to_numpy(dtype=float)
    return 1.0 / np.median(np.diff(t))


def estimate_gravity_vector(
    data: pd.DataFrame,
    stationary_seconds: float = 3.0,
    time_col: str = COL_T,
) -> np.ndarray:
    """
    Estimate the gravity vector in phone coordinates.

    Assumption: the first stationary_seconds of the recording are almost still,
    so their mean acceleration is mostly gravity.
    """
    col_x, col_y, col_z = find_accelerometer_columns(data)
    fs = estimate_fs(data, time_col=time_col)
    n_samples = max(int(stationary_seconds * fs), 1)
    stationary = data[[col_x, col_y, col_z]].iloc[:n_samples]
    return stationary.mean().to_numpy(dtype=float)


def correct_gravity(
    data: pd.DataFrame,
    stationary_seconds: float = 3.0,
    expected_g: float | None = 9.81,
    time_col: str = COL_T,
) -> tuple[pd.DataFrame, dict[str, np.ndarray | float]]:
    """
    Remove gravity from a 3-axis accelerometer signal.

    Returns:
      corrected_data:
        original data plus:
        - ax_no_g, ay_no_g, az_no_g: phone-axis acceleration with gravity removed
        - a_vertical_no_g: acceleration along the estimated vertical direction
          with the constant gravity component removed
      info:
        gravity_vector, gravity_norm, vertical_unit_vector
    """
    col_x, col_y, col_z = find_accelerometer_columns(data)
    acceleration = data[[col_x, col_y, col_z]].to_numpy(dtype=float)

    gravity_vector = estimate_gravity_vector(
        data,
        stationary_seconds=stationary_seconds,
        time_col=time_col,
    )
    gravity_norm = float(np.linalg.norm(gravity_vector))
    if gravity_norm == 0:
        raise ValueError("Cannot estimate gravity direction: gravity norm is zero.")

    vertical_unit_vector = gravity_vector / gravity_norm
    gravity_to_remove = gravity_vector
    if expected_g is not None:
        gravity_to_remove = vertical_unit_vector * expected_g

    linear_acceleration = acceleration - gravity_to_remove
    vertical_acceleration = acceleration @ vertical_unit_vector
    vertical_without_g = vertical_acceleration - np.linalg.norm(gravity_to_remove)

    corrected = data.copy()
    corrected["ax_no_g"] = linear_acceleration[:, 0]
    corrected["ay_no_g"] = linear_acceleration[:, 1]
    corrected["az_no_g"] = linear_acceleration[:, 2]
    corrected["a_vertical_no_g"] = vertical_without_g

    info = {
        "gravity_vector": gravity_vector,
        "gravity_norm": gravity_norm,
        "vertical_unit_vector": vertical_unit_vector,
    }
    return corrected, info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove the constant gravity component from Accelerometer.csv."
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path. Default: <input stem>_no_g.csv",
    )
    parser.add_argument(
        "--stationary-seconds",
        type=float,
        default=3.0,
        help="Initial stationary interval used to estimate gravity.",
    )
    parser.add_argument(
        "--use-measured-g",
        action="store_true",
        help="Remove the measured gravity norm instead of forcing 9.81 m/s^2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv.resolve()
    output_csv = args.output_csv
    if output_csv is None:
        output_csv = input_csv.with_name(f"{input_csv.stem}_no_g.csv")

    data = pd.read_csv(input_csv)
    corrected, info = correct_gravity(
        data,
        stationary_seconds=args.stationary_seconds,
        expected_g=None if args.use_measured_g else 9.81,
    )
    corrected.to_csv(output_csv, index=False)

    gravity = info["gravity_vector"]
    vertical = info["vertical_unit_vector"]
    print(f"Gravity vector [x, y, z]: {gravity}")
    print(f"Gravity norm: {info['gravity_norm']:.4f} m/s^2")
    print(f"Vertical unit vector [x, y, z]: {vertical}")
    print(f"Saved corrected file: {output_csv}")


if __name__ == "__main__":
    main()
