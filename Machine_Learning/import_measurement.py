"""
Import raw measurement files into the Clean_measurements_ML layout.

The dataset construction code expects this structure:

    Clean_measurements_ML/<Surface>_<id>/
        Accelerometer.csv
        Location.csv
        meta/               optional

and accelerometer columns named:

    Time (s), X (m/s^2), Y (m/s^2), Z (m/s^2)

This script copies a raw measurement folder into that layout and normalizes
known column-name variants produced by the recording app.

# example usage in terminal:
python Measurements_Project\Machine_Learning\import_measurement.py "G:\Il mio Drive\measurements_Project\Project_Guidelines\Measurments\pothole measurements (g included)\pothole_1" --surface Pothole
python Measurements_Project\Machine_Learning\import_measurement.py "G:\Il mio Drive\measurements_Project\Project_Guidelines\Measurments\pothole measurements (g included)\pothole_2" --surface Pothole
python Measurements_Project\Machine_Learning\import_measurement.py "G:\Il mio Drive\measurements_Project\Project_Guidelines\Measurments\pothole measurements (g included)\pothole_3" --surface Pothole
python Measurements_Project\Machine_Learning\import_measurement.py "G:\Il mio Drive\measurements_Project\Project_Guidelines\Measurments\pothole measurements (g included)\pothole_4" --surface Pothole
python Measurements_Project\Machine_Learning\import_measurement.py "G:\Il mio Drive\measurements_Project\Project_Guidelines\Measurments\pothole measurements (g included)\pothole_5" --surface Pothole
python Measurements_Project\Machine_Learning\import_measurement.py "G:\Il mio Drive\measurements_Project\Project_Guidelines\Measurments\pothole measurements (g included)\pothole_6" --surface Pothole
python Measurements_Project\Machine_Learning\import_measurement.py "G:\Il mio Drive\measurements_Project\Project_Guidelines\Measurments\pothole measurements (g included)\pothole_7" --surface Pothole
python Measurements_Project\Machine_Learning\import_measurement.py "G:\Il mio Drive\measurements_Project\Project_Guidelines\Measurments\pothole measurements (g included)\pothole_8" --surface Pothole
python Measurements_Project\Machine_Learning\import_measurement.py "G:\Il mio Drive\measurements_Project\Project_Guidelines\Measurments\pothole measurements (g included)\pothole_9" --surface Pothole
python Measurements_Project\Machine_Learning\import_measurement.py "G:\Il mio Drive\measurements_Project\Project_Guidelines\Measurments\pothole measurements (g included)\pothole_10" --surface Pothole
python Measurements_Project\Machine_Learning\import_measurement.py "G:\Il mio Drive\measurements_Project\Project_Guidelines\Measurments\pothole measurements (g included)\pothole_11" --surface Pothole



"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "Clean_measurements_ML"
DEGREE_SIGN = "\N{DEGREE SIGN}"


ACCELEROMETER_COLUMN_MAP = {
    "time (s)": "Time (s)",
    "x (m/s^2)": "X (m/s^2)",
    "y (m/s^2)": "Y (m/s^2)",
    "z (m/s^2)": "Z (m/s^2)",
    "acceleration x (m/s^2)": "X (m/s^2)",
    "acceleration y (m/s^2)": "Y (m/s^2)",
    "acceleration z (m/s^2)": "Z (m/s^2)",
}

LOCATION_COLUMN_MAP = {
    "time (s)": "Time (s)",
    "latitude (°)": f"Latitude ({DEGREE_SIGN})",
    "latitude (â°)": f"Latitude ({DEGREE_SIGN})",
    "latitude (â°)": f"Latitude ({DEGREE_SIGN})",
    "longitude (°)": f"Longitude ({DEGREE_SIGN})",
    "longitude (â°)": f"Longitude ({DEGREE_SIGN})",
    "height (m)": "Height (m)",
    "velocity (m/s)": "Velocity (m/s)",
    "direction (°)": f"Direction ({DEGREE_SIGN})",
    "direction (â°)": f"Direction ({DEGREE_SIGN})",
    "horizontal accuracy (m)": "Horizontal Accuracy (m)",
    "vertical accuracy (m)": "Vertical Accuracy (m)",
    "vertical accuracy (°)": f"Vertical Accuracy ({DEGREE_SIGN})",
    "vertical accuracy (â°)": f"Vertical Accuracy ({DEGREE_SIGN})",
}

REQUIRED_ACCELEROMETER_COLUMNS = ["Time (s)", "X (m/s^2)", "Y (m/s^2)", "Z (m/s^2)"]
REQUIRED_LOCATION_COLUMNS = [
    "Time (s)",
    f"Latitude ({DEGREE_SIGN})",
    f"Longitude ({DEGREE_SIGN})",
    "Velocity (m/s)",
]


def clean_surface_name(surface: str) -> str:
    cleaned = re.sub(r"\s+", "_", surface.strip())
    cleaned = re.sub(r"[^A-Za-z0-9_]", "", cleaned)
    if not cleaned:
        raise ValueError("Surface label cannot be empty after cleaning.")
    return cleaned


def normalized_key(column_name: str) -> str:
    return " ".join(column_name.strip().strip('"').lower().split())


def normalize_columns(data: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
    renamed_columns = {
        column: column_map.get(normalized_key(column), column.strip().strip('"'))
        for column in data.columns
    }
    return data.rename(columns=renamed_columns)


def normalize_location_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = normalize_columns(data, LOCATION_COLUMN_MAP)
    semantic_names = {
        "latitude": f"Latitude ({DEGREE_SIGN})",
        "longitude": f"Longitude ({DEGREE_SIGN})",
        "direction": f"Direction ({DEGREE_SIGN})",
        "vertical accuracy": "Vertical Accuracy (m)",
    }
    renamed_columns = {}
    for column in data.columns:
        key = normalized_key(column)
        for prefix, normalized_name in semantic_names.items():
            if key.startswith(prefix):
                renamed_columns[column] = normalized_name
                break
    return data.rename(columns=renamed_columns)


def require_columns(data: pd.DataFrame, required_columns: list[str], file_name: str) -> None:
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        available = ", ".join(data.columns)
        missing = ", ".join(missing_columns)
        raise ValueError(
            f"{file_name} is missing required columns: {missing}. "
            f"Available columns: {available}"
        )


def next_measurement_id(output_dir: Path, surface: str) -> int:
    pattern = re.compile(rf"^{re.escape(surface)}_(\d+)$", re.IGNORECASE)
    existing_ids = []
    for child in output_dir.iterdir() if output_dir.exists() else []:
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if match:
            existing_ids.append(int(match.group(1)))
    return max(existing_ids, default=0) + 1


def import_measurement(
    source_dir: Path,
    surface: str,
    output_dir: Path,
    measurement_id: int | None,
    overwrite: bool,
    min_dataset_speed: float,
) -> Path:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    surface = clean_surface_name(surface)
    measurement_id = measurement_id or next_measurement_id(output_dir, surface)

    accelerometer_file = source_dir / "Accelerometer.csv"
    location_file = source_dir / "Location.csv"
    if not accelerometer_file.exists():
        raise FileNotFoundError(f"Missing file: {accelerometer_file}")
    if not location_file.exists():
        raise FileNotFoundError(f"Missing file: {location_file}")

    target_dir = output_dir / f"{surface}_{measurement_id}"
    if target_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Target folder already exists: {target_dir}. "
                "Use --overwrite or choose another --id."
            )

    accelerometer_data = pd.read_csv(accelerometer_file)
    accelerometer_data = normalize_columns(accelerometer_data, ACCELEROMETER_COLUMN_MAP)
    require_columns(accelerometer_data, REQUIRED_ACCELEROMETER_COLUMNS, "Accelerometer.csv")

    location_data = pd.read_csv(location_file)
    location_data = normalize_location_columns(location_data)
    require_columns(location_data, REQUIRED_LOCATION_COLUMNS, "Location.csv")
    usable_speed_samples = (location_data["Velocity (m/s)"] > min_dataset_speed).sum()

    target_dir.mkdir(parents=True, exist_ok=True)
    accelerometer_data.to_csv(target_dir / "Accelerometer.csv", index=False)
    location_data.to_csv(target_dir / "Location.csv", index=False)

    source_meta = source_dir / "meta"
    if source_meta.exists() and source_meta.is_dir():
        shutil.copytree(source_meta, target_dir / "meta", dirs_exist_ok=True)

    if usable_speed_samples == 0:
        print(
            "Warning: no GPS velocity samples are above "
            f"{min_dataset_speed} m/s. Dataset construction currently filters "
            "out slower samples, so this measurement may produce no rows."
        )

    return target_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import a raw measurement folder into Clean_measurements_ML."
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Folder containing Accelerometer.csv and Location.csv.",
    )
    parser.add_argument(
        "--surface",
        required=True,
        help="Surface label to use in the dataset, e.g. Grass or Rough_asphalt.",
    )
    parser.add_argument(
        "--id",
        type=int,
        default=None,
        help="Measurement id. If omitted, the next available id for the surface is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Destination clean-measurements directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the destination folder if it already exists.",
    )
    parser.add_argument(
        "--min-dataset-speed",
        type=float,
        default=1.0,
        help="Speed threshold used only for the import warning. Default: 1.0 m/s.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_dir = import_measurement(
        source_dir=args.source_dir,
        surface=args.surface,
        output_dir=args.output_dir,
        measurement_id=args.id,
        overwrite=args.overwrite,
        min_dataset_speed=args.min_dataset_speed,
    )
    print(f"Imported measurement into: {target_dir}")


if __name__ == "__main__":
    main()
