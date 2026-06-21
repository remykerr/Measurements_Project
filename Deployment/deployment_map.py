import ast
from pathlib import Path
import shutil
import sys
import warnings

import folium
import joblib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Machine_Learning.Surface_Classification.Dataset_construction import (  # noqa: E402
    build_surface_dataset as build_surface_classification_dataset,
)
from Machine_Learning.Defect_identification.Dataset_construction import (  # noqa: E402
    build_surface_dataset as build_defect_identification_dataset,
)


BASE_DIR = Path(__file__).resolve().parent
TEST_FILES_DIR = PROJECT_ROOT / "Test_files"
TEST_CIRCUIT_DIR = TEST_FILES_DIR / "trial_run_random_long"
MEASUREMENTS_DIR = TEST_FILES_DIR / "Measurements"

# Model no Medium Asphalt 
#SURFACE_MODEL_DIR = PROJECT_ROOT / "Machine_Learning" / "Surface_Classification" / "final_models" / "noMedium"

# Model no Unpaved
SURFACE_MODEL_DIR = PROJECT_ROOT / "Machine_Learning" / "Surface_Classification" / "final_models" / "noUnpaved"

# Model No Medium Asphalt, No Unpaved
#SURFACE_MODEL_DIR = PROJECT_ROOT / "Machine_Learning" / "Surface_Classification" / "final_models" / "noMediumnoUnpaved"

#Model 3 classes (Rough, Smooth, Medium asphalt)
#SURFACE_MODEL_DIR = PROJECT_ROOT / "Machine_Learning" / "Surface_Classification" / "final_models" / "3classes"

MODEL_FILE = SURFACE_MODEL_DIR / "knn_surface_classifier.joblib"
SURFACE_SCALER_FILE = SURFACE_MODEL_DIR / "surface_scaler.joblib"
FEATURE_COLUMNS_FILE = SURFACE_MODEL_DIR / "feature_columns.joblib"
AE_MODEL_FILE = TEST_FILES_DIR / "autoencoder_model.keras"
AE_SCALER_FILE = TEST_FILES_DIR / "ae_scaler.joblib"
AE_THRESHOLD_FILE = TEST_FILES_DIR / "ae_threshold.joblib"
TEST_6_FILE = TEST_FILES_DIR / "Test_6.py"
OUTPUT_MAP_FILE = BASE_DIR / "Rugosity_Map_Milan_Deployment.html"

DATA_SOURCE = "test_circuit"
ENABLE_HAZARD_DETECTION = True
COMFORT_SMOOTHING_WINDOW = 3
SURFACE_SMOOTHING_WINDOW = 5
HAZARD_SMOOTHING_WINDOW = 3
SURFACE_COLORS = {
    "Smooth_asphalt": "#2ca02c",
    "Medium_asphalt": "#ffbf00",
    "Rough_asphalt": "#d62728",
    "cobble": "#D955D9",
    "Grass": "#49F527",
    "Unpaved": "#0724ED",
}
DEFAULT_SURFACE_COLOR = "#6c757d"


def load_test6_build_prediction_dataset():
    if str(TEST_FILES_DIR) not in sys.path:
        sys.path.insert(0, str(TEST_FILES_DIR))

    source = TEST_6_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TEST_6_FILE))
    reusable_nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
    ]
    module = ast.Module(body=reusable_nodes, type_ignores=[])
    ast.fix_missing_locations(module)

    namespace = {"__file__": str(TEST_6_FILE)}
    exec(compile(module, str(TEST_6_FILE), "exec"), namespace)
    return namespace["build_prediction_dataset"]


build_comfort_dataset = load_test6_build_prediction_dataset()


def iso_color(aw):
    if aw < 0.315:
        return "blue"
    if aw < 0.63:
        return "green"
    if aw < 1:
        return "yellow"
    if aw < 1.6:
        return "orange"
    if aw < 2.5:
        return "red"
    return "black"


def surface_color(surface):
    return SURFACE_COLORS.get(surface, DEFAULT_SURFACE_COLOR)


def surface_image_html(surface):
    if surface in ("Pothole", "Speedbump", "Defect"):
        return ""

    image_dir = TEST_FILES_DIR / "images"
    if not image_dir.is_dir():
        return ""

    valid_exts = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    target_name = surface.lower()
    for file_name in image_dir.iterdir():
        if file_name.suffix.lower() in valid_exts and file_name.stem.lower() == target_name:
            image_url = file_name.resolve().as_uri()
            return (
                '<tr><td colspan="2" style="text-align:center;">'
                f'<img src="{image_url}" '
                'style="width:240px;max-width:100%;border:1px solid #aaa;margin-top:8px;"/>'
                "</td></tr>"
            )
    return ""


def stage_measurement(staging_dir, surface, measurement_id, accel_file, gps_file):
    measure_dir = staging_dir / "Clean_measurements_ML" / f"{surface}_{measurement_id}"
    measure_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(accel_file, measure_dir / "Accelerometer.csv")
    shutil.copy2(gps_file, measure_dir / "Location.csv")


def build_surface_prediction_input(accel_file, gps_file, run_id):
    staging_dir = BASE_DIR / "_staging_surface"
    train_surface = "DeploymentTrain"
    test_surface = "DeploymentTest"
    stage_measurement(staging_dir, train_surface, 1, accel_file, gps_file)
    stage_measurement(staging_dir, test_surface, 2, accel_file, gps_file)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Skipping missing measurement: .*",
            category=UserWarning,
        )
        _, _, full_window_dataset = build_surface_classification_dataset(
            base_dir=staging_dir,
            surface_types=(train_surface, test_surface),
            train_measurements=(1,),
            test_measurements=(2,),
            plot_debug_acc=False,
            plot_debug=False,
            return_debug=False,
        )

    run_df = full_window_dataset[full_window_dataset["srf"] == train_surface].copy()
    run_df = run_df.drop(columns=["srf"]).reset_index(drop=True)
    run_df["run_id"] = run_id
    return run_df


def build_defect_prediction_input(accel_file, gps_file, run_id):
    staging_dir = BASE_DIR / "_staging_defect"
    surface = "DeploymentDefect"
    stage_measurement(staging_dir, surface, 1, accel_file, gps_file)

    ae_windows = build_defect_identification_dataset(
        base_dir=staging_dir,
        surface_types=(surface,),
        measurement_ids=(1,),
        target_window_size=100,
        min_speed=1.0,
        plot_debug_acc=False,
        plot_debug=False,
        return_debug=False,
    )
    ae_windows = ae_windows.drop(columns=["surface", "measurement_id"]).reset_index(drop=True)
    ae_windows["run_id"] = run_id
    return ae_windows


def iter_input_runs(data_source):
    if data_source == "test_circuit":
        yield 1, TEST_CIRCUIT_DIR / "Accelerometer.csv", TEST_CIRCUIT_DIR / "Location.csv"
        return

    if data_source == "measurements":
        for run_id in range(1, 8):
            yield (
                run_id,
                MEASUREMENTS_DIR / f"Accelerometer_{run_id}.csv",
                MEASUREMENTS_DIR / f"Location_{run_id}.csv",
            )
        return

    raise ValueError("DATA_SOURCE must be 'test_circuit' or 'measurements'.")


def build_graph_input(data_source):
    frames = []
    for run_id, accel_file, gps_file in iter_input_runs(data_source):
        surface_df = build_surface_prediction_input(accel_file, gps_file, run_id)
        comfort_df = build_comfort_dataset(accel_file, gps_file)[["t", "aw"]]
        surface_df = pd.merge_asof(
            surface_df.sort_values("t"),
            comfort_df.sort_values("t"),
            on="t",
            direction="nearest",
        )
        frames.append(surface_df)
    return pd.concat(frames, ignore_index=True)


def build_ae_input(data_source):
    frames = []
    for run_id, accel_file, gps_file in iter_input_runs(data_source):
        frames.append(build_defect_prediction_input(accel_file, gps_file, run_id))
    return pd.concat(frames, ignore_index=True)


def smooth_labels(label_series, window=3):
    labels = label_series.tolist()
    smoothed = []
    half = window // 2
    for i in range(len(labels)):
        start = max(0, i - half)
        end = min(len(labels), i + half + 1)
        window_labels = pd.Series(labels[start:end])
        modes = window_labels.mode()
        smoothed.append(modes.iloc[0] if len(modes) > 0 else window_labels.iloc[-1])
    return pd.Series(smoothed, index=label_series.index)


def mark_stable_hazards(label_series, min_block=2):
    labels = label_series.tolist()
    keep = [False] * len(labels)
    i = 0
    while i < len(labels):
        if labels[i] == "Defect":
            j = i
            while j + 1 < len(labels) and labels[j + 1] == labels[i]:
                j += 1
            if j - i + 1 >= min_block:
                for k in range(i, j + 1):
                    keep[k] = True
            i = j + 1
        else:
            i += 1
    return pd.Series(keep, index=label_series.index)


def add_predictions(full_df):
    feature_columns = joblib.load(FEATURE_COLUMNS_FILE)
    classifier = joblib.load(MODEL_FILE)
    surface_scaler = joblib.load(SURFACE_SCALER_FILE)

    x_surface = surface_scaler.transform(full_df[feature_columns])
    full_df["surface_prediction"] = classifier.predict(x_surface)

    if not ENABLE_HAZARD_DETECTION:
        full_df["hazard_prediction"] = "None"
        return full_df

    import tensorflow as tf

    ae_model = tf.keras.models.load_model(AE_MODEL_FILE)
    ae_scaler = joblib.load(AE_SCALER_FILE)
    ae_threshold = joblib.load(AE_THRESHOLD_FILE)
    ae_windows = build_ae_input(DATA_SOURCE)

    az_cols = [f"az_{i}" for i in range(100)]
    x_ae = ae_scaler.transform(ae_windows[az_cols])
    reconstructions = ae_model.predict(x_ae, verbose=0)
    point_errors = np.abs(x_ae - reconstructions)
    top_k = min(10, point_errors.shape[1])

    ae_windows["reconstruction_error"] = np.mean(
        np.sort(point_errors, axis=1)[:, -top_k:],
        axis=1,
    )
    ae_windows["is_anomaly"] = ae_windows["reconstruction_error"] > ae_threshold

    from sklearn.neighbors import BallTree

    ae_coords = np.radians(ae_windows[["lat", "long"]].values)
    full_coords = np.radians(full_df[["lat", "long"]].values)
    tree = BallTree(ae_coords, metric="haversine")
    _, indices = tree.query(full_coords, k=1)

    nearest = indices.flatten()
    full_df["reconstruction_error"] = ae_windows["reconstruction_error"].values[nearest]
    full_df["hazard_prediction"] = np.where(
        ae_windows["is_anomaly"].values[nearest],
        "Defect",
        "None",
    )
    full_df.loc[full_df["surface_prediction"] != "Smooth_asphalt", "hazard_prediction"] = "None"
    return full_df


def prepare_visualization_data():
    full_df = build_graph_input(DATA_SOURCE)
    full_df = full_df.sort_values(["run_id", "t"]).reset_index(drop=True)
    full_df = add_predictions(full_df)

    full_df["aw_smooth"] = full_df.groupby("run_id")["aw"].transform(
        lambda s: s.rolling(
            window=COMFORT_SMOOTHING_WINDOW,
            center=True,
            min_periods=1,
        ).mean()
    )
    full_df["surface_prediction_smoothed"] = (
        full_df.groupby("run_id")["surface_prediction"]
        .apply(lambda s: smooth_labels(s, window=SURFACE_SMOOTHING_WINDOW))
        .reset_index(level=0, drop=True)
    )
    full_df["hazard_prediction_smoothed"] = (
        full_df.groupby("run_id")["hazard_prediction"]
        .apply(lambda s: smooth_labels(s, window=HAZARD_SMOOTHING_WINDOW))
        .reset_index(level=0, drop=True)
    )
    full_df["show_hazard_marker"] = (
        full_df.groupby("run_id")["hazard_prediction_smoothed"]
        .apply(lambda s: mark_stable_hazards(s, min_block=2))
        .reset_index(level=0, drop=True)
    )
    full_df["iso_color"] = full_df["aw_smooth"].apply(iso_color)
    return full_df


def plot_comfort_scatter(full_df):
    plt.figure(figsize=(10, 8))
    plt.scatter(full_df["long"], full_df["lat"], c=full_df["iso_color"], s=40)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("ISO 2631 Weighted Comfort Map")
    plt.axis("equal")

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Comfortable (<0.315 m/s^2)", markerfacecolor="blue", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="A Little Uncomfortable (0.315-0.63 m/s^2)", markerfacecolor="green", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="Fairly Uncomfortable (0.63-1 m/s^2)", markerfacecolor="yellow", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="Uncomfortable (1-1.6 m/s^2)", markerfacecolor="orange", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="Very Uncomfortable (1.6-2.5 m/s^2)", markerfacecolor="red", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="Extremely Uncomfortable (>2.5 m/s^2)", markerfacecolor="black", markersize=10),
    ]
    plt.legend(handles=legend_elements)
    plt.grid(True)
    plt.show()


def add_route_layers(complete_map, full_df):
    comfort_layer = folium.FeatureGroup(name="Comfort", show=True)
    surface_layer = folium.FeatureGroup(name="Surface prediction", show=False)
    comfort_layer.add_to(complete_map)
    surface_layer.add_to(complete_map)

    for i in range(len(full_df) - 1):
        current_line = full_df.iloc[i]
        next_line = full_df.iloc[i + 1]

        if current_line["run_id"] != next_line["run_id"]:
            continue

        aw = current_line["aw_smooth"]
        color = current_line["iso_color"]
        surface = current_line["surface_prediction_smoothed"]
        image_html = surface_image_html(surface)
        segment_locations = [
            [current_line["lat"], current_line["long"]],
            [next_line["lat"], next_line["long"]],
        ]

        comfort_class = {
            "blue": "Comfortable",
            "green": "A Little Uncomfortable",
            "yellow": "Fairly Uncomfortable",
            "orange": "Uncomfortable",
            "red": "Very Uncomfortable",
            "black": "Extremely Uncomfortable",
        }.get(color, "Unknown")

        folium.PolyLine(
            locations=segment_locations,
            color=color,
            weight=4,
            opacity=0.8,
            popup=f"""
                <table>
                <tr><td colspan="2"><b><u>ISO Comfort Analysis</u></b></td></tr>
                <tr><td><u>aw:</u></td><td>{aw:.3f} m/s^2</td></tr>
                <tr><td><u>Velocity:</u></td><td>{current_line['v']:.1f} m/s</td></tr>
                <tr><td><u>Predicted Surface:</u></td><td><b>{surface}</b></td></tr>
                <tr><td><u>Comfort Class:</u></td><td>{comfort_class}</td></tr>
                {image_html}
                </table>
                """,
        ).add_to(comfort_layer)

        folium.PolyLine(
            locations=segment_locations,
            color=surface_color(surface),
            weight=4,
            opacity=0.8,
            popup=f"""
                <table>
                <tr><td colspan="2"><b><u>Surface Prediction</u></b></td></tr>
                <tr><td><u>Predicted Surface:</u></td><td><b>{surface}</b></td></tr>
                <tr><td><u>aw:</u></td><td>{aw:.3f} m/s^2</td></tr>
                <tr><td><u>Velocity:</u></td><td>{current_line['v']:.1f} m/s</td></tr>
                {image_html}
                </table>
                """,
        ).add_to(surface_layer)

    return comfort_layer, surface_layer


def add_hazard_markers(complete_map, full_df):
    for _, line in full_df.iterrows():
        if not line.get("show_hazard_marker", False):
            continue

        if line["hazard_prediction_smoothed"] == "Defect":
            folium.CircleMarker(
                location=[line["lat"], line["long"]],
                radius=6,
                color="darkred",
                fill=True,
                fill_color="red",
                fill_opacity=0.9,
                weight=3,
                popup=f"""
                    <table>
                    <tr><td colspan="2"><b><u>DEFECT DETECTED</u></b></td></tr>
                    <tr><td><u>aw:</u></td><td>{line['aw_smooth']:.3f} m/s^2</td></tr>
                    <tr><td><u>Velocity:</u></td><td>{line['v']:.1f} m/s</td></tr>
                    <tr><td><u>Latitude:</u></td><td>{line['lat']:.6f}</td></tr>
                    <tr><td><u>Longitude:</u></td><td>{line['long']:.6f}</td></tr>
                    </table>
                    """,
            ).add_to(complete_map)


def add_custom_controls(complete_map, comfort_layer, surface_layer):
    legend_html = """
<button id="route-mode-toggle" style="position: fixed; top: 20px; left: 100px; z-index: 10000; padding: 8px 12px; background-color: white; border: 2px solid grey; border-radius: 4px; cursor: pointer; font-size: 14px; font-weight: bold;">Show Surface Colors</button>
<button id="legend-toggle" onclick="const legend = document.querySelector('.route-legend[data-active=&quot;true&quot;]'); if (legend.style.display === 'none') { legend.style.display = 'block'; this.textContent = 'Hide Legend'; } else { legend.style.display = 'none'; this.textContent = 'Show Legend'; }" style="position: fixed; top: 20px; left: 280px; z-index: 10000; padding: 8px 12px; background-color: white; border: 2px solid grey; border-radius: 4px; cursor: pointer; font-size: 14px; font-weight: bold;">Hide Legend</button>
<div id="iso-legend" class="route-legend" data-active="true" style="position: fixed; bottom: 100px; left: 100px; width: 280px; height: 420px; background-color: white; border:2px solid grey; z-index:9999; font-size:14px; padding:10px; overflow-y: auto;">
<b>ISO 2631 Comfort Levels</b><br><br>
<i style="background:blue;width:14px;height:14px;float:left;margin-right:8px;opacity:0.8;"></i>Comfortable (&lt;0.315 m/s^2)<br><br>
<i style="background:green;width:14px;height:14px;float:left;margin-right:8px;opacity:0.8;"></i>A Little Uncomfortable (0.315-0.63)<br><br>
<i style="background:yellow;width:14px;height:14px;float:left;margin-right:8px;opacity:0.8;"></i>Fairly Uncomfortable (0.63-1)<br><br>
<i style="background:orange;width:14px;height:14px;float:left;margin-right:8px;opacity:0.8;"></i>Uncomfortable (1-1.6)<br><br>
<i style="background:red;width:14px;height:14px;float:left;margin-right:8px;opacity:0.8;"></i>Very Uncomfortable (1.6-2.5)<br><br>
<i style="background:black;width:14px;height:14px;float:left;margin-right:8px;opacity:0.8;"></i>Extremely Uncomfortable (&gt;2.5)<br><br>
<hr style="margin:10px 0;"><b>Road Hazards</b><br><br>
<i style="background:red;border: 2px solid darkred;width:14px;height:14px;border-radius: 50%;float:left;margin-right:8px;opacity:0.9;"></i>Defect<br><br>
</div>
<div id="surface-legend" class="route-legend" data-active="false" style="position: fixed; bottom: 100px; left: 100px; width: 280px; background-color: white; border:2px solid grey; z-index:9999; font-size:14px; padding:10px; display:none;">
<b>Surface Prediction</b><br><br>
<i style="background:#2ca02c;width:14px;height:14px;float:left;margin-right:8px;opacity:0.8;"></i>Smooth asphalt<br><br>
<i style="background:#ffbf00;width:14px;height:14px;float:left;margin-right:8px;opacity:0.8;"></i>Medium asphalt<br><br>
<i style="background:#d62728;width:14px;height:14px;float:left;margin-right:8px;opacity:0.8;"></i>Rough asphalt<br><br>
<i style="background:#D955D9;width:14px;height:14px;float:left;margin-right:8px;opacity:0.8;"></i>Cobble<br><br>
<i style="background:#49F527;width:14px;height:14px;float:left;margin-right:8px;opacity:0.8;"></i>Grass<br><br>
<i style="background:#0724ED;width:14px;height:14px;float:left;margin-right:8px;opacity:0.8;"></i>Unpaved<br><br>
</div>
"""
    complete_map.get_root().html.add_child(folium.Element(legend_html))

    mode_toggle_js = f"""
setTimeout(function() {{
const routeMap = {complete_map.get_name()};
const comfortLayer = {comfort_layer.get_name()};
const surfaceLayer = {surface_layer.get_name()};
window.currentRouteColorMode = "comfort";

function setRouteColorMode(mode) {{
  const comfortLegend = document.getElementById("iso-legend");
  const surfaceLegend = document.getElementById("surface-legend");
  const modeButton = document.getElementById("route-mode-toggle");
  const legendButton = document.getElementById("legend-toggle");

  if (mode === "surface") {{
    if (routeMap.hasLayer(comfortLayer)) routeMap.removeLayer(comfortLayer);
    if (!routeMap.hasLayer(surfaceLayer)) routeMap.addLayer(surfaceLayer);
    comfortLegend.style.display = "none";
    comfortLegend.dataset.active = "false";
    surfaceLegend.style.display = "block";
    surfaceLegend.dataset.active = "true";
    modeButton.textContent = "Show Comfort Colors";
  }} else {{
    if (routeMap.hasLayer(surfaceLayer)) routeMap.removeLayer(surfaceLayer);
    if (!routeMap.hasLayer(comfortLayer)) routeMap.addLayer(comfortLayer);
    surfaceLegend.style.display = "none";
    surfaceLegend.dataset.active = "false";
    comfortLegend.style.display = "block";
    comfortLegend.dataset.active = "true";
    modeButton.textContent = "Show Surface Colors";
  }}

  legendButton.textContent = "Hide Legend";
  window.currentRouteColorMode = mode;
}}

document.getElementById("route-mode-toggle").addEventListener("click", function() {{
  const nextMode = window.currentRouteColorMode === "comfort" ? "surface" : "comfort";
  setRouteColorMode(nextMode);
}});

setRouteColorMode("comfort");
}}, 0);
"""
    complete_map.get_root().script.add_child(folium.Element(mode_toggle_js))


def create_map(full_df):
    complete_map = folium.Map(
        location=[full_df["lat"].mean(), full_df["long"].mean()],
        zoom_start=15,
        tiles="OpenStreetMap",
    )
    comfort_layer, surface_layer = add_route_layers(complete_map, full_df)
    add_hazard_markers(complete_map, full_df)
    add_custom_controls(complete_map, comfort_layer, surface_layer)
    complete_map.save(OUTPUT_MAP_FILE)


def main():
    full_df = prepare_visualization_data()
    plot_comfort_scatter(full_df)
    create_map(full_df)
    print(f"Map saved to: {OUTPUT_MAP_FILE}")


if __name__ == "__main__":
    main()
