
import ast
import sys
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import folium
import joblib

Base_Dir = Path(__file__).resolve().parent
Measurements_Dir = Base_Dir / "Measurements"
Test_Circuit_Dir = Base_Dir / "Test circuit"
Model_File = Base_Dir / "Road_Surface_Classifier.pkl"
Hazard_Model_File = Base_Dir / "Road_Hazard_Classifier.pkl"
Output_Map_File = Base_Dir / "Rugosity_Map_Milan_CL.html"
Test_6_File = Base_Dir / "Test_6.py"
Data_Source = "test_circuit"
Enable_Hazard_Detection = False
Comfort_Smoothing_Window = 3
Surface_Smoothing_Window = 5
Hazard_Smoothing_Window = 3
Surface_Colors = {
    "Smooth_asphalt": "#2ca02c",
    "Medium_asphalt": "#ffbf00",
    "Rough_asphalt": "#d62728",
    "cobble": "#9467bd",
    "Unpaved": "#8c564b",
    "Grass": "#17becf",
}
Default_Surface_Color = "#6c757d"

if str(Base_Dir) not in sys.path:
    sys.path.insert(0, str(Base_Dir))


def load_test6_functions():
    """Load helpers from Test_6.py without running its plotting script."""
    source = Test_6_File.read_text(encoding="utf-8")
    source = source.replace(
        "segment_times.append(Accel_z.index[k])",
        "center_index = k + window_size // 2\n        "
        "segment_times.append(Accel_z.index[center_index])",
    )
    tree = ast.parse(source, filename=str(Test_6_File))
    reusable_nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
    ]
    module = ast.Module(body=reusable_nodes, type_ignores=[])
    ast.fix_missing_locations(module)

    namespace = {"__file__": str(Test_6_File)}
    exec(compile(module, str(Test_6_File), "exec"), namespace)
    return namespace["build_prediction_dataset"], namespace["iso_color"]


build_prediction_dataset, iso_color = load_test6_functions()


def surface_image_html(surface):
    if surface in ("Pothole", "Speedbump"):
        return ''

    image_dir = Base_Dir / "images"
    if not image_dir.is_dir():
        return ''

    target_name = surface.lower()
    valid_exts = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

    for file_name in image_dir.iterdir():
        if file_name.suffix.lower() not in valid_exts:
            continue
        if file_name.stem.lower() != target_name:
            continue

        image_url = file_name.resolve().as_uri()
        return (
            f'<tr><td colspan="2" style="text-align:center;">'
            f'<img src="{image_url}" '
            f'style="width:240px;max-width:100%;border:1px solid #aaa;margin-top:8px;"/>'
            f'</td></tr>'
        )

    return ''


def surface_color(surface):
    return Surface_Colors.get(surface, Default_Surface_Color)


def comfort_class(color):
    if color == "blue":
        return "Comfortable"
    if color == "green":
        return "A Little Uncomfortable"
    if color == "yellow":
        return "Fairly Uncomfortable"
    if color == "orange":
        return "Uncomfortable"
    if color == "red":
        return "Very Uncomfortable"
    return "Extremely Uncomfortable"


def build_graph_input(data_source):
    if data_source == "test_circuit":
        single_run_df = build_prediction_dataset(
            Test_Circuit_Dir / "Accelerometer.csv",
            Test_Circuit_Dir / "Location.csv",
        )
        single_run_df["run_id"] = 1
        return single_run_df

    if data_source == "measurements":
        full_df = pd.DataFrame({})
        for i in range(1, 8):
            accel_file = Measurements_Dir / f"Accelerometer_{i}.csv"
            gps_file = Measurements_Dir / f"Location_{i}.csv"
            single_run_df = build_prediction_dataset(accel_file, gps_file)
            single_run_df["run_id"] = i
            full_df = pd.concat([full_df, single_run_df])
        return full_df

    raise ValueError(
        "DATA_SOURCE must be 'test_circuit' or 'measurements'."
    )


Full_df = build_graph_input(Data_Source)

Full_df = Full_df.sort_values(["run_id", "t"]).reset_index(drop=True)

feature_columns = [
    "Norm_az_avg",
    "Norm_az_rms",
    "Norm_az_std",
    "Norm_az_peak",
    "Norm_az_kurt",
    "Norm_az_crest",
    "Norm_az_skew",
    "spec_energy",
    "comfort_energy",
    "texture_energy",
    "dominant_freq",
    "periodicity_strength",]

X = Full_df[feature_columns]

classifier = joblib.load(Model_File)

surface_predictions = classifier.predict(X)

Full_df["surface_prediction"] = surface_predictions

if Enable_Hazard_Detection:
    hazard_classifier = joblib.load(Hazard_Model_File)
    Full_df["hazard_prediction"] = hazard_classifier.predict(X)
else:
    Full_df["hazard_prediction"] = "None"

# Smooth the ISO comfort score and the surface label to reduce second-to-second noise.
Full_df["aw_smooth"] = Full_df.groupby("run_id")["aw"].transform(
    lambda s: s.rolling(
        window=Comfort_Smoothing_Window,
        center=True,
        min_periods=1,
    ).mean()
)

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

Full_df["surface_prediction_smoothed"] = (
    Full_df.groupby("run_id")["surface_prediction"]
           .apply(lambda s: smooth_labels(s, window=Surface_Smoothing_Window))
           .reset_index(level=0, drop=True)
)

Full_df["hazard_prediction_smoothed"] = (
    Full_df.groupby("run_id")["hazard_prediction"]
           .apply(lambda s: smooth_labels(s, window=Hazard_Smoothing_Window))
           .reset_index(level=0, drop=True)
)


def mark_stable_hazards(label_series, min_block=2):
    labels = label_series.tolist()
    keep = [False] * len(labels)
    i = 0
    while i < len(labels):
        if labels[i] in ("Pothole", "Speedbump"):
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

Full_df["show_hazard_marker"] = (
    Full_df.groupby("run_id")["hazard_prediction_smoothed"]
           .apply(lambda s: mark_stable_hazards(s, min_block=2))
           .reset_index(level=0, drop=True)
)

Full_df["iso_color"] = Full_df["aw_smooth"].apply(iso_color)

 # ==========================================
 # Plotting
 # ==========================================
plt.figure(figsize=(10,8))

plt.scatter(
    Full_df["long"],
    Full_df["lat"],
    c=Full_df["iso_color"],
    s=40)

# Labels
plt.xlabel('Longitude')
plt.ylabel('Latitude')
# Title
plt.title('ISO 2631 Weighted Comfort Map')
# Equal scaling
plt.axis('equal')

# Manual legend
from matplotlib.lines import Line2D
legend_elements = [

    Line2D([0], [0],
           marker='o',
           color='w',
           label='Comfortable (<0.315 m/s²)',
           markerfacecolor='blue',
           markersize=10),

    Line2D([0], [0],
           marker='o',
           color='w',
           label='a Little Uncomfortable (0.315–0.63 m/s²)',
           markerfacecolor='green',
           markersize=10),

    Line2D([0], [0],
           marker='o',
           color='w',
           label='Fairly Uncomfortable (0.63-1 m/s²)',
           markerfacecolor='yellow',
           markersize=10),
    
    Line2D([0], [0],
           marker='o',
           color='w',
           label='Uncomfortable (1-1.6 m/s²)',
           markerfacecolor='orange',
           markersize=10),

    Line2D([0], [0],
           marker='o',
           color='w',
           label='Very Uncomfortable (1.6-2.5 m/s²)',
           markerfacecolor='red',
           markersize=10),

    Line2D([0], [0],
           marker='o',
           color='w',
           label='Extremly Uncomfortable (>2 m/s²)',
           markerfacecolor='black',
           markersize=10)]


plt.legend(handles=legend_elements)
plt.grid(True)
plt.show()

# ==========================================
# CREATE FOLIUM MAP
# ==========================================

# Create map centered on measurements
lat_med = Full_df["lat"].mean()
lon_med = Full_df["long"].mean()

complete_map = folium.Map(
    location=[lat_med, lon_med],
    zoom_start=15,
    tiles='OpenStreetMap'
)

comfort_layer = folium.FeatureGroup(name="Comfort", show=True)
surface_layer = folium.FeatureGroup(name="Surface prediction", show=False)
comfort_layer.add_to(complete_map)
surface_layer.add_to(complete_map)

# ==========================================
# ADD CONTINUOUS ROUTE LAYERS WITH POPUPS
# ==========================================

for i in range(len(Full_df) - 1):

    current_line = Full_df.iloc[i]
    next_line = Full_df.iloc[i + 1]

    # Only connect points from the same run/series
    if current_line["run_id"] != next_line["run_id"]:
        continue

    aw = current_line["aw_smooth"]
    color = current_line["iso_color"]
    surface = current_line["surface_prediction_smoothed"]
    surface_line_color = surface_color(surface)
    image_html = surface_image_html(surface)
    segment_locations = [
        [current_line["lat"], current_line["long"]],
        [next_line["lat"], next_line["long"]]
    ]

    # Create line segment between consecutive points
    folium.PolyLine(
        locations=segment_locations,
        color=color,
        weight=4,
        opacity=0.8,
        popup=(f"""
                <table>
                <tr><td colspan="2"><b><u>ISO Comfort Analysis</u></b></td></tr>
                
                <tr>
                    <td><u>aw:</u></td>
                    <td>{aw:.3f} m/s²</td>
                </tr>
                
                <tr>
                    <td><u>Velocity:</u></td>
                    <td>{current_line['v']:.1f} m/s</td>
                </tr>
                
                <tr>
                    <td><u>Predicted Surface:</u></td>
                    <td><b>{current_line['surface_prediction_smoothed']}</b></td>
                </tr>
                
                <tr>
                    <td><u>Comfort Class:</u></td>
                    <td>
                        {'Comfortable' if color == 'blue' else ''}
                        {'A Little Uncomfortable' if color == 'green' else ''}
                        {'Fairly Uncomfortable' if color == 'yellow' else ''}
                        {'Uncomfortable' if color == 'orange' else ''}
                        {'Very Uncomfortable' if color == 'red' else ''}
                        {'Extremely Uncomfortable' if color == 'black' else ''}
                    </td>
                </tr>
                {image_html}
                </table>
                """
        )
    ).add_to(comfort_layer)

    folium.PolyLine(
        locations=segment_locations,
        color=surface_line_color,
        weight=4,
        opacity=0.8,
        popup=(f"""
                <table>
                <tr><td colspan="2"><b><u>Surface Prediction</u></b></td></tr>
                
                <tr>
                    <td><u>Predicted Surface:</u></td>
                    <td><b>{surface}</b></td>
                </tr>
                
                <tr>
                    <td><u>aw:</u></td>
                    <td>{aw:.3f} m/sÂ²</td>
                </tr>
                
                <tr>
                    <td><u>Velocity:</u></td>
                    <td>{current_line['v']:.1f} m/s</td>
                </tr>
                {image_html}
                </table>
                """
        )
    ).add_to(surface_layer)

# ==========================================
# ADD POTHOLE AND SPEEDBUMP MARKERS
# ==========================================

for i in range(len(Full_df)):
    
    line = Full_df.iloc[i]
    surface = line["hazard_prediction_smoothed"]
    
    # Only mark Potholes and Speedbumps that are stable blocks.
    if not line.get("show_hazard_marker", False):
        continue

    if surface == "Pothole":
        folium.CircleMarker(
            location=[line["lat"], line["long"]],
            radius=6,
            color="darkred",
            fill=True,
            fill_color="red",
            fill_opacity=0.9,
            weight=3,
            popup=(f"""
                    <table>
                    <tr><td colspan="2"><b><u>POTHOLE DETECTED</u></b></td></tr>
                    
                    <tr>
                        <td><u>aw:</u></td>
                        <td>{line['aw_smooth']:.3f} m/s²</td>
                    </tr>
                    
                    <tr>
                        <td><u>Velocity:</u></td>
                        <td>{line['v']:.1f} m/s</td>
                    </tr>
                    
                    <tr>
                        <td><u>Latitude:</u></td>
                        <td>{line['lat']:.6f}</td>
                    </tr>
                    
                    <tr>
                        <td><u>Longitude:</u></td>
                        <td>{line['long']:.6f}</td>
                    </tr>
                    </table>
                    """
            )
        ).add_to(complete_map)
    
    elif surface == "Speedbump":
        folium.CircleMarker(
            location=[line["lat"], line["long"]],
            radius=6,
            color="darkviolet",
            fill=True,
            fill_color="violet",
            fill_opacity=0.9,
            weight=3,
            popup=(f"""
                    <table>
                    <tr><td colspan="2"><b><u>SPEEDBUMP DETECTED</u></b></td></tr>
                    
                    <tr>
                        <td><u>aw:</u></td>
                        <td>{line['aw_smooth']:.3f} m/s²</td>
                    </tr>
                    
                    <tr>
                        <td><u>Velocity:</u></td>
                        <td>{line['v']:.1f} m/s</td>
                    </tr>
                    
                    <tr>
                        <td><u>Latitude:</u></td>
                        <td>{line['lat']:.6f}</td>
                    </tr>
                    
                    <tr>
                        <td><u>Longitude:</u></td>
                        <td>{line['long']:.6f}</td>
                    </tr>
                    </table>
                    """
            )
        ).add_to(complete_map)

# ==========================================
# CUSTOM ISO LEGEND
# ==========================================

legend_html = """

<button id="route-mode-toggle" style="
  position: fixed;
  top: 20px;
  left: 100px;
  z-index: 10000;
  padding: 8px 12px;
  background-color: white;
  border: 2px solid grey;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  font-weight: bold;
">
  Show Surface Colors
</button>

<button id="legend-toggle" onclick="
  const legend = document.querySelector('.route-legend[data-active=&quot;true&quot;]');
  if (legend.style.display === 'none') {
    legend.style.display = 'block';
    this.textContent = 'Hide Legend';
  } else {
    legend.style.display = 'none';
    this.textContent = 'Show Legend';
  }
" style="
  position: fixed;
  top: 20px;
  left: 280px;
  z-index: 10000;
  padding: 8px 12px;
  background-color: white;
  border: 2px solid grey;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  font-weight: bold;
">
  Hide Legend
</button>

<div id="iso-legend" class="route-legend" data-active="true" style="
position: fixed;
bottom: 100px;
left: 100px;
width: 280px;
height: 420px;
background-color: white;
border:2px solid grey;
z-index:9999;
font-size:14px;
padding:10px;
overflow-y: auto;
">

<b>ISO 2631 Comfort Levels</b><br><br>

<i style="
background:blue;
width:14px;
height:14px;
float:left;
margin-right:8px;
opacity:0.8;"></i>

Comfortable (<0.315 m/s²)<br><br>

<i style="
background:green;
width:14px;
height:14px;
float:left;
margin-right:8px;
opacity:0.8;"></i>

a Little Uncomfortable (0.315–0.63)<br><br>

<i style="
background:yellow;
width:14px;
height:14px;
float:left;
margin-right:8px;
opacity:0.8;"></i>

Fairly Uncomfortable (0.63–1)<br><br>

<i style="
background:orange;
width:14px;
height:14px;
float:left;
margin-right:8px;
opacity:0.8;"></i>

Uncomfortable (1–1.6)<br><br>

<i style="
background:red;
width:14px;
height:14px;
float:left;
margin-right:8px;
opacity:0.8;"></i>

Very Uncomfortable (1.6–2.5)<br><br>

<i style="
background:black;
width:14px;
height:14px;
float:left;
margin-right:8px;
opacity:0.8;"></i>

Extremely Uncomfortable (>2.5)<br><br>

<hr style="margin:10px 0;">

<b>Road Hazards</b><br><br>

<i style="
background:red;
border: 2px solid darkred;
width:14px;
height:14px;
border-radius: 50%;
float:left;
margin-right:8px;
opacity:0.9;"></i>

Pothole<br><br>

<i style="
background:violet;
border: 2px solid darkviolet;
width:14px;
height:14px;
border-radius: 50%;
float:left;
margin-right:8px;
opacity:0.9;"></i>

Speedbump

</div>

<div id="surface-legend" class="route-legend" data-active="false" style="
position: fixed;
bottom: 100px;
left: 100px;
width: 280px;
background-color: white;
border:2px solid grey;
z-index:9999;
font-size:14px;
padding:10px;
display:none;
">

<b>Surface Prediction</b><br><br>

<i style="background:#2ca02c;width:14px;height:14px;float:left;margin-right:8px;opacity:0.8;"></i>
Smooth asphalt<br><br>

<i style="background:#ffbf00;width:14px;height:14px;float:left;margin-right:8px;opacity:0.8;"></i>
Medium asphalt<br><br>

<i style="background:#d62728;width:14px;height:14px;float:left;margin-right:8px;opacity:0.8;"></i>
Rough asphalt<br><br>

<i style="background:#9467bd;width:14px;height:14px;float:left;margin-right:8px;opacity:0.8;"></i>
Cobble<br><br>

<i style="background:#8c564b;width:14px;height:14px;float:left;margin-right:8px;opacity:0.8;"></i>
Unpaved<br><br>

<i style="background:#17becf;width:14px;height:14px;float:left;margin-right:8px;opacity:0.8;"></i>
Grass

</div>
"""

complete_map.get_root().html.add_child(
    folium.Element(legend_html)
)

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

complete_map.get_root().script.add_child(
    folium.Element(mode_toggle_js)
)

# ==========================================
# SAVE MAP
# ==========================================

complete_map.save(
    Output_Map_File
)
