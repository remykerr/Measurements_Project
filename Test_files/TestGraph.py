
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt
import folium
import re
from gravity_correction import correct_gravity
from scipy.stats import kurtosis, skew
import joblib

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
    raise KeyError(f"Expected a column starting with {prefix}. Available columns: {available}")
def estimate_sampling_frequency(time_values):
    time_values = np.asarray(time_values, dtype=float)
    time_steps = np.diff(time_values)
    time_steps = time_steps[time_steps > 0]
    if len(time_steps) == 0:
        raise ValueError("Cannot estimate sampling frequency from non-increasing timestamps.")
    return 1.0 / np.median(time_steps)
def iso_color(aw):

    if aw < 0.315:
        return 'blue'
    elif aw < 0.63:
        return 'green'
    elif aw < 1:
        return 'yellow'
    elif aw < 1.6:
        return 'orange'
    elif aw < 2.5:
        return 'red'
    else:
        return 'black'

def surface_image_html(surface):
    if surface in ("Pothole", "Speedbump"):
        return ''

    image_dir = Path(__file__).resolve().parent / "images"
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

def build_prediction_dataset(
    accel_file,
    gps_file,
    v_ref=2.5,
    fsamp=None,
    stationary_seconds=3.0,
    min_speed=1.0,
):
    """
    Computes exactly the same features used during training
    and returns GPS coordinates together with ML features.

    Returns
    -------
    DataFrame
        lat
        long
        v
        feature columns...
    """

    Accel_data = pd.read_csv(accel_file)
    GPS_data = pd.read_csv(gps_file)
    # ==========================================
    # ISO 2631-1 Wk weighting table
    # ==========================================
    
    ISO_FREQ = np.array([
        0.5,0.63,0.8,1,1.25,1.6,2,2.5,3.15,4,
        5,6.3,8,10,12.5,16,20,25,31.5,40,50
    ])
    
    ISO_WK = np.array([
        0.418,0.459,0.477,0.482,0.484,0.494,
        0.531,0.631,0.804,0.967,
        1.039,1.054,1.036,0.988,0.902,0.768,
        0.636,0.513,0.405,0.314,0.246
    ])
    # ==================================
    # GRAVITY CORRECTION
    # ==================================

    Accel_corrected, _ = correct_gravity(
        Accel_data,
        stationary_seconds=stationary_seconds,
        expected_g=9.81,
    )

    True_z_accel_data = Accel_corrected["a_vertical_no_g"]

    current_fsamp = (
        estimate_sampling_frequency(Accel_data["Time (s)"])
        if fsamp is None
        else fsamp
    )

    # ==================================
    # HIGH PASS FILTER
    # ==================================

    def hp_filter(signal, fs, fc=0.5, order=4):
        b, a = butter(order, fc/(fs/2), btype="high")
        return filtfilt(b, a, signal)

    True_z_accel_data = hp_filter(
        True_z_accel_data,
        fs=current_fsamp
    )

    Accel_z = pd.DataFrame({
        "t": Accel_data["Time (s)"],
        "az": True_z_accel_data
    })

    # ==================================
    # GPS DATA
    # ==================================

    latitude_col = find_column(GPS_data, "latitude")
    longitude_col = find_column(GPS_data, "longitude")

    GPS = pd.DataFrame({
        "t": GPS_data["Time (s)"],
        "lat": GPS_data[latitude_col],
        "long": GPS_data[longitude_col],
        "v": GPS_data["Velocity (m/s)"]
    })

    # ==================================
    # TIME INDEXING
    # ==================================

    Accel_z["t"] = pd.to_timedelta(Accel_z["t"], unit="s")
    GPS["t"] = pd.to_timedelta(GPS["t"], unit="s")

    Accel_z = Accel_z.set_index("t")

    # ==================================
    # FFT FEATURES
    # ==================================

    window_size = int(round(current_fsamp))

    signal = Accel_z["az"].values

    segments = []
    segment_times = []

    for k in range(0, len(signal)-window_size, window_size):
        segments.append(signal[k:k+window_size])
        center_index = k + window_size // 2
        segment_times.append(Accel_z.index[center_index])

    segments = np.array(segments)

    N = window_size

    window = np.hanning(N)

    segments_windowed = segments * window

    dft = np.fft.rfft(
        segments_windowed,
        axis=1
    ) / N

    freq = np.fft.rfftfreq(
        N,
        d=1/current_fsamp
    )

    ps = np.abs(dft)**2

    if N % 2 == 0:
        ps[:,1:-1] *= 2
    else:
        ps[:,1:] *= 2

    df = current_fsamp / N

    psd = ps / df
    
    # ==========================================
    # ISO 2631-1 Wk weighted acceleration
    # ==========================================
    
    wk_interp = np.interp(
        freq,
        ISO_FREQ,
        ISO_WK,
        left=0,
        right=0
    )
    
    psd_weighted = psd * (wk_interp ** 2)
    
    # Frequency weighted RMS acceleration
    aw = np.sqrt(
        np.sum(psd_weighted * df, axis=1)
    )

    band0 = (freq >= 0.5) & (freq <= 50)
    band1 = (freq >= 3) & (freq <= 10)
    band2 = (freq >= 10) & (freq <= 20)

    spec_energy = np.sum(psd[:, band0], axis=1)
    comfort_energy = np.sum(psd[:, band1], axis=1)
    texture_energy = np.sum(psd[:, band2], axis=1)

    dominant_freq = freq[np.argmax(psd, axis=1)]

    periodicity_strengths = []

    for seg in segments:

        seg_centered = seg - np.mean(seg)

        acf = np.correlate(
            seg_centered,
            seg_centered,
            mode="full"
        )

        acf = acf[len(acf)//2:]

        acf = acf / np.max(acf)

        periodicity_strengths.append(
            np.max(acf[1:])
        )

    FFT_metrics = pd.DataFrame({
    "t": segment_times,
    # PSD metrics
    "spec_energy": spec_energy,
    "comfort_energy": comfort_energy,
    "texture_energy": texture_energy,
    "dominant_freq": dominant_freq,
    # Periodicity metrics
    "periodicity_strength": periodicity_strengths,
    # ISO comfort metrics
    "aw": aw,})
    
    # ==================================
    # TIME DOMAIN FEATURES
    # ==================================

    def rms(x):
        return np.sqrt(np.mean(x**2))

    def kurt(x):
        return kurtosis(x)

    def crest(x):
        return abs(np.max(x)) / (rms(x) + 1e-12)

    def skew_(x):
        return skew(x)

    Accel_avg = Accel_z.resample("1s").mean()
    Accel_rms = Accel_z.resample("1s").apply(rms)
    Accel_std = Accel_z.resample("1s").std()
    Accel_peak = Accel_z.resample("1s").max()
    Accel_kurt = Accel_z.resample("1s").apply(kurt)
    Accel_crest = Accel_z.resample("1s").apply(crest)
    Accel_skew = Accel_z.resample("1s").apply(skew_)

    Accel_avg.columns = ["az_avg"]
    Accel_rms.columns = ["az_rms"]
    Accel_std.columns = ["az_std"]
    Accel_peak.columns = ["az_peak"]
    Accel_kurt.columns = ["az_kurt"]
    Accel_crest.columns = ["az_crest"]
    Accel_skew.columns = ["az_skew"]

    Accel_metrics = pd.concat([
        Accel_avg,
        Accel_rms,
        Accel_std,
        Accel_peak,
        Accel_kurt,
        Accel_crest,
        Accel_skew
    ], axis=1)

    # ==================================
    # MERGING
    # ==================================

    Merged = pd.merge_asof(
        GPS.sort_values("t"),
        Accel_metrics.reset_index().sort_values("t"),
        on="t"
    )

    Merged["Norm_az_avg"] = (
        Merged["az_avg"] /
        np.sqrt(v_ref / Merged["v"])
    )

    Merged["Norm_az_rms"] = (
        Merged["az_rms"] /
        np.sqrt(v_ref / Merged["v"])
    )

    Merged["Norm_az_std"] = (
        Merged["az_std"] /
        np.sqrt(v_ref / Merged["v"])
    )

    Merged["Norm_az_peak"] = (
        Merged["az_peak"] /
        np.sqrt(v_ref / Merged["v"])
    )

    Merged["Norm_az_kurt"] = (
        Merged["az_kurt"] /
        np.sqrt(v_ref / Merged["v"])
    )

    Merged["Norm_az_crest"] = (
        Merged["az_crest"] /
        np.sqrt(v_ref / Merged["v"])
    )

    Merged["Norm_az_skew"] = (
        Merged["az_skew"] /
        np.sqrt(v_ref / Merged["v"])
    )

    Merged = pd.merge_asof(
        Merged.sort_values("t"),
        FFT_metrics.sort_values("t"),
        on="t"
    )

    # Remove low speed samples

    Merged = Merged[Merged["v"] > min_speed]

    return Merged.reset_index(drop=True)


Full_df = pd.DataFrame({})
for i in range (1,8) :
    Accel_file = f"Test_files/Measurements/Accelerometer_{i}.csv"
    GPS_file = f"Test_files/Measurements/Location_{i}.csv"
    single_run_df = build_prediction_dataset(Accel_file, GPS_file)
    single_run_df["run_id"] = i
    Full_df = pd.concat([Full_df,single_run_df])

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

classifier = joblib.load(f"Test_files/Road_Surface_Classifier.pkl")

surface_predictions = classifier.predict(X)

Full_df["surface_prediction"] = surface_predictions

# Smooth the ISO comfort score and the surface label to reduce second-to-second noise.
Full_df["aw_smooth"] = Full_df.groupby("run_id")["aw"].transform(
    lambda s: s.rolling(window=3, center=True, min_periods=1).mean()
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
           .apply(lambda s: smooth_labels(s, window=3))
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
    Full_df.groupby("run_id")["surface_prediction_smoothed"]
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

# ==========================================
# ADD ISO-COLORED CONTINUOUS LINE WITH POPUPS
# ==========================================

for i in range(len(Full_df) - 1):

    current_line = Full_df.iloc[i]
    next_line = Full_df.iloc[i + 1]

    # Only connect points from the same run/series
    if current_line["run_id"] != next_line["run_id"]:
        continue

    aw = current_line["aw_smooth"]
    color = current_line["iso_color"]
    image_html = surface_image_html(current_line["surface_prediction_smoothed"])

    # Create line segment between consecutive points
    folium.PolyLine(
        locations=[
            [current_line["lat"], current_line["long"]],
            [next_line["lat"], next_line["long"]]
        ],
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
    ).add_to(complete_map)

# ==========================================
# ADD POTHOLE AND SPEEDBUMP MARKERS
# ==========================================

for i in range(len(Full_df)):
    
    line = Full_df.iloc[i]
    surface = line["surface_prediction_smoothed"]
    
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

<button id="legend-toggle" onclick="
  const legend = document.getElementById('iso-legend');
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
  Hide Legend
</button>

<div id="iso-legend" style="
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
"""

complete_map.get_root().html.add_child(
    folium.Element(legend_html)
)

# ==========================================
# SAVE MAP
# ==========================================

complete_map.save(
    "Rugosity_Map_Milan_CL.html"
)





    