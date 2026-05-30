# -*- coding: utf-8 -*-
"""
Created on Wed May 13 15:15:37 2026

@author: remyk
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt
import folium

v_ref = 3

fsamp = 100.5

# ==========================================
# ISO 2631-1 Wk weighting table
# ==========================================

iso_freq = np.array([
    0.5,0.63,0.8,1,1.25,1.6,2,2.5,3.15,4,
    5,6.3,8,10,12.5,16,20,25,31.5,40,50
])

iso_wk = np.array([
    0.418,0.459,0.477,0.482,0.484,0.494,
    0.531,0.631,0.804,0.967,
    1.039,1.054,1.036,0.988,0.902,0.768,
    0.636,0.513,0.405,0.314,0.246
])

Full_df = pd.DataFrame({})


for i in range (1,8) :
    Accel_file = f"Measurements/Accelerometer_{i}.csv"
    GPS_file = f"Measurements/Location_{i}.csv"
    
    Phone_angle = 35 * (np.pi/180)
    
    Accel_data = pd.read_csv(Accel_file)
    GPS_data = pd.read_csv(GPS_file)
    
    #Since phone was at an angle on the bike, we recreate the z component using trig and we substract gravity g = 9.81
    True_z_accel_data = (np.cos(Phone_angle)*Accel_data["Z (m/s^2)"] + np.sin(Phone_angle)*Accel_data["Y (m/s^2)"])-9.81 
    
    #Applying a High Pass filter with cutoff at 0.5hz to remove drift and slow tilt change from measurments
    

    def hp_filter(signal, fs=100.0, fc=0.5, order=4):
        b, a = butter(order, fc / (fs/2.0), btype='high')
        return filtfilt(b, a, signal)   # filtfilt = zero-phase, no distortion
    
    True_z_accel_data = hp_filter(True_z_accel_data)
    True_z_accel_data = pd.DataFrame(True_z_accel_data)

    
    # plt.figure()
    # plt.plot(Accel_data["Time (s)"], True_z_accel_data, label='Recorded Acceleration')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Acc [m/s^2]')
    # plt.title('Recorded Acceleration Over Time')
    # plt.legend()
    # plt.show()
    
    Accel_z = pd.concat([Accel_data["Time (s)"],True_z_accel_data], axis=1)
    Accel_z.columns = ['t', "az"]
    
    # plt.figure()
    # plt.plot(GPS_data["Time (s)"], GPS_data["Velocity (m/s)"], label='Recorded GPS')
    # plt.xlabel('Time [s]')
    # plt.ylabel('V [m/s]')
    # plt.title('Recorded Speed Over Time')
    # plt.legend()
    # plt.show()
    
    GPS = pd.concat([GPS_data["Time (s)"],GPS_data["Latitude (°)"],GPS_data["Longitude (°)"],GPS_data["Velocity (m/s)"]], axis=1)
    GPS.columns = ['t', "lat", "long", "v"]
    
    #Different and Unstable sampling frequency (100,5hz for accel, 1hz for GPS) -> use pandas time resampling to connect the two data sets 
    # Convert to timedelta
    Accel_z['t'] = pd.to_timedelta(Accel_z['t'], unit='s')
    
    GPS['t'] = pd.to_timedelta(GPS['t'], unit='s')
    
    # Set index
    Accel_z = Accel_z.set_index('t')
    
        
    # ===============================
    # FFT / PSD ANALYSIS
    # ===============================
    
    window_size = int(fsamp)
    
    segments = []
    segment_times = []
    
    signal = Accel_z["az"].values
    for k in range(0, len(signal)-window_size, window_size):
    
        seg = signal[k:k+window_size]
    
        segments.append(seg)
    
        segment_times.append(
            Accel_z.index[k]
        )
    
    segments = np.array(segments)
    N = window_size
    window = np.hanning(N)
    segments_windowed = segments * window
    

    # FFT
    dft = np.fft.rfft(segments_windowed, axis=1) / N

    # Frequency vector
    freq = np.fft.rfftfreq(N, d=1/fsamp)
    
    # Power spectrum
    ps = np.abs(dft)**2
    
    # One-sided correction
    if N % 2 == 0:
        ps[:,1:-1] *= 2
    else:
        ps[:,1:] *= 2
    
    # Power density spectrum
    df = fsamp / N
    psd = ps / df
    
    # ==========================================
    # ISO weighting
    # ==========================================
    
    wk_interp = np.interp(freq,iso_freq,iso_wk,left=0,right=0)
    
    psd_weighted = psd * (wk_interp**2)
    
    # ISO weighted RMS
    aw = np.sqrt(np.sum(psd_weighted * df, axis=1))
    
    # ==========================================
    # ISO 2631 comfort classification
    # ==========================================
    
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
    
    FFT_metrics = pd.DataFrame({
    't': segment_times,
    "aw" : aw})
    FFT_metrics["iso_color"] = FFT_metrics["aw"].apply(iso_color)
    
    #merging Gps data with accelerometer metrics with repect to timestamp
    Merged = pd.merge_asof(
        GPS.sort_values('t'),FFT_metrics.reset_index().sort_values('t'), on='t')
    
    # Remove very low speeds
    Merged = Merged[Merged["v"] > 1]
    
    #adding final result to full data frame
    Full_df = pd.concat([Full_df,Merged])
    





































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
# ADD ISO-COLORED POINTS
# ==========================================

for i in range(len(Full_df)):

    line = Full_df.iloc[i]

    aw = line["aw"]

    # Use already computed ISO color
    color = line["iso_color"]

    folium.CircleMarker(

        location=[line["lat"], line["long"]],

        radius=4,

        color=color,

        fill=True,

        fill_color=color,

        fill_opacity=0.8,

        weight=1,

        popup=(

            f"<b>ISO Comfort Analysis</b><br>"

            f"Time: {line['t']}<br>"

            f"aw: {aw:.3f} m/s²<br>"

            f"Velocity: {line['v']:.1f} m/s<br>"

            f"Class: "

            f"{'Comfortable' if color == 'blue' else ''}"

            f"{'a Little Uncomfortable' if color == 'green' else ''}"

            f"{'Fairly Uncomfortable' if color == 'yellow' else ''}"

            f"{'Uncomfortable' if color == 'orange' else ''}"

            f"{'Very Uncomfortable' if color == 'red' else ''}"

            f"{'Extremely Uncomfortable' if color == 'black' else ''}"

        )

    ).add_to(complete_map)

# ==========================================
# CUSTOM ISO LEGEND
# ==========================================

legend_html = """

<div style="
position: fixed;
bottom: 100px;
left: 100px;
width: 260px;
height: 300px;
background-color: white;
border:2px solid grey;
z-index:9999;
font-size:14px;
padding:10px;
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

Extremely Uncomfortable (>2.5)

</div>
"""

complete_map.get_root().html.add_child(
    folium.Element(legend_html)
)

# ==========================================
# SAVE MAP
# ==========================================

complete_map.save(
    "Full_Detailed_Rugosity_Map_Milan.html"
)
