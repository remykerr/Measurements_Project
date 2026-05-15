# -*- coding: utf-8 -*-
"""
Created on Wed May 13 15:15:37 2026

@author: remyk
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis, skew
import folium
from branca.colormap import LinearColormap

v_ref = 3

fsamp = 100.5

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
    dft = np.fft.rfft(segments, axis=1) / N

    # Frequency vector
    freq = np.fft.rfftfreq(N, d=1/fsamp)
    
    # Power spectrum
    ps = np.abs(dft)**2
    
    # One-sided correction
    if N % 2 == 0:
        ps[:,1:-1] *= 2
    else:
        ps[:,1:] *= 2
    df = fsamp / N

    psd = ps / df
    spec_energy = np.sum(psd, axis=1)
    band1 = (freq >= 3) & (freq <= 10)
    comfort_energy = np.sum(psd[:, band1],axis=1)
    band2 = (freq >= 10) & (freq <= 20)
    texture_energy = np.sum(psd[:, band2],axis=1)
    dominant_freq = freq[np.argmax(psd, axis=1)]
    
    FFT_metrics = pd.DataFrame({
    't': segment_times,
    'spec_energy': spec_energy,
    'comfort_energy': comfort_energy,
    'texture_energy': texture_energy,
    'dominant_freq': dominant_freq
})
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Compute signal metrics on a 1s window which also downsamples accelerometer data to 1 Hz 
    def rms(x):
        return np.sqrt(np.mean(x**2))
    def kurt(x):
        return kurtosis(x)
    def crest(x):
        return abs(max(x))/rms(x + 1e-12)
    def skew_(x):
        return skew(x)
    
    
    Accel_avg = Accel_z.resample('1s').mean()
    Accel_rms = Accel_z.resample("1s").apply(rms)
    Accel_std = Accel_z.resample('1s').std()
    Accel_peak = Accel_z.resample('1s').max()
    Accel_kurt = Accel_z.resample('1s').apply(kurt)
    Accel_crest = Accel_z.resample('1s').apply(crest)
    Accel_skew = Accel_z.resample('1s').apply(skew_)
    
    
    Accel_avg.columns  = ['az_avg']
    Accel_rms.columns  = ['az_rms']
    Accel_std.columns  = ['az_std']
    Accel_peak.columns = ['az_peak']
    Accel_kurt.columns = ['az_kurt']
    Accel_crest.columns = ['az_crest']
    Accel_skew.columns = ['az_skew']
    
    Accel_metrics = pd.concat([Accel_avg, Accel_rms, Accel_std, Accel_peak, Accel_kurt, Accel_crest, Accel_skew ],axis=1)
    
    #merging Gps data with accelerometer metrics with repect to timestamp
    Merged = pd.merge_asof(
        GPS.sort_values('t'),Accel_metrics.reset_index().sort_values('t'), on='t')
    
    
    # Add normalized metrics vith repesct to speed
    Merged["Norm_az_avg"]  = Merged["az_avg"]  / np.sqrt (v_ref/Merged["v"])
    Merged["Norm_az_rms"]  = Merged["az_rms"]  / np.sqrt (v_ref/Merged["v"])
    Merged["Norm_az_std"]  = Merged["az_std"]  / np.sqrt (v_ref/Merged["v"])
    Merged["Norm_az_peak"] = Merged["az_peak"] / np.sqrt (v_ref/Merged["v"])
    Merged["Norm_az_kurt"] = Merged['az_kurt'] / np.sqrt (v_ref/Merged["v"])
    Merged["Norm_az_crest"] = Merged['az_crest'] / np.sqrt (v_ref/Merged["v"])
    Merged["Norm_az_skew"] = Merged['az_skew'] / np.sqrt (v_ref/Merged["v"])
    
    #Merging FFt data and accel data
    Merged = pd.merge_asof(Merged.sort_values('t'),FFT_metrics.sort_values('t'), on='t')
    
    
    #Filtering out zones where speed is to low
    # Remove very low speeds
    Merged = Merged[Merged["v"] > 1]
    #adding final result to full data frame
    Full_df = pd.concat([Full_df,Merged])
    




plt.figure(figsize=(10,8))

scatter = plt.scatter(
    Full_df["long"],      # x = longitude
    Full_df["lat"],       # y = latitude
    c=Full_df["Norm_az_std"],      # color = averaged acceleration
    cmap='turbo',      # color map
    s=40                 # point size
)

# Color bar
cbar = plt.colorbar(scatter)
cbar.set_label('RMS Vertical Acceleration [m/s²]')

# Labels
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Title
plt.title('GPS Position Colored by Vertical Acceleration')

# Equal scaling for map-like appearance
plt.axis('equal')

plt.show()


#Create a folium map centered around the mean location of the data points
lat_med = Full_df["lat"].mean()
lon_med = Full_df["long"].mean()
complete_map = folium.Map(location=[lat_med, lon_med], zoom_start=15, tiles='OpenStreetMap')

# Define a colormap for the RMS values to visually differentiate areas of different vibration intensity
# We use a linear colormap that goes from green (low RMS) to red (high RMS), with yellow in the middle. 
# The vmin and vmax parameters are set to the minimum and maximum of the normalized RMS values to ensure that the colors are scaled appropriately.
colormap = LinearColormap(
    colors=['green', 'yellow', 'red'],
    vmin=Full_df["Norm_az_rms"].min(),
    vmax=Full_df["Norm_az_rms"].max()
)
colormap.add_to(complete_map) # Add the legend to the map

# We iterate through the data points and add a CircleMarker for each location, colored according to the normalized RMS value.
for i in range(len(Full_df)):
    line = Full_df.iloc[i]
    rms_norm = Full_df["Norm_az_rms"].iloc[i]
    
    folium.CircleMarker(
        location=[line["lat"], line["long"]],
        radius=4,
        color=colormap(rms_norm),
        fill=True,
        fill_opacity=0.7,
        # Data displayed when clicking on the point
        popup=(f"Time: {line['t']}\n"
               f"RMS: {line['Norm_az_rms']:.2f} m/s²\n"
               f"Velocity: {line['v']:.1f} m/s")
    ).add_to(complete_map)

# Save the map as an HTML file that can be opened in a web browser. 
# This file will contain the interactive map with all the markers and the legend.
complete_map.save("Full_Detailed_Rugosity_Map_Milan.html")

