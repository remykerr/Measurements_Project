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

v_ref = 3

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
    
    #Filtering out zones where speed is to low
    # Remove very low speeds
    Merged = Merged[Merged["v"] > 1]
    
    print(Merged.head())
    
    # Add normalized metrics vith repesct to speed
    Merged["Norm_az_avg"]  = Merged["az_avg"]  / np.sqrt (v_ref/Merged["v"])
    Merged["Norm_az_rms"]  = Merged["az_rms"]  / np.sqrt (v_ref/Merged["v"])
    Merged["Norm_az_std"]  = Merged["az_std"]  / np.sqrt (v_ref/Merged["v"])
    Merged["Norm_az_peak"] = Merged["az_peak"] / np.sqrt (v_ref/Merged["v"])
    Merged["Norm_az_kurt"] = Merged['az_kurt'] / np.sqrt (v_ref/Merged["v"])
    Merged["Norm_az_crest"] = Merged['az_crest'] / np.sqrt (v_ref/Merged["v"])
    Merged["Norm_az_skew"] = Merged['az_skew'] / np.sqrt (v_ref/Merged["v"])
    
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

