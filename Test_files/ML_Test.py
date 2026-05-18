# -*- coding: utf-8 -*-
"""
Created on Mon May 18 19:21:17 2026

@author: remyk
"""

#same base as for basic code, without GPS so no Speed normalizing and speed filtering

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis, skew
import folium
from branca.colormap import LinearColormap


fsamp = 100.5

# surface_data = {}

# #Run this for each measurement for each surface type, accounting for 5 measures per type : asphalt, dmg asphalt, cobble, dirt, pothole, speed_bump

# surface_types = ["asphalt","dmg asphalt","cobble","dirt","pothole","speed_bump"]

# for surface in surface_types :
#     surface_data[surface] = pd.DataFrame({})
#     for i in range (1,6) :
#         Accel_file = f"{surface}_{i}"
#         #create the data frame with accel and metrics called temp_data
#         temp_data["surface"] = surface
#         surface_data[surface] = pd.concat([surface_data[surface], temp_data],ignore_index=True)
        
    

Accel_file = f"Measurements/Accelerometer_{1}.csv"

Phone_angle = 35 * (np.pi/180)

Accel_data = pd.read_csv(Accel_file)

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


#Different and Unstable sampling frequency (100,5hz for accel, 1hz for GPS) -> use pandas time resampling to connect the two data sets 
# Convert to timedelta
Accel_z['t'] = pd.to_timedelta(Accel_z['t'], unit='s')


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



#Merging FFt data and accel data
Merged = pd.merge_asof(Accel_metrics.sort_values('t'),FFT_metrics.sort_values('t'), on='t')

#adding final result to full data frame
Full_df = pd.concat([Full_df,Merged])
