# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 17:10:49 2026

@author: remyk
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt

#Use to generate graphs for presentation

Accel_file_1 = "Measurements/Accelerometer_rough.csv"
Accel_file_3 = "Measurements/Accelerometer_smooth.csv"
Accel_file_2 = "Measurements/Accelerometer_cobble.csv"

Accel_data_1 = pd.read_csv(Accel_file_1)
Accel_data_2 = pd.read_csv(Accel_file_2)
Accel_data_3 = pd.read_csv(Accel_file_3)




plt.figure()
plt.plot(Accel_data_1["Time (s)"], Accel_data_1["Z (m/s^2)"], label='Rough Asphalt')
plt.plot(Accel_data_2["Time (s)"], Accel_data_2["Z (m/s^2)"], label='Cobble')
plt.plot(Accel_data_3["Time (s)"], Accel_data_3["Z (m/s^2)"], label='Smooth Asphalt')
plt.xlabel('Time [s]')
plt.ylabel('Acc [m/s^2]')
plt.title('Recorded Acceleration Over Time')
plt.legend()
plt.show()

Phone_angle = 35 * (np.pi/180)
True_z_accel_data_1 = (np.cos(Phone_angle)*Accel_data_1["Z (m/s^2)"] + np.sin(Phone_angle)*Accel_data_1["Y (m/s^2)"])-9.81 
True_z_accel_data_2 = (np.cos(Phone_angle)*Accel_data_2["Z (m/s^2)"] + np.sin(Phone_angle)*Accel_data_2["Y (m/s^2)"])-9.81 
True_z_accel_data_3 = (np.cos(Phone_angle)*Accel_data_3["Z (m/s^2)"] + np.sin(Phone_angle)*Accel_data_3["Y (m/s^2)"])-9.81 
def hp_filter(signal, fs=100.0, fc=0.5, order=4):
    b, a = butter(order, fc / (fs/2.0), btype='high')
    return filtfilt(b, a, signal)   # filtfilt = zero-phase, no distortion

True_z_accel_data_1 = hp_filter(True_z_accel_data_1)
True_z_accel_data_2 = hp_filter(True_z_accel_data_2)
True_z_accel_data_3 = hp_filter(True_z_accel_data_3)


# ===============================
# FFT / PSD ANALYSIS
# ===============================
fsamp = 100.5


segment_1 = np.array(True_z_accel_data_1).flatten()
segment_1 = segment_1[1000:3024]
segment_2 = np.array(True_z_accel_data_2).flatten()
segment_2 = segment_2[1000:3024]
segment_3 = np.array(True_z_accel_data_3).flatten()
segment_3 = segment_3[1000:3024]


N = len(segment_1)

# FFT
dft_1 = np.fft.rfft(segment_1) / N
dft_2 = np.fft.rfft(segment_2) / N
dft_3 = np.fft.rfft(segment_3) / N

# Frequency vector
freq = np.fft.rfftfreq(N, d=1/fsamp)

# Power spectrum
ps_1 = np.abs(dft_1)**2
ps_2 = np.abs(dft_2)**2
ps_3 = np.abs(dft_3)**2
# One-sided correction
if N % 2 == 0:
    ps_1[1:-1] *= 2
    ps_2[1:-1] *= 2
    ps_3[1:-1] *= 2
else:
    ps_1[1:] *= 2
    ps_2[1:] *= 2
    ps_3[1:] *= 2

# Power density spectrum
df = fsamp / N
psd_1 = ps_1 / df
psd_2 = ps_2 / df
psd_3 = ps_3 / df




# ===============================
# PSD PLOT
# ===============================
plt.figure(figsize=(8,4))
plt.semilogy(freq, psd_1,label='Rough Asphalt')
plt.semilogy(freq, psd_2,label='Cobble')
plt.semilogy(freq, psd_3,label='Smooth Asphalt')
plt.xlabel('Frequency [Hz]')
plt.ylabel(r'PSD [(m/s²)²/Hz]')
plt.title('Power Spectral Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

