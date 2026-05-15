import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import folium
from branca.colormap import LinearColormap

acc_data = pd.read_csv('Project/Raw Data/Accelerometer.csv')
location_data = pd.read_csv('Project/Raw Data/Location.csv')

#Assigning column names to the dataframes
acc_data.columns = ['time_acc', 'x', 'y', 'z']
location_data.columns = ['time_loc', 'latitude', 'longitude','altitude','velocity','direction', 'H_accuracy', 'V_accuracy']

#Calculate sampling frequency of accelerometer and location data
time_diffs = acc_data['time_acc'].diff().dropna()
time_diffs_location = location_data['time_loc'].diff().dropna()
sf_acc = 1 / time_diffs.mean()
sf_location = 1 / time_diffs_location.mean()


#Getting the real z and y axis values assuming an inclination of 35 degrees
inclination_angle = 35
acc_z_real = (acc_data['z'] * np.cos(np.radians(inclination_angle))) + (acc_data['y'] * np.sin(np.radians(inclination_angle)))
acc_y_real = (acc_data['y'] * np.cos(np.radians(inclination_angle))) - (acc_data['z'] * np.sin(np.radians(inclination_angle)))


# Cleaning the data
# Remove DC offset per axis
acc_x_clean = acc_data['x'] - acc_data['x'].mean()
acc_y_clean = acc_y_real - acc_y_real.mean()
acc_z_clean = acc_z_real - acc_z_real.mean()

def Triggers(acc_y_clean, sf_acc):
    #Estimating base noise level using the first 0.5s samples
    noise_level_y = acc_y_clean.iloc[:int(0.5 * sf_acc)].std()
    #Define Trigger level depending on initial noise level
    if noise_level_y > 0.2: 
        Trigger_Level_y = noise_level_y * 1.5  # Si ya hay movimiento, bajamos el multiplicador
    else:
        Trigger_Level_y = noise_level_y * 5    # Si hay silencio real, mantenemos el estándar 5-sigma

    #Calculate stability in a 1s window
    window_size = int(sf_acc)  # 1 second window
    stability_y = acc_y_clean.rolling(window=window_size).std()

    #Search for the moment when stability exceeds the trigger level
    t_trigger = acc_data[stability_y > Trigger_Level_y]['time_acc'].iloc[0]

    #Cut initial data depending on the trigger time
    if t_trigger < 0.5:
            t_trigger = 0.0
    else:
        t_trigger = t_trigger + 2.0
    
    #Identify times were acceleration exceeds the trigger level
    trigger_met = acc_data[stability_y > Trigger_Level_y]
    #Get the last time where the trigger condition is met
    t_end = trigger_met['time_acc'].iloc[-1]
    #Subtract a safety margin of 2 seconds to the end time
    t_end = t_end - 2.0

    return t_trigger, t_end
    
t0= Triggers(acc_y_clean, sf_acc)[0]
tf= Triggers(acc_y_clean, sf_acc)[1]

#Cut the acceleration and location data to the time interval between t0 and tf
acc_final = acc_data[(acc_data['time_acc'] >= t0) & (acc_data['time_acc'] <= tf)]
loc_final = location_data[(location_data['time_loc'] >= t0) & (location_data['time_loc'] <= tf)]

#Resample the location data to match the sampling frequency of the accelerometer data

#Convert time to timedelta for resampling
acc_final['t'] = pd.to_timedelta(acc_data['time_acc'], unit='s')
loc_final['t'] = pd.to_timedelta(loc_final['time_loc'], unit='s')

#Set time as index for resampling
acc_final.set_index('t', inplace=True)
loc_final.set_index('t', inplace=True)

def metrics_acc(acc_final,coluumn='z'):
    resampler=acc_final[coluumn].resample('1s')  # Resample to 1s intervals
    metrics=pd.DataFrame()

    #Calculate RMS, std, peak, kurtosis, and skewness for each window
    metrics['RMS'] = np.sqrt(resampler.apply(lambda x: np.sqrt(np.mean(x**2))))
    metrics['std'] = resampler.std()
    metrics['peak'] = resampler.max()
    metrics['kurtosis'] = resampler.apply(lambda x: kurtosis(x, fisher=False))
    metrics['skewness'] = resampler.apply(lambda x: skew(x))

    #Calculate Crest factor for each window
    metrics['Crest_Factor'] = resampler.apply(lambda x: np.max(np.abs(x)) / np.sqrt(np.mean(x**2)))

    return metrics

#Calculate the metrics for the z-axis
metrics_z = metrics_acc(acc_final, 'z')

#Combine the metrics with the location data

merged_data = pd.merge_asof(
     metrics_z.reset_index(), 
     loc_final.reset_index(), 
     on='t')

#Filter very slow speeds (velocity < 1 m/s) to focus on relevant movement data
merged_data_final = merged_data[merged_data['velocity'] > 1.0].copy()

#Continuity Analysis: Check for gaps in the data
stops= merged_data['velocity'] < 1
#Percentage of time spent still
T_still= stops.sum()
Percentage_still = (T_still / len(merged_data)) * 100
print(f"Percentage of time spent still: {Percentage_still:.2f}%")

#Normalize metrics with filtered data to make them comparable across different conditions and subjects
metrics_columns = ['RMS', 'std', 'peak', 'kurtosis', 'skewness', 'Crest_Factor']
filtered_metrics = merged_data_final[metrics_columns]

metrics_mean = filtered_metrics.mean()
metrics_std = filtered_metrics.std()
normalized_metrics = (filtered_metrics - metrics_mean) / metrics_std

#Plot Rugosity Map

# We use normalized RMS as the color for the scatter plot, which represents the vibration intensity at each location
scatter = plt.scatter(
    merged_data_final["longitude"], 
    merged_data_final["latitude"], 
    c=normalized_metrics["RMS"], 
    cmap='plasma', 
    s=40)

cbar = plt.colorbar(scatter)
cbar.set_label('Vibration Intensity (Normalized RMS)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Quality of the Ride: Rugosity Map')
plt.axis('equal') 
plt.show()


#Create a folium map centered around the mean location of the data points
lat_med = merged_data_final["latitude"].mean()
lon_med = merged_data_final["longitude"].mean()
complete_map = folium.Map(location=[lat_med, lon_med], zoom_start=15, tiles='OpenStreetMap')

# Define a colormap for the RMS values to visually differentiate areas of different vibration intensity
# We use a linear colormap that goes from green (low RMS) to red (high RMS), with yellow in the middle. 
# The vmin and vmax parameters are set to the minimum and maximum of the normalized RMS values to ensure that the colors are scaled appropriately.
colormap = LinearColormap(
    colors=['green', 'yellow', 'red'],
    vmin=normalized_metrics["RMS"].min(),
    vmax=normalized_metrics["RMS"].max()
)
colormap.add_to(complete_map) # Add the legend to the map

# We iterate through the data points and add a CircleMarker for each location, colored according to the normalized RMS value.
for i in range(len(merged_data_final)):
    line = merged_data_final.iloc[i]
    rms_norm = normalized_metrics["RMS"].iloc[i]
    
    folium.CircleMarker(
        location=[line["latitude"], line["longitude"]],
        radius=4,
        color=colormap(rms_norm),
        fill=True,
        fill_opacity=0.7,
        # Data displayed when clicking on the point
        popup=(f"Time: {line['t']}\n"
               f"RMS: {line['RMS']:.2f} m/s²\n"
               f"Velocity: {line['velocity']:.1f} m/s")
    ).add_to(complete_map)

# Save the map as an HTML file that can be opened in a web browser. 
# This file will contain the interactive map with all the markers and the legend.
complete_map.save("Detailed_Rugosity_Map_Milan.html")











