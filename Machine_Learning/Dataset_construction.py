import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis, skew

BASE_DIR = Path(__file__).resolve().parent


def find_column(data, exact_name=None, prefix=None):
    if exact_name in data.columns:
        return exact_name
    if prefix is not None:
        prefix = prefix.lower()
        for column in data.columns:
            if column.strip().lower().startswith(prefix):
                return column
    available = ", ".join(data.columns)
    expected = exact_name or f"a column starting with {prefix}"
    raise KeyError(f"Expected {expected}. Available columns: {available}")


def build_surface_dataset(
    base_dir=BASE_DIR,
    surface_types=("cobble", "Rough_asphalt", "Smooth_asphalt"),
    train_measurements=(1, 2, 3, 4, 5, 6),
    test_measurements=(7,),
    v_ref=2.5,
    fsamp=100.5,
    phone_angle_deg=42,
    skip_missing_measurements=True,
):
    """
    Build the machine-learning dataset from accelerometer and GPS measurements.

    The function reads each measurement folder in Clean_measurements_ML, computes
    the corrected vertical acceleration, filters it, extracts time-domain and
    frequency-domain features on 1-second windows, merges those features with GPS
    speed/time data, removes very slow samples, and adds the surface label.

    Measurements listed in train_measurements are returned in the training
    DataFrame, while measurements listed in test_measurements are returned in the
    test DataFrame. Both returned DataFrames contain only model-ready feature
    columns plus the target column "srf".

    If skip_missing_measurements is True, missing measurement folders are skipped
    with a warning. This allows adding a new surface class before all measurement
    IDs exist for that class.
    """
    base_dir = Path(base_dir)
    surface_data = {}        
    test_data = {}
    measurement_ids = sorted(set(train_measurements) | set(test_measurements))

    for surface in surface_types :
        surface_data[surface] = pd.DataFrame({})
        test_data[surface] = pd.DataFrame({})
        for i in measurement_ids :
            measure_dir = base_dir / "Clean_measurements_ML" / f"{surface}_{i}"
            Accel_file = measure_dir / "Accelerometer.csv"
            GPS_file = measure_dir / "Location.csv"

            if not Accel_file.exists() or not GPS_file.exists():
                if skip_missing_measurements:
                    warnings.warn(f"Skipping missing measurement: {measure_dir}")
                    continue
                missing_files = [
                    str(file_path)
                    for file_path in (Accel_file, GPS_file)
                    if not file_path.exists()
                ]
                raise FileNotFoundError(
                    "Missing measurement files: " + ", ".join(missing_files)
                )
            
            Phone_angle = phone_angle_deg * (np.pi/180)
            
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
            
            latitude_col = find_column(GPS_data, prefix="latitude")
            longitude_col = find_column(GPS_data, prefix="longitude")
            GPS = pd.concat([GPS_data["Time (s)"],GPS_data[latitude_col],GPS_data[longitude_col],GPS_data["Velocity (m/s)"]], axis=1)
            GPS.columns = ['t', "lat", "long", "v"]
            
            #Different and Unstable sampling frequency (100,5hz for accel, 1hz for GPS) -> use pandas time resampling to connect the two data sets 
            # Convert to timedelta
            Accel_z['t'] = pd.to_timedelta(Accel_z['t'], unit='s')
            
            GPS['t'] = pd.to_timedelta(GPS['t'], unit='s')
            
            # Set index
            Accel_z = Accel_z.set_index('t')
            
            # ===============================
            # FFT / PSD + PERIODICITY ANALYSIS
            # ===============================
            
            window_size = int(fsamp)
            
            segments = []
            segment_times = []
            
            signal = Accel_z["az"].values
            
            for k in range(0, len(signal)-window_size, window_size):
                seg = signal[k:k+window_size]
                segments.append(seg)
                segment_times.append(Accel_z.index[k])
            segments = np.array(segments)
            
            N = window_size
            
            # Apply Hanning window
            window = np.hanning(N)
            segments_windowed = segments * window
            
            # ===============================
            # FFT
            # ===============================
            dft = np.fft.rfft(segments_windowed, axis=1) / N
            # Frequency vector
            freq = np.fft.rfftfreq(N, d=1/fsamp)
            
            # ===============================
            # POWER SPECTRUM
            # ===============================
            ps = np.abs(dft)**2
            # One-sided correction
            if N % 2 == 0:
                ps[:,1:-1] *= 2
            else:
                ps[:,1:] *= 2
            df = fsamp / N
            # Power Spectral Density
            psd = ps / df
            
            # ===============================
            # PSD FEATURES
            # ===============================
            # Total spectral energy
            spec_energy = np.sum(psd, axis=1)
            # Human comfort band (ISO-sensitive region)
            band1 = (freq >= 3) & (freq <= 10)
            comfort_energy = np.sum(psd[:, band1], axis=1)
            # Fine texture / harshness band
            band2 = (freq >= 10) & (freq <= 20)
            texture_energy = np.sum(psd[:, band2], axis=1)
            # Dominant frequency
            dominant_freq = freq[np.argmax(psd, axis=1)]
            
            # ===============================
            # PERIODICITY ANALYSIS
            # ===============================
            
            periodicity_strengths = []
            for seg in segments:
                # Remove mean
                seg_centered = seg - np.mean(seg)
                # Autocorrelation
                acf = np.correlate(seg_centered,seg_centered,mode='full')
                # Keep positive lags only
                acf = acf[len(acf)//2:]
                # Normalize
                acf = acf / np.max(acf)
                # Remove zero-lag peak
                acf_no0 = acf[1:]
                # Periodicity strength
                periodicity_strength = np.max(acf_no0)
                periodicity_strengths.append(periodicity_strength)


                
            # ===============================
            # FINAL FFT FEATURE DATAFRAME
            # ===============================
            
            FFT_metrics = pd.DataFrame({
            
                't': segment_times,
            
                # PSD metrics
                'spec_energy': spec_energy,
                'comfort_energy': comfort_energy,
                'texture_energy': texture_energy,
                'dominant_freq': dominant_freq,
            
                # Periodicity metrics
                'periodicity_strength': periodicity_strengths,
            
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
            # Merged = Merged[Merged["t"] > pd.to_timedelta(5, unit='s')]
            
            #added the classifier 
            Merged["srf"] = surface
            if i in train_measurements :
                #adding final result to full data frame
                surface_data[surface] = pd.concat([surface_data[surface], Merged],ignore_index=True)
            elif i in test_measurements :
                test_data[surface] = pd.concat([test_data[surface], Merged],ignore_index=True)

    train_frames = [
        frame for frame in surface_data.values()
        if not frame.empty
    ]
    test_frames = [
        frame for frame in test_data.values()
        if not frame.empty
    ]
    data_set = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame({})
    test_data = pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame({})
    #keep only data we want to use as attributs
    data_set = data_set.iloc[:, 11:]
    test_data = test_data.iloc[:, 11:]

    # ==================================
    # SHUFFLE TRAIN DATA
    # ==================================

    data_set = data_set.sample(frac=1,random_state=42).reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    return data_set, test_data
    
