import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis, skew
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
MACHINE_LEARNING_DIR = PROJECT_DIR / "Machine_Learning"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from Machine_Learning.Surface_Classification.gravity_correction import correct_gravity

BASE_DIR = MACHINE_LEARNING_DIR
DEBUG_OUTPUT_KEYS = (
    "sampling_frequency",
    "raw_acceleration",
    "vertical_no_g",
    "vertical_no_g_highpass",
    "frequency",
    "dft",
    "power_spectrum",
    "psd",
    "merged_before_speed_filter",
    "merged_after_speed_filter",
)


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


def build_surface_dataset_2(
    base_dir=BASE_DIR,
    surface_types=("cobble", "Rough_asphalt", "Smooth_asphalt", "Grass", "Unpaved"), #, "Speedbump", "Pothole"), # "Speedbump" and "Pothole" are not relevant for statistical checks as they are very short events, not continuous surfaces
    train_measurements=None,
    test_measurements=(4,),
    v_ref=2.5,
    fsamp=None,
    stationary_seconds=3.0,
    min_speed=1.0,
    plot_debug_acc = False,
    plot_debug=False,
    return_debug=False,
    debug_surface=None,
    debug_measurement=None,
    debug_window_index=0,
):
    """
    Build the machine-learning dataset from accelerometer and GPS measurements.

    The function reads each measurement folder in Clean_measurements_ML, computes
    the corrected vertical acceleration (removing gravity), filters it, extracts time-domain and
    frequency-domain features on 1-second windows, merges those features with GPS
    speed/time data, removes very slow samples, and adds the surface label.

    By default, train_measurements=None means: use every available measurement
    except the IDs listed in test_measurements. Both returned DataFrames contain
    only model-ready feature columns plus the target column "srf".

    If return_debug=True, the function returns (train_df, test_df, debug_data).
    Use debug_surface/debug_measurement to collect diagnostics for a specific
    measurement without storing debug data for the whole dataset.
    """
    base_dir = Path(base_dir)
    clean_measurements_dir = base_dir / "Clean_measurements_ML"
    surface_data = {}        
    test_data = {}
    debug_data = {}
    test_measurements = set(test_measurements)
    requested_train_measurements = (
        None if train_measurements is None else set(train_measurements)
    )

    full_window_dataset = pd.DataFrame({}) # full window dataset for statistical checks
    
    for surface in surface_types :
        surface_data[surface] = pd.DataFrame({})
        test_data[surface] = pd.DataFrame({})
        available_ids = discover_measurement_ids(clean_measurements_dir, surface)
        if requested_train_measurements is None:
            train_measurements_for_surface = set(available_ids) - test_measurements
        else:
            train_measurements_for_surface = requested_train_measurements

        measurement_ids = sorted(train_measurements_for_surface | test_measurements)

        for i in measurement_ids :
            collect_debug = (
                plot_debug
                or return_debug
                or debug_surface is not None
                or debug_measurement is not None
            )
            collect_debug = collect_debug and (
                debug_surface is None or surface.lower() == debug_surface.lower()
            )
            collect_debug = collect_debug and (
                debug_measurement is None or i == debug_measurement
            )

            measure_dir = clean_measurements_dir / f"{surface}_{i}"
            Accel_file = measure_dir / "Accelerometer.csv"
            GPS_file = measure_dir / "Location.csv"

            if not Accel_file.exists() or not GPS_file.exists():
                warnings.warn(f"Skipping missing measurement: {measure_dir}")
                continue
            
            Accel_data = pd.read_csv(Accel_file)
            GPS_data = pd.read_csv(GPS_file)
            
            
            if plot_debug_acc is True and collect_debug:
                plt.figure()
                plt.plot(Accel_data["Time (s)"], Accel_data["Z (m/s^2)"], label='Vertical Acceleration (g_included)')
                plt.xlabel('Time [s]')
                plt.ylabel('Acc [m/s^2]')
                plt.title(f'{surface}_{i} raw vertical acceleration (g included)')
                plt.legend()
                plt.show()
            
            # Estimate the gravity direction from the initial stationary samples
            # and project the signal on the real vertical axis with g removed.
            Accel_corrected, _ = correct_gravity(
                Accel_data,
                stationary_seconds=stationary_seconds,
                expected_g=9.81,
            )
            True_z_accel_data = Accel_corrected["a_vertical_no_g"]
            True_z_accel_data_raw = True_z_accel_data.copy()
            current_fsamp = (
                estimate_sampling_frequency(Accel_data["Time (s)"])
                if fsamp is None
                else fsamp
            )
            if collect_debug:
                print(f"{surface}_{i}: sampling frequency = {current_fsamp:.2f} Hz")
            
            #Applying a High Pass filter with cutoff at 0.5hz to remove drift and slow tilt change from measurments
        
            def hp_filter(signal, fs, fc=0.5, order=4):
                b, a = butter(order, fc / (fs/2.0), btype='high')
                return filtfilt(b, a, signal)   # filtfilt = zero-phase, no distortion
            
            True_z_accel_data = hp_filter(True_z_accel_data, fs=current_fsamp)
            True_z_accel_data = pd.DataFrame(True_z_accel_data)
        
            
            if plot_debug and collect_debug:
                plt.figure()

                plt.plot(
                    Accel_data["Time (s)"],
                    True_z_accel_data_raw,
                    label="Vertical Acceleration (g removed)"
                )

                plt.plot(
                    Accel_data["Time (s)"],
                    Accel_data["Z (m/s^2)"],
                    label="Vertical Acceleration (g included)"
                )

                plt.xlabel("Time [s]")
                plt.ylabel("Acc [m/s^2]")
                plt.title(f"{surface}_{i} vertical acceleration comparison")
                plt.legend()
                plt.grid(True)
                plt.show()
                
                plt.figure()
                plt.plot(Accel_data["Time (s)"], True_z_accel_data, label='Vertical Acceleration (g removed + high-pass)')
                plt.xlabel('Time [s]')
                plt.ylabel('Acc [m/s^2]')
                plt.title(f'{surface}_{i} Vertical Acceleration (g removed + high-pass)')
                plt.legend()
                plt.show()
                
        
            Accel_z = pd.concat([Accel_data["Time (s)"],True_z_accel_data], axis=1)
            Accel_z.columns = ['t', "az"]
            
            # plt.figure()
            # plt.plot(GPS_data["Time (s)"], GPS_data["Velocity (m/s)"], label='Recorded GPS')
            # plt.xlabel('Time [s]')
            # plt.ylabel('V [m/s]')
            # plt.title('Recorded Speed Over Time')
            # plt.legend()
            # plt.show()
            
            latitude_col = find_column(GPS_data, "latitude")
            longitude_col = find_column(GPS_data, "longitude")
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
            
            window_size = int(round(current_fsamp))
            
            segments = []
            segment_times = []
            
            signal = Accel_z["az"].values
            
            for k in range(0, len(signal)-window_size, window_size):
                seg = signal[k:k+window_size]
                segments.append(seg)
                segment_times.append(Accel_z.index[k])
            segments = np.array(segments)

            # Weak-stationarity check support: compare the normalized
            # autocorrelation shape of each 1 s window against the average one.
            max_lag_seconds = 0.3
            max_lag = int(max_lag_seconds * current_fsamp)
            acf_windows = []

            for seg in segments:
                seg_centered = seg - np.mean(seg)
                acf = np.correlate(seg_centered, seg_centered, mode="full")
                acf = acf[len(acf)//2:]

                if acf[0] != 0:
                    acf = acf / acf[0]
                else:
                    acf = np.zeros_like(acf)

                acf_windows.append(acf[:max_lag + 1])

            acf_windows = np.array(acf_windows)
            acf_reference = np.mean(acf_windows, axis=0)
            acf_reference_norm = np.linalg.norm(acf_reference)

            if acf_reference_norm != 0:
                acf_errors = (
                    np.linalg.norm(acf_windows - acf_reference, axis=1)
                    / acf_reference_norm
                )
            else:
                acf_errors = np.zeros(len(acf_windows))
            
            N = window_size
            
            # Apply Hanning window
            window = np.hanning(N)
            segments_windowed = segments * window
            
            # ===============================
            # FFT
            # ===============================
            dft = np.fft.rfft(segments_windowed, axis=1) / N
            # Frequency vector
            freq = np.fft.rfftfreq(N, d=1/current_fsamp)
            
            # ===============================
            # POWER SPECTRUM
            # ===============================
            ps = np.abs(dft)**2
            # One-sided correction
            if N % 2 == 0:
                ps[:,1:-1] *= 2
            else:
                ps[:,1:] *= 2
            df = current_fsamp / N
            # Power Spectral Density
            psd = ps / df

            if collect_debug and len(segments) > 0:
                debug_window_index_safe = int(
                    np.clip(debug_window_index, 0, len(segments) - 1)
                )
                debug_key = f"{surface}_{i}"
                debug_data[debug_key] = {
                    "surface": surface,
                    "measurement_id": i,
                    "sampling_frequency": current_fsamp,
                    "raw_acceleration": Accel_data.copy(),
                    "vertical_no_g": pd.Series(
                        True_z_accel_data_raw,
                        name="a_vertical_no_g",
                    ),
                    "vertical_no_g_highpass": pd.Series(
                        True_z_accel_data.iloc[:, 0].to_numpy(),
                        name="a_vertical_no_g_highpass",
                    ),
                    "window_size": window_size,
                    "segment_times": segment_times,
                    "debug_window_index": debug_window_index_safe,
                    "debug_segment": segments[debug_window_index_safe].copy(),
                    "frequency": freq.copy(),
                    "dft": dft[debug_window_index_safe].copy(),
                    "power_spectrum": ps[debug_window_index_safe].copy(),
                    "psd": psd[debug_window_index_safe].copy(),
                }

                if plot_debug:
                    plt.figure()
                    plt.plot(freq, np.abs(dft[debug_window_index_safe]))
                    plt.xlabel("Frequency [Hz]")
                    plt.ylabel("DFT amplitude")
                    plt.title(f"{debug_key} DFT window {debug_window_index_safe}")
                    plt.grid(True)
                    plt.show()

                    plt.figure()
                    plt.plot(freq, psd[debug_window_index_safe])
                    plt.xlabel("Frequency [Hz]")
                    plt.ylabel("PSD [m^2/s^4/Hz]")
                    plt.title(f"{debug_key} PSD window {debug_window_index_safe}")
                    plt.grid(True)
                    plt.show()
            
            # ===============================
            # PSD FEATURES
            # ===============================
            # Total spectral energy
            band0 = (freq >= 0.5) & (freq <= 50)  # the band is 0.5 - 50 (minimum nyquist frequency across all measurements) (remy fsamp = 100 hz)
            spec_energy = np.sum(psd[:, band0], axis=1) 
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

                # Weak-stationarity autocorrelation metrics
                'acf_error': acf_errors,
                'acf_stationary': acf_errors < 1,
            
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
            
            
            # Add normalized metrics vith repesct to speed (There was an error the normalization was accel * sqrt(v/v_ref))
            Merged["Norm_az_avg"]  = Merged["az_avg"]  * np.sqrt (v_ref/Merged["v"])
            Merged["Norm_az_rms"]  = Merged["az_rms"]  * np.sqrt (v_ref/Merged["v"])
            Merged["Norm_az_std"]  = Merged["az_std"]  * np.sqrt (v_ref/Merged["v"])
            Merged["Norm_az_peak"] = Merged["az_peak"] * np.sqrt (v_ref/Merged["v"])
            Merged["Norm_az_kurt"] = Merged['az_kurt'] * np.sqrt (v_ref/Merged["v"])
            Merged["Norm_az_crest"] = Merged['az_crest'] * np.sqrt (v_ref/Merged["v"])
            Merged["Norm_az_skew"] = Merged['az_skew'] * np.sqrt (v_ref/Merged["v"])
            
            #Merging FFt data and accel data
            Merged = pd.merge_asof(Merged.sort_values('t'),FFT_metrics.sort_values('t'), on='t')
            
            if collect_debug and f"{surface}_{i}" in debug_data:
                debug_data[f"{surface}_{i}"]["merged_before_speed_filter"] = Merged.copy()
            
            #Filtering out zones where speed is to low
            # Remove very low speeds < 1 m/s
            Merged = Merged[Merged["v"] > min_speed]
            # Merged = Merged[Merged["t"] > pd.to_timedelta(5, unit='s')]

            if collect_debug and f"{surface}_{i}" in debug_data:
                debug_data[f"{surface}_{i}"]["merged_after_speed_filter"] = Merged.copy()
            
            #added the surface classifier 
            Merged["srf"] = surface
            if i in train_measurements_for_surface :
                #adding final result to full data frame
                surface_data[surface] = pd.concat([surface_data[surface], Merged],ignore_index=True)
            elif i in test_measurements :
                test_data[surface] = pd.concat([test_data[surface], Merged],ignore_index=True)
            
            #add id of the measurement example: Cobble_1, Cobble_2, etc...
            Merged["measurement_id"] = f"{surface}_{i}"
            
            full_window_dataset = pd.concat([full_window_dataset, Merged], ignore_index=True)
            
    return  full_window_dataset
    
