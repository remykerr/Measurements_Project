""""
This script extract the final dataset containing all thes statistical features comouted over 1s windows:

1. Checking the stationary of the data
2. Checking the ergodicity of the data
3. Verify coherence between the different measurements of the same surface


"""

from Dataset_construction_statisticalChecks import build_surface_dataset_2

# importing the dataset with all the features computed on 1s windows
full_window_dataset = build_surface_dataset_2()

print("Dataset shape:", full_window_dataset.shape)
print("Dataset columns:", full_window_dataset.columns)


# Stationarity Check: 

percentage_stable_full = [] # list to store the percentage of stable windows for each measurement
percentage_acf_stable_full = [] # list to store the percentage of ACF-stable windows for each measurement

for surface in full_window_dataset["srf"].unique():
    for measurement_id in full_window_dataset["measurement_id"].unique():
        
        print(f"Surface: {surface}, Measurement ID: {measurement_id}")
        df = full_window_dataset[
            (full_window_dataset["srf"] == surface) &
            (full_window_dataset["measurement_id"] == measurement_id)
        ].copy()

        df = df.sort_values("t")

        features = ["az_avg"]#, "az_std", "az_rms", "spec_energy"]

        for feature in features:
            mu = df[feature].mean() # mean of the feature over the whole measurement
            sigma = df[feature].std() # standard deviation of the feature over the whole measurement
            
            
            df["az_avg_dev"] = abs(df["az_avg"] - df["az_avg"].mean()) / df["az_avg"].std() # Deviation of mu_i from the global mean, normalized by the global std
            percentage_stable = (df["az_avg_dev"] < 2).mean() * 100  # to be better defined
            percentage_stable_full.append(percentage_stable)
            
            acf_stable_percentage = df["acf_stationary"].mean() * 100
            print("ACF stable windows [%]:", acf_stable_percentage)
            percentage_acf_stable_full.append(acf_stable_percentage)

            print("Feature:", feature)
            print("mean:", mu)
            print("std:", sigma)
            print("Deviation of mu_i from global mean (normalized):", df["az_avg_dev"].mean()) # Average deviation of the feature mean in each window from the global mean, normalized by the global std
            print("Stable windows [%]:", percentage_stable)
            
