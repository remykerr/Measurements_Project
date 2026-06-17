"""
Utility functions for training, evaluating, and visualizing AE for Road Defect Identification.
"""

import numpy as np
import matplotlib.pyplot as plt
import keras


def show_plot():
    if "agg" not in plt.get_backend().lower():
        plt.show()


def model_training(X_train, X_val):
    
    # Build the AutoEncoder model
    model = keras.Sequential([
        keras.Input(shape=(X_train.shape[1],)),

        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.4),

        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.4),

        keras.layers.Dense(8, activation="relu"),

        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(X_train.shape[1]),
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=30,
        restore_best_weights=True,
    )

    # Train the model
    history = model.fit(
        X_train,
        X_train,
        epochs=300,
        batch_size=32,
        validation_data=(X_val, X_val),
        callbacks=[early_stop],
        verbose=1,
    )
    

    # plotting loss over epochs
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    show_plot()
    
    return model

# Function to calculate reconstruction loss with differents metrics
def calculate_reconstruction_loss(data, model, method="top_k", top_k=10):
    reconstructions = model.predict(data)
    point_errors = np.abs(data - reconstructions)

    if method == "mean":
        return np.mean(point_errors, axis=1)

    if method == "max":
        return np.max(point_errors, axis=1)

    if method == "top_k":
        safe_top_k = min(top_k, point_errors.shape[1])
        return np.mean(np.sort(point_errors, axis=1)[:, -safe_top_k:], axis=1)

    raise ValueError(f"Unknown reconstruction loss method: {method}")

# def calculate_reconstruction_loss(data, model):
#     reconstructions = model.predict(data)
#     reconstruction_errors = np.mean(np.abs(data - reconstructions), axis=1)
#     return reconstruction_errors


def plot_prediction_histogram(normal_errors, anomaly_errors, threshold):
    predicted_normal = anomaly_errors[anomaly_errors <= threshold]
    predicted_anomaly = anomaly_errors[anomaly_errors > threshold]

    plt.figure(figsize=(7, 4))
    plt.hist(
        normal_errors,
        bins=40,
        alpha=0.6,
        color="green",
        label="Validation Data (Asphalt w/o Defects)",
    )
    plt.hist(
        predicted_normal,
        bins=40,
        alpha=0.6,
        color="orange",
        label="Speedbump/Pothole below threshold",
    )
    plt.hist(
        predicted_anomaly,
        bins=40,
        alpha=0.8,
        color="red",
        label="Predicted anomaly",
    )
    plt.axvline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Threshold",
    )
    plt.yscale("log")
    plt.xlabel("Reconstruction error")
    plt.ylabel("Frequency (log scale)")
    plt.title("AE predictions by reconstruction error")
    plt.legend()
    plt.grid(True, alpha=0.3)
    show_plot()


def select_defect_windows(anomaly_results):
    selected = anomaly_results.copy()
    selected["is_selected_defect"] = False

    group_columns = ["surface", "measurement_id"]
    for _, group in selected.sort_values("start_time").groupby(group_columns):
        current_cluster = []

        for row_index, row in group.iterrows():
            if row["is_anomaly"]:
                current_cluster.append(row_index)
                continue

            if current_cluster:
                best_index = selected.loc[current_cluster, "reconstruction_error"].idxmax()
                selected.loc[best_index, "is_selected_defect"] = True
                current_cluster = []

        if current_cluster:
            best_index = selected.loc[current_cluster, "reconstruction_error"].idxmax()
            selected.loc[best_index, "is_selected_defect"] = True

    return selected


def plot_reconstruction_error_over_time(
    anomaly_results,
    threshold,
    surface=None,
    measurement_id=None,
    max_plots=5,
):
    if anomaly_results.empty:
        return

    selected = anomaly_results
    if surface is not None:
        selected = selected[selected["surface"] == surface]
    if measurement_id is not None:
        selected = selected[selected["measurement_id"] == measurement_id]

    if selected.empty:
        return

    if surface is not None and measurement_id is not None:
        groups_to_plot = [(surface, measurement_id)]
    else:
        groups_to_plot = (
            selected.groupby(["surface", "measurement_id"])["reconstruction_error"]
            .max()
            .sort_values(ascending=False)
            .head(max_plots)
            .index
            .tolist()
        )

    for plot_surface, plot_measurement_id in groups_to_plot:
        plot_data = selected[
            (selected["surface"] == plot_surface)
            & (selected["measurement_id"] == plot_measurement_id)
        ].sort_values("start_time")

        if plot_data.empty:
            continue

        time_seconds = plot_data["start_time"].dt.total_seconds()

        plt.figure(figsize=(8, 4))
        plt.plot(
            time_seconds,
            plot_data["reconstruction_error"],
            marker="o",
            linewidth=1.5,
            label="Reconstruction error",
        )
        plt.axhline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=2,
            label="Threshold",
        )
        plt.xlabel("Time [s]")
        plt.ylabel("Reconstruction error")
        plt.title(
            f"Reconstruction error over time - {plot_surface} {plot_measurement_id}"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        show_plot()


def plot_signal_with_detected_anomalies(
    anomaly_dataset,
    anomaly_results,
    surface=None,
    measurement_id=None,
    window_duration=1.0,
    max_plots=5,
):
    if anomaly_dataset.empty or anomaly_results.empty:
        return

    selected_results = anomaly_results
    if surface is not None:
        selected_results = selected_results[selected_results["surface"] == surface]
    if measurement_id is not None:
        selected_results = selected_results[
            selected_results["measurement_id"] == measurement_id
        ]

    if selected_results.empty:
        return

    if surface is not None and measurement_id is not None:
        groups_to_plot = [(surface, measurement_id)]
    else:
        defect_column = (
            "is_selected_defect"
            if "is_selected_defect" in selected_results.columns
            else "is_anomaly"
        )
        if selected_results[defect_column].any():
            ranking_source = selected_results[selected_results[defect_column]]
        else:
            ranking_source = selected_results

        groups_to_plot = (
            ranking_source.groupby(["surface", "measurement_id"])["reconstruction_error"]
            .max()
            .sort_values(ascending=False)
            .head(max_plots)
            .index
            .tolist()
        )

    for plot_surface, plot_measurement_id in groups_to_plot:
        _plot_single_signal_with_detected_anomalies(
            anomaly_dataset,
            anomaly_results,
            plot_surface,
            plot_measurement_id,
            window_duration,
        )


def _plot_single_signal_with_detected_anomalies(
    anomaly_dataset,
    anomaly_results,
    surface,
    measurement_id,
    window_duration,
):

    selected_dataset = anomaly_dataset[
        (anomaly_dataset["surface"] == surface)
        & (anomaly_dataset["measurement_id"] == measurement_id)
    ].sort_values("start_time")
    selected_results = anomaly_results[
        (anomaly_results["surface"] == surface)
        & (anomaly_results["measurement_id"] == measurement_id)
    ].sort_values("start_time")

    if selected_dataset.empty or selected_results.empty:
        return

    az_columns = [column for column in selected_dataset.columns if column.startswith("az_")]
    az_columns = sorted(az_columns, key=lambda column: int(column.split("_")[1]))
    samples_per_window = len(az_columns)

    signal_times = []
    signal_values = []
    anomaly_window_count = 0

    result_by_time = {
        row.start_time: row
        for row in selected_results.itertuples(index=False)
    }

    plt.figure(figsize=(10, 4))
    for row in selected_dataset.itertuples(index=False):
        start_time = row.start_time.total_seconds()
        values = np.asarray([getattr(row, column) for column in az_columns], dtype=float)
        times = start_time + np.linspace(
            0,
            window_duration,
            samples_per_window,
            endpoint=False,
        )
        signal_times.extend(times)
        signal_values.extend(values)

        result = result_by_time.get(row.start_time)
        is_defect = False
        if result is not None:
            if hasattr(result, "is_selected_defect"):
                is_defect = result.is_selected_defect
            elif hasattr(result, "is_anomaly"):
                is_defect = result.is_anomaly

        if is_defect:
            anomaly_window_count += 1
            plt.axvspan(
                start_time,
                start_time + window_duration,
                color="red",
                alpha=0.12,
            )

    plt.plot(
        signal_times,
        signal_values,
        color="steelblue",
        linewidth=1.2,
        label="Reconstructed Vertical acceleration window signal",
    )
    if anomaly_window_count > 0:
        plt.axvspan(
            signal_times[0],
            signal_times[0],
            color="red",
            alpha=0.12,
            label="Detected defect window",
        )

    plt.xlabel("Time [s]")
    plt.ylabel("Vertical acceleration [m/s²]")
    plt.title(f"Detected anomalous windows on signal - {surface} {measurement_id}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    show_plot()


def evaluate_autoencoder(model, x_test_normal, x_test_anomalous):
    # Evaluate the model
    reconstruction_loss_normal = calculate_reconstruction_loss(x_test_normal, model, method="top_k", top_k=10)
    reconstruction_loss_anomalous = calculate_reconstruction_loss(x_test_anomalous, model, method="top_k", top_k=10)

    # Print average reconstruction loss
    print(f"Average Reconstruction Loss for Normal Data: {np.mean(reconstruction_loss_normal)}")
    print(f"Reconstruction Loss for Anomalous Data: {np.mean(reconstruction_loss_anomalous)}")
    print(f"Normal Reconstruction Loss Std: {np.std(reconstruction_loss_normal)}")
    print(f"Anomalous Reconstruction Loss Std: {np.std(reconstruction_loss_anomalous)}")

    # Visualization of reconstruction error distribution
    plt.figure(figsize=(6, 4))
    plt.hist(reconstruction_loss_normal, bins=50, alpha=0.6, color='g', label='Normal')
    plt.hist(reconstruction_loss_anomalous, bins=50, alpha=0.6, color='r', label='Anomalous')
    plt.axvline(x=np.mean(reconstruction_loss_normal), color='g', linestyle='dashed', linewidth=2, label='Normal')
    plt.axvline(x=np.mean(reconstruction_loss_anomalous), color='r', linestyle='dashed', linewidth=2, label='Anomalous')
    plt.title('Reconstruction Error Distribution')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.legend()
    show_plot()

    return reconstruction_loss_normal, reconstruction_loss_anomalous
