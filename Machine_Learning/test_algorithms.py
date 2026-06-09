"""
Train and evaluate classifiers on the prepared road-surface dataset.

The script builds the train/test DataFrames, separates features from labels,
scales the features, and calls the shared evaluation utility for each selected
machine-learning model.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn import neighbors, svm
from sklearn.preprocessing import StandardScaler

from Dataset_construction import build_surface_dataset
from ML_utils import evaluate_model


OUTPUT_DIR = Path(__file__).resolve().parent / "model_results"


train_df, test_df, _ = build_surface_dataset() 

# sanity check of dimensions
print("train_df shape:", train_df.shape)
print("test_df shape:", test_df.shape)

# ==================================
# FEATURES / LABELS
# ==================================
X_train = train_df.drop(columns=["srf"])
Y_train = train_df["srf"]

X_test = test_df.drop(columns=["srf"])
Y_test = test_df["srf"]

# ==================================
# SCALING INPUT FEATURES
# ==================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================================
# MODELS TRAINING AND EVALUATION
# ==================================
evaluate_model(
    svm.SVC(kernel="rbf"),
    X_train_scaled,
    Y_train,
    X_test_scaled,
    Y_test,
    "SVM Classifier",
)
evaluate_model(
    neighbors.KNeighborsClassifier(n_neighbors=5, weights="uniform"),
    X_train_scaled,
    Y_train,
    X_test_scaled,
    Y_test,
    "KNN Classifier",
)

OUTPUT_DIR.mkdir(exist_ok=True)
for figure_number in plt.get_fignums():
    figure = plt.figure(figure_number)
    figure.savefig(
        OUTPUT_DIR / f"confusion_matrix_{figure_number}.png",
        dpi=150,
        bbox_inches="tight",
    )

print(f"\nConfusion matrix figures saved to: {OUTPUT_DIR}")
