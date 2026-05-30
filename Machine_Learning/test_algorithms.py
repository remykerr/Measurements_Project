"""
Train and evaluate classifiers on the prepared road-surface dataset.

The script builds the train/test DataFrames, separates features from labels,
scales the features, and calls the shared evaluation utility for each selected
machine-learning model.
"""

import matplotlib.pyplot as plt
from sklearn import neighbors, svm
from sklearn.preprocessing import StandardScaler

from Dataset_construction import build_surface_dataset
from ML_utils import evaluate_model


train_df, test_df = build_surface_dataset() 

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

plt.show()

# ==================================
# Exporting trained model
# ==================================
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

surface_classifier = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(
        kernel="rbf",
        probability=True
    ))
])

surface_classifier.fit(X_train, Y_train)

joblib.dump(
    surface_classifier,
    "Road_Surface_Classifier.pkl")
