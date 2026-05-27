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
