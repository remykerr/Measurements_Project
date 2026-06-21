"""
Train and evaluate classifiers on the prepared road-surface dataset.

The script builds the train/test DataFrames, separates features from labels,
scales the features, and calls the shared evaluation utility for each selected
machine-learning model.
"""

from pathlib import Path

import matplotlib
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn import neighbors, svm
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

try:
    from .Dataset_construction import build_surface_dataset
    from .ML_utils import evaluate_model
except ImportError:
    from Dataset_construction import build_surface_dataset
    from ML_utils import evaluate_model


OUTPUT_DIR = Path(__file__).resolve().parent / "model_results_KNN_SVM_noUnpaved"
OUTPUT_DIR.mkdir(exist_ok=True)

    
#results = []

#for i in range(1, 8):

train_df, test_df, _ = build_surface_dataset(test_measurements=(5,)) 

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
svm.SVC(kernel="rbf", class_weight="balanced"),
X_train_scaled,
Y_train,
X_test_scaled,
Y_test,
"SVM Classifier",
test_id=5
)


_, _, f1, accuracy, recall, precision = evaluate_model(
neighbors.KNeighborsClassifier(n_neighbors=5, weights="uniform"),
X_train_scaled,
Y_train,
X_test_scaled,
Y_test,
"KNN Classifier",
test_id=5
)


# results.append({
# "test_id": i,
# "model_name": "KNN Classifier",
# "f1_score": f1,
# "accuracy": accuracy,
# "recall": recall,
# "precision": precision,
# })

# evaluate_model(
#     RandomForestClassifier(
#     n_estimators=300,
#     max_depth=4,
#     min_samples_leaf=10,
#     min_samples_split=20,
#     class_weight="balanced",
#     random_state=42,
#     ),
#     X_train_scaled,
#     Y_train,
#     X_test_scaled,
#     Y_test,
#     "Random Forest Classifier",


# evaluate_model(
#     HistGradientBoostingClassifier(
#     max_iter=100,
#     learning_rate=0.03,
#     max_leaf_nodes=8,
#     min_samples_leaf=20,
#     l2_regularization=1.0,
#     random_state=42,
#     ),
#     X_train_scaled,
#     Y_train,
#     X_test_scaled,
#     Y_test,
#     "HistGradientBoostingClassifier",
# )

# save the final KNN Model



# final statistics for Training data sensibility analysis
#results_df = pd.DataFrame(results)

# print(f"Average F1-score across all test_ids: {results_df['f1_score'].mean():.3f}")
# print(f"Standard deviation of F1-scores across all test_ids: {results_df['f1_score'].std():.3f}")

# print(f"Average Accuracy across all test_ids: {results_df['accuracy'].mean():.3f}")
# print(f"Standard deviation of Accuracy across all test_ids: {results_df['accuracy'].std():.3f}")

# print(f"Average Recall across all test_ids: {results_df['recall'].mean():.3f}")
# print(f"Standard deviation of Recall across all test_ids: {results_df['recall'].std():.3f}")

# print(f"Average Precision across all test_ids: {results_df['precision'].mean():.3f}")
# print(f"Standard deviation of Precision across all test_ids: {results_df['precision'].std():.3f}")

# # export csv with results
# OUTPUT_DIR.mkdir(exist_ok=True)
# results_df.to_csv(OUTPUT_DIR / "model_results_KNN.csv", index=False)

for figure_number in plt.get_fignums():
    figure = plt.figure(figure_number)
    figure.savefig(
    OUTPUT_DIR / f"confusion_matrix_{figure_number}.png",
    dpi=150,
    bbox_inches="tight",
    )

print(f"\nResults saved to: {OUTPUT_DIR}")
