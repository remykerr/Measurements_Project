"""
Utility functions for training, evaluating, and visualizing ML classifiers.

The main helper fits a scikit-learn model, computes standard classification
metrics, prints the classification report, and plots both raw and normalized
confusion matrices.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, test_id=None):
    """
    Fit a classifier and report its performance on the test set.

    Parameters are generic scikit-learn inputs: the model must implement
    fit/predict, X_train and X_test contain the feature matrices, y_train and
    y_test contain the labels, and model_name is used in printed output/plots.
    """
    # Train model
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # Results
    print(f"\n===== {model_name} RESULTS =====")
    print(
        f"Accuracy : "
        f"{metrics.accuracy_score(y_test, y_pred):.3f}"
    )
    print(
        f"Recall : "
        f"{metrics.recall_score(y_test, y_pred, average='macro'):.3f}"
    )
    print(
        f"Precision : "
        f"{metrics.precision_score(y_test, y_pred, average='macro'):.3f}"
    )
    print(
        f"F1-score : "
        f"{metrics.f1_score(y_test, y_pred, average='macro'):.3f}"
    )
    
    f1 = metrics.f1_score(y_test, y_pred, average='macro')
    accuracy = metrics.accuracy_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred, average='macro')
    precision = metrics.precision_score(y_test, y_pred, average='macro')

    # Confusion matrix
    print("\nConfusion Matrix :")
    print(metrics.confusion_matrix(y_test, y_pred))

    ### TTest Confusion matrix visualization
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    titles_options = [
        (f"Confusion matrix |Test Set| {model_name} | test_id={test_id}", None),
        (f"Normalized confusion matrix |Test Set| {model_name} | test_id={test_id}", "true"),
    ]
    for title, normalize in titles_options:
        _, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay.from_estimator(
            model,
            X_test,
            y_test,
            display_labels=model.classes_,
            cmap=plt.cm.Blues,
            normalize=normalize,
            ax=ax,
        )
        disp.ax_.set_title(title)
        disp.ax_.tick_params(axis="x", labelrotation=30)

        for label in disp.ax_.get_xticklabels():
            label.set_horizontalalignment("right")

        print(title)
        print(disp.confusion_matrix)
        
    ### Train Confusion matrix visualization
    np.set_printoptions(precision=2)

    # # Plot non-normalized confusion matrix
    # titles_options = [
    #     (f"Confusion matrix |Train Set| {model_name} | test_id={test_id}", None),
    #     (f"Normalized confusion matrix |Train Set| {model_name} | test_id={test_id}", "true"),
    # ]
    # for title, normalize in titles_options:
    #     _, ax = plt.subplots(figsize=(8, 6))
    #     disp = ConfusionMatrixDisplay.from_estimator(
    #         model,
    #         X_train,
    #         y_train,
    #         display_labels=model.classes_,
    #         cmap=plt.cm.Blues,
    #         normalize=normalize,
    #         ax=ax,
    #     )
    #     disp.ax_.set_title(title)
    #     disp.ax_.tick_params(axis="x", labelrotation=30)

    #     for label in disp.ax_.get_xticklabels():
    #         label.set_horizontalalignment("right")

    #     print(title)
    #     print(disp.confusion_matrix)
    # Detailed report
    print("\nClassification Report :")
    print(metrics.classification_report(y_test, y_pred))

    return model, y_pred, f1, accuracy, recall, precision
