import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class ModelIdentifier:
    """
    Trains classifiers to identify LLMs from behavioral fingerprints.
    """

    def __init__(self):
        self.logreg = LogisticRegression(max_iter=1000)
        self.svm = SVC(kernel="linear")

    def train_and_evaluate(self, X, y):
        results = {}

        # Logistic Regression
        self.logreg.fit(X, y)
        preds_lr = self.logreg.predict(X)
        results["logistic_regression"] = {
            "accuracy": accuracy_score(y, preds_lr),
            "confusion_matrix": confusion_matrix(y, preds_lr),
            "report": classification_report(y, preds_lr)
        }

        # SVM
        self.svm.fit(X, y)
        preds_svm = self.svm.predict(X)
        results["svm"] = {
            "accuracy": accuracy_score(y, preds_svm),
            "confusion_matrix": confusion_matrix(y, preds_svm),
            "report": classification_report(y, preds_svm)
        }

        return results
