import numpy as np
from src.features.semantic_drift import SemanticDriftAnalyzer
from src.features.length_variance import LengthVarianceAnalyzer
from src.features.hedging_features import HedgingFeatureAnalyzer


class FingerprintDatasetBuilder:
    """
    Builds a dataset of behavioral fingerprints for classification.
    """

    def __init__(self):
        self.drift = SemanticDriftAnalyzer()
        self.length = LengthVarianceAnalyzer()
        self.hedge = HedgingFeatureAnalyzer()

    def build(self, model_dirs, labels):
        X = []
        y = []

        for model_dir, label in zip(model_dirs, labels):
            drift = self.drift.analyze_model(model_dir)
            length = self.length.analyze_model(model_dir)
            hedge = self.hedge.analyze_model(model_dir)

            for p in ["negation", "ambiguity", "contradiction", "context_shift"]:
                vector = [
                    drift[p]["mean_drift"],
                    length[p]["mean_length"],
                    length[p]["std_length"],
                    hedge[p]["mean_hedge_density"]
                ]

                X.append(vector)
                y.append(label)

        return np.array(X), np.array(y)
