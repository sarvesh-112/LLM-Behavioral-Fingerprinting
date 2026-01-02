import numpy as np
from features.semantic_drift import SemanticDriftAnalyzer
from features.length_variance import LengthVarianceAnalyzer
from features.hedging_features import HedgingFeatureAnalyzer


class FeatureFusion:
    """
    Fuses multiple behavioral features into a single fingerprint vector.
    """

    def __init__(self):
        self.drift_analyzer = SemanticDriftAnalyzer()
        self.length_analyzer = LengthVarianceAnalyzer()
        self.hedge_analyzer = HedgingFeatureAnalyzer()

    def extract_features(self, model_response_dir: str):
        # Semantic drift (ordered)
        drift = self.drift_analyzer.analyze_model(model_response_dir)
        drift_vector = [
            drift[p]["mean_drift"]
            for p in ["negation", "ambiguity", "contradiction", "context_shift"]
        ]

        # Length features
        length_stats = self.length_analyzer.analyze_model(model_response_dir)
        length_vector = [
            length_stats[p]["mean_length"]
            for p in ["base", "negation", "ambiguity", "contradiction", "context_shift"]
        ]
        length_vector.append(
            length_stats["stability"]["length_variance_across_perturbations"]
        )

        # Hedging features
        hedge = self.hedge_analyzer.analyze_model(model_response_dir)
        hedge_vector = [
            hedge[p]["mean_hedge_density"]
            for p in ["base", "negation", "ambiguity", "contradiction", "context_shift"]
        ]

        # Final fingerprint
        fingerprint = np.array(
            drift_vector + length_vector + hedge_vector,
            dtype=float
        )

        return {
            "drift": drift_vector,
            "length": length_vector,
            "hedging": hedge_vector,
            "fingerprint": fingerprint
        }
