import json
from pathlib import Path
import numpy as np
from embeddings.sentence_encoder import SentenceEncoder


class SemanticDriftAnalyzer:
    """
    Computes semantic drift between base prompts and perturbed prompts.
    """

    def __init__(self):
        self.encoder = SentenceEncoder()

    def load_responses(self, response_dir: Path, prompt_type: str):
        """
        Load responses for a given prompt type.
        """
        file_path = response_dir / f"{prompt_type}.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def compute_drift(self, base_responses, perturbed_responses):
        """
        Compute semantic drift between base and perturbed responses.
        """
        base_texts = [r["response"] for r in base_responses]
        perturbed_texts = [r["response"] for r in perturbed_responses]

        base_emb = self.encoder.encode(base_texts)
        pert_emb = self.encoder.encode(perturbed_texts)

        drifts = []
        for i in range(len(base_emb)):
            similarity = float(np.dot(base_emb[i], pert_emb[i]))
            drift = 1.0 - similarity
            drifts.append(drift)

        return drifts

    def analyze_model(self, model_response_dir: str):
        """
        Analyze semantic drift for all perturbation types of a model.
        """
        model_dir = Path(model_response_dir)

        base_responses = self.load_responses(model_dir, "base")

        results = {}
        for ptype in ["negation", "ambiguity", "contradiction", "context_shift"]:
            perturbed = self.load_responses(model_dir, ptype)
            drifts = self.compute_drift(base_responses, perturbed)

            results[ptype] = {
                "mean_drift": float(np.mean(drifts)),
                "std_drift": float(np.std(drifts)),
                "all_drifts": drifts
            }

        return results
