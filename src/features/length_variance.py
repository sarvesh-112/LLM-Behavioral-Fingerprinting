import json
from pathlib import Path
import numpy as np


class LengthVarianceAnalyzer:
    """
    Analyzes response length variance across prompt perturbations,
    ignoring empty or trivial responses.
    """

    def _load_responses(self, model_dir: Path, prompt_type: str):
        file_path = model_dir / f"{prompt_type}.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _response_length(text: str):
        """
        Computes length in number of words.
        Ignores empty or trivial responses.
        """
        words = text.strip().split()
        return len(words)

    def analyze_model(self, model_response_dir: str):
        model_dir = Path(model_response_dir)

        results = {}
        perturb_means = []

        for ptype in ["base", "negation", "ambiguity", "contradiction", "context_shift"]:
            responses = self._load_responses(model_dir, ptype)
            lengths = [
                self._response_length(r["response"])
                for r in responses
                if self._response_length(r["response"]) > 0
            ]

            if len(lengths) == 0:
                mean_len = 0.0
                std_len = 0.0
            else:
                mean_len = float(np.mean(lengths))
                std_len = float(np.std(lengths))

            results[ptype] = {
                "mean_length": mean_len,
                "std_length": std_len
            }

            if ptype != "base":
                perturb_means.append(mean_len)

        results["stability"] = {
            "length_variance_across_perturbations": float(
                np.var(perturb_means) if len(perturb_means) > 0 else 0.0
            )
        }

        return results
