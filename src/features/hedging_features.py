import json
from pathlib import Path
import numpy as np
import re


class HedgingFeatureAnalyzer:
    """
    Analyzes linguistic uncertainty (hedging) normalized by response length.
    """

    HEDGING_TERMS = [
        "may", "might", "could", "would", "can be",
        "generally", "typically", "often", "usually",
        "likely", "possibly", "approximately",
        "in many cases", "it depends", "sometimes",
        "suggests", "indicates", "appears to"
    ]

    def _load_responses(self, model_dir: Path, prompt_type: str):
        with open(model_dir / f"{prompt_type}.json", "r", encoding="utf-8") as f:
            return json.load(f)

    def _hedge_density(self, text: str):
        text = text.lower()
        word_count = len(text.split())
        if word_count == 0:
            return 0.0

        hedge_count = 0
        for term in self.HEDGING_TERMS:
            hedge_count += len(re.findall(re.escape(term), text))

        return hedge_count / word_count

    def analyze_model(self, model_response_dir: str):
        model_dir = Path(model_response_dir)
        results = {}

        for ptype in ["base", "negation", "ambiguity", "contradiction", "context_shift"]:
            responses = self._load_responses(model_dir, ptype)
            densities = [self._hedge_density(r["response"]) for r in responses]

            results[ptype] = {
                "mean_hedge_density": float(np.mean(densities)),
                "std_hedge_density": float(np.std(densities))
            }

        return results
