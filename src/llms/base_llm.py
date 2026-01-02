from abc import ABC, abstractmethod
from typing import Dict
import time


class BaseLLM(ABC):
    """
    Abstract base class for all Large Language Models.

    This class defines a unified interface for querying LLMs
    to ensure reproducibility and fair behavioral comparison.
    """

    def __init__(self, model_name: str, temperature: float = 0.0, max_tokens: int = 512):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def generate(self, prompt: str) -> Dict:
        """
        Generate a response for a given prompt.

        Returns a dictionary containing:
        - prompt
        - response
        - model_name
        - generation_time
        """
        pass

    def _wrap_response(self, prompt: str, response_text: str, start_time: float) -> Dict:
        """
        Standardized response wrapper used by all subclasses.
        """
        return {
            "model": self.model_name,
            "prompt": prompt,
            "response": response_text,
            "generation_time": round(time.time() - start_time, 4)
        }
