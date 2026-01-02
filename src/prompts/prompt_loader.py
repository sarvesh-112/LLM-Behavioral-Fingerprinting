from pathlib import Path
from typing import Dict, List


PROMPT_TYPES = [
    "base",
    "negation",
    "ambiguity",
    "contradiction",
    "context_shift"
]


class PromptLoader:
    def __init__(self, prompt_dir: str):
        self.prompt_dir = Path(prompt_dir)
        self.prompts = {}

    def load_prompts(self) -> Dict[str, List[str]]:
        """
        Loads prompts from text files and ensures alignment across prompt types.
        """
        for ptype in PROMPT_TYPES:
            file_path = self.prompt_dir / f"{ptype}.txt"
            if not file_path.exists():
                raise FileNotFoundError(f"Missing prompt file: {file_path}")

            with open(file_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            self.prompts[ptype] = lines

        self._validate_alignment()
        return self.prompts

    def _validate_alignment(self):
        lengths = {ptype: len(lines) for ptype, lines in self.prompts.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"Prompt files are misaligned: {lengths}"
            )
