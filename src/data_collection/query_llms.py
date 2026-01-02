import json
from pathlib import Path
from tqdm import tqdm
from prompts.prompt_loader import PromptLoader


class LLMQueryEngine:
    """
    Queries multiple LLMs using aligned prompt sets and stores raw responses.
    """

    def __init__(self, prompt_dir: str, output_dir: str):
        self.prompt_loader = PromptLoader(prompt_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, llm, experiment_name: str):
        prompts = self.prompt_loader.load_prompts()

        exp_dir = self.output_dir / experiment_name / llm.model_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        for prompt_type, prompt_list in prompts.items():
            results = []

            for idx, prompt in enumerate(tqdm(prompt_list, desc=f"{llm.model_name} | {prompt_type}")):
                response = llm.generate(prompt)
                response["prompt_type"] = prompt_type
                response["prompt_id"] = idx
                results.append(response)

            with open(exp_dir / f"{prompt_type}.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
