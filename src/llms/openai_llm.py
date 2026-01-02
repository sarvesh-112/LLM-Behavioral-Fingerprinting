import time
from llms.base_llm import BaseLLM

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class OpenAILLM(BaseLLM):
    """
    OpenAI-compatible LLM wrapper.
    """

    def __init__(self, model_name: str, api_key: str, temperature: float = 0.0, max_tokens: int = 512):
        super().__init__(model_name, temperature, max_tokens)

        if OpenAI is None:
            raise ImportError("openai package not installed")

        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str):
        start_time = time.time()

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        response_text = completion.choices[0].message.content.strip()
        return self._wrap_response(prompt, response_text, start_time)
