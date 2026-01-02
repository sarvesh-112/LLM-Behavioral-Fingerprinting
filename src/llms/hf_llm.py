import time
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from llms.base_llm import BaseLLM


class HuggingFaceLLM(BaseLLM):
    """
    Hugging Face local LLM wrapper supporting both
    decoder-only (GPT) and encoder-decoder (T5) models.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 150
    ):
        super().__init__(model_name, temperature, max_tokens)

        self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Detect model type
        if "t5" in model_name.lower():
            self.model_type = "seq2seq"
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, torch_dtype=torch.float32
            )
        else:
            self.model_type = "causal"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float32
            )

        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str):
        start_time = time.time()

        # Instruction forcing (works for both)
        instruction_prompt = (
            "Answer the following question clearly and completely.\n\n"
            f"{prompt}\n\nAnswer:"
        )

        inputs = self.tokenizer(
            instruction_prompt,
            return_tensors="pt",
            truncation=True
        )

        with torch.no_grad():
            if self.model_type == "seq2seq":
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=self.max_tokens,
                    do_sample=False
                )
            else:
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=self.max_tokens,
                    min_new_tokens=30,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )

        generated_text = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True
        ).strip()

        return self._wrap_response(prompt, generated_text, start_time)
