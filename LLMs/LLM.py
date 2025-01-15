import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from LLMs.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

from utils.logger import get_logger

from typing import List

logger = get_logger()


def build_chat(prompt, model_name, tokenizer):
    if "llama-2-7b-chat" in model_name.lower():
        prompt = f"[INST]{prompt}[/INST]"

    return prompt


class LLM:
    def __init__(
        self,
        args
    ):
        self.model_name = args.llm_params.model_name
        self.temperature = args.llm_params.temperature
        self.do_sample = args.llm_params.do_sample
        self.num_beams = args.llm_params.num_beams
        self.max_length = args.llm_params.max_length
        self.half = int(self.max_length / 2)

        self.device = args.device
        self.batch_size = args.batch_size

        logger.info(f"Loading model: {self.model_name}")

        if "llama-2" in self.model_name.lower() or "vicuna" in self.model_name.lower():
            replace_llama_attn_with_flash_attn()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        ).to(self.device)

        self.model.eval()

        if "vicuna" in self.model_name.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, use_fast=False)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token  # To avoid an error

    def generate(self,
                 batched_prompts: List[str],
                 max_new_tokens: int
                ):
        responses = []
        inputs = self.tokenizer(batched_prompts, return_tensors="pt", padding=True).to(self.device)

        truncated_batched_prompts = []

        tokenized_prompts = self.tokenizer(batched_prompts, truncation=False, return_tensors="pt", padding=True).input_ids

        for tokenized_pmt, pmt in zip(tokenized_prompts, batched_prompts):
            if len(tokenized_pmt) > self.max_length:
                prompt = self.tokenizer.decode(tokenized_pmt[:self.half], skip_special_tokens=True) + self.tokenizer.decode(tokenized_pmt[-self.half:], skip_special_tokens=True)
            else:
                prompt = pmt

            truncated_batched_prompts.append(build_chat(prompt, self.model_name, self.tokenizer))

        inputs = self.tokenizer(truncated_batched_prompts, truncation=False, return_tensors="pt", padding=True).to(self.device)

        try:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=self.do_sample,
                temperature=self.temperature if self.do_sample else None,
                top_k=self.top_k if self.do_sample else None,
                top_p=self.top_p if self.do_sample else None,
                use_cache=True
            )

            texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for text, pmt in zip(texts, truncated_batched_prompts):
                new_text = text[len(pmt):]
                responses.append(new_text.strip())

        except Exception as e:
            logger.error(f"Error: {e}")
            responses.extend(["" for _ in range(len(batched_prompts))])

        return responses
