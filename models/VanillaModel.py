from tqdm import tqdm
from LLMs.LLM import LLM
from utils.logger import get_logger
from utils.model_utils import chunks

from typing import List

logger = get_logger()


class VanillaModel:
    def __init__(
            self,
            args,
        ):
        if args.multichoice:
            logger.warning("Using multichoice prompt template")
            self.max_new_tokens = args.MC_task.max_new_tokens
            self.prompt_template = args.MC_task.prompt_template
        else:
            logger.warning("Using QA prompt template")
            self.max_new_tokens = args.QA_task.max_new_tokens
            self.prompt_template = args.QA_task.prompt_template

        self.model_name = args.llm_params.model_name
        self.batch_size = args.batch_size

        self.model = LLM(args)

    def get_prompts(
        self,
        data: List[dict],
    ):
        prompts = []
        for sample in data:
            pmt = self.prompt_template.format(
                input=sample["question"],
                context=sample["context"]
            )

            prompts.append(pmt)

        return prompts

    def run(self,
            data: List[dict]
        ):
        prompts = self.get_prompts(data)

        responses = []
        for batched_prompts in tqdm(chunks(prompts, self.batch_size), total=len(prompts) // self.batch_size):
            texts = self.model.generate(batched_prompts, max_new_tokens=self.max_new_tokens)
            responses.extend(texts)

        logger.info(f"responses[0]: {responses[:1]}")

        return responses, prompts
