import random
import string
from algorithms.BaseAlgorithm import BaseAlgorithm
from datasets import load_dataset, Dataset
from vllm import LLM, SamplingParams


class ZeroShotAlgorithm(BaseAlgorithm):
    """
    This does not utilize the meta_learning_database, and generates from vllm using previous rows only.

    """

    def __init__(self, args):
        super().__init__(args)
        self.llm = LLM(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            trust_remote_code=True,
            tensor_parallel_size=1,
            download_dir="/shared/share_mala/andrew/huggingface/cache",
            disable_log_stats=True,
        )

    def generate_prompt(self, row: dict) -> str:
        prompt = "User: " + row['test_prompt']
        return prompt

    def generate_responses(self, prompts):
        sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=512)
        outputs = self.llm.generate(prompts, sampling_params)
        return outputs

    def generate_evaluation_responses(self, args) -> Dataset:
        debug = args.debug
        dataset = self.eval_dataset
        if debug:
            dataset = dataset.select(range(20))
        prompts = [self.generate_prompt(dataset[i]) for i in range(len(dataset))]
        outputs = self.generate_responses(prompts)
        responses = [output.outputs[0].text for output in outputs]
        dataset = dataset.add_column("test_response", responses)
        return dataset

if __name__ == "__main__":
    test_gen = ZeroShotAlgorithm()
    updated_dataset = test_gen.generate_evaluation_responses(debug=True)
    print(updated_dataset[0]["test_response"])
