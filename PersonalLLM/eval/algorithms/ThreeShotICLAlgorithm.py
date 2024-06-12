import random
import string
from algorithms.BaseAlgorithm import BaseAlgorithm
from datasets import load_dataset, Dataset
from vllm import LLM, SamplingParams


class ThreeShotICLAlgorithm(BaseAlgorithm):
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

    def generate_response(self, row: dict) -> str:
        prompt = "Below are some examples of the user's past conversation history."
        for i in range(int(row["user_history_length"])):
            past_prompt = row["prompt_" + str(i + 1)]
            past_response = row["response_" + str(i + 1)]
            past_reward = str(row["reward_" + str(i + 1)])
            prompt += (
                "User: "
                + past_prompt
                + "\nAssistant: "
                + past_response
                + "\nReward: "
                + past_reward
                + "\n\n"
            )
        prompt += "Use the contexts above to generate a good response for the user prompt below. Stop after answering the User Prompt, don't give a reward.\n"
        prompt += "User: " + row['test_prompt']
        sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=512)
        output = self.llm.generate([prompt], sampling_params)
        return output[0].outputs[0].text

    def generate_evaluation_responses(self, debug=False) -> Dataset:
        dataset = self.eval_dataset
        if debug:
            dataset = dataset.select(range(20))
        dataset = dataset.add_column(
            "test_response",
            [self.generate_response(dataset[i]) for i in range(len(dataset))],
        )
        return dataset
    

    def generate_reward_prompt(self, row: dict) -> str:
        prompt = "Below are some examples of the user's past conversation history."
        shots = 3 
        for i in range(shots):
            past_prompt = row["prompt_" + str(i + 1)]
            past_response = row["response_" + str(i + 1)]
            past_reward = str(row["reward_" + str(i + 1)])
            prompt += (
                "User: "
                + past_prompt
                + "\nAssistant: "
                + past_response
                + "\nReward: "
                + past_reward
                + "\n\n"
            )
        prompt += "Use the contexts above to generate a good response for the user prompt below. Stop after answering the User Prompt, don't give a reward.\n"
        prompt += "User: " + row['test_prompt']
        return prompt
    
    def generate_pairwise_pref_prompt(self, row: dict) -> str:
        prompt = "Below are some examples of the user's past conversation history with a chosen response per prompt."
        history = []
        shots = 3 
        for i in range(shots):
            past_prompt = row["prompt_" + str(i + 1)]
            chosen_response = row["chosen_" + str(i + 1)]
            history.append(
                "User: "
                + past_prompt
                + "\nAssistant: "
                + chosen_response
                + "\n\n"
            )
        # Check if the total length of the history exceeds the maximum token limit
        while len(''.join(history)) > 6000:
            # If it does, remove the earliest history
            history.pop(0)
        prompt += ''.join(history)
        prompt += "Use the contexts above to generate a good response for the user prompt below. Stop after answering the User Prompt, don't give a reward.\n"
        prompt += "User: " + row['test_prompt']
        return prompt
    
    def generate_responses(self, prompts):
        sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=512)
        outputs = self.llm.generate(prompts, sampling_params)
        return outputs

    def generate_evaluation_responses(self, args) -> Dataset:
        debug = args.debug
        dataset = self.eval_dataset
        if debug:
            dataset = dataset.select(range(20))
        
        if args.eval_type == "reward":
            prompts = [self.generate_reward_prompt(dataset[i]) for i in range(len(dataset))]
        elif args.eval_type == "pairwise_pref":
            prompts = [self.generate_pairwise_pref_prompt(dataset[i]) for i in range(len(dataset))]
        outputs = self.generate_responses(prompts)
        responses = [output.outputs[0].text for output in outputs]
        dataset = dataset.add_column("test_response", responses)
        return dataset

if __name__ == "__main__":
    test_gen = ThreeShotICLAlgorithm()
    updated_dataset = test_gen.generate_evaluation_responses(debug=True)
    print(updated_dataset[0]["test_response"])
