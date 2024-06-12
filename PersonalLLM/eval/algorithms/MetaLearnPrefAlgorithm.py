import random
import string
from algorithms.BaseAlgorithm import BaseAlgorithm
from datasets import load_dataset, Dataset
import torch
import numpy as np
from vllm import LLM, SamplingParams
import os
import pandas as pd
import torch.nn.functional as F
import random


class MetaLearnPrefAlgorithm(BaseAlgorithm):
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
        self.meta_learning_database = load_dataset("andrewsiah/MetaLearningPrefDatabase10K_toembed")["train"]
        self.eval_dataset = load_dataset("andrewsiah/Eval_Pref_Dataset")["test"]
        # self.eval_dataset = load_dataset("andrewsiah/Eval_Pref_Dataset_toembed")["train"]
        self.load_dataset()
        self.create_tensor_and_mask(self.train_df_embeddings)

    def generate_prompt(self, i) -> str:
        full_prompt = "#########[INSTRUCTIONS]:Below are some examples of the user's past conversation history and the most similar conversation history from the other users data.#########\n\n"
        row = self.eval_df.iloc[i]
        total_similar_pairs = []
        full_prompt += f'''###Current User Histories###\n\n'''
        for j in range(1, int(row["user_history_length"]) + 1):
            full_prompt += f'---Current User Interaction {j}:---\n\n'
            past_prompt = row[f'prompt_{j}']
            past_response = row[f'response_{j}_a'] if row[f'chosen_{j}'] == "a" else row[f'response_{j}_b']
            full_prompt += f'User:\n{past_prompt}\n\nAssistant:\n\n{past_response}\n\n\n'
            similar_pairs = self.extract_similar_prr(self.train_df, i, j, k = 2)
            total_similar_pairs.extend(similar_pairs)

        sampled_pairs = random.sample(total_similar_pairs, 8 - int(row["user_history_length"]))
        full_prompt += f'''###Most Similar Users' Histories From Database###\n\n'''
        for idx, (prompt, response) in enumerate(sampled_pairs):
            full_prompt += f'---Similar User Interaction {idx + 1}:---\n\n'
            full_prompt += f'User:\n\n{prompt}\nAssistant:\n\n{response}\n\n'

        full_prompt += "#########[INSTRUCTIONS]:Use the above histories to generate a response for the following prompt that would make the user satisfied based on the user's preferences and values. Generate the response below after 'Your Response:'. Only give the response, without any other explanations.#########\n\n"
        full_prompt += f'User:\n\n{row["test_prompt"]}\n\nYour Response:'
        return full_prompt

    def generate_responses(self, prompts):
        sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=512)
        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]

    def generate_evaluation_responses(self, args) -> Dataset:
        debug = args.debug    
        dataset = self.eval_dataset
        if debug:
            dataset = dataset.select(range(3))
        prompts = [self.generate_prompt(i) for i in range(len(dataset))]
        responses = self.generate_responses(prompts)
        dataset = dataset.add_column("test_response", responses)
        return dataset


    def load_dataset(self):
        self.train_df = pd.DataFrame.from_dict(self.meta_learning_database)
        self.train_df_embeddings = pd.read_json('/shared/share_mala/leon/train_1k_embeddings.json')
        self.eval_df = pd.DataFrame.from_dict(self.eval_dataset)
        self.eval_df_embeddings = pd.read_json('/shared/share_mala/leon/eval_pref_embeddings.json')

    def create_tensor_and_mask(self, df):
        embedding_length = 256
        mask = df.notnull().applymap(lambda x: 1.0 if x else -1.0)
        df_filled = df.applymap(lambda x: x if x is not None else [0] * embedding_length)
        tensor_table = torch.tensor(np.array(df_filled.values.tolist()))
        self.train_tensor_table = tensor_table
        self.mask = mask.values
    
    def get_top_k_pairs(self, cosine_similarities, k=3):
        cosine_similarities = (cosine_similarities * self.mask).to('cuda')
        flattened_similarities = cosine_similarities.flatten()

        topk_values, topk_indices = torch.topk(flattened_similarities, k=k, largest=True)

        topk_indices_2d = torch.stack(torch.unravel_index(topk_indices, cosine_similarities.shape)).T

        topk_indices_2d = topk_indices_2d.cpu()
        topk_values = topk_values.cpu()

        xy_pairs = [(int(x), int(y+1)) for x, y in topk_indices_2d]
        similarities = topk_values.tolist()

        return xy_pairs, similarities

    def extract_similar_prr(self, df, i, j, k):
        new_user_embed = torch.tensor(self.eval_df_embeddings.iloc[i][f'embedding_{j}'])
        cosine_similarities = F.cosine_similarity(self.train_tensor_table, new_user_embed, dim=-1)
        xy_pairs, similarities = self.get_top_k_pairs(cosine_similarities, k)

        similar_pairs = []
        for x, y in xy_pairs:
            prompt = df.iloc[x][f'prompt_{y}']
            response = df.iloc[x][f'response_{y}']
            similar_pairs.append((prompt, response))

        return similar_pairs
    

if __name__ == "__main__":
    test_gen = MetaLearnPrefAlgorithm()
    test_gen.generate_response(0)
    updated_dataset = test_gen.generate_evaluation_responses(debug=True)
    print(updated_dataset[0]["test_response"])   





