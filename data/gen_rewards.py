import os
import re
import json
import yaml
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
from multiprocessing import Process, Manager
import multiprocessing
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, pipeline, PreTrainedTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from fastchat.conversation import get_conv_template, Conversation
from rewardbench import (
    DPO_MODEL_CONFIG,
    REWARD_MODEL_CONFIG,
    check_tokenizer_chat_template,
)
from constants import REWARD_MODELS

class DotDict(dict):
    """Dictionary with dot notation access to attributes."""
    def __getattr__(self, name):
        return self[name]
    def __setattr__(self, name, value):
        self[name] = value
    def __delattr__(self, name):
        del self[name]

def set_environment_variables(args):
    """Set environment variables for cache."""
    os.environ['HF_HOME'] = args.cache_dir
    os.environ['TRANSFORMERS_CACHE'] = args.cache_dir

def load_and_combine_datasets(dataset_names):
    """Load datasets and combine them into a single dataset."""
    datasets = [load_dataset(name)['train'] if 'train' in load_dataset(name) else load_dataset(name) for name in dataset_names]
    combined_data = {key: sum([list(dataset[key]) for dataset in datasets], []) for key in datasets[0].features}
    return Dataset.from_dict(combined_data)


def init_models(args, model_builder, pipeline_builder, quantized, tokenizer, device):
    if quantized:
        model_kwargs = {
            "load_in_8bit": True,
            "device_map": {"": device},
            "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
        }
    else:
        model_kwargs = {"device_map": {"": device}}

    model = model_builder(args.model_name, **model_kwargs, cache_dir=args.cache_dir, trust_remote_code=args.trust_remote_code)
    model.eval()
    reward_pipe = pipeline_builder(
        "text-classification",  # often not used
        model=model,
        tokenizer=tokenizer
    )

    # set pad token to eos token if not set
    if reward_pipe.tokenizer.pad_token_id is None:
        reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.eos_token_id
        reward_pipe.tokenizer.pad_token_id = reward_pipe.tokenizer.eos_token_id
    # For models whose config did not contains `pad_token_id`
    if reward_pipe.model.config.pad_token_id is None:
        reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.pad_token_id

    # if using fastchat template (no template in tokenizer), make the RM tokenizer output an EOS token
    if not check_tokenizer_chat_template(tokenizer):
        reward_pipe.tokenizer.add_eos_token = True

    return tokenizer, reward_pipe


def generate_reward_model_formatted_responses(
    dataset, tokenizer: PreTrainedTokenizer = None, conv: Conversation = None
):
    """Generate rewards model formatted responses and create a new dataset with additional columns."""
    stylized_columns = {}
    updated_rows = []
    if tokenizer is not None and tokenizer.chat_template is not None and not conv:
        print("Warning: No tokenizer chat template or fastchat conversation passed. Using Default Template. Please check if that's alright for current model.")
    for row in tqdm(dataset, desc="Processing rows"):
        new_row = row.copy()
        for key, value in row.items():
            if re.match(r"response_\d+$", key):
                if tokenizer is not None and tokenizer.chat_template is not None:
                    stylized_response = tokenizer.apply_chat_template(
                        [
                            {"role": "user", "content": row["prompt"]},
                            {"role": "assistant", "content": value},
                        ],
                        tokenize=False,
                        add_generation_prompt=False,
                    ).replace(tokenizer.bos_token, "")
                elif conv:
                    # AS: 3 models in reward bench takes in a conv when chat_template is None.
                    # Refer to utils.py in rewardbench (prepare_dialogue function) for `conv`
                    conv.messages = [
                        [conv.roles[0], row["prompt"]],
                        [conv.roles[1], value],
                    ]
                    stylized_response = conv.get_prompt()
                else:
                    stylized_response = f"User: {row['prompt']}\nAssistant: {value}"
                if key not in stylized_columns:
                    stylized_columns[key] = []
                stylized_columns[key].append(stylized_response)
                new_row[f"rformatted_promptresponse_{key.split('_')[1]}"] = (
                    stylized_response
                )
        updated_rows.append(new_row)
    dataset = Dataset.from_dict(
        {key: [dic[key] for dic in updated_rows] for key in updated_rows[0]}
    )
    return dataset

def worker(gpu, start_idx, end_idx, return_dict, worker_id, args, model_builder, pipeline_builder, quantized):
    set_environment_variables(args)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    tokenizer_path = args.tokenizer if args.tokenizer else args.tokenizer_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    tokenizer, model_pipeline = init_models(args, model_builder, pipeline_builder, quantized, tokenizer, device)
    dataset = load_and_combine_datasets(args.dataset_names)
    new_dataset = generate_reward_model_formatted_responses(dataset, tokenizer)
    # TODO: AS: When batch_size > 1, the rewards change. Find out why.
    # Suspicion, does using dataset instead of dataloader change things?
    pipe_kwargs = {
        "batch_size": args.batch_size,  # eval_args.inference_batch_size,
        "truncation": True,
        "padding": "longest", # True
        "max_length": args.max_length,
        "function_to_apply": "none",  # Compute raw logits
        "return_token_type_ids": False,
    }
    rewarded_dataset = compute_rewards(args, new_dataset, args.columns, start_idx, end_idx, args.batch_size, model_pipeline, pipe_kwargs)
    return_dict[worker_id] = rewarded_dataset


def compute_rewards(args, new_dataset, columns, start, end, batch_size, model_pipeline, pipe_kwargs):
    """Compute rewards for each response column and return a dataset with rewards."""
    rewarded_dataset = {}
    for i in trange(1, columns + 1, desc="Computing columns"):
        column_name = f'rformatted_promptresponse_{i}'
        rewards_column_name = f'reward_{i}'
        rewards = []
        for batch_start in trange(start, end, batch_size, desc="Batch:"):
            batch_end = min(batch_start + batch_size, end)
            batch_texts = new_dataset[batch_start:batch_end][column_name]
            pipe_outputs = model_pipeline(batch_texts, **pipe_kwargs)

            if args.debug:
                with open("pipe_outputs.txt", "a") as f:
                    for text in batch_texts:
                        f.write(str(text))
                        f.write('\n')
                    f.write(str(pipe_outputs))
                    f.write("-"*40)
                    f.write('\n')
            if isinstance(pipe_outputs[0], dict):
                
                batch_rewards = [output["score"] for output in pipe_outputs]
            elif isinstance(pipe_outputs[0][0], dict):
                batch_rewards = [output[0]["score"] for output in pipe_outputs]
            else:
                batch_rewards = [output[0].cpu().numpy().tolist() for output in pipe_outputs]
            
            rewards.extend(batch_rewards)
            if args.debug:
                with open("pipe_outputs.txt", "a") as f:
                    f.write(f"BATCH_REWARDS: {str(batch_rewards)}\n")
                    f.write(f"REWARDS: {str(rewards)}\n")
        
        rewarded_dataset[rewards_column_name] = rewards
    for column in new_dataset.features:
        rewarded_dataset[column] = new_dataset[column][start:end] 
    return Dataset.from_dict(rewarded_dataset)


def mp_process_and_push_dataset(args):
    """
    Process and push dataset with rewards using multiple GPUs on multiprocess.

    Args:
        args (Namespace): Arguments containing configuration for processing and pushing the dataset.
    """
    device_count = len(args.gpus)
    total_samples = args.end - args.start
    samples_per_gpu = total_samples // device_count

    is_dpo = False
    MODEL_CONFIGS = REWARD_MODEL_CONFIG

    if args.chat_template:
        from fastchat.conversation import get_conv_template
        conv = get_conv_template(args.chat_template)
    else:
        conv = None

    if args.model_name in MODEL_CONFIGS:
        config = MODEL_CONFIGS[args.model_name]
    else:
        config = MODEL_CONFIGS["default"]

    print(f"MODEL_CONFIGS: {config}")

    quantized = config["quantized"]  # only Starling isn't quantized for now
    custom_dialogue = config["custom_dialogue"]
    pipeline_builder = config["pipeline_builder"]
    model_builder = config["model_builder"]

    print(f"PIPELINE_BUILDER: {pipeline_builder}")

    if custom_dialogue:
        raise NotImplementedError("Custom dialogue not implemented yet for simpler data formatting.")

    processes = []

    manager = Manager()
    return_dict = manager.dict()
    worker_id = 0

    for i, gpu in enumerate(args.gpus):
        gpu_start = args.start + i * samples_per_gpu
        gpu_end = args.start + (i + 1) * samples_per_gpu if i < device_count - 1 else args.end

        p = Process(target=worker, args=(gpu, gpu_start, gpu_end, return_dict, worker_id, args, model_builder, pipeline_builder, quantized))
        worker_id += 1
        processes.append(p)

    try:
        for p in processes:
            p.start()

        for p in processes:
            p.join()
    except Exception as e:
        print(f"Exception occurred: {e}, terminating processes...")
    finally:
        for p in processes:
            p.terminate()
            p.join()

    # Concatenate all rewarded datasets
    all_rewarded_datasets = []
    for i in range(worker_id):
        try:
            all_rewarded_datasets.append(return_dict[i])
        except KeyError:
            print(f"Warning: Worker {i} did not return a dataset.")

    if all_rewarded_datasets:
        concatenated_dataset = Dataset.from_dict({
            column: sum([list(ds[column]) for ds in all_rewarded_datasets], [])
            for column in all_rewarded_datasets[0].features
        })
    else:
        concatenated_dataset = Dataset.from_dict({})

    if concatenated_dataset:
        concatenated_dataset.push_to_hub(args.postreward_dataset_hfhub_name)

def parse_args():
    parser = argparse.ArgumentParser(description="Process and push dataset with rewards.")
    parser.add_argument("--dataset_names", nargs='+', required=True, help="List of dataset names.")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for processing.")
    parser.add_argument("--start", type=int, required=True, help="Start index for processing.")
    parser.add_argument("--end", type=int, required=True, help="End index for processing.")
    parser.add_argument("--columns", type=int, required=True, help="Number of columns to process.")
    parser.add_argument("--gpu", type=int, required=True, help="GPU index to use.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name to use.")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Tokenizer name to use.")
    parser.add_argument("--generate_reward_model_dataset", action='store_true', help="Flag to generate reward model formatted dataset.")
    return parser.parse_args()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

    with open("../data/gen_rewards_conf.yaml", "r") as file:
        config = yaml.safe_load(file)

    args = DotDict(config)

    reward_model = args.reward_model
    if reward_model in REWARD_MODELS:
        args.model_name = REWARD_MODELS[reward_model]["model_name"]
        args.tokenizer_name = REWARD_MODELS[reward_model]["tokenizer_name"]
        args.chat_template = REWARD_MODELS[reward_model]["fastchat_chat_template"]
        if args.postreward_dataset_hfhub_name is None:
            args.postreward_dataset_hfhub_name = (
                args.prereward_dataset_hfhub_name + f"_{reward_model}"
            )
    else:
        raise ValueError(f"Reward model '{reward_model}' not found in REWARD_MODELS.")

    print(json.dumps(args, indent=4))

    mp_process_and_push_dataset(args)
    print("Done")
