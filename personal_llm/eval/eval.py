import time
import argparse
from algorithms.BaseAlgorithm import BaseAlgorithm
from datasets import Dataset, load_dataset
from typing import Type
import pandas as pd
import numpy as np
import os
import random
import numpy as np
import torch
import multiprocessing as mp
from constants import REWARD_MODELS
import os
import importlib


directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'algorithms')
for file in os.listdir(directory):
    if file.endswith('.py') and file != '__init__.py':
        module_name = file[:-3]  # removes the .py at the end
        module = importlib.import_module('algorithms.' + module_name)
        globals()[module_name] = module

reward_models_names = sorted(list(REWARD_MODELS.keys()))

def set_seed(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def persona_score_dataset(updated_dataset, args):
    from gen_score import score_dataframe

    results_dict = {}
    updated_df = pd.DataFrame(updated_dataset)
    updated_df.rename(columns={'test_prompt': 'prompt', 'test_response': 'response'}, inplace=True)

    for reward_model in reward_models_names:
        scored_df = score_dataframe(updated_df, args.conf_filepath, reward_model)

        updated_df['score_' + reward_model] = scored_df['score']

        mean_score = REWARD_MODELS[reward_model]['mean']
        std_score = REWARD_MODELS[reward_model]['sd']
        updated_df['score_' + reward_model] = (updated_df['score_' + reward_model] - mean_score) / std_score
        model_mean = updated_df['score_' + reward_model].mean()
        results_dict[f'mean_test_{reward_model}'] = model_mean
        results_dict[f'mean_train_{reward_model}'] = mean_score
        results_dict[f'std_train_{reward_model}'] = std_score
        print(f"Mean for {reward_model}: {model_mean}; Train Set Mean: {mean_score}, Train Set Std: {std_score}")

    for i, row in updated_df.iterrows():
        reward_scores = [row[f'score_{model_name}'] for model_name in reward_models_names]
        updated_df.loc[i, 'persona_weighted_score'] = np.dot(row['person_weight'][:len(reward_models_names)], reward_scores)

    results_dict['total_score'] = updated_df['persona_weighted_score'].mean()
    results_dict['win_rate_against_bestresponse'] = len(updated_df[updated_df['persona_weighted_score'] > updated_df['best_response_reward']]) / len(updated_df) * 100
    results_dict['win_rate_against_gpt4o'] = len(updated_df[updated_df['persona_weighted_score'] > updated_df['gpt4o_reward']]) / len(updated_df) * 100

    print("Total Score: ", results_dict['total_score'])
    print("Win Rate Against Best_Response_Reward: ", results_dict['win_rate_against_bestresponse'], "%")
    print("Win Rate Against gpt4o: ", results_dict['win_rate_against_gpt4o'], "%")

    return results_dict, updated_df

def evaluate_response_algorithm(response_algorithm_class: Type[BaseAlgorithm]) -> float:
    response_algorithm = response_algorithm_class(args)
    responsed_dataset = response_algorithm.generate_evaluation_responses(args)
    responded_df = pd.DataFrame(responsed_dataset)
    
    results, scored_df = persona_score_dataset(responded_df, args)
    scored_df.to_csv(args.csv_output, index=False)
    return results


if __name__ == "__main__":
    start_time = time.time()

    mp.set_start_method('spawn')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_filepath", type=str, default="../eval/eval_conf.yaml", help="Path to the configuration file")
    parser.add_argument("--meta_db_path", type=str, default='andrewsiah/MetaLearningDatabase10K', help="Path to the meta learning database")
    parser.add_argument("--csv_output", type=str, default=None, help="Path to the output CSV file")
    parser.add_argument("--debug", action='store_true', default=False, help="Enable debug mode")
    parser.add_argument("--gpu", type=int, default=0, help="GPU CUDA number to use for computation")
    parser.add_argument("--algorithm", type=str, default="TestAlgorithm", help="Name of the response algorithm to use")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generation")
    parser.add_argument("--eval_type", type=str, default="reward", choices=["reward", "pairwise_pref"], help="Type of evaluation to perform: 'reward' or 'pref'")
    args = parser.parse_args()

    set_seed(args.seed)

    from datetime import datetime
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M")
    default_output_path = f'scores/score_df_{args.algorithm}_{current_datetime}.csv'
    if args.csv_output is None:
        args.csv_output = default_output_path 
    # imports the class from the module
    algorithm_class = getattr(globals()[args.algorithm], args.algorithm)

    if args.debug:
        reward_models_names = ['oasst_pythia_1b']
        pass

    results = evaluate_response_algorithm(algorithm_class)
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Start time: {start_time}, Time taken: {time_taken}")

    if os.path.isfile('scores/all_scores.csv') and os.path.getsize('scores/all_scores.csv') > 0:
        df = pd.read_csv('scores/all_scores.csv')
    else:
        df = pd.DataFrame(columns=['eval_type', 'algorithm', 'debug', 'total_score', 'win_rate_against_bestresponse', 'win_rate_against_gpt4o', 'time_taken', 'seed', 'conf_filepath', 'meta_db_path', 'csv_output', 'gpu'])

    new_row = {
        'eval_type': args.eval_type,
        'algorithm': args.algorithm,
        'debug': args.debug,
        'total_score': results['total_score'],
        'win_rate_against_bestresponse': results['win_rate_against_bestresponse'],
        'win_rate_against_gpt4o': results['win_rate_against_gpt4o'],
        'time_taken': time_taken,
        'seed': args.seed,
        'conf_filepath': args.conf_filepath,
        'meta_db_path': args.meta_db_path,
        'csv_output': args.csv_output,
        'gpu': args.gpu
    }
    for model in reward_models_names:
        new_row[f'mean_test_{model}'] = results[f'mean_test_{model}']
        new_row[f'mean_train_{model}'] = results[f'mean_train_{model}']
        new_row[f'std_train_{model}'] = results[f'std_train_{model}']

    df = pd.concat([df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
    df.to_csv('scores/all_scores.csv', index=False)
