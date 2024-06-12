from constants import REWARD_MODELS
from datasets import load_dataset
import numpy as np
from numba import jit, prange
from scipy.stats import dirichlet
from tqdm import tqdm
import pandas as pd
from datasets import Dataset
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--n_persons", type=int, default=1000, help="Number of personas to generate"
)
parser.add_argument("--n_responses", type=int, default=8, help="Number of responses")
parser.add_argument(
    "--start", type=int, default=0, help="Start index for processing database"
)
parser.add_argument(
    "--end", type=int, default=1000, help="End index for processing database"
)
parser.add_argument(
    "--alpha", type=float, default=0.01, help="Alpha value for generating personas"
)
parser.add_argument(
    "--push_to_hub_path",
    type=str,
    default="MetaLearningPrefDatabase2",
    help="Path to push the huggingface dataset",
)
parser.add_argument(
    "--seed", type=int, default=42, help="Seed for random number generation"
)
parser.add_argument(
    "--max_interaction_length", type=int, default=50, help="Maximum interaction length"
)
parser.add_argument(
    "--min_interaction_length", type=int, default=25, help="Minimum interaction length"
)


args = parser.parse_args()

dataset = load_dataset("andrewsiah/Personalization_Bench_Cleaned")
ds = dataset["train"]

reward_models_names = sorted(list(REWARD_MODELS.keys()))
n_rm = len(reward_models_names)


# Define functions for calculating distances and filtering similar personas
@jit(nopython=True, parallel=True)
def calculate_distances(persons):
    n = len(persons)
    distances = np.empty((n, n), dtype=np.float64)
    for i in prange(n):
        for j in prange(i + 1, n):
            distances[i, j] = np.sqrt(np.sum((persons[i] - persons[j]) ** 2))
            distances[j, i] = distances[i, j]
    return distances


@jit(nopython=True)
def filter_similar_personas(persons, threshold=1e-5):
    n = len(persons)
    distances = calculate_distances(persons)
    keep_mask = np.ones(n, dtype=np.bool_)
    for i in range(n):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, n):
            if distances[i, j] < threshold:
                keep_mask[j] = False
    return persons[keep_mask]


# Define function for generating personas
def generate_personas(
    alpha_values,
    n_rm,
    n_persons,
    filter_persona_threshold=None,
    same_alpha=True,
    random_alpha=False,
):
    all_persons = []
    random_state = args.seed
    if same_alpha:
        for alpha in alpha_values:
            alphas = np.array([alpha] * n_rm)
            persons = dirichlet.rvs(alphas, size=n_persons, random_state=random_state)
            all_persons.append(persons)
    if random_alpha:
        alphas = np.random.choice(alpha_values, size=n_rm)
        persons = dirichlet.rvs(alphas, size=n_persons, random_state=random_state)
        all_persons.append(persons)
    all_persons = np.vstack(all_persons)
    if filter_persona_threshold:
        all_persons = filter_similar_personas(all_persons, filter_persona_threshold)
    return all_persons


# Generate personas
n_persons = args.n_persons
n_responses = args.n_responses
alpha_values = [args.alpha]
persons = generate_personas(
    alpha_values,
    n_rm,
    n_persons,
    filter_persona_threshold=0,
    same_alpha=True,
    random_alpha=False,
)

# Generate historical database
np.random.seed(args.seed)

database_index = [
    [
        (
            np.random.randint(0, high=len(ds)),
            np.random.choice(np.arange(1, n_responses), size=2, replace=False),
        )
        for _ in range(
            np.random.randint(args.min_interaction_length, high=args.max_interaction_length)
        )
    ]
    for _ in range(len(persons))
]

historical_data = []
start = args.start
end = args.end

# iterate over each person
for i, person in tqdm(enumerate(database_index[start:end])):
    # create a list to store the interactions for each person
    person_data = []
    length_count = 0
    # iterate over each interaction
    for j, interaction in enumerate(person):
        j += 1
        length_count += 1 
        # get the prompt and response indices
        prompt_n, (response_1, response_2) = interaction
        # get the reward scores

        rewards = []
        for response_n in [response_1, response_2]:
            reward_scores = []
            for model_name in reward_models_names:
                column_data = ds[f'response_{response_n}_{model_name}']

                mean = REWARD_MODELS[model_name]['mean']
                std = REWARD_MODELS[model_name]['sd']
                normalized_score = (column_data[prompt_n] - mean) / std
                reward_scores.append(normalized_score)
            reward = np.dot(persons[i], reward_scores)
            rewards.append(reward)
        reward_1, reward_2 = rewards
        reward_1, reward_2 = reward_1, reward_2
        response_1, response_2 = int(response_1), int(response_2)

        # print(prompt_n)
        person_data.append(
            {
                "person_id": i,
                "person_weight": persons[i],
                f"prompt_{j}": ds["prompt"][prompt_n],
                f"response_{j}_a": ds[f"response_{response_1}"][prompt_n],
                f"response_{j}_b": ds[f"response_{response_2}"][prompt_n],
                f'chosen_{j}': "a" if reward_1 > reward_2 else "b",
            }
        )
    person_data[-1]['length'] = length_count
    historical_data.extend(person_data)

historical_database = pd.DataFrame(historical_data)
historical_database = historical_database.groupby("person_id").first()

historical_database_hf = Dataset.from_pandas(historical_database)
historical_database_hf.push_to_hub(args.push_to_hub_path)
