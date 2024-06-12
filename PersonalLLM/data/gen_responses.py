from datasets import load_dataset, Dataset
from datetime import datetime
import os
from dotenv import load_dotenv
import json
import yaml
from tqdm import tqdm
import asyncio
import aiohttp
import random


load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


async def test_get_model_response():
    """
    Tests the get_model_response function by sending a sample prompt to a specific model.
    """
    async with aiohttp.ClientSession() as session:
        content, response_json = await get_model_response(
            session,
            "How are you doing today?",
            "anthropic/claude-3-haiku",
            0.7,
            150,
            1.0,
            40,
        )
        print("Content:", content)
        print("Response JSON:", response_json)


async def get_model_response(
    session, prompt, model, temperature, max_length, top_p, top_k, retries=5
):
    """
    Asynchronously fetches a model response from the OpenRouter API.

    Args:
        session (aiohttp.ClientSession): The aiohttp session to use for the request.
        prompt (str): The prompt to send to the model.
        model (str): The model to query.
        temperature (float): The temperature setting for the model.
        max_length (int): The maximum length of the generated response.
        top_p (float): The top-p sampling parameter.
        top_k (int): The top-k sampling parameter.
        retries (int): The number of retries for rate limiting.

    Returns:
        tuple: A tuple containing the content of the response and the raw response JSON.
    """

    def log_message(message):
        with open("gen_response_logs.txt", "a", buffering=1) as f:  # Line buffering
            f.write(str(message) + "\n")  # Ensure new line after each log message
            f.flush()  # Force write to file immediately

    while retries > 0:
        try:
            async with session.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_length": max_length,
                    "top_p": top_p,
                    "top_k": top_k,
                },
            ) as response:
                response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
                response_json = await response.json()
                if response.status == 200:
                    content = response_json["choices"][0]["message"]["content"]
                    return content, response_json
                else:
                    raise
        except Exception as e:
            log_message(f"Error: {type(e).__name__}, Message: {str(e)}, Model: {model}")
            log_message(f"Exception details: {e}")
            backoff_time = min(10, 2 ** (5 - retries))
            await asyncio.sleep(random.uniform(1, backoff_time))
        retries -= 1
    log_message(f"Exhausted all retries for model {model} and prompt {prompt}")
    # Returned this Error message because None might cause a dataset to shrink one column.
    return "!!!ERROR: FILTER THIS ROW OUT LATER!!!", None


async def get_responses_for_prompts(conf, data):
    """
    Asynchronously fetches responses for a list of prompts using multiple models.

    Args:
        data (list): A list of dictionaries containing prompts and other metadata.

    Returns:
        tuple: A tuple containing two lists - one with the responses and one with the raw responses.
    """
    rate_limit = conf["openrouter_rate_limit"]

    all_responses = []
    all_raw_responses = []
    tasks = []
    async with aiohttp.ClientSession() as session:
        for i, item in tqdm(
            enumerate(data), total=len(data), desc="Processing prompts"
        ):
            prompt = item["prompt"]
            for model in conf["models"]:
                temperature = conf["temperature"]
                task = asyncio.create_task(
                    get_model_response(
                        session,
                        prompt,
                        model,
                        temperature,
                        conf["max_length"],
                        conf["top_p"],
                        conf["top_k"],
                    )
                )
                tasks.append((task, item, model))

                if len(tasks) % rate_limit == 0:
                    await asyncio.sleep(7)

        results = []
        for i in tqdm(
            range(0, len(tasks), rate_limit),
            desc=f"Querying Openrouter w/ Rate Limit {rate_limit}",
        ):
            chunk = tasks[i : i + rate_limit]
            print(f"Exiting Chunk {i}")
            chunk_results = await asyncio.gather(*[task for task, _, _ in chunk])
            results.extend(chunk_results)

        response_dict = {}
        raw_response_dict = {}
        for (result, item, model), (response, raw_response) in zip(tasks, results):
            prompt = item["prompt"]
            if prompt not in response_dict:
                response_dict[prompt] = {}
                raw_response_dict[prompt] = {}
            if response is not None and response.strip() != "":
                response_key = f"response_{len(response_dict[prompt]) // 2 + 1}"
                response_dict[prompt][response_key] = response
                response_dict[prompt][f"{response_key}_model"] = model
                raw_response_dict[prompt][response_key] = raw_response
                raw_response_dict[prompt][f"{response_key}_model"] = model
        for item in data:
            prompt = item["prompt"]
            all_responses.append({**item, **response_dict.get(prompt, {})})
            all_raw_responses.append({**item, **raw_response_dict.get(prompt, {})})
        # Push the data with responses to Hugging Face
        dataset = Dataset.from_list(all_responses)
        dataset_path = conf["output_huggingface_dataset_hub"]
        dataset.push_to_hub(dataset_path)

        local_save_fp = conf["local_save_filepath"]
        all_response_fp = f'{local_save_fp}/prompt_responses_{datetime.now().strftime("%Y%m%d%H%M%S")}.json'
        all_raw_response_fp = f'{local_save_fp}/raw_prompt_responses_{datetime.now().strftime("%Y%m%d%H%M%S")}.json'

        os.makedirs(os.path.dirname(all_response_fp), exist_ok=True)
        with open(all_response_fp, "w") as f:
            json.dump(all_responses, f)
        os.makedirs(os.path.dirname(all_raw_response_fp), exist_ok=True)
        with open(all_raw_response_fp, "w") as f:
            json.dump(all_raw_responses, f)
    return all_responses, all_raw_responses


if __name__ == "__main__":
    with open(
        "../../personal_llm/data/gen_responses_conf.yaml", "r"
    ) as f:
        conf = yaml.safe_load(f)
    dataset = load_dataset(conf["prompts_dataset"])
    dataset = dataset["train"]
    if conf["shuffle_dataset"]:
        dataset = dataset.shuffle(seed=41)

    dataset = dataset.select(
        range(conf["dataset_start_index"], conf["dataset_end_index"])
    )

    preexisting_dataset = load_dataset(conf["pre_existing_dataset"])["train"]
    if preexisting_dataset is not None:
        existing_ids = set(preexisting_dataset["id"])
        print(f"Number of existing rows to be excluded: {len(existing_ids)}")
        dataset = dataset.filter(lambda item: item["id"] not in existing_ids)

    if conf["test"]:
        conf["models"] = conf["test_models"]

    asyncio.run(get_responses_for_prompts(conf, dataset))
