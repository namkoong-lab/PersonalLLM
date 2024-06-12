import subprocess

# Define the seeds to sweep through
seeds = [27]

# Define the algorithms and their corresponding eval types
algorithms_eval_types = {
    "ZeroShotAlgorithm": ["reward"],
    "VanillaICLAlgorithm": [ "pairwise_pref"],
    "OneShotICLAlgorithm": ["pairwise_pref"],
    "ThreeShotICLAlgorithm": ["pairwise_pref"],
    "VanillaICLAlgorithm": ["reward"]
}

# Define debug mode
debug = False

# Loop through the seeds, algorithms, and eval types
for seed in seeds:
    for algorithm, eval_types in algorithms_eval_types.items():
        for eval_type in eval_types:
            command = ['python', 'eval.py', '--algorithm=' + algorithm, '--seed=' + str(seed)]
            if eval_type:
                command.append('--eval_type=' + eval_type)
            if debug:
                command.append('--debug')

            try:
                subprocess.check_call(command)
            except Exception as e:
                with open('sweep_logs.txt', 'a') as log_file:
                    log_file.write(f"Failed to run {algorithm} with {eval_type if eval_type else ''} and seed {seed}: {str(e)}\n")
