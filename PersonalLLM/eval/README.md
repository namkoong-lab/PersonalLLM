## Execution Guide:

1. Change directory to 'eval': `cd eval`
2. Activate the 'personal' conda environment: `conda activate personal`

Run the evaluation script with different algorithms:
- TestAlgorithm in Debug Mode: `python eval.py --algorithm=TestAlgorithm --debug`
- VanillaICLAlgorithm on Pairwise Preference Setting: `python eval.py --algorithm=VanillaICLAlgorithm --eval_type=pairwise_pref`
- MetaLearnTenKAlgorithm: `python eval.py --algorithm=MetaLearnTenKAlgorithm`

To execute the sweep script:
`python sweep.py`

## Personalization Algorithm:

The algorithm processes unique personalities (3-5 conversations long), optionally using external datasets. It adds a `test_response` column to the dataset, containing responses to test prompts.

The modified dataset is evaluated using 'eval.py'. Configuration parameters can be passed via argparse (e.g., `--debug`) or by modifying 'eval.py'.
