from datasets import load_dataset, Dataset


class BaseAlgorithm:
    def __init__(self, args):

        if args.eval_type == "pairwise_pref":
            self.eval_dataset = load_dataset("andrewsiah/Eval_Pref_Dataset")["test"]
            self.meta_learning_database = load_dataset(
                "andrewsiah/MetaLearningRewardDatabase"
            )["train"]
        if args.eval_type == "reward":
            self.eval_dataset = load_dataset("andrewsiah/Eval_Reward_Dataset")["test"]
            self.meta_learning_database = load_dataset(
                "andrewsiah/MetaLearningPrefDatabase"
            )["train"]

    def generate_evaluation_responses(self, args) -> Dataset:
        raise NotImplementedError
