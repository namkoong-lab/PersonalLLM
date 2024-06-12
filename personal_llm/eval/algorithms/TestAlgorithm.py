from algorithms.BaseAlgorithm import BaseAlgorithm
from datasets import Dataset


class TestAlgorithm(BaseAlgorithm):
    """
    This is a test algorithm class that inherits from the BaseAlgorithm class.
    It is used for testing purposes.
    """
    def __init__(self, args):
        """
        Initialize the TestAlgorithm with a meta learning database.
        If no database is provided, it defaults to None.
        """
        super().__init__(args)

    def generate_string(self) -> str:
        """
        Generate a string to be used in the response.
        In this test case, it always returns the string "ANSWER."
        """
        result_str = "ANSWER."
        return result_str

    def generate_evaluation_responses(self, args) -> Dataset:
        """
        Generate a response by adding a new column 'test_response' to the test dataset.
        The response is a combination of the test prompt and the generated string.
        """
        debug = args.debug
        dataset = self.eval_dataset
        if debug:
            dataset = dataset.select(range(20))
        dataset = dataset.add_column(
            "test_response",
            [
                dataset["test_prompt"][i] + self.generate_string()
                for i in range(len(dataset))
            ],
        )
        return dataset

if __name__ == "__main__":
    test_gen = TestAlgorithm()
    updated_dataset = test_gen.generate_response()
    print(updated_dataset[0]["test_response"])
