from typing import List, Optional, Dict
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

from deepeval.dataset import Golden
from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmark
from deepeval.models import DeepEvalBaseLLM
from deepeval.benchmarks.custom_mcq.template import CustomMCQTemplate
from deepeval.benchmarks.utils import should_use_batch
from deepeval.scorer import Scorer
import json

class CustomMCQ(DeepEvalBaseBenchmark):

    def __init__(
        self,dataset_path, n_shots: int , **kwargs
    ):
        assert n_shots <= 5, "MMLU only supports n_shots <= 5"
        super().__init__(**kwargs)
        self.scorer = Scorer()
        self.dataset_path=dataset_path
        self.shots_dataset: List[Dict] = None
        self.n_shots: int = n_shots
        self.predictions: Optional[pd.DataFrame] = None
        self.overall_score: Optional[float] = None

    def evaluate(
        self, model: DeepEvalBaseLLM, batch_size: Optional[int] = None
    ) -> Dict:
        overall_correct_predictions = 0
        overall_total_predictions = 0
        predictions_row = []
        use_batch = should_use_batch(model, batch_size)


        goldens = self.load_benchmark_dataset()
        overall_total_predictions += len(goldens)

        # Calculate task accuracy
        if use_batch:
            for i in tqdm(
                range(0, len(goldens), batch_size),
                desc=f"Batch Processing (batch_size={batch_size})",
            ):
                goldens_batch = goldens[i : i + batch_size]
                batch_predictions = self.batch_predict(
                    model, goldens_batch
                )
                for golden, prediction_dict in zip(
                    goldens_batch, batch_predictions
                ):
                    prediction = prediction_dict["prediction"]
                    score = prediction_dict["score"]
                    if score:
                        overall_correct_predictions += 1
                    predictions_row.append(
                        ( golden.input, prediction, score)
                    )
        else:
            for golden in tqdm(goldens, desc=f"Processing"):
                prediction, score = self.predict(
                    model, golden
                ).values()
                if score:
                    overall_correct_predictions += 1
                predictions_row.append(
                    (golden.input, prediction, score)
                )



        # Calculate overall accuracy
        overall_accuracy = (
            overall_correct_predictions / overall_total_predictions
        )
        print(f"Overall MMLU Accuracy: {overall_accuracy}")

        # Create a DataFrame from task_results_data
        # Columns: 'Task', 'Input', 'Prediction', 'Score'
        self.predictions = pd.DataFrame(
            predictions_row, columns=[ "Input", "Prediction", "Correct"]
        )
        self.overall_score = overall_accuracy

        return overall_accuracy

    def predict(
        self, model: DeepEvalBaseLLM, golden: Golden
    ) -> Dict:
        # Define prompt template
        assert (
            self.shots_dataset != None
        ), "Example dataset is empty. Call load_benchmark."
        prompt: dict = CustomMCQTemplate.generate_output(
            train_set=self.shots_dataset,
            input=golden.input,
            n_shots=self.n_shots,
        )

        print("MMLU Prompt: ",prompt)

        prediction = model.generate(prompt)
        print("LLM Prediction: ",prediction)
        # For native models, shouldn't happen but just in case
        if isinstance(prediction, tuple):
            prediction = prediction[0]

        # Define Metric
        score = self.scorer.exact_match_score(
            golden.expected_output, prediction
        )
        print("Golden Expected Output: ",golden.expected_output)
        return {"prediction": prediction, "score": score}

    def batch_predict(
        self, model: DeepEvalBaseLLM, goldens: List[Golden]
    ) -> List[Dict]:
        # Define prompt template
        assert (
            self.shots_dataset != None
        ), "Example dataset is empty. Call load_benchmark."

        prompts = []
        for golden in goldens:
            prompt: dict = CustomMCQTemplate.generate_output(
                train_set=self.shots_dataset,
                input=golden.input,
                n_shots=self.n_shots,
            )
            prompts.append(prompt)

        predictions = model.batch_generate(prompts)
        if len(predictions) is not len(goldens):
            raise ValueError(
                "Custom `batch_generate` method did not return the same number of generations as the number of prompts."
            )

        res = []
        for i in range(len(predictions)):
            prediction = predictions[i]
            print("Prediction: ",prediction)
            golden = goldens[i]
            # Define Metric
            score = self.scorer.exact_match_score(
                golden.expected_output, prediction
            )
            print("Score: ", score)
            print("Golden Expected Output: ",golden.expected_output)
            res.append({"prediction": prediction, "score": score})

        return res

    def load_benchmark_dataset(self) -> List[Golden]:
        # If dataset has been previously loaded, load from
        # instance var (to save time)
        if self.dataset:
            dataset = self.dataset
        else:
            with open(self.dataset_path, 'r',encoding="utf-8") as f:
                dataset = json.load(f)
                self.dataset=dataset

        # If dataset has not been previously loaded, construct
        # dataset of examples and save as instance var (to save time)
        if not self.shots_dataset:
            train_set = dataset["train"]
            shots_set = []
            for data in train_set:
                shots_set.append(data)
            self.shots_dataset = shots_set

        # Construct test set
        goldens: List[Golden] = []
        for data in dataset["test"]:
            print("Golden Dataset data Point: ",data)
            input = CustomMCQTemplate.format_question(data, include_answer=False)
            golden = Golden(input=input, expected_output=data["target"])
            goldens.append(golden)
        return goldens
