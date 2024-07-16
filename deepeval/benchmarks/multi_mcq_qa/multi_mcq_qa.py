from typing import List, Optional, Dict
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

from deepeval.dataset import Golden
from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmark
from deepeval.models import DeepEvalBaseLLM
from deepeval.benchmarks.multi_mcq_qa.template import Multi_MCQ_QA_PromptTemplate
from deepeval.benchmarks.utils import should_use_batch
from deepeval.scorer import Scorer
import json

class Multi_MCQ_QA(DeepEvalBaseBenchmark):
    def __init__(
        self, dataset_path,system_message_template:str="You are a helpful AI Assistant",user_message_template:str="Question: {question} \n Options: {options} \n {context}  ",assistant_message_template:str="Assistant: {}",assistant_start_template:str="Assistant: ",number_of_rows_to_consider: int=None,tasks: List[str] = None, n_shots: int = 5,**kwargs
    ):
        assert n_shots <= 5, "Multi_MCQ_QA only supports n_shots <= 5"
        super().__init__(**kwargs)
        self.tasks=tasks
        self.scorer = Scorer()
        self.shots_dataset: List[Dict] = None
        self.n_shots: int = n_shots
        self.predictions: Optional[pd.DataFrame] = None
        self.task_scores: Optional[pd.DataFrame] = None
        self.overall_score: Optional[float] = None
        self.number_of_rows_to_consider=number_of_rows_to_consider
        self.dataset_path=dataset_path

        self.system_message_template=system_message_template
        self.user_message_template=user_message_template
        self.assistant_message_template=assistant_message_template
        self.assistant_start_template=assistant_start_template


    def evaluate(
        self, model: DeepEvalBaseLLM, batch_size: Optional[int] = None
    ) -> Dict:
        overall_correct_predictions = 0
        overall_total_predictions = 0
        predictions_row = []
        scores_row = []
        use_batch = should_use_batch(model, batch_size)

        for task in self.tasks:
            goldens = self.load_benchmark_dataset(task)
            task_correct_predictions = 0
            task_total_predictions = len(goldens)
            overall_total_predictions += len(goldens)

            # Calculate task accuracy
            if use_batch:
                for i in tqdm(
                    range(0, len(goldens), batch_size),
                    desc=f"Batch Processing {task} (batch_size={batch_size})",
                ):
                    goldens_batch = goldens[i : i + batch_size]
                    batch_predictions = self.batch_predict(
                        model, task, goldens_batch, batch_size
                    )
                    for golden, prediction_dict in zip(
                        goldens_batch, batch_predictions
                    ):
                        prediction = prediction_dict["prediction"]
                        score = prediction_dict["score"]
                        if score:
                            task_correct_predictions += 1
                            overall_correct_predictions += 1
                        predictions_row.append(
                            (task, golden.input, prediction,golden.context,golden.expected_output,score)
                        )
            else:
                for golden in tqdm(goldens, desc=f"Processing {task}"):
                    prediction, score = self.predict(
                        model, task, golden
                    ).values()
                    if score:
                        task_correct_predictions += 1
                        overall_correct_predictions += 1
                    predictions_row.append(
                        (task, golden.input, prediction,golden.context,golden.expected_output,score)
                    )

            task_accuracy = task_correct_predictions / task_total_predictions
            print(f"Task Accuracy (task={task}): {task_accuracy}")
            scores_row.append((task, task_accuracy))

        # Calculate overall accuracy
        overall_accuracy = (
            overall_correct_predictions / overall_total_predictions
        )
        print(f"Overall Accuracy: {overall_accuracy}")

        # Create a DataFrame from task_results_data
        # Columns: 'Task', 'Input', 'Prediction', 'Score'
        self.predictions = pd.DataFrame(
            predictions_row, columns=["Task", "Input", "Prediction","Context","Correct Answer","Score"]
        )
        self.task_scores = pd.DataFrame(scores_row, columns=["Task", "Score"])
        self.overall_score = overall_accuracy

        return overall_accuracy

    def predict(
        self, model: DeepEvalBaseLLM, task: str, golden: Golden
    ) -> Dict:
        # Define prompt template
        assert (
            self.shots_dataset != None
        ), "Example dataset is empty. Call load_benchmark."
        prompt: dict = Multi_MCQ_QA_PromptTemplate.generate_output(
            train_set=self.shots_dataset,
            input=golden.input,
            task=task,
            n_shots=self.n_shots,
            system_message_template= self.system_message_template,
            user_message_template=self.user_message_template,
            assistant_message_template=self.assistant_message_template,
            assistant_start_template=self.assistant_message_template
            
        )

        #print("Final Prompt: ",prompt)

        prediction = model.generate(prompt)
        #print("LLM Prediction: ",prediction)
        # For native models, shouldn't happen but just in case
        if isinstance(prediction, tuple):
            prediction = prediction[0]

        # Define Metric
        score = self.scorer.exact_match_score(
            golden.expected_output, prediction
        )
        #print("Golden Expected Output: ",golden.expected_output)
        return {"prediction": prediction, "score": score}

    def batch_predict(
        self, model: DeepEvalBaseLLM, task: str, goldens: List[Golden], batch_size:int
    ) -> List[Dict]:
        # Define prompt template
        assert (
            self.shots_dataset != None
        ), "Example dataset is empty. Call load_benchmark."

        prompts = []
        for golden in goldens:

            prompt: dict = Multi_MCQ_QA_PromptTemplate.generate_output(
                train_set=self.shots_dataset,
                input=golden.input,
                task=task,
                n_shots=self.n_shots,
                system_message_template= self.system_message_template,
                user_message_template=self.user_message_template,
                assistant_message_template=self.assistant_message_template,
                assistant_start_template=self.assistant_message_template
            )
            prompts.append(prompt)

        predictions = model.batch_generate(prompts,batch_size)
        if len(predictions) is not len(goldens):
            raise ValueError(
                "Custom `batch_generate` method did not return the same number of generations as the number of prompts."
            )

        res = []
        for i in range(len(predictions)):
            prediction = predictions[i]
            #print("Prediction: ",prediction)
            golden = goldens[i]
            # Define Metric
            score = self.scorer.exact_match_score(
                golden.expected_output, prediction
            )
            #print("Score: ", score)
            #print("Golden Expected Output: ",golden.expected_output)
            res.append({"prediction": prediction, "score": score})

        return res

    def load_benchmark_dataset(self, task: str) -> List[Golden]:
        # If dataset has been previously loaded, load from
        # instance var (to save time)

        with open(self.dataset_path+task+".json", 'r',encoding="utf-8") as f:
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
        for data in dataset["test"][:self.number_of_rows_to_consider]:
            #print("Golden Dataset data Point: ",data)
            input = Multi_MCQ_QA_PromptTemplate.format_question(data, self.user_message_template, self.assistant_message_template, self.assistant_start_template, include_answer=False)
            context_data = data["data"].get("Context", None)

            if context_data:
                golden = Golden(input=input, expected_output=data["data"]["Correct Option"],context=context_data)
            else:
                golden = Golden(input=input, expected_output=data["data"]["Correct Option"])
            goldens.append(golden)

        return goldens
