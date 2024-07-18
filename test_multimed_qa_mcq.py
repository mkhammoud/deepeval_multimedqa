from deepeval.models.base_model import DeepEvalBaseLLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
import torch
import gc
import os
import json
import time


# Define benchmark with specific tasks and shots
from deepeval.benchmarks.multi_mcq_qa.multi_mcq_qa import Multi_MCQ_QA
from typing import List

from datetime import datetime

run_id = datetime.now().strftime("%y%m%d%H%M%S")



class CustomModel(DeepEvalBaseLLM):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def clear_memory(self):
        torch.cuda.empty_cache()
        gc.collect()
        gc.collect()

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()

        pipe = pipeline(
            "text-generation",
            model=model,
            max_new_tokens=256,
            tokenizer=tokenizer,
            temperature=0.5,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            return_full_text=False,
        )

        answer = pipe(prompt)[0]["generated_text"].strip()
        self.clear_memory()
        return answer

    def batch_generate(self, prompts: List[str], batch_size: int) -> List[str]:
        model = self.load_model()

        pipe = pipeline(
            "text-generation",
            model=model,
            max_new_tokens=256,
            tokenizer=tokenizer,
            temperature=0.5,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            return_full_text=False,
            batch_size=batch_size,
        )

        answers = [item[0]["generated_text"] for item in pipe(prompts)]
        self.clear_memory()
        return answers

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Mistral 7B"


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
)

model_path = "amew0/l3-8b-medical-v240623023136"
# model_path = "amew0/Meta-Llama-3-8B-Instruct-v240714045919"
# model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
cache_dir = "/dpc/kunf0097/l3-8b"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    # config=quantization_config,
    cache_dir=f"{cache_dir}/model",
    device_map="auto",
    offload_buffers=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path, cache_dir=f"{cache_dir}/tokenizer"
)

custom_model = CustomModel(model=model, tokenizer=tokenizer)

# tb saved in config. change here NOT in the params
number_of_rows_to_consider = None # None for all
batch_size_number = None # None for no batching
tasks = ["mmlu_anatomy"]

benchmark = Multi_MCQ_QA(
    tasks=tasks,
    dataset_path="./eval/",
    n_shots=5,
    number_of_rows_to_consider=number_of_rows_to_consider,
    system_message_template="<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are being assessed on your knowledge of medical subjects to evaluate your understanding and reasoning capabilities in this field. Below, you will find multiple-choice questions related to medicine and biology. Each question has four options, and you must choose only the single correct one, represented by its corresponding letter. You should only reply with the correct Letter without any additional explanation or words, only one single letter. <|eot_id|>",
    user_message_template="<|start_header_id|>user<|end_header_id|> Question: {question}. {context}.  \n Here are the Options: {options}. <|eot_id|>",
    assistant_message_template="<|start_header_id|>assistant<|end_header_id|>{}<|eot_id|>",
    assistant_start_template="<|start_header_id|>assistant<|end_header_id|>",
)

start = time.time()
benchmark.evaluate(model=custom_model, batch_size=batch_size_number)
end = time.time()

print("Task Scores: ", benchmark.task_scores)
print(run_id)


model_name = model_path.split("/")[-1]
if not os.path.exists(f"{model_name}/{run_id}"):
    os.makedirs(f"{model_name}/{run_id}")

benchmark.predictions.to_json(
    f"{model_name}/{run_id}/predicitions_{run_id}.json",
    orient="records",
    force_ascii=False,
    indent=4,
)

benchmark.task_scores.to_json(
    f"{model_name}/{run_id}/score_summary_{run_id}.json",
    orient="records",
    force_ascii=False,
    indent=4,
)

with open(f"{model_name}/{run_id}/config_{run_id}.json", "w") as f:
    json.dump(
        {
            "number_of_rows_to_consider": number_of_rows_to_consider,
            "batch_size": batch_size_number,
            "time": end - start,
            "tasks": tasks,
        },
        f,
    )
