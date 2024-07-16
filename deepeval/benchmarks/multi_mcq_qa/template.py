

from typing import List, Optional
import re


class Multi_MCQ_QA_PromptTemplate:

    # Most of this template was taken from MMLU Github Repo
    # The output confinement is a novel addition, since the original code
    # outputted log_probabilties for each answer choice

    @staticmethod
    def generate_output(
        input: str, train_set: object, task:str , n_shots: int , system_message_template:str, user_message_template:str, assistant_message_template:str,assistant_start_template:str
    ):
        
        prompt = system_message_template

        for i in range(n_shots):
            prompt += Multi_MCQ_QA_PromptTemplate.format_question(train_set[i],user_message_template, assistant_message_template,assistant_start_template)
            #print(f"FORMAT QUESTION OF N SHOTS: {i}",Multi_MCQ_QA_PromptTemplate.format_question(train_set[i],user_message_template, assistant_message_template,assistant_start_template))
        prompt += input

        return prompt
    

    @staticmethod
    def format_question(data: dict,user_message_template:str, assistant_message_template:str,assistant_start_template:str , include_answer: bool = True, ):
        
        context_data = data["data"].get("Context", None)

        question=data["data"]["Question"]

        user_message_template=user_message_template.replace("{question}",question)

        if context_data:
            context_paragraph = '\n\n'.join(context_data)
            user_message_template=user_message_template.replace("{context}","\n\nMake use of the following context to formulate your final answer: " + context_paragraph + "\n\n")
        else:
            user_message_template=user_message_template.replace("{context}","")
            
            
        choices = data["data"]["Options"]
        options=""
        for key, value in choices.items():
            options += f"\n{key}:{value}"

        user_message_template=user_message_template.replace("{options}",options)

        prompt=user_message_template

        if include_answer:
            prompt += assistant_message_template.format(data["data"]["Correct Option"])
        else:
            prompt+=assistant_start_template

        return prompt

