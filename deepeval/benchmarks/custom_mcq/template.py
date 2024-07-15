

class CustomMCQTemplate:

    # Most of this template was taken from MMLU Github Repo
    # The output confinement is a novel addition, since the original code
    # outputted log_probabilties for each answer choice

    @staticmethod
    def generate_output(
        input: str, train_set: object, n_shots: int
    ):
        prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are being assessed on your knowledge of medical subjects to evaluate your understanding and reasoning capabilities in this field. Below, you will find multiple-choice questions related to medicine and biology. Each question has four options, and you must choose only the single correct one, represented by its corresponding letter. You should only reply with the correct Letter without any additional explanation or words, only one single letter. <|eot_id|>\n\n"
        for i in range(n_shots):
            prompt += CustomMCQTemplate.format_question(train_set[i])
        prompt += input

        # define ouptut confinement
        #prompt += "Output 'A', 'B', 'C', or 'D'. Full answer not needed."
        #print(prompt)
        return prompt

    @staticmethod
    def format_question(data: dict, include_answer: bool = True):
        prompt = "<|start_header_id|>user<|end_header_id|> "
        prompt += data["input"]
        choices = ["A", "B", "C", "D"]
        for j in range(len(choices)):
            choice = choices[j]
            prompt += "\n{}. {}".format(choice, data[choice]) 
        prompt += "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
        if include_answer:
            prompt += "{}".format(data["target"]) +"<|eot_id|>"
        return prompt

    def format_subject(subject: str):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s
