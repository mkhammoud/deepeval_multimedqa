class Open_G_EvalTemplate:
    @staticmethod
    def generate_evaluation_steps(parameters, criteria,generate_evaluation_steps_prompt_template):
        template=generate_evaluation_steps_prompt_template
        formatted_template = template.format(parameters=parameters, criteria=criteria)

        return formatted_template

    @staticmethod
    def generate_evaluation_results(evaluation_steps, text, parameters,generate_evaluation_results_prompt_template):
        template=generate_evaluation_results_prompt_template
        formatted_template = template.format(parameters=parameters, evaluation_steps=evaluation_steps, text=text)


        return formatted_template
