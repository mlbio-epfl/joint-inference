class BaseTemplate:
    def __call__(self, input_dict, y):
        """
            input_dict: dict of str
            y: str
        """
        pass


class OpenFlamingoImageClassificationTemplate(BaseTemplate):
    def __call__(self, input_dict, y):
        return f"<image>An image of {y}.<|endofchunk|>"


class OpenFlamingoVQATemplate(BaseTemplate):
    def __call__(self, input_dict, y):
        return f'<image>Question: {input_dict["q"]}? Short answer: {y}.<|endofchunk|>'


''' Text classification dataset'''
class SST2Template(BaseTemplate):
    def __call__(self, input_dict, y):
        return f"{input_dict['sentence']}\nThe sentiment of the sentence is {y}."
    
class AmazonPolarityTemplate(BaseTemplate):
    def __call__(self, input_dict, y):
        return f"{input_dict['title']} {input_dict['content']}\nThe sentiment of the sentence is {y}."

class AGNewsTemplate(BaseTemplate):
    def __call__(self, input_dict, y):
        return f"{input_dict['text']}\nThe topic of the sentence is about {y}."
    
class TRECTemplate(BaseTemplate):
    def __call__(self, input_dict, y):
        return f"{input_dict['text']}\nThe topic of the sentence is about {y}."
    
class DBPedia14Template(BaseTemplate):
    def __call__(self, input_dict, y):
        return f"{input_dict['title']} {input_dict['content']}\nThe topic of the sentence is about {y}."
    
class SUBJTemplate(BaseTemplate):
    def __call__(self, input_dict, y):
        return f"{input_dict['text']}\nThe sentence is {y}."

''' Natural language inference dataset '''

class RTETemplate(BaseTemplate):
    def __call__(self, input_dict, y):
        return f"{input_dict['premise']}\nQuestion: Does this imply that \"{input_dict['hypothesis']}\", yes or no?\nAnswer: {y}."

class QNLITemplate(BaseTemplate):
    def __call__(self, input_dict, y):
        return f"{input_dict['sentence']}\nQuestion: Does that sentence have all you need to answer the question \"{input_dict['question']}\", yes or no?\nAnswer: {y}."

class MNLITemplate(BaseTemplate):
    def __call__(self, input_dict, y):
        return f"{input_dict['premise']}\nBased on the previous passage, is it true that \"{input_dict['hypothesis']}\"?\n {y}."

''' Question answering dataset '''
class COPATemplate(BaseTemplate):
    def __call__(self, input_dict, y):
        return f"Consider the following premise: ''' {input_dict['premise']} '''\nChoice 1: {input_dict['choice1']}\nChoice 2: {input_dict['choice2']}\nQ: Which one is more likely to be the {input_dict['question']}, choice 1 or choice 2?\nA: choice {y}."
    
class BoolQTemplate(BaseTemplate):
    def __call__(self, input_dict, y):
        return f"{input_dict['passage']}\n\nQuestion: After reading this passage, the answer to the question {input_dict['question']} is yes or no?\nAnswer: {y}."

class PIQATemplate(BaseTemplate):
    def __call__(self, input_dict, y):
        return f"Goal: {input_dict['goal']}\nSolution 1: {input_dict['sol1']}\nSolution 2: {input_dict['sol2']}\nQuestion: Given the goal, what is the correct solution, solution 1 or solution 2?\nAnswer: solution {y}."

class HellaSwagTemplate(BaseTemplate):
    def __call__(self, input_dict, y):
        return f"Consider the following description: ''' {input_dict['ctx']} '''\nChoice 1: {input_dict['endings0']}\nChoice 2: {input_dict['endings1']}\nChoice 3: {input_dict['endings2']}\nChoice 4: {input_dict['endings3']}\nQuestion: Which is the most plausible ending, choice 1, choice 2, choice 3 or choice 4?\nAnswer: choice {y}."
    
''' Other datasets '''
class MMLUTempalte(BaseTemplate):
    def __call__(self, input_dict, y):
        choices = ["A", "B", "C", "D"]
        subject = input_dict['subject']
        prompt_head = f'The following are multiple choice questions (with answers) about {subject}.\n'
        prompt_question = f'{input_dict["question"]}\n'
        for i, opt in enumerate(input_dict['choices']):
            prompt_question += f'{choices[i]}. {opt}\n'
        prompt_answer = f"Answer: {y}\n"

        return prompt_head + prompt_question + prompt_answer