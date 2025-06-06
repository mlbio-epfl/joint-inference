import os
import random
import re 
import ast
import base64
from io import BytesIO

from PIL import Image
import numpy as np
from openai import OpenAI
from tqdm import tqdm
import json
from argparse import ArgumentParser
from utils.mmmu import MMMU, MMMUPro


GPT_IMG_DETAIL = 'low'
SYSTEM_PROMPT = {
    'multiple-choice': "Answer the following multiple-choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of the options. Think step by step before answering.",
    'open': "Answer the following open-ended question. Do not use latex. Think step by step before answering. Provide the final answer in the last line of your response in the following format 'The final answer is $ANSWER' (without quotes) where $ANSWER is a single word, phrase or number."
}
QUESTION_SUFFIX = {
    "multiple-choice": "The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of the options. Think step by step before answering.",
    "open": "Do not use latex. Think step by step before answering. Provide the final answer in the last line of your response in the following format 'The final answer is $ANSWER' (without quotes) where $ANSWER is a single word, phrase or number."
}


def get_multi_choice_info(options):
    """
    Given the list of options for multiple choice question
    Return the index2ans and all_choices
    
    taken from: https://github.com/MMMU-Benchmark/MMMU/blob/aa9b70da92c2825b3d544d1a11b36856bd92f6c3/mmmu/utils/data_utils.py#L58 
    """
    start_chr = 'A'
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))

    return index2ans, all_choices


def extract_answer(text):
    pattern = r"The best answer is \(?([A-Z])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1), 1
    else:
        #print("1st answer extract failed\n" + text)
        return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-Z])', text)
    if match:
        return match.group(1), 2
    else:
        return extract_final(text)


def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-Z]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0), 3
    else:
        return None, 4


def encode_image(image: Image.Image) -> str:
    """Function to encode the image."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")  # You can specify the format if needed
    encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()  # closing the buffer
    return encoded_string


def replace_images_tokens(input_string):
    for i in range(1, 8):
        question_text = f"<image {i}>"
        query_text = f"image {i}"
        if question_text in input_string:
            input_string = input_string.replace(question_text, query_text)
    return input_string


def mmmu_doc_to_text(doc):
    question = construct_prompt(doc)
    return replace_images_tokens(question)
    #return question


def origin_mmmu_doc_to_visual(doc):
    visual = []
    for i in range(1,8):
        if not doc[f'image_{i}']:
            break
        visual.append(doc[f'image_{i}'].convert("RGB"))
    return visual


def parse_options(options):
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
    return choices_str


def construct_prompt(doc):
    question = doc["question"]
    if doc['question_type'] == "multiple-choice":
        parsed_options = parse_options(ast.literal_eval(str(doc["options"])))
    else:
        parsed_options = ""
        print(f'Question type: {doc["question_type"]}')

    question = f"{question}\n{parsed_options}\n{QUESTION_SUFFIX[doc['question_type']]}"
    return question


def prepare_example(data):
    prompt = mmmu_doc_to_text(data)
    images = origin_mmmu_doc_to_visual(data)
    conversation_content = [{"type": "text", "text": prompt}]
    # add picture content
    for image in images:
        conversation_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encode_image(image)}", "detail": GPT_IMG_DETAIL}
        })
    return conversation_content


def get_icl_examples(query_example, icl_path, dataset, N):
    query_id = query_example['id']
    if isinstance(dataset, MMMUPro):
        icl_indices = np.random.permutation(len(dataset))
    elif query_example['question_type'] == 'multiple-choice':
        icl_indices = np.random.permutation(len(dataset))
        icl_indices = [idx for idx in icl_indices if dataset[int(idx)]['question_type'] == 'multiple-choice']
    elif query_example['question_type'] == 'open':
        icl_indices = []
        # add "open" questions first
        for idx, example in enumerate(dataset):
            if example['question_type'] == 'open':
                icl_indices.append(idx)
        # add "multiple-choice" questions
        for idx, example in enumerate(dataset):
            if example['question_type'] == 'multiple-choice':
                icl_indices.append(idx)

    i = 0
    icl_examples = []
    while len(icl_examples) < N:
        icl_example = dataset[icl_indices[i]]
        i += 1
        # load answer
        icl_example_path = os.path.join(icl_path, f'{icl_example["id"]}.json')
        with open(icl_example_path, 'r') as f:
            icl_data = json.load(f)
        icl_response_func_id = extract_answer(icl_data['response'])[1]
        # for multiple-choice questions, choose ICL examples matching Answer: $LETTER
        if (icl_response_func_id == 2 or icl_example['question_type'] == "open") and query_id != icl_example['id']:
            if isinstance(dataset, MMMU) and (query_example['question_type'] == 'multiple-choice'):
                assert icl_example['question_type'] == 'multiple-choice'
            icl_examples.append({
                'id': icl_example['id'],
                'user_content': prepare_example(icl_example),
                'assistant_content': [
                    {"type": "text", "text": icl_data['response']},
                ]
            })
    # assume that ordering doesn't matter for MCQ and put open at the end
    icl_examples = icl_examples[::-1]

    icl_messages = []
    icl_ids = []
    for icl_example in icl_examples:
        icl_ids.append(icl_example['id'])
        icl_messages.append({
            'role': 'user',
            'content': icl_example['user_content']
        })
        icl_messages.append({
            'role': 'assistant',
            'content': icl_example['assistant_content']
        })
     
    return icl_messages, icl_ids


def example2message(example, icl_messages=None):
    icl_messages = [] if icl_messages is None else icl_messages
    message_content = prepare_example(example)
    messages = [
        {
            'role': 'system',
            'content': [
                {"type": "text", "text": SYSTEM_PROMPT[example['question_type']]},
            ]
        },
        *icl_messages,
        {
            'role': 'user',
            'content': message_content
        }
    ]
    return messages


def gpt_predict(messages, model, client, seed=0):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=4000,
        seed=seed,
        response_format={"type": "text"},
        temperature=0,  # greedy sampling
    )
    return response.choices[0].message.content


def eval_outputs(outputs):
    all_predictions = []
    all_answers = []
    for out in outputs:
        prediction = extract_answer(out['response'])[0]
        all_predictions.append(prediction)
        all_answers.append(out['answer'])

    return np.mean([p == a for p, a in zip(all_predictions, all_answers)])


def main(args):
    if args.dataset == 'mmmu-pro':
        dataset = MMMUPro(
            option="standard (10 options)",
            subject=args.subject_name,
            support_difficulty=["Easy", "Medium", "Hard"],
            query_difficulty=["Easy", "Medium", "Hard"],
            single_image=False
        )
    elif args.dataset == 'mmmu':
        dataset = MMMU(args.subject_name)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if args.uicl_path != '':
        args.uicl_path = os.path.join(args.uicl_path, args.subject_name, 'outputs')

    client =  OpenAI()

    save_path = args.save_path
    all_outputs_dir = os.path.join(save_path, args.subject_name, 'outputs')
    os.makedirs(all_outputs_dir, exist_ok=True)

    all_outputs = []

    for example in tqdm(dataset):
        assert args.subject_name in example['id'], f"Example ID {example['id']} does not match subject name {args.subject_name}"

        # read the output if it already exists
        curr_path = os.path.join(all_outputs_dir, f'{example["id"]}.json')
        if os.path.exists(curr_path):
            with open(curr_path, 'r') as f:
                output = json.load(f)
            all_outputs.append(output)
            continue

        # prepare messages for ICL examples
        icl_messages = None
        icl_ids = []
        if args.uicl_path != '':
            icl_messages, icl_ids = get_icl_examples(example, args.uicl_path, dataset, N=args.uicl_n)

        # prepare the query message for the current example
        messages = example2message(example, icl_messages=icl_messages)

        # get the response from the model
        response = gpt_predict(messages, args.model, client, seed=args.seed)

        options = ast.literal_eval(example['options'])
        output = {
            'id': example['id'],
            'question_type': example['question_type'],
            'answer': example['answer'],
            'all_choices': get_multi_choice_info(options)[1],
            'index2ans': get_multi_choice_info(options)[0],
            'response': response,
            'icl_ids': icl_ids,
            'icl_pred_path': args.uicl_path,
            'parsed_answer': extract_answer(response)[0],
        }

        with open(curr_path, 'w') as f:
            json.dump(output, f, indent=4)

        all_outputs.append(output)

    with open(os.path.join(save_path, args.subject_name, 'output.json'), 'w') as f:
        json.dump(all_outputs, f, indent=4)

    acc = eval_outputs(all_outputs)
    print(f"{args.subject_name} accuracy: {acc}")
    with open(os.path.join(save_path, args.subject_name, 'accuracy.txt'), 'w') as f:
        f.write(f"{acc}\n")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--save_path', type=str, default='./mmmu-pro-t0/', help="The path to save the outputs.")
    parser.add_argument('--dataset', type=str, default='mmmu-pro', help="The dataset to use.")
    parser.add_argument('--uicl_path', type=str, default='', help="Path to the labels for ICL. Default: '' (no ICL).")
    parser.add_argument('--uicl_n', type=int, default=8, help="The number of ICL examples to use.")
    parser.add_argument('--model', type=str, default="gpt-4o-2024-08-06", help="The model to use for prediction.")
    parser.add_argument('--subject_name', type=str, default="History", help="The subject to evaluate.")
    parser.add_argument('--seed', type=int, default=0, help="The seed to use for prediction.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    main(args)