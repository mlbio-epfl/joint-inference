import os
import argparse
from collections import Counter
import json
import random
import copy
import re

import numpy as np
import torch
from vllm import LLM, SamplingParams
 
from utils.text_utils import get_mmlu_pro
from misc.constants import CACHE_DIR


choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

def extract_answer(text):
    if type(text) is not str:
        return ''
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        # print("1st answer extract failed\n" + text)
        return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return ''

def format_question(ex):
    prompt = f'Question:\n{ex["question"]}\nOptions:\n'
    for i, opt in enumerate(ex['options']):
        prompt += f'{choices[i]}. {opt}\n'
    
    prompt += "Answer: Let's think step by step."

    return prompt

def format_zs(example):
    category = example['category']
    prompt_head = f'The following are multiple choice questions (with answers) about {category}. Think step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice.\n'
    prompt_query = format_question(example)
    return prompt_head + prompt_query

def format_icl(support_ds, query_example, n=5):
    category = query_example['category']
    support_ds = [example for example in support_ds if example['category'] == category]
    
    prompt_head = f'The following are multiple choice questions (with answers) about {category}. Think step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice.\n'
    prompt_demon = ""
    support_set = np.random.choice(support_ds, n, replace=False)
    for example in support_set:
        prompt_demon += format_question(example) + example['raw_response'] + '\n\n'
    
    prompt_query = format_question(query_example)

    return prompt_head + prompt_demon + prompt_query


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--n', type=int, default=5)
    parser.add_argument('--T', type=int, default=8, help='number of rounds of unsupervised ICL')
    parser.add_argument('--num_repeats', type=int, default=5, help='number of trials for each query for majority vote')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--temperature', type=float, default=0.)
    parser.add_argument('--top_p', type=float, default=1.)
    parser.add_argument('--max_new_tokens', type=int, default=1024)

    parser.add_argument('--save_preds', action='store_true', help='set True to save the prediction of the model')
    parser.add_argument('--save_path', type=str, default="./exp_local/unsup_icl_llm")    
    return parser.parse_args()

if __name__ == '__main__':
    args = _parse_args()
    np.random.seed(args.seed)

    model = LLM(args.model_name, download_dir=CACHE_DIR, tensor_parallel_size=torch.cuda.device_count())
    sampling_params = SamplingParams(temperature=args.temperature, 
                                     top_p=args.top_p, 
                                     max_tokens=args.max_new_tokens,
                                     n=1,
                                     seed=args.seed,
                                     stop=[f'answer is ({a})' for a in choices],
                                     include_stop_str_in_output=True)

    # load dataset
    val_ds_all, test_ds_all = get_mmlu_pro()
    categories = set([example['category'] for example in val_ds_all])

    def post_processing_response(outputs, gt_answers, n=1):
        # outputs: output of vllm, with N questions and N_samples for each question
        n_te = len(gt_answers)
        output_texts = sum([[o.text for o in output.outputs] for output in outputs], [])
        output_texts = [output_texts[i*n: (i+1)*n] for i in range(n_te)]
        raw_response, answer, correct = [], [], []
        for ys, gt_ans in zip(output_texts, gt_answers):
            answers = [extract_answer(y) for y in ys if extract_answer(y) != '']
            if len(answers) > 0:
                major_answer = Counter(answers).most_common()[0][0]
                response = np.random.choice([y for y in ys if extract_answer(y) == major_answer])
                
                raw_response.append(response)
                answer.append(major_answer)
                correct.append(gt_ans == major_answer)
            else:
                raw_response.append(np.random.choice(ys))
                answer.append('')
                correct.append(False)

        return {'raw_response': raw_response, 'answer': answer, 'correct': correct}

    for category in categories:
        print(f"starting experiments on {category}")
        val_ds = [example for example in val_ds_all if example['category'] == category]
        test_ds = [example for example in test_ds_all if example['category'] == category]
        gt_answers = [x['answer'] for x in test_ds]

        # 0. zero-shot inference
        zs_prompts = [format_zs(example) for example in test_ds]
        
        outputs = model.generate(zs_prompts, sampling_params)
        outputs_zs = post_processing_response(outputs, gt_answers)
        
        acc_zs = np.mean(outputs_zs["correct"])
        print(f'{category}, Finishing zero-shot inference, accurecy: {acc_zs}')
        
        # 1. n-shot supervised ICL inference
        sup_icl_prompts = [format_icl(val_ds, example, args.n) for example in test_ds]

        outputs = model.generate(sup_icl_prompts, sampling_params)
        outputs_sup_icl = post_processing_response(outputs, gt_answers)

        acc_sup_icl = np.mean(outputs_sup_icl["correct"])
        print(f'{category}, Finishing {args.n}-shot supervised ICL inference, accurecy: {acc_sup_icl}')

        # 2. Run unsupervised_ICL with zero-shot as init
        def update_support(outputs):
            support_ds = []
            for ex, response, answer in zip(copy.deepcopy(test_ds), outputs['raw_response'], outputs['answer']):
                ex['raw_response'] = response
                # filter results that is not formated to improve the quality of demonstration
                if answer == '' or 'answer is' not in response:
                    continue
                else:
                    support_ds.append(ex)
                
            return support_ds
        
        outputs_curr = outputs_zs
        per_round_outputs = []
        acc_unsup_icl = []
        for round in range(args.num_rounds):
            support_ds = update_support(outputs_curr)
            unsup_icl_prompts = sum([[format_icl(support_ds, example) for _ in range(args.num_samples)] for example in test_ds], [])

            outputs = model.generate(unsup_icl_prompts, sampling_params)
            outputs_unsup_icl = post_processing_response(outputs, gt_answers)
            
            per_round_outputs.append(outputs_unsup_icl)
            outputs_curr = outputs_unsup_icl

            acc_unsup_icl.append(np.mean(outputs_unsup_icl["correct"]))
            print(f'{category}, Finishing round{round+1} unsupervised ICL inference, accurecy: {acc_unsup_icl[-1]}')

        records = {
            'args': vars(args),
            'acc_zs': acc_zs,
            'acc_sup_icl': acc_sup_icl,
            'acc_unsup_icl': acc_unsup_icl,
        }
        if args.save_preds:
            records['outputs_zs'] = outputs_zs
            records['outputs_sup_icl'] = outputs_sup_icl,
            records['outputs_unsup_icl'] = per_round_outputs
        
        # save results
        model_name = args.model_name.replace('/', '_')
        save_path = f'{args.save_path}/{model_name}/n{args.n}_mmlu_pro'
        os.makedirs(save_path, exist_ok=True)

        with open(f'{save_path}/{category}.json', 'w') as f:
            json.dump(records, f, indent=2)