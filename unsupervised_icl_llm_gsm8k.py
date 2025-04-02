import os
import argparse
from collections import Counter
import json
import copy
import numpy as np

import torch
import regex
from vllm import LLM, SamplingParams

from utils.text_utils import get_gsm8k
from misc.constants import CACHE_DIR


# raw response: [reasoning] + "The final answer is [answer]"
# answer: the number of the answer
def process_answers_of_ds(ds):
    for example in ds:
        # get answer
        match = regex.search(r"#### *(\-?[0-9\%\$]+(?:[,.]\d+)*)", example['answer'])
        raw_answer = match.group()
        for remove_char in ['$', '%', 'g', ',']:
            raw_answer = raw_answer.replace(remove_char, '')
        answer_num = eval(raw_answer.split('####')[-1].strip('').strip('.'))
        
        # get reasoning
        reasoning = example['answer'].split('\n####')[0]
        answer = f"The final answer is {answer_num}."
        
        raw_response = reasoning + "\n" + answer
        example['raw_response'] = raw_response
        example['answer'] = answer_num

    return ds


def extract_answer(response):
    match = regex.search(r"The(?: final)? answer is .*?(\d[\d\\$€,(){}.]*|\d)", response)
    if match:
        answer = match.group(1).rstrip('.')
        for remove_char in ['$', '%', 'g', ',', '{', '}', '(', ')']:
            answer = answer.replace(remove_char, '')
        try:
            answer_number = eval(answer)
        except:
            answer_number = ''

    else:
        answer_number = ''

    return answer_number

def format_zs(example):
    if 'qwen' not in args.model_name.lower():
        return "Given the following question, reason and give a final answer to the question. Your response should end with \"The answer is [answer]\" where [answer] is the response to the problem.\n\n" + \
                    f"Q: {example['question']}\n" + \
                    "A: Let's think step by step."
    else:
        return "Given the following question, reason step by step, and put your final answer within \\boxed{}. Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n\n" + \
                    f"Problem: {example['question']}\n" + \
                    "Answer: Let's think step by step."


def format_icl(support_ds, query_example, n=8):
    prompt_head = "Given the following problem, reason and give a final answer to the problem. Your response should end with \"The answer is [answer]\" where [answer] is the response to the problem.\n\n"
    
    prompt_demon = ""
    support_set = np.random.choice(support_ds, n, replace=False)
    for example in support_set:
        prompt_demon +=  f"Q: {example['question']}\nA:{example['raw_response']}\n\n"

    prompt_query = f"Q: {query_example['question']}\nA:"

    return prompt_head + prompt_demon + prompt_query

def format_sup_icl(example):
    if 'qwen' not in args.model_name.lower():
        return f"""
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.

Q: {example['question']}
A:
""".strip()
    else:
        # check Qwen2.5-Math paper https://arxiv.org/pdf/2409.12122v1 page 35
        sup_icl_examples = [
        (
            "In 2004, there were 60 kids at a cookout. In 2005, half the number of kids came to the cookout as compared to 2004. In 2006, 2/3 as many kids came to the cookout as in 2005. How many kids came to the cookout in 2006?",
            "Let's think step by step. In 2005, 60/2=30 kids came to the cookout. In 2006, 30/3*2=20 kids came to the cookout. The answer is 20.",
        ),
        (
            "Zilla spent 7% of her monthly earnings on rent, half of it on her other monthly expenses, and put the rest in her savings. If she spent $133 on her rent, how much does she deposit into her savings account in a month?",
            "Let's think step by step. Since $133 is equal to 7% of her earnings, then 1% is equal to $133/7 = $19. The total monthly earning of Zilla is represented by 100%, so $19 x 100 = $1900 is her monthly earnings. So, $1900/2 = $950 is spent on her other monthly expenses. The total amount spent on the rent and other monthly expenses is $133 + $950 = $1083. Hence, she saves $1900 - $1083 = $817 per month. The answer is 817.",
        ),
        (
            "If Buzz bought a pizza with 78 slices at a restaurant and then decided to share it with the waiter in the ratio of 5:8, with Buzz's ratio being 5, what's twenty less the number of slices of pizza that the waiter ate?",
            "Let's think step by step. The total ratio representing the slices of pizza that Buzz bought is 5+8=13. If he shared the slices of pizza with the waiter, the waiter received a fraction of 8/13 of the total number of slices, which totals 8/13 * 78 = 48 slices. Twenty less the number of slices of pizza that the waiter ate is 48-20 = 28. The answer is 28.",
        ),
        (
            "Jame gets a raise to $20 per hour and works 40 hours a week. His old job was $16 an hour for 25 hours per week. How much more money does he make per year in his new job than the old job if he works 52 weeks a year?",
            "Let's think step by step. He makes 20*40=$800 per week. He used to make 16*25=$400 per week. So his raise was 800-400=$400 per week. So he makes 400*52=$20,800 per year more. The answer is 20800.",
        ),
        (
            "Mr. Gardner bakes 20 cookies, 25 cupcakes, and 35 brownies for his second-grade class of 20 students. If he wants to give each student an equal amount of sweet treats, how many sweet treats will each student receive?",
            "Let's think step by step. Mr. Gardner bakes a total of 20 + 25 + 35 = 80 sweet treats. Each student will receive 80 / 20 = 4 sweet treats. The answer is 4.",
        ),
        (
            "A used car lot has 24 cars and motorcycles (in total) for sale. A third of the vehicles are motorcycles, and a quarter of the cars have a spare tire included. How many tires are on the used car lot's vehicles in all?",
            "Let's think step by step. The used car lot has 24 / 3 = 8 motorcycles with 2 tires each. The lot has 24 - 8 = 16 cars for sale. There are 16 / 4 = 4 cars with a spare tire with 5 tires each. The lot has 16 - 4 = 12 cars with 4 tires each. Thus, the used car lot's vehicles have 8 * 2 + 4 * 5 + 12 * 4 = 16 + 20 + 48 = 84 tires in all. The answer is 84.",
        ),
        (
            "Norma takes her clothes to the laundry. She leaves 9 t-shirts and twice as many sweaters as t-shirts in the washer. When she returns she finds 3 sweaters and triple the number of t-shirts. How many items are missing?",
            "Let's think step by step. Norma left 9 t-shirts and twice as many sweaters, she took 9 * 2 = 18 sweaters. Adding the t-shirts and sweaters, Norma left 9 + 18 = 27 clothes. When she came back, she found 3 sweaters and triple the number of t-shirts, she found 3 * 3 = 9 t-shirts. Adding the t-shirts and sweaters, Norma found 3 + 9 = 12 clothes. Subtracting the clothes she left from the clothes she found, 27 - 12 = 15 clothes are missing. The answer is 15.",
        ),
        (
            "Adam has an orchard. Every day for 30 days he picks 4 apples from his orchard. After a month, Adam has collected all the remaining apples, which were 230. How many apples in total has Adam collected from his orchard?",
            "Let's think step by step. During 30 days Adam picked 4 * 30 = 120 apples. So in total with all the remaining apples, he picked 120 + 230 = 350 apples from his orchard. The answer is 350.",
        )
        ]

        demonstrations = ""
        for q, a in sup_icl_examples:
            demonstrations += f"Q: {q}\nA:{a}\n\n"
        
        demonstrations += f"Q: {example['question']}\nA:"
        return demonstrations


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-Math-7B") 
    parser.add_argument('--T', type=int, default=8, help='number of rounds of unsupervised ICL')
    parser.add_argument('--num_repeats', type=int, default=5, help='number of trials to answer each question for majority vote')
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
                                     stop=['\n\nQ:'])

    # load dataset
    _, test_ds = get_gsm8k()

    n_te = len(test_ds)
    test_ds = process_answers_of_ds(test_ds)
    test_ds = [x for x in test_ds]

    gt_answers = [x['answer'] for x in test_ds]

    def truncate(text):
        # for ICL output, LLM might keep producing (question, answer) pairs after answering the original question
        # return text.split('Q:')[0].rstrip()
        match = regex.search(r"The(?: final)? answer is .*?(\d[\d\\$€,(){}.]*|\d)", text)
        if match:
            return text[:match.span()[1]]
        else: 
             return text

    def post_processing_response(outputs, gt_answers, n=1):
        # outputs: output of vllm, with N questions and N_samples for each question
        output_texts = sum([[truncate(o.text) for o in output.outputs] for output in outputs], [])
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

    # 0. zero-shot inference
    zs_prompts = [format_zs(example) for example in test_ds]
    
    outputs = model.generate(zs_prompts, sampling_params)
    outputs_zs = post_processing_response(outputs, gt_answers)
    
    acc_zs = np.mean(outputs_zs["correct"])
    print(f'Finishing zero-shot inference, accurecy: {acc_zs}')
    
    # 1. n-shot supervised ICL inference
    sup_icl_prompts = [format_sup_icl(example) for example in test_ds]

    outputs = model.generate(sup_icl_prompts, sampling_params)
    outputs_sup_icl = post_processing_response(outputs, gt_answers)

    acc_sup_icl = np.mean(outputs_sup_icl["correct"])
    print(f'Finishing 8-shot supervised ICL inference, accurecy: {acc_sup_icl}')

    # 2. Run unsupervised_ICL with zero-shot as init
    def update_support(outputs):
        support_ds = []
        for ex, response, answer in zip(copy.deepcopy(test_ds), outputs['raw_response'], outputs['answer']):
            ex['raw_response'] = response
            # filter results that is not formated to improve the quality of demonstration
            if answer == '':
                continue
            else:
                support_ds.append(ex)
            
        return support_ds
    
    outputs_curr = outputs_zs
    per_round_outputs = []
    acc_unsup_icl = []
    for round in range(args.T):
        support_ds = update_support(outputs_curr)
        unsup_icl_prompts = sum([[format_icl(support_ds, example) for _ in range(args.num_repeats)] for example in test_ds], [])

        outputs = model.generate(unsup_icl_prompts, sampling_params)
        outputs_unsup_icl = post_processing_response(outputs, gt_answers, n=args.num_repeats)

        per_round_outputs.append(outputs_unsup_icl)
        outputs_curr = outputs_unsup_icl

        acc_unsup_icl.append(np.mean(outputs_unsup_icl["correct"]))
        print(f'Finishing round{round+1} unsupervised ICL inference, accurecy: {acc_unsup_icl[-1]}')

    records = {
        'args': vars(args),
        'acc_zs': acc_zs,
        'acc_sup_icl': acc_sup_icl,
        'acc_unsup_icl': acc_unsup_icl,
    }
    if args.save_preds:
        records['outputs_zs'] = outputs_zs,
        records['outputs_sup_icl'] = outputs_sup_icl,
        records['outputs_unsup_icl'] = per_round_outputs

    # save results
    model_name = args.model_name.replace('/', '_')
    save_path = f'{args.save_path}/{model_name}'
    os.makedirs(save_path, exist_ok=True)

    with open(f'{save_path}/n8_gsm8k.json', 'w') as f:
        json.dump(records, f, indent=2)

