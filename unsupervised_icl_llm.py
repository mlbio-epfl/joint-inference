import os
import argparse
from collections import defaultdict

import json
from tqdm import tqdm
import numpy as np
import torch
from scipy.stats import mode
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.text_utils import get_dataset, get_template, get_label_set
from utils.common_utils import get_location, get_location_prefix
from misc.constants import LOC_FINDER_TOKEN, CACHE_DIR

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='sst2')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--n', type=int, default=16, help='number of examples in a sequence')
    parser.add_argument('--num_repeats', type=int, default=5, help='number of trials for a single query for majority vote')
    parser.add_argument('--T', type=int, default=8, help='number of rounds for updating the labels of query set')
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--save_preds', action='store_true', help='set True to save the prediction of the model')
    parser.add_argument('--save_path', type=str, default="./exp_local/unsup_icl_llm")

    return parser.parse_args()


def balanced_sampling(y_train):
    n_sample = np.floor(np.ones(num_classes) * args.n / num_classes)
    remaining = int(args.n - np.sum(n_sample))
    n_sample[:remaining] += 1
    np.random.shuffle(n_sample)

    idxs_per_classes = [np.arange(n_tr)[y_train == i] for i in range(num_classes)]
    idxs = np.concatenate([np.random.choice(idx, int(n_s)) for n_s, idx in zip(n_sample, idxs_per_classes) if len(idx) > 0])

    left = np.random.choice(np.arange(n_tr), args.n - len(idxs))
    idxs = np.concatenate([idxs, left])
    np.random.shuffle(idxs)
    
    return idxs

def icl_inference(train_ds, test_ds, y_train, y_test):
    n_tr, n_te = len(train_ds), len(test_ds)
    preds = []
    for i in tqdm(range(n_te)):
        x_query = test_ds[i]
        query_str = template(x_query, label_set[x_query['label']])
            
        pred_i = []
        for repreat in range(args.num_repeats): 
            # support_idxs = np.random.permutation(n_tr)[:args.n]
            support_idxs = balanced_sampling(y_train)
                
            x_support = [train_ds[idx] for idx in support_idxs]
            y_support = y_train[support_idxs]

            support_str = tokenizer.eos_token.join([template(x, label_set[y]) for x, y in zip(x_support, y_support)]) + tokenizer.eos_token
            _, location = get_location_prefix(template, tokenizer, x_query, label_set[x_query['label']], prefix=support_str)
                
            sent_toks = tokenizer(support_str + query_str, return_tensors='pt')
            sent_toks['input_ids'] = torch.where(sent_toks['input_ids'] != tokenizer.eos_token_id, 
                                                sent_toks['input_ids'], 
                                                sep_token_id)
            # print(tokenizer.decode(toks['input_ids'][0, :location+1]))
            output = model(**sent_toks.to(device))
            logits_cls = output.logits[0, location, label_token_indices]
            pred_i.append(logits_cls.argmax().item())

        pred_i = np.array(pred_i)
        preds.append(pred_i)

    preds = np.stack(preds) # (N, num_repeats)
    majority_preds, _ = mode(preds, axis=1)
    acc = np.mean(majority_preds==y_test)

    return majority_preds, acc

def zs_inference(ds):
    preds = []
    for example in tqdm(ds):
        sentence =  template(example, label_set[example['label']])
        _, loc = get_location_prefix(template, tokenizer, example, label_set[example['label']])
        tok = tokenizer(sentence, return_tensors='pt')
        output = model(**tok.to(device), output_hidden_states=True)

        logits = output.logits # (bs, PADDED_SEQLEN, |V|)
        logits_cls = logits[torch.arange(logits.shape[0]), loc][:, label_token_indices]
        preds.append(int(logits_cls.argmax(dim=1)[0]))

    return np.array(preds)

if __name__ == '__main__':
    args = _parse_args()
    np.random.seed(args.seed)
    device = 'cuda'

    model = AutoModelForCausalLM.from_pretrained(args.model_name, 
                                                 torch_dtype=torch.float16, 
                                                 attn_implementation="flash_attention_2", 
                                                 cache_dir=CACHE_DIR, 
                                                 device_map='auto')

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens(
                {'additional_special_tokens': [LOC_FINDER_TOKEN]},
                replace_additional_special_tokens=False
            )
    sep_token_id = tokenizer('\n', add_special_tokens=False)['input_ids'][0]

    # load dataset
    train_ds, test_ds = get_dataset(args.dataset)
    template = get_template(args.dataset)
    label_set = get_label_set(args.dataset)

    n_tr, n_te = len(train_ds), len(test_ds)
    labels_tr = np.array([x['label'] for x in train_ds])
    labels_te = np.array([x['label'] for x in test_ds])

    num_classes = len(label_set)

    # Prepare label token indices for inference
    label_token_indices = []
    first_tokenized, first_loc = get_location(template, tokenizer, defaultdict(str), label_set[0])
    for classname in label_set:
        tokenized_i, loc_i = get_location(template, tokenizer, defaultdict(str), classname)
        assert loc_i == first_loc, \
            "Check your label_set or template, because given fixed input_dict, locations MUST be the same for all the classnames"
        label_token_indices.append(tokenized_i[loc_i + 1])

    print(f'Prefix is "{tokenizer.decode(first_tokenized[:first_loc + 1])}"')
    print("Classname & Classnames first tokens:")
    for i, token_id in enumerate(label_token_indices):
        print(f'"{label_set[i]}" -> "{tokenizer.decode(token_id)}"')

    # round 0, zero-shot inference
    preds_test = zs_inference(test_ds)
    labels_support = zs_inference(train_ds)

    support_acc = np.mean(labels_support == labels_tr)
    test_acc = np.mean(preds_test == labels_te)
    print(f'Round 0, zero-shot accuracy on support set: {support_acc:.3f}')
    print(f'Round 0, zero-shot accuracy on test set: {test_acc:.3f}')

    # round 0, supervised ICL
    preds_test_sup_icl, acc_sup_icl = icl_inference(train_ds, test_ds, labels_tr, labels_te) 
    print(f'Round 0, {args.n}-shot Supervised In-context Inference, test set accuracy {acc_sup_icl:.3f}')

    records = {
        'args': vars(args),
        'acc_zs': test_acc,
        'acc_sup_icl': acc_sup_icl,
        'acc_unsup_icl_support': [],
        'acc_unsup_icl_test': [],
    }

    if args.save_preds:
        records['preds_support'] = [labels_support]
        records['preds_test'] = [preds_test]
        records['labels_tr'] = labels_tr
        records['labels_te'] = labels_te

    # t-th round of unsupervised ICL
    for t in range(args.T):
        # eval test set accuracy 
        preds_test, acc_test = icl_inference(train_ds, test_ds, labels_support, labels_te)
        print(f'Round {t+1}, {args.n}-shot In-context Inference, test set accuracy {acc_test:.3f}')

        # update labels of support set
        preds_support, acc_support = icl_inference(train_ds, train_ds, labels_support, labels_tr)
        print(f'Round {t+1}, update support labels, support set accuracy {acc_support:.3f}')

        labels_support = preds_support

        records['acc_unsup_icl_test'].append(acc_test)
        records['acc_unsup_icl_support'].append(acc_support)
        if args.save_preds:
            records['preds_support'].append(preds_support)
            records['preds_test'].append(preds_test)


    model_name = args.model_name.replace('/', '_')
    save_path = f'{args.save_path}/{model_name}'
    os.makedirs(save_path, exist_ok=True)
    
    with open(f'{save_path}/n{args.n}_{args.dataset}.json', 'w') as f:
        json.dump(records, f, indent=2)
