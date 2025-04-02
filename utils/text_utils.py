import os

import pandas as pd
from datasets import load_dataset, Dataset

from misc.templates import *

ALL_DATASETS = ['sst2', 'amazon_polarity', 'ag_news', 'trec', 'dbpedia_14', 'subj',
                'qnli', 'mnli', 'rte',
                'copa', 'boolq', 'piqa', 'hellaswag',
                'mmlu', 'mmlu_pro', 'gsm8k']

DATASETS_TO_CLASSNAMES = {
        # text classification
        "sst2": ["negative", "positive"],
        "imdb": ["negative", "positive"],
        "amazon_polarity": ["negative", "positive"],
        "ag_news": ["world", "sports", "business", "technology"],
        "trec": ["description", "entity", "expression", "human", "location", "number"],
        "dbpedia_14": ["company", "educational institution", "artist", "athlete", "office holder", "mean of transportation", "building", "natural place", "village", "animal",  "plant",  "album",  "film",  "written work"],
        "subj": ["objective", "subjective"],
        # nli
        "rte": ["yes", "no"],
        "qnli": ["yes", "no"],
        "mnli": ["yes", "maybe", "no"],
        # qa
        "copa": ['1', '2'],
        "boolq": ['yes', 'no'],
        "piqa": ['1', '2'],
        "hellaswag": ['1', '2', '3', '4'],

        "mmlu": ["A", "B", "C", "D"]
        }

DATASET_NAME_HUGGINGFACE = {'subj': ['SetFit/subj'], 
                            'rte': ['super_glue', 'rte'], 
                            'qnli': ['glue', 'qnli'], 
                            'mnli': ['glue', 'mnli'],
                            'copa': ['super_glue', 'copa']}

def get_dataset(name, num_examples=3000, seed=42):
    '''Load dataset for NLP tasks (Table 1 in the paper)'''
    # return a list of dict [{'input': ---, 'label': ---}, ..., {}]
    if not os.path.exists(f'./data/raw_dataset/'):
        os.makedirs(f'./data/raw_dataset/')
    if not os.path.exists(f'./data/raw_dataset/{name}.csv'):
        # get HuggingFace dataset name
        ds_name_hf = DATASET_NAME_HUGGINGFACE.get(name, [name])

        # load dataset
        if name not in ['rte', 'copa']:
            dataset = load_dataset(*ds_name_hf, split='train', trust_remote_code=True).to_pandas()
        elif name in ['rte', 'copa']:
            dataset = load_dataset(*ds_name_hf, trust_remote_code=True)
            dataset = pd.concat([dataset['train'].to_pandas(), dataset['validation'].to_pandas()]).reset_index(drop=True)
        
        # information of label
        if name == 'trec':
            dataset['label'] = dataset['coarse_label']
        elif name == 'boolq':
            dataset['label'] = [0 if y else 1 for y in dataset['answer']]
        elif name == 'hellaswag':
            dataset['label'] = [int(y) for y in dataset['label']]
            dataset['endings0'] = [endings[0] for endings in dataset['endings']]
            dataset['endings1'] = [endings[1] for endings in dataset['endings']]
            dataset['endings2'] = [endings[2] for endings in dataset['endings']]
            dataset['endings3'] = [endings[3] for endings in dataset['endings']]

        num_available_examples = dataset.shape[0]
        label_counts = dataset['label'].value_counts().sort_index()
        num_per_class = min(num_examples, num_available_examples) // label_counts.shape[0]
        print(list(label_counts.index))
        ds_df = pd.concat([dataset[dataset['label'] == label].sample(min(count, num_per_class), random_state=seed) for label, count in label_counts.items()]).reset_index(drop=True)
        ds_df.to_csv(f'./data/raw_dataset/{name}.csv', index=False)
    else:
        ds_df = pd.read_csv(f'./data/raw_dataset/{name}.csv')
    
    # train-test split 3/2 vs. 1/3
    label_counts = ds_df['label'].value_counts().sort_index()
    ds_train = Dataset.from_pandas(pd.concat([ds_df[ds_df['label']==label].iloc[:int(count*2/3)] for label, count in label_counts.items()]))
    ds_test = Dataset.from_pandas(pd.concat([ds_df[ds_df['label']==label].iloc[int(count*2/3):] for label, count in label_counts.items()]))

    return [row for row in ds_train], [row for row in ds_test]


def get_gsm8k():
    ds_train = load_dataset('gsm8k', 'main', split='train')
    ds_test = load_dataset('gsm8k', 'main', split='test')
        
    return [row for row in ds_train], [row for row in ds_test]


def get_mmlu(category=None):
    dataset = load_dataset("cais/mmlu", category)
    val_ds, test_ds = dataset["validation"].to_pandas(), dataset["test"].to_pandas()
    val_ds['label'] = val_ds['answer']
    test_ds['label'] = test_ds['answer']
    if category is not None:
        val_ds = val_ds[val_ds['subject'] == category]
        test_ds = test_ds[test_ds['subject'] == category]
    
    return [example for example in Dataset.from_pandas(val_ds)], [example for example in Dataset.from_pandas(test_ds)]

def get_mmlu_pro():
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    val_ds, test_ds = dataset["validation"], dataset["test"]

    val_ds = val_ds.to_pandas()
    val_ds['raw_response'] = [ans.replace("A: Let's think step by step.", "") for ans in val_ds['cot_content']]
    val_ds = Dataset.from_pandas(val_ds)

    return val_ds, test_ds

def get_label_set(name):
    assert name in ALL_DATASETS
    return DATASETS_TO_CLASSNAMES[name]

def get_template(name):
    if name == 'sst2':
        return SST2Template()
    elif name == 'amazon_polarity':
        return AmazonPolarityTemplate()
    elif name == 'ag_news':
        return AGNewsTemplate()
    elif name == 'trec':
        return TRECTemplate()
    elif name == 'dbpedia_14':
        return DBPedia14Template()
    elif name == 'subj':
        return SUBJTemplate()
    elif name == 'rte':
        return RTETemplate()
    elif name == 'qnli':
        return QNLITemplate()
    elif name == 'mnli':
        return MNLITemplate()
    elif name == 'copa':
        return COPATemplate()
    elif name == 'piqa':
        return PIQATemplate()
    elif name == 'boolq':
        return BoolQTemplate()
    elif name == 'hellaswag':
        return HellaSwagTemplate()
    elif name == "mmlu":
        return MMLUTempalte()
    else:
        raise f"Invalid dataset name: {name}"
