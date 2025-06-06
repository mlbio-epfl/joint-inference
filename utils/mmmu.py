# This code is adapted from https://github.com/Zheng0428/MMMU-Pro/blob/main/mmmu-pro/infer/infer_llava_onevision.py
import ast

import numpy as np
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets


class MMMUPro(Dataset):
    possible_subjects = [
        'History', 'Art', 'Design', 'Literature', 'Agriculture',
        'Finance', 'Sociology', 'Accounting', 'Energy_and_Power',
        'Pharmacy', 'Architecture_and_Engineering', 'Clinical_Medicine',
        'Public_Health', 'Physics', 'Art_Theory', 'Electronics',
        'Psychology', 'Biology', 'Manage', 'Economics', 'Mechanical_Engineering',
        'Diagnostics_and_Laboratory_Medicine', 'Basic_Medical_Science', 'Computer_Science',
        'Math', 'Music', 'Materials', 'Marketing', 'Chemistry', 'Geography' 
    ]
    possible_difficulties = ["Easy", "Medium", "Hard"]
    def __init__(self, subject, option="standard (4 options)", query_difficulty=["Hard"], support_difficulty=["Easy", "Medium"], single_image=True):
        # Check args
        assert not single_image
        assert option == "standard (10 options)"
        assert option in ["standard (4 options)", "standard (10 options)"]
        assert isinstance(query_difficulty, list)
        for level in query_difficulty:
            assert level in self.possible_difficulties
        assert isinstance(support_difficulty, list)
        for level in support_difficulty:
            assert level in self.possible_difficulties
        assert subject in self.possible_subjects
        
        base_dataset = load_dataset("MMMU/MMMU_Pro", option, split="test")
        base_dataset = base_dataset.filter(lambda x: x["subject"] == subject)
        if single_image:
            base_dataset = base_dataset.filter(lambda x: x["image_2"] is None)
        self.support_dataset = concatenate_datasets(
            [base_dataset.filter(lambda x: x["topic_difficulty"] == difficulty) for difficulty in support_difficulty]
        )
        self.query_dataset = concatenate_datasets(
            [base_dataset.filter(lambda x: x["topic_difficulty"] == difficulty) for difficulty in query_difficulty]
        )
    
    def __len__(self):
        return len(self.query_dataset)
    
    def __getitem__(self, idx):
        example =  self.query_dataset[int(idx)]
        example['question_type'] = 'multiple-choice'
        return example
    
    def get_support(self, N=4):
        indices = np.random.choice(len(self.support_dataset), size=(N, ), replace=False).tolist()
        return [self.support_dataset[idx] for idx in indices]


class MMMU(Dataset):
    possible_subjects = [
        'Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art',
        'Art_Theory', 'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine',
        'Computer_Science', 'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics',
        'Electronics', 'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature',
        'Manage', 'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 'Music', 'Pharmacy',
        'Physics', 'Psychology', 'Public_Health', 'Sociology'
    ]

    def __init__(self, subject):
        # Check args
        assert subject in self.possible_subjects

        self.query_dataset = load_dataset("MMMU/MMMU", subject, split="validation")

    def __len__(self):
        return len(self.query_dataset)

    def __getitem__(self, idx):
        return self.query_dataset[int(idx)]
    
    def get_support(self, N=4):
        raise NotImplementedError("This dataset does not support support set sampling.")
