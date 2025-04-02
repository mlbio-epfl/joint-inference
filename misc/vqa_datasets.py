import os
import time
import json
from PIL import Image

import pandas as pd
from torch.utils.data import Dataset


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class COCOQADataset(Dataset):
    """
        Based on https://github.com/phiyodr/vqaloader/blob/master/vqaloader/loaders.py
    """
    IMAGE_PATH = {
        "train": "train2014",
        "test": "test2015"}

    def __init__(self, root, split, type, transform=None):
        """
        split train, test
        """
        start_time = time.time()
        self.root = root
        self.type = type
        self.split = split
        assert type in ["color", "number"]
        # TODO: TEST DOES NOT WORK NOW
        assert split in ["train"]
        self.transform = transform
        self.base_path = os.path.expanduser(os.path.join(root, "COCOQA", self.split))

        print(f"Start loading COCOQA Dataset from {self.base_path}/*.txt", flush=True)

        # Load data and create df

        with open(os.path.expanduser(os.path.join(self.base_path, "img_ids.txt"))) as f:
            img_ids = f.readlines()
            img_ids = [line.rstrip() for line in img_ids]
        with open(os.path.expanduser(os.path.join(self.base_path, "questions.txt"))) as f:
            questions = f.readlines()
            questions = [line.rstrip() for line in questions]
        with open(os.path.expanduser(os.path.join(self.base_path, "answers.txt"))) as f:
            answers = f.readlines()
            answers = [line.rstrip() for line in answers]
        with open(os.path.expanduser(os.path.join(self.base_path, "types.txt"))) as f:
            types = f.readlines()
            types = [line.rstrip() for line in types]
            f.close()
            for index, line in enumerate(types):
                if line == "0":
                    types[index] = "object"
                elif line == "1":
                    types[index] = "number"
                elif line == "2":
                    types[index] = "color"
                else:
                    types[index] = "location"

        df = pd.DataFrame(list(zip(img_ids, questions, answers, types)),
                          columns=["image_id", "question", "answer", "type"])

        df["image_path"] = df["image_id"].apply(
            lambda x: f"{self.IMAGE_PATH[split]}/COCO_{self.IMAGE_PATH[split]}_{int(x):012d}.jpg")

        df = df[df["type"] == self.type].reset_index(drop=True)
        if self.type == "number":
            self.classname_to_idx = {
                "one": 0,
                "two": 1,
                "three": 2,
                "four": 3,
                "five": 4,
                "six": 5,
                "seven": 6,
                "eight": 7,
                "nine": 8,
                "ten": 9
            }
        else:
            self.classname_to_idx = {
                "brown": 0,
                "purple": 1,
                "black": 2,
                "red": 3,
                "orange": 4,
                "yellow": 5,
                "white": 6,
                "gray": 7,
                "blue": 8,
                "green": 9
            }

        self.df = df
        self.n_samples = self.df.shape[0]
        print(f"Loading COCOQA Dataset done in {time.time() - start_time:.1f} seconds. Loaded {self.n_samples} samples.")

    def __getitem__(self, index):
        # image input
        image_id = self.df.iloc[index]["image_id"]
        image_path = os.path.join(self.root, "COCOQA", self.df.iloc[index]["image_path"])
        # question input
        question = self.df.iloc[index]["question"]
        # answer and question type
        answer = self.df.iloc[index]["answer"]
        question_type = self.df.iloc[index]["type"]
        # split
        split = self.split

        # Load and transform image
        img = pil_loader(image_path)
        if self.transform:
            img = self.transform(img)

        return img, question, self.classname_to_idx[answer]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.df.shape[0]