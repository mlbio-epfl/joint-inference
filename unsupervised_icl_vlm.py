from collections import defaultdict
import os
import random
from itertools import islice
from pathlib import Path
import time

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler
from tqdm import tqdm
import numpy as np
from einops import rearrange
import wandb

from misc.constants import PIXELS_INPUT, PIXELS_PHI, PIXELS_PHI_SEQ, TEXT_INPUT, INPUT_ID
import utils.common_utils as utils


def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def labeler(dataset_train, y_train, dataset_eval, reward, N, num_repeats, batch_size, device):
    # labels dataset_eval 
    assert len(dataset_train) == len(y_train)
    assert N >= 0

    eval_loader = DataLoader(
        dataset_eval,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=10,
        pin_memory=True,
        persistent_workers=True,
    )

    if N == 0:
        # just zero-shot preds, dataset_train, y_train, num_repeats are not used
        gt_eval_labels = []
        y_eval_labels = [] # stack then will be of shape (len(dataset_eval))

        all_logprobs = []
        for input_dict, gt_labels in tqdm(eval_loader):
            cur_batch_size = input_dict[PIXELS_PHI].shape[0]
            input_dict = {k: (v.to(device) if k != TEXT_INPUT else v) for k, v in input_dict.items()}

            for k in input_dict:
                if k == TEXT_INPUT:
                    input_dict[k] = utils.reshape_list(input_dict[k], (cur_batch_size, 1))
                else:
                    input_dict[k] = input_dict[k].view(cur_batch_size, 1, *input_dict[k].shape[1:])

            cur_placeholder_labels = np.zeros((cur_batch_size, 1), dtype=int)
            cur_logprobs = F.log_softmax(reward(input_dict, cur_placeholder_labels).squeeze(1), dim=1) # (batch_size, |Y|)

            all_logprobs.append(cur_logprobs.cpu().detach())
            gt_eval_labels.append(gt_labels.cpu().detach())


        all_logprobs = torch.cat(all_logprobs)
        y_eval_labels = all_logprobs.argmax(1).to(device)
        gt_eval_labels = torch.cat(gt_eval_labels).to(device)
        
    else:

        unique_classes, class_counts = torch.unique(y_train, return_counts=True)
        class_weights = {}
        for c, count in zip(unique_classes, class_counts):
            class_weights[c.item()] = 1./count

        sample_weights = [class_weights[class_id.item()] for class_id in y_train]

        balanced_train_sampler = WeightedRandomSampler(sample_weights, len(y_train))
        train_loader = DataLoader(
            dataset_train,
            batch_size = N * batch_size,
            sampler=balanced_train_sampler,
            drop_last=True,
            num_workers=10,
            pin_memory=True,
            persistent_workers=True
        )


        support_set = cycle(train_loader)

        gt_eval_labels = []
        y_eval_labels = [[] for i in range(num_repeats)] # stack then will be of shape (num_repeats, len(dataset_eval))

        for input_dict_query, gt_labels_query in tqdm(eval_loader):
            # prepare query set
            cur_batch_size = input_dict_query[PIXELS_PHI].shape[0]
            input_dict_query = {k: (v.to(device) if k != TEXT_INPUT else v) for k, v in input_dict_query.items()}

            for k in input_dict_query:
                if k == TEXT_INPUT:
                    input_dict_query[k] = utils.reshape_list(input_dict_query[k], (cur_batch_size, 1))
                else:
                    input_dict_query[k] = input_dict_query[k].view(cur_batch_size, 1, *input_dict_query[k].shape[1:])

            # generate different support sets
            for i in range(num_repeats):
                # prepare support set
                input_dict_support, _ = next(support_set)
                input_dict_support = {k: v[:cur_batch_size * N] for k, v in input_dict_support.items()}
                input_dict_support = {k: (v.to(device) if k != TEXT_INPUT else v) for k, v in input_dict_support.items()}
                
                labels_support = y_train[input_dict_support[INPUT_ID][:cur_batch_size * N]]

                # import pdb
                # pdb.set_trace()
                for k in input_dict_support:
                    if k == TEXT_INPUT:
                        input_dict_support[k] = utils.reshape_list(input_dict_support[k], (cur_batch_size, N))
                    else:
                        #print(k)
                        input_dict_support[k] = input_dict_support[k].view(cur_batch_size, N, *input_dict_support[k].shape[1:])
                # merge support and query
                input_dict_merged = {}
                for k in input_dict_query:
                    if k == TEXT_INPUT:
                        input_dict_merged[k] = [input_dict_support[k][j] + input_dict_query[k][j] for j in range(cur_batch_size)]
                    else:
                        input_dict_merged[k] = torch.cat((input_dict_support[k], input_dict_query[k]), dim=1)

                cur_placeholder_labels = torch.zeros((cur_batch_size, 1), dtype=int).to(device)
                labels_support = labels_support.view((cur_batch_size, N))
                labels_merged = torch.cat((labels_support, cur_placeholder_labels), dim=1)
                # make predictions
                cur_rewards = reward(input_dict_merged, labels_merged) # (batch_size, N+1, |Y|)
                preds = cur_rewards[:, -1, :].argmax(1)
                y_eval_labels[i].append(preds)

        y_eval_labels = torch.stack([torch.cat(elem) for elem in y_eval_labels])

    return y_eval_labels, gt_eval_labels




@hydra.main(version_base=None, config_path="./configs", config_name="openflamingo_icl_base")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # data preparation
    transforms = instantiate(cfg.dataset_transforms.transforms_eval)

    train_dataset = instantiate(cfg.dataset.dataset, transform=transforms)
    train_dataset = instantiate(
        cfg.dataset.wrapper_train,
        original_dataset=train_dataset,
    )

    eval_dataset = instantiate(cfg.dataset.dataset_eval, transform=transforms)
    eval_dataset = instantiate(
        cfg.dataset.wrapper_eval,
        original_dataset=eval_dataset,
    )

    # instantiate reward
    reward = instantiate(cfg.reward.reward)
    reward.set_template_and_labelset(
       instantiate(cfg.template),
       OmegaConf.to_object(cfg.labelset)
    )
    seed_everything(cfg.seed)

    y_train_labels = torch.zeros((cfg.T, len(train_dataset)), dtype=int).to(cfg.device)
    y_eval_labels = torch.zeros((cfg.T, cfg.num_repeats, len(eval_dataset)), dtype=int).to(cfg.device)

    # labeler(dataset_train, y_train, dataset_eval, model, N, num_repeats, batch_size, device):
    y_zs_train, gt_labels_train = labeler(train_dataset, y_train_labels[0], train_dataset, reward, N=0, num_repeats=0, batch_size=cfg.batch_size, device=cfg.device) # zero-shot
    y_train_labels[0] = y_zs_train
    # gt_labels_train of shape (len(dataset_train))
    y_zs_eval, gt_labels_eval = labeler(eval_dataset, y_eval_labels[0, 0], eval_dataset, reward, N=0, num_repeats=0, batch_size=cfg.batch_size, device=cfg.device)
    # gt_labels_eval of shape(len(dataset_eval))

    print(f"Zero-shot accuracy Dtrain: {(y_zs_train == gt_labels_train).float().mean().item() * 100:.2f}")
    print(f"Zero-shot accuracy Dval (Turn t=0): {(y_zs_eval == gt_labels_eval).float().mean().item() * 100:.2f}")
    for t in range(cfg.T):
        y_eval_labels[t] = labeler(train_dataset, y_train_labels[t], eval_dataset, reward, N=cfg.N, num_repeats=cfg.num_repeats, batch_size=cfg.batch_size, device=cfg.device)[0]

        acc_t_eval = (y_eval_labels[t] == gt_labels_eval.unsqueeze(0)).float().mean(1).cpu().numpy() * 100
        print(f"Turn t={t+1}, acc on Dval: {np.mean(acc_t_eval):.2f} +- {np.std(acc_t_eval):.2f}")
        if t < cfg.T - 1:
            y_train_labels[t + 1] = labeler(train_dataset, y_train_labels[t], train_dataset, reward, N=cfg.N, num_repeats=1, batch_size=cfg.batch_size, device=cfg.device)[0][0]
            print(f"Turn t={t+2}, acc on Dtrain: {(y_train_labels[t + 1] == gt_labels_train).float().mean().item() * 100:.2f}")


if __name__ == "__main__":
    main()
