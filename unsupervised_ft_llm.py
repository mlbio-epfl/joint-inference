from collections import defaultdict
import os
import random
from itertools import islice
from pathlib import Path
from tqdm import tqdm
import time

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np

from misc.constants import TEXT_INPUT
from utils.text_utils import get_dataset, get_label_set, get_template
import utils.common_utils as utils

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

def replace_none_with_default(data):
    for key in data.keys():
        if data[key] is None:
            data[key] = ""
    return data

@hydra.main(version_base=None, config_path="./configs", config_name="llama_sst2")
def main(cfg: DictConfig) -> None:
    WORKING_DIR = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    if cfg.job_id == "":
        cfg.job_id = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}"
        WORKING_DIR = WORKING_DIR / cfg.job_id
        os.makedirs(WORKING_DIR, exist_ok=True)

    cfg.work_dir = str(WORKING_DIR)
    cfg.ckpt_path = os.path.join(cfg.work_dir, 'ckpt.pth')

    print(OmegaConf.to_yaml(cfg))
    if cfg.wandb is not None:
        wandb.init(
            **cfg.wandb,
            config=OmegaConf.to_object(cfg),
        )

    # dataset preparation
    training_set, test_set = get_dataset(cfg.text_dataset)
    label_set = get_label_set(cfg.text_dataset)
    template = get_template(cfg.text_dataset)

    n_tr, n_te = len(training_set), len(test_set)
    labels_tr, labels_te = np.array([example['label'] for example in training_set]), np.array([example['label'] for example in test_set])

    training_set = [replace_none_with_default(d) for d in training_set]
    test_set = [replace_none_with_default(d) for d in test_set]

    loader = DataLoader(
        training_set,
        batch_size=cfg.batch_size * cfg.N,
        shuffle=True,
        drop_last=True,
        num_workers=10,
        pin_memory=True,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size_eval,
        shuffle=False,
        drop_last=False,
        num_workers=10,
        pin_memory=True,
        persistent_workers=True,
    )

    # instantiate reward
    reward = instantiate(cfg.reward)
    reward.set_template_and_labelset(
       template,
       label_set
    )

    # instantiate task encoder
    task_encoder = instantiate(cfg.task_encoder)
    task_encoder.set_template_and_label_set(
       template,
       label_set
    )
    task_encoder.set_train()
    print(f"Loading Task Encoder model from {cfg.task_encoder.model_name} and Reward model from {cfg.reward.model_name}")

    # instantiate optimizer
    parameters = [(n, p) for n, p in task_encoder.encoder.named_parameters() if p.requires_grad]
    print(f"Number of trainable parameters: {sum(p.numel() for _, p in parameters)/1e6:.3f}M")
    print(f"Trainable parameters:\n- " + "\n- ".join([n for n, p in parameters]))
    parameters = [p for n, p in parameters]
    optimizer = instantiate(cfg.optimization.optimizer, params=parameters)

    if cfg.optimization.lr_schedule_name == "constant":
        lr_schedule_values = defaultdict(lambda: cfg.optimization.lr_schedule.lr)
    elif cfg.optimization.lr_schedule_name == "cosine":
        lr_schedule_values = utils.cosine_scheduler(**cfg.optimization.lr_schedule)
    else:
        raise ValueError(f"Unknown lr schedule {cfg.optimization.lr_schedule_name}")

    # instantiate gradient estimator
    estimator = instantiate(cfg.optimization.gradient_estimator, reward=reward)

    # train
    reward_history_sample = []
    reward_history_argmax = []
    batch_acc_history_sample = []
    batch_acc_history_argmax = []
    history_surr_loss = []
    history_grad_norm = []
    history_entr = []
    train_acc_history = []
    test_acc_history = []
    pred_dist_history = []

    seed_everything(cfg.seed)

    # load task encoder and optimizer state dicts if exist
    start_iter = 0
    if os.path.exists(cfg.ckpt_path):
        checkpoint = torch.load(cfg.ckpt_path)
        msg = task_encoder.load_state_dict(checkpoint['task_encoder_state_dict'], strict=False)
        print(f"Loaded task encoder state dict with msg: {msg}")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint['iter']
        print(f"Resuming from iteration {start_iter}")
    else:
        print(f"No checkpoint found at {cfg.ckpt_path}, starting from scratch")

    t0 = time.time()
    test_acc = 0.
    pred_dist = 0.
    iters_bar = tqdm(islice(cycle(loader), start_iter, cfg.num_iterations), total=cfg.num_iterations - start_iter)

    for i, examples in enumerate(iters_bar):
        idx_iter = start_iter + i
        # update learning rate
        for _, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule_values[idx_iter]

        logs_dict = {
            'opt/lr': optimizer.param_groups[0]['lr']
        }
        optimizer.zero_grad()

        # task encoder forward pass (bsize, *) - > (bsize, |S|)
        task_encoder_logits = task_encoder(examples)
        task_encoder_logits = task_encoder_logits.reshape(cfg.batch_size, cfg.N, -1).to(cfg.device)

        examples_input = [{key: int(value) if key in ['idx', 'label'] else value for key, value in zip(examples.keys(), data)} for data in zip(*examples.values())]
        examples_input = [examples_input[i * cfg.N:(i + 1) * cfg.N] for i in range(cfg.batch_size)]

        input_dict = {TEXT_INPUT: examples_input}
        
        surrogate_loss, stats = estimator(input_dict, task_encoder_logits)

        # entropy reg
        entr_reg = utils.entropy_regularizer(task_encoder_logits)

        # backward and opt step
        loss = (surrogate_loss - cfg.gamma * entr_reg) 
        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(task_encoder.parameters(), max_norm=cfg.N)
        optimizer.step()
        optimizer.zero_grad()
                
        # statistics monitoring
        history_grad_norm.append(float(grad_norm.item()))
        gt_labels = examples['label'].reshape(cfg.batch_size, cfg.N)
        reward_history_sample.append(stats["R_sample"])
        reward_history_argmax.append(stats["R_argmax"])
        batch_acc_sample = (gt_labels == stats["y_sample"].cpu()).float().mean().item()
        batch_acc_argmax = (gt_labels == stats["y_argmax"].cpu()).float().mean().item()
        batch_acc_history_sample.append(batch_acc_sample)
        batch_acc_history_argmax.append(batch_acc_argmax)
        history_surr_loss.append(surrogate_loss.item())
        history_entr.append(entr_reg.item())

        logs_dict.update({
            "reward/R_sample": reward_history_sample[-1],
            "reward/R_argmax": reward_history_argmax[-1],
            "acc/batch acc sample": batch_acc_sample,
            "acc/batch acc argmax": batch_acc_argmax,
            "opt/surr loss": history_surr_loss[-1],
            "opt/grad norm": history_grad_norm[-1],
            "opt/entr reg": history_entr[-1]
        })
        
        if cfg.wandb is not None:
            wandb.log(logs_dict, step=idx_iter)

        if (idx_iter + 1) % cfg.eval_freq == 0 or (idx_iter + 1) in [1, cfg.num_iterations]:
            task_encoder.set_eval()

            preds = []
            for examples in test_loader: 
                with torch.no_grad():
                    preds.extend(task_encoder(examples).argmax(dim=1).detach().cpu().numpy())
            preds = np.array(preds)
            test_acc = np.mean(preds == labels_te)
            pred_dist = np.round(torch.bincount(torch.tensor(preds), minlength=len(label_set)).numpy() / n_te, 3).tolist()
    
            test_acc_history.append(test_acc)
            pred_dist_history.append(pred_dist)

            task_encoder.set_train()
            if cfg.wandb is not None:
                wandb.log({'acc/test acc': test_acc}, step=idx_iter)

        t1 = time.time()
        iters_bar.set_description(f"Iter {idx_iter:05d}, Time {(t1-t0)/60:.2f} mins, Test Acc: {test_acc:.4f}, R_argmax: {reward_history_argmax[-1]:.4f}, Surr. Loss {history_surr_loss[-1]:.4f}, Grad Norm {history_grad_norm[-1]:.4f}, Pred dist: {pred_dist}, Entr reg: {history_entr[-1]:.4f}")

        if idx_iter % cfg.ckpt_freq == 0:
            torch.save(
                {
                    'task_encoder_state_dict': task_encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iter': idx_iter,
                },
                cfg.ckpt_path,
            )

        torch.cuda.empty_cache()
        
    np.savez(
        os.path.join(cfg.work_dir, 'record.npz'),
        **{
            "reward_history_argmax" : reward_history_argmax,
            "reward_history_sample" : reward_history_sample,
            "batch_acc_history_argmax" : batch_acc_history_argmax,
            "batch_acc_history_sample" : batch_acc_history_sample,
            "history_surr_loss": history_surr_loss,
            "history_entr": history_entr,
            "history_grad_norm": history_grad_norm,
            "train_acc_history": train_acc_history,
            "test_acc_history": test_acc_history,
            "pred_dist_history": pred_dist_history
        }
    )

    torch.save(
        {
            'task_encoder_state_dict': task_encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },
        os.path.join(cfg.work_dir, 'ckpt.pth')
    )

if __name__ == '__main__':
    main()
