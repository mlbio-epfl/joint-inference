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
from torch.utils.data import DataLoader
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


@hydra.main(version_base=None, config_path="./configs", config_name="openflamingo_cifar10")
def main(cfg : DictConfig) -> None:
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

    # data preparation
    transforms = instantiate(cfg.dataset_transforms.transforms_train)
    dataset = instantiate(cfg.dataset.dataset, transform=transforms)
    dataset = instantiate(
        cfg.dataset.wrapper_train,
        original_dataset=dataset,
    )
    latest_labelling = torch.zeros(len(dataset), device=cfg.device, dtype=torch.long) - 1
    latest_logp = torch.zeros(len(dataset), cfg.dataset.K, device=cfg.device)
    latest_logp = F.log_softmax(latest_logp, dim=1)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size * cfg.N,
        shuffle=True,
        drop_last=True,
        num_workers=10,
        pin_memory=True,
        persistent_workers=True,
    )
    latest_test_labels = None
    if 'dataset_eval' in cfg.dataset:
        eval_transforms = instantiate(cfg.dataset_transforms.transforms_eval)
        eval_dataset = instantiate(cfg.dataset.dataset_eval, transform=eval_transforms)
        eval_dataset = instantiate(
            cfg.dataset.wrapper_eval,
            original_dataset=eval_dataset,
        )
        latest_test_labels = torch.zeros(len(eval_dataset), device=cfg.device, dtype=torch.long) - 1
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=cfg.batch_size_eval,
            shuffle=False,
            drop_last=False,
            num_workers=10,
            pin_memory=True
        )

    # instantiate reward
    reward = instantiate(cfg.reward.reward)
    reward.set_template_and_labelset(
       instantiate(cfg.template),
       OmegaConf.to_object(cfg.labelset)
    )

    if cfg.calibrated:
        with torch.no_grad():
            reward.set_logprior(dataset, batch_size=cfg.batch_size_eval)

    # instantiate task encoder
    seed_everything(cfg.seed)
    task_encoder = instantiate(cfg.task_encoder.task_encoder)
    task_encoder = task_encoder.to(cfg.device)

    # instantiate optimizer
    parameters = [(n, p) for n, p in task_encoder.named_parameters() if p.requires_grad]
    print(f"Number of trainable parameters: {sum(p.numel() for _, p in parameters)/1e6:.3f}M")
    print(f"Trainable parameters:", *[n for n, p in parameters], sep="\n- ")
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

    iters_bar = tqdm(islice(cycle(loader), start_iter, cfg.num_iterations), total=cfg.num_iterations - start_iter)
    i = start_iter
    
    if cfg.eval:
        eval_log_dict = eval(cfg, task_encoder, eval_loader, estimator, latest_test_labels)
        eval_log_dict['iter'] = i
        if cfg.wandb is not None:
            wandb.log(eval_log_dict, step=i)

    start_time = time.time()
    threshold_window = [1] * cfg.threshold_window

    for input_dict, gt_labels in iters_bar:
        for _, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule_values[i]

        i += 1
        logs_dict = {
            'iter': i,
            'lr': optimizer.param_groups[0]['lr']
        }
        optimizer.zero_grad()
        
        input_dict = {k: (v.to(cfg.device) if k != TEXT_INPUT else v) for k, v in input_dict.items()}
        data_loading_time = time.time() - start_time

        start_time = time.time()
        # task encoder forward pass (bsize, *) - > (bsize, |S|)
        task_encoder_logits = task_encoder(input_dict)

        if getattr(cfg.task_encoder, 'calibrated', False):
            assert cfg.calibrated
            task_encoder_logits = F.log_softmax(task_encoder_logits, dim=1) - reward.logprior[None].expand_as(task_encoder_logits)
        
        # compare with latest labelling and update
        img_indices = input_dict[INPUT_ID]
        logp = F.log_softmax(task_encoder_logits.detach(), dim=1)
        pred = logp.argmax(dim=1)
        kl = F.kl_div(logp, latest_logp[img_indices], log_target=True)
        kl_rev = F.kl_div(latest_logp[img_indices], logp, log_target=True)
        ratio_flipped = (pred != latest_logp[img_indices].argmax(dim=1)).float().mean()
        latest_logp[img_indices] = logp
        logs_dict.update({
            "batch_ratio_flipped": ratio_flipped.item(),
            "batch_kl": kl.mean().item(),
            "batch_kl_rev": kl_rev.mean().item(),
        })
        threshold_window.pop(0)
        threshold_window.append(ratio_flipped.item())
        if all([x < cfg.ratio_flipped_threshold for x in threshold_window]):
            print(f"Flipping threshold reached, breaking!")
            break

        task_encoder_logits = task_encoder_logits.view(cfg.batch_size, cfg.N, -1) # -> (bsize, N, |S|)
        task_encoder_forward_time = time.time() - start_time

        # TODO: not sure where we want to handle this here
        for k in input_dict:
            if k == TEXT_INPUT:
                input_dict[k] = utils.reshape_list(input_dict[k], (cfg.batch_size, cfg.N))
            else:
                input_dict[k] = input_dict[k].view(cfg.batch_size, cfg.N, *input_dict[k].shape[1:])

        start_time = time.time()
        surrogate_loss, stats = estimator(input_dict, task_encoder_logits)
        estimator_forward_time = time.time() - start_time

        # entropy reg
        entr_reg = utils.entropy_regularizer(task_encoder_logits)
        sample_logprobs = F.log_softmax(task_encoder_logits, dim=2)
        sample_entropy = - (torch.exp(sample_logprobs) * sample_logprobs).sum(2)
        logs_dict.update({
            "sample_entropy_mean": sample_entropy.mean().item(),
            "sample_entropy_q25": np.percentile(sample_entropy.detach().cpu().numpy(), 25),
            "sample_entropy_q75": np.percentile(sample_entropy.detach().cpu().numpy(), 75),
        })

        start_time = time.time()
        # backward and opt step
        (surrogate_loss - cfg.gamma * entr_reg).backward()
        grad_norm = nn.utils.clip_grad_norm_(task_encoder.parameters(), max_norm=cfg.max_norm)
        optimizer.step()
        backward_time = time.time() - start_time

        # statistics monitoring
        reward_history_sample.append(stats["R_sample"])
        reward_history_argmax.append(stats["R_argmax"])
        batch_acc_sample = (gt_labels == stats["y_sample"].view(-1).cpu()).float().mean().item()
        batch_acc_argmax = (gt_labels == stats["y_argmax"].view(-1).cpu()).float().mean().item()
        batch_acc_history_sample.append(batch_acc_sample)
        batch_acc_history_argmax.append(batch_acc_argmax)
        history_surr_loss.append(surrogate_loss.item())
        history_grad_norm.append(grad_norm.item())
        history_entr.append(entr_reg.item())
        logs_dict.update({
            "R_sample": reward_history_sample[-1],
            "R_argmax": reward_history_argmax[-1],
            "batch acc sample": batch_acc_sample,
            "batch acc argmax": batch_acc_argmax,
            "surr loss": history_surr_loss[-1],
            "grad norm": history_grad_norm[-1],
            "entr reg": history_entr[-1],
            "data_loading_time": data_loading_time,
            "task_encoder_forward_time": task_encoder_forward_time,
            "estimator_forward_time": estimator_forward_time,
            "backward_time": backward_time
        })
        iters_bar.set_description(
            f"R_samp: {reward_history_sample[-1]:.2f}, R_am: {reward_history_argmax[-1]:.2f}, sur: {history_surr_loss[-1]:.2f}, lr: {logs_dict['lr']:.4f}, g_norm: {history_grad_norm[-1]:.2f}, acc_samp: {batch_acc_sample:.3f}, acc_amax: {batch_acc_argmax:.3f}, entr_reg: {history_entr[-1]:.2f}, data_t: {data_loading_time:.2f}, task_t: {task_encoder_forward_time:.2f}, est_t: {estimator_forward_time:.2f}, bckwd_t: {backward_time:.2f}"
        )

        if cfg.eval and i % cfg.eval_freq == 0:
            eval_log_dict = eval(cfg, task_encoder, eval_loader, estimator, latest_test_labels)
            logs_dict.update(eval_log_dict)

        if cfg.wandb is not None:
            wandb.log(logs_dict, step=i)

        if i % cfg.ckpt_freq == 0:
            torch.save(
                {
                    'task_encoder_state_dict': task_encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iter': i,
                },
                cfg.ckpt_path,
            )

    if cfg.eval:
        eval_log_dict = eval(cfg, task_encoder, eval_loader, estimator, latest_test_labels)
        eval_log_dict['iter'] = i
        if cfg.wandb is not None:
            wandb.log(eval_log_dict, step=i)

    np.savez(
        os.path.join(cfg.work_dir, 'results.npz'),
        **{
            "reward_history_argmax" : reward_history_argmax,
            "reward_history_sample" : reward_history_sample,
            "batch_acc_history_argmax" : batch_acc_history_argmax,
            "batch_acc_history_sample" : batch_acc_history_sample,
            "history_surr_loss": history_surr_loss,
            "history_entr": history_entr,
            "history_grad_norm": history_grad_norm
        }
    )

    torch.save(
        {
            'task_encoder_state_dict': task_encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iter': i,
        },
        os.path.join(cfg.work_dir, 'ckpt.pth'),
    )

def eval(cfg, task_encoder, eval_loader, estimator, latest_test_labels=None):
    start_eval_time = time.time()
    task_encoder.eval()
    history = defaultdict(list)
    n_correct = 0
    n_total = 0
    n_flipped = 0

    with torch.no_grad():
        for input_dict, gt_labels in tqdm(eval_loader):
            input_dict = {k: (v.to(cfg.device) if k != TEXT_INPUT else v) for k, v in input_dict.items()}
            gt_labels = gt_labels.to(cfg.device)
            task_encoder_logits = task_encoder(input_dict)

            if getattr(cfg.task_encoder, 'calibrated', False):
                assert cfg.calibrated
                task_encoder_logits = F.log_softmax(task_encoder_logits, dim=1) - estimator.reward.logprior[None].expand_as(task_encoder_logits)

            # compare with latest labelling and update
            if latest_test_labels is not None:
                pred = task_encoder_logits.argmax(dim=1)
                img_indices = input_dict[INPUT_ID]    
                n_flipped += (latest_test_labels[img_indices] != pred).sum().item()
                latest_test_labels[input_dict[INPUT_ID]] = pred.detach().clone()

            n_correct += (gt_labels == task_encoder_logits.argmax(dim=1)).sum().item()
            n_total += gt_labels.size(0)

            # task_encoder_logits = task_encoder_logits.view(-1, cfg.N, *task_encoder_logits.shape[1:]) # -> (bsize, N, |S|)
            # for k in input_dict:
            #     input_dict[k] = input_dict[k].view(-1, cfg.N, *input_dict[k].shape[1:])

            # surrogate_loss, stats = estimator(input_dict, task_encoder_logits)
            # entr_reg = torch.special.entr( F.softmax(task_encoder_logits, dim=2).mean((0, 1)) ).sum()

            # history["R_sample"].append(stats["R_sample"])
            # history["R_argmax"].append(stats["R_argmax"])
            # history["surr_loss"].append(surrogate_loss.item())
            # history["entr_reg"].append(entr_reg.item())

    task_encoder.train()
    eval_time = time.time() - start_eval_time
    # reduce history
    log_dict = {
        'acc': n_correct / n_total,
        'flipped': n_flipped / n_total,
        'eval_time': eval_time,
    }
    for k, v in history.items():
        log_dict[k] = np.mean(v)
    
    print_str = " | ".join([f"{k}: {v:.3f}" for k, v in log_dict.items()])
    print(f"[Eval] {print_str} \n", flush=True)

    return {f"eval/{k}": v for k, v in log_dict.items()}

if __name__ == "__main__":
    main()
