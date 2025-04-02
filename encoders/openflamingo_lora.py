from typing import List
from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F
from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
from einops import rearrange

from misc.constants import PIXELS_INPUT, PIXELS_PHI_SEQ, TEXT_INPUT
from misc.constants import CACHE_DIR, LOC_FINDER_TOKEN
from misc.templates import BaseTemplate, OpenFlamingoImageClassificationTemplate
from encoders.lora_layers import inject_trainable_LoRA, LoRAWrapper
from utils.openflamingo_wrapper import PrecomputedCLIPFLamingo
import utils.common_utils as utils

LANG_ATTENTION_MODULES = ['GPTNeoXAttention', 'GPTNeoXSdpaAttention']
PERCIVER_ATTENTION_MODULES = ["PerceiverAttention"]
TARGET_LINEAR_LAYERS = ["query_key_value", "dense", "to_q", "to_kv", "to_out"]


class OpenFlamingoLoRAEncoder(nn.Module):
    def __init__(
        self,
        clip_vision_encoder_path: str,
        clip_vision_encoder_pretrained: bool,
        lang_encoder_path: str,
        tokenizer_path: str,
        cross_attn_every_n_layers: int,
        checkpoint_path: str,
        class_names: List[str],
        template: BaseTemplate,
        rank: int = 2,
        scale: float = 1.,
        apply_lora_to: List[str] = ['lang'],
        save_lora_only: bool = True,
        vision_x_feat_key: str = PIXELS_PHI_SEQ,
        inference_type: str = 'first',
    ):
        super().__init__()
        print(f'Initializing OpenFlamingoLoRAEncoder with rank={rank}, scale={scale}, apply_lora_to={apply_lora_to}')
        assert inference_type in ["first", "BoT_minimal"]

        self.rank = rank
        self.scale = scale
        self.save_lora_only = save_lora_only
        self.vision_x_feat_key = vision_x_feat_key
        self.template = template
        self.class_names = class_names
        self.inference_type = inference_type

        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path=clip_vision_encoder_path,
            clip_vision_encoder_pretrained=clip_vision_encoder_pretrained,
            lang_encoder_path=lang_encoder_path,
            tokenizer_path=tokenizer_path,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            cache_dir=CACHE_DIR,
        )
        tokenizer.padding_side = "right"
        self.tokenizer = tokenizer

        checkpoint_path = hf_hub_download(checkpoint_path, "checkpoint.pt")
        model.load_state_dict(torch.load(checkpoint_path), strict=False)
        
        self.transform_train = image_processor
        self.transform_val = image_processor
        self.model = PrecomputedCLIPFLamingo(model)

        self._add_lora_and_set_grad(apply_lora_to)

        # get class ids and logit location for inference
        tokenizer.add_special_tokens(
			{'additional_special_tokens': [LOC_FINDER_TOKEN]},
			replace_additional_special_tokens=False
		)
        tokens_locs = [utils.get_location(template, tokenizer, defaultdict(str), cn) for cn in class_names]
        self._logit_loc = tokens_locs[0][1]
        assert all(self._logit_loc == t[1] for t in tokens_locs), "All class names must be at the same location"
        self._class_ids = torch.LongTensor([t[0][self._logit_loc + 1] for t in tokens_locs])
        assert len(set(self._class_ids)) == len(class_names), "Class names must have different first tokens"
        print(f'Prefix ids: {tokens_locs[0][0][:self._logit_loc + 1]}')
        print(f'Class ids: {self._class_ids}')
        print(f'Prefix: "{tokenizer.decode(tokens_locs[0][0][:self._logit_loc + 1])}"')
        print(f'Classes first tokens:', *[f'"{tokenizer.decode(t)}" -- {cn}' for t, cn in zip(self._class_ids, class_names)], sep='\n- ')

        if inference_type == 'BoT_minimal':
            print("Using Reward with BoT minimal token approximation......")
            label_set_tokenized = [utils.get_BoT(self.template, self.tokenizer, defaultdict(str), classname) for classname in self.class_names]
            self.label_token_indices = utils.get_minimal_to_distinguish(label_set_tokenized)
            mask = torch.zeros((len(self.class_names), len(self.tokenizer.vocab) - 1))
            for k, idx in enumerate(self.label_token_indices):
                mask[k, idx] = 1.0 / len(idx)
            
            print("Classname & Classnames BoT minimal tokens:")
            for i, token_ids in enumerate(self.label_token_indices):
                cur_bots = ", ".join([f'"{self.tokenizer.decode(token_id)}"' for token_id in token_ids])
                print(f'"{self.class_names[i]}" -> {cur_bots}')
            self.register_buffer("mask", mask)

        lang_x = tokenizer(
            [template(defaultdict(str), class_names[0])],
            return_tensors="pt",
        )
        assert (lang_x["input_ids"][0].numpy() == tokens_locs[0][0]).all(), 'Smth went wrong with tokenization'

        self.register_buffer("_input_ids", lang_x["input_ids"])
        self.register_buffer("_attention_mask", lang_x["attention_mask"])

    def _add_lora_and_set_grad(self, apply_lora_to):
        target_modules = []
        if 'lang' in apply_lora_to:
            target_modules.extend(LANG_ATTENTION_MODULES)
        if 'resampler' in apply_lora_to:
            target_modules.extend(PERCIVER_ATTENTION_MODULES)

        inject_trainable_LoRA(self.model, rank=self.rank, scale=self.scale, target_names=TARGET_LINEAR_LAYERS, target_replace_modules=target_modules)

        for p in self.model.parameters():
            p.requires_grad = False

        for n, p in self.model.named_parameters():
            if 'lora' in n:
                p.requires_grad = True

    def reset_parameters(self):
        for m in self.model.modules():
            if isinstance(m, LoRAWrapper):
                m.reset_parameters()

    def forward(self, input_dict):        
        vision_x_feats = input_dict[self.vision_x_feat_key] # (B, N_IMG_TOK, TOK_DIM)
        B = vision_x_feats.shape[0]
        assert vision_x_feats.dim() == 3, f'vision_x_feats must be 2D (B, N_IMG_TOK, TOK_DIM), got {vision_x_feats.shape}'
        vision_x_feats = rearrange(vision_x_feats, "(b T F) v d -> b T F v d", T=1, F=1)

        prompts = []
        if TEXT_INPUT in input_dict:
            logit_locs = []
            prompts = []
            for i in range(B):
                _inp_dict = {'q': input_dict[TEXT_INPUT][i]}
                ids_and_locs = [utils.get_location(self.template, self.tokenizer, _inp_dict, cn) for cn in self.class_names]
                logit_loc = ids_and_locs[0][1]
                logit_locs.append(logit_loc)
                assert all(logit_loc == t[1] for t in ids_and_locs), "All class names must be at the same location"
                class_ids = torch.LongTensor([t[0][logit_loc + 1] for t in ids_and_locs])
                assert class_ids.equal(self._class_ids), "Class names must have the same first tokens"

                prompts.append(self.template(_inp_dict, self.class_names[0]))

            lang_x = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
            )
            logit_locs = torch.LongTensor(logit_locs).to(vision_x_feats.device)
            max_loc = logit_locs.max().item()
            input_ids = lang_x["input_ids"][:, :max_loc + 1].to(vision_x_feats.device)
            attention_mask = lang_x["attention_mask"][:, :max_loc + 1].to(vision_x_feats.device)
        else:
            input_ids = self._input_ids.expand(B, -1)
            attention_mask = self._attention_mask.expand(B, -1)
            logit_locs = torch.LongTensor([self._logit_loc] * B).to(vision_x_feats.device)

        logits = self.model(
            vision_x_feat=vision_x_feats,
            lang_x=input_ids,
            attention_mask=attention_mask,
        ).logits

        if self.inference_type == 'first':
            class_logits = logits[torch.arange(B), logit_locs][:, self._class_ids]
        elif self.inference_type == 'BoT_minimal':
            logits = logits[torch.arange(B), logit_locs] # (B, V)
            class_logits = torch.einsum('bv,kv->bk', logits, self.mask) # (bsize, |Y|)
        else:
            raise ValueError(f"Unknown inference type {self.inference_type}")

        return class_logits

    def state_dict(self):
        state_dict = super().state_dict()
        if self.save_lora_only:
            state_dict = {k: v for k, v in state_dict.items() if 'lora' in k}
        return state_dict


def openflamingo_lora(
    class_names: List[str],
    template: BaseTemplate = OpenFlamingoImageClassificationTemplate(),
    rank: int = 2,
    scale: float = 1.,
    apply_lora_to: List[str] = ['lang'],
    **kwargs,
):
    return OpenFlamingoLoRAEncoder(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
        tokenizer_path="togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
        cross_attn_every_n_layers=2,
        checkpoint_path='openflamingo/OpenFlamingo-4B-vitl-rpj3b-langinstruct',
        class_names=class_names,
        template=template,
        rank=rank,
        scale=scale,
        apply_lora_to=apply_lora_to,
        **kwargs,
    )


if __name__ == "__main__":
    # 10 classes
    from torchvision.datasets import CIFAR10
    from torch.utils.data import DataLoader, Subset
    import numpy as np
    from templates import OpenFlamingoImageClassificationTemplate
    from vision_utils import OpenFlamingoVisualFeaturesCache

    cifar10_classes = [
        'airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck',
    ]
    template = OpenFlamingoImageClassificationTemplate()
    model = openflamingo_lora(cifar10_classes, template, rank=2, scale=1., apply_lora_to=['lang'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    X = torch.randn(32, 3, 224, 224).to(device)
    Y = torch.randint(0, 10, (32,)).to(device)

    dataset = CIFAR10('./data/', download=True, transform=model.transform_train, train=False)
    dataset = OpenFlamingoVisualFeaturesCache(dataset, './data/representations/cifar10_eval.pth')
    batch_size = 16
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    print('Number of parameters:', sum(p.numel() for p in parameters))
    print('Trainable parameters:', *[n for n, p in model.named_parameters() if p.requires_grad], sep='\n- ')
    opt = torch.optim.Adam(parameters, lr=1e-4)

    print('Training...')
    i = 0
    for e in range(100):
        for input_dict, y in dataloader:
            y = y.to(device)
            input_dict = {k: v.to(device) for k, v in input_dict.items()}
            # add random strings to input_dict
            ascii_lowercase = 'a'
            input_dict[TEXT_INPUT] = [f'What is {"".join(np.random.choice(list(ascii_lowercase), 10))}?' for _ in range(batch_size)]

            logits = model(input_dict)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)

            if i % 10 == 0:
                accuracy = (logits.argmax(dim=-1) == y).float().mean().item()
                print(f'[iter: {i}] loss: {loss.item():.3f} | acc: {accuracy:.3f} | grad: {grad_norm:.2f}')

            opt.step()
            opt.zero_grad()
            i += 1

    print('Done!')
