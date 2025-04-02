from collections import defaultdict
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from misc.constants import PIXELS_INPUT, TEXT_INPUT, CACHE_DIR, LOC_FINDER_TOKEN, PIXELS_PHI_SEQ, PIXELS_PHI
from utils.common_utils import get_location, find_repeated_indices, get_BoT, get_minimal_to_distinguish, reshape_list


class BaseReward:
    def __init__(self, device, type="BoT_minimal", **init_kwargs):
        self.device = device
        assert type in ["first", "BoT", "BoT_minimal"]
        self.type = type
        self.initialize(**init_kwargs)

        assert hasattr(self, 'tokenizer')
        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': [LOC_FINDER_TOKEN]},
            replace_additional_special_tokens=False
        )


    def initialize(self, ):
        """
            This module initializes the LLM/VLM to use as a reward and sets the following attributes to the class
                self.tokenizer

            This function must set the self.tokenizer
        """
        raise NotImplementedError


    def get_logits(self, input_dict, y, return_right_locations=False):
        """
            input_dict: whatever we need to compute log probs 

            y: tensor of shape (bsize, N), each y[i, j] is an int from 0 to |Y| - 1, corresponding to the classname

            return_right_locations: Useful for multi-token labels y=t_1, ..., t_m. Whether to return left locations only,
                                    i.e., location preceding t_1 or also return right locations, i.e., location preceding t_m.
                                    then basically logits[left:right + 1] will give you logits to predict t_1, ..., t_m

            
            Returns
                logits of shape (bsize, full_num_tokens, |V|), where |V| is the vocab size
                locations: LongTensor of size 
        """
        raise NotImplementedError


    def set_template_and_labelset(self, template, label_set):
        """
            It sets:
                self.template: callable(input_dict, y), where input_dict is dict of str and y is str

                self.label_set: list[str]

                self.label_token_indices: list[int] or list[list[int]]
                    1) for self.type == "first", self.label_token_indices[i] is the integer
                    corresponding to the first token of classname[i]
                    2) for self.type == "BoT" and "BoT_minimal", self.label_token_indices[i] is the list
                    containing integers corresponding to tokens of classname[i]

                self.mask: Tensor of shape (K, vocab_len)
                    1) for self.type == "first" implements reward computation
                       considering only first token for each label
                    2) for self.type == "BoT" implements reward computation
                       with Bag-of-Tokens approximation
                    3) for self.type == "BoT" implements reward computation
                       with Bag-of-Tokens approximation considering minimal prefixes to
                       distinguish a classname from the rest classnames
        """
        self.template = template
        self.label_set = label_set
        self.label_token_indices = []
        if self.type == "first":
            print("Using Reward with first token approximation......")
            first_tokenized, first_loc = self.get_location(defaultdict(str), self.label_set[0])
            for classname in self.label_set:
                tokenized_i, loc_i = self.get_location(defaultdict(str), classname)
                assert loc_i == first_loc, \
                      "Check your label_set or template, because given fixed input_dict, locations MUST be the same for all the classnames"
                self.label_token_indices.append(tokenized_i[loc_i + 1])

            token_repetitions = find_repeated_indices(self.label_token_indices)
            if len(token_repetitions) > 0:
                print("The following classnames share the first token")
                for repetition in token_repetitions:
                    print(f'Shared token is "{self.tokenizer.decode(self.label_token_indices[repetition[0]])}" for ', end="")
                    print(" | ".join([f'"{self.label_set[idx]}"' for idx in repetition]))

                assert len(set(self.label_token_indices)) == len(self.label_set), \
                   "Check your label_set or template, some multi-token labels share first token"

            print(f'Prefix is "{self.tokenizer.decode(first_tokenized[:first_loc + 1])}"')
            print("Classname & Classnames first tokens:")
            for i, token_id in enumerate(self.label_token_indices):
                print(f'"{self.label_set[i]}" -> "{self.tokenizer.decode(token_id)}"')

            self.mask = torch.zeros((len(self.label_set), len(self.tokenizer.vocab) - 1))
            self.mask[torch.arange(len(self.label_set)), self.label_token_indices] = 1.0

        elif self.type == "BoT":
            print("Using Reward with BoT token approximation......")
            self.mask = torch.zeros((len(self.label_set), len(self.tokenizer.vocab) - 1))
            for k, classname in enumerate(self.label_set):
                cur_idx = get_BoT(self.template, self.tokenizer, defaultdict(str), classname)
                self.label_token_indices.append(cur_idx)
                self.mask[k, cur_idx] = 1 / len(cur_idx)
            
            print("Classname & Classnames BoT tokens:")
            for i, token_ids in enumerate(self.label_token_indices):
                cur_bots = ", ".join([f'"{self.tokenizer.decode(token_id)}"' for token_id in token_ids])
                print(f'"{self.label_set[i]}" -> {cur_bots}')
        else:
            print("Using Reward with BoT minimal token approximation......")
            label_set_tokenized = [get_BoT(self.template, self.tokenizer, defaultdict(str), classname) for classname in self.label_set]
            self.label_token_indices = get_minimal_to_distinguish(label_set_tokenized)
            self.mask = torch.zeros((len(self.label_set), len(self.tokenizer.vocab) - 1))
            for k, idx in enumerate(self.label_token_indices):
                self.mask[k, idx] = 1.0 / len(idx)
            
            print("Classname & Classnames BoT minimal tokens:")
            for i, token_ids in enumerate(self.label_token_indices):
                cur_bots = ", ".join([f'"{self.tokenizer.decode(token_id)}"' for token_id in token_ids])
                print(f'"{self.label_set[i]}" -> {cur_bots}')

        self.mask = self.mask.to(self.device)



    def get_location(self, input_dict, y, return_right_locations=False):
        return get_location(self.template, self.tokenizer, input_dict, y, return_right_locations)


    def set_logprior(self, dataset, batch_size=128):
        print("Setting Log Prior for Calibrated Reward......")
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=10,
            pin_memory=True,
            persistent_workers=True,
        )

        all_logprobs = []
        for input_dict, gt_labels in tqdm(loader):
            cur_batch_size = input_dict[PIXELS_PHI].shape[0]
            input_dict = {k: (v.to(self.device) if k != TEXT_INPUT else v) for k, v in input_dict.items()}

            for k in input_dict:
                if k == TEXT_INPUT:
                    input_dict[k] = reshape_list(input_dict[k], (cur_batch_size, 1))
                else:
                    input_dict[k] = input_dict[k].view(cur_batch_size, 1, *input_dict[k].shape[1:])

            cur_placeholder_labels = np.zeros((cur_batch_size, 1), dtype=int)
            cur_logprobs = F.log_softmax(self(input_dict, cur_placeholder_labels).squeeze(1), dim=1) # (batch_size, |Y|)

            all_logprobs.append(cur_logprobs.cpu().detach())

        all_logprobs = torch.cat(all_logprobs)
        self.logprior = torch.logsumexp(all_logprobs, dim=0) - math.log(all_logprobs.shape[0])
        self.logprior = self.logprior.to(self.device)
        print("Log Prior is set!")


    def __call__(self, input_dict, y):
        """
            input_dict: whatever we need to compute log probs 
            y: tensor of shape (bsize, N), each y[i, j] is an int from 0 to |Y| - 1, corresponding to the classname

            Returns normalized rewards tensor of shape (bsize, N, |Y|), i.e., log p(y_{n} | y_{i < n})
        """

        with torch.no_grad():
            logits, locations = self.get_logits(input_dict, y)  # (bsize, PADDED_SEQLEN, |V|)
            gathered = torch.gather(logits, 1, locations)  # (bsize, N, |V|)
            to_renormalize = torch.einsum('bnv,kv->bnk', gathered, self.mask) # (bsize, N, |Y|)
            reward = F.log_softmax(to_renormalize, dim=2)  # (bsize, N, |Y|)

        if hasattr(self, "logprior"):
            reward = reward - self.logprior.expand_as(reward)

        return reward / reward.shape[1]


class OpenFlamingoReward(BaseReward):
    def initialize(
        self,
        vision_x_feat_key: str = PIXELS_PHI_SEQ,
        allow_non_precomputed: bool = False,
    ):
        self.vision_x_feat_key = vision_x_feat_key
        assert allow_non_precomputed or vision_x_feat_key is not None, "Use PrecomputedCLIPFLamingo model."

        from utils.openflamingo_wrapper import PrecomputedCLIPFLamingo
        from open_flamingo import create_model_and_transforms
        from huggingface_hub import hf_hub_download

        self.model, self.image_processor, self.tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
            tokenizer_path="togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
            cross_attn_every_n_layers=2,
            cache_dir=CACHE_DIR,
        )


        checkpoint_path = hf_hub_download(
            "openflamingo/OpenFlamingo-4B-vitl-rpj3b-langinstruct",
            "checkpoint.pt",
            cache_dir=CACHE_DIR
        )

        self.model.load_state_dict(torch.load(checkpoint_path), strict=False)
        if self.vision_x_feat_key is not None:
            self.model = PrecomputedCLIPFLamingo(self.model)
        self.model = self.model.to(self.device)
        self.tokenizer.padding_side = "right"

    def get_logits(self, input_dict, y, return_right_locations=False):

        # input_dict values assumed to be on the correct device, 
        # i.e., the same as the model to avoid device transitioning within the code
        if self.vision_x_feat_key is None:
            device = input_dict[PIXELS_INPUT].device
            bsize, N = input_dict[PIXELS_INPUT].shape[:2]
            vision_x = input_dict[PIXELS_INPUT].view(bsize, N, 1, 3, 224, 224)
        else:
            vision_x_feat = input_dict[PIXELS_PHI_SEQ]
            device = vision_x_feat.device
            bsize, N, N_IMG_TOK, TOK_DIM = vision_x_feat.shape
            # add dummy frame dimension for the OF model -> (B, N, N_IMG_TOK, TOK_DIM) -> (B, N, 1, N_IMG_TOK, TOK_DIM)
            vision_x_feat = vision_x_feat[:, :, None]

        to_tokenize = []
        locations = []
        if return_right_locations:
            right_locations = []
        for i in range(bsize):
            cur_example = ""
            cur_locations = []
            if return_right_locations:
                cur_right_locations = []
            cur_running_length = 0
            for j in range(N):
                # TODO
                # Somehow we should put this out of get_logits logic
                # like input_dict[TEXT_INPUT] should either provide defaultdict(str)
                # or provide necessary things for the template, like question and etc
                if TEXT_INPUT in input_dict:
                    cur_text_input = {"q": input_dict[TEXT_INPUT][i][j]}
                else:
                    cur_text_input = defaultdict(str)
                cur_example += self.template(cur_text_input, self.label_set[y[i, j]])
                cur_example_tokenized, cur_loc, *cur_right_loc = self.get_location(cur_text_input, self.label_set[y[i, j]], return_right_locations)
                cur_locations.append(cur_running_length + cur_loc)
                cur_running_length += len(cur_example_tokenized)
                if return_right_locations:
                    cur_right_locations.append(cur_right_loc[0])

            to_tokenize.append(cur_example)
            locations.append(cur_locations)
            if return_right_locations:
                right_locations.append(cur_right_locations)

        lang_x = self.tokenizer(
            to_tokenize,
            return_tensors="pt",
            padding=True
        ).to(device)

        if self.vision_x_feat_key is None:
            output = self.model(
                vision_x=vision_x,
                lang_x=lang_x["input_ids"],
                attention_mask=lang_x["attention_mask"]
            )
        else:
            output = self.model.forward(
                vision_x_feat=vision_x_feat,
                lang_x=lang_x["input_ids"],
                attention_mask=lang_x["attention_mask"]
            )

        locations = torch.LongTensor(locations).unsqueeze(-1).expand(-1, -1, output.logits.shape[-1]).to(device)
        if return_right_locations:
            right_locations = torch.LongTensor(right_locations).unsqueeze(-1).expand(-1, -1, output.logits.shape[-1]).to(device)
            return output.logits, locations, right_locations
        else:
            return output.logits, locations
        

class LlamaReward(BaseReward):
    def __init__(self, device, sep='\n', model_name="meta-llama/Meta-Llama-3.1-8B"):
        self.sep = sep
        self.model_name = model_name
        super().__init__(device)

        self.eos_token_id = self.tokenizer.eos_token_id
        self.sep_token_id = self.tokenizer(sep, add_special_tokens=False)['input_ids'][0]

    def initialize(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=CACHE_DIR)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, 
                                                          torch_dtype=torch.float16, 
                                                          attn_implementation="flash_attention_2",
                                                          cache_dir=CACHE_DIR, 
                                                          device_map='auto')
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def get_logits(self, input_dict, y):
        examples = input_dict[TEXT_INPUT]
        bsize, N = len(examples), len(examples[0])

        to_tokenize = []
        locations = []
        for i in range(bsize):
            cur_example = ""
            cur_locations = []
            cur_running_length = 0
            for j in range(N):
                cur_example += self.template(examples[i][j], self.label_set[y[i][j]]) + self.tokenizer.eos_token
                cur_example_tokenized, cur_loc = self.get_location(examples[i][j], self.label_set[y[i][j]])
                cur_locations.append(cur_running_length + cur_loc)
                cur_running_length += len(cur_example_tokenized)

            to_tokenize.append(cur_example)
            locations.append(cur_locations)

        sent_toks = self.tokenizer(to_tokenize, return_tensors='pt', padding=True)
        # replace the token of "<|eos_token|>" by the token of "\n"
        sent_toks['input_ids'] = torch.where(sent_toks['input_ids'] != self.eos_token_id, 
                                                sent_toks['input_ids'], 
                                                self.sep_token_id)
        # ids = sent_toks['input_ids']
        # print(self.tokenizer.decode(ids[0, torch.tensor(locations[0])+1]))
        output = self.model(**sent_toks.to(self.device))
        locations = torch.LongTensor(locations).unsqueeze(-1).expand(-1, -1, output.logits.shape[-1]).to(self.device)

        return output.logits.to(torch.float32), locations

class UnslothLlamaReward(LlamaReward):    
    def initialize(self, device='cuda', sep='\n', model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit"):
        from unsloth import FastLanguageModel

        max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
        dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
            # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
        )
        self.model.eval()
