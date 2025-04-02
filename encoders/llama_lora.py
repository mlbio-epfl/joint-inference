from typing import List, Dict
from collections import defaultdict

import torch
from torch import nn

from misc.constants import CACHE_DIR, LOC_FINDER_TOKEN
from misc.templates import BaseTemplate
from utils.common_utils import get_location

class LlamaLoRAEncoder(nn.Module):
    def __init__(
        self,
        model_name = 'unsloth/Meta-Llama-3.1-8B-bnb-4bit',
        use_lora = True,
    ):
        super().__init__()

        self.use_lora = use_lora
        if use_lora:
            from unsloth import FastLanguageModel
            max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
            dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
            load_in_4bit = True, # Use 4bit quantization to reduce memory usage. Can be False.

            model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name = model_name, # "unsloth/llama-3-8b-bnb-4bit"
                max_seq_length = max_seq_length,
                dtype = dtype,
                load_in_4bit = load_in_4bit,
                # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
            )

            self.encoder = FastLanguageModel.get_peft_model(
                model,
                r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128, set to 16 by default.
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj",],
                lora_alpha = 16,
                lora_dropout = 0, # Supports any, but = 0 is optimized
                bias = "none",    # Supports any, but = "none" is optimized
                # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
                use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
                random_state = 3407,
                use_rslora = False,  # We support rank stabilized LoRA
                loftq_config = None, # And LoftQ
            )
            self.encoder = self.encoder.to(torch.bfloat16)

        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
            self.encoder = AutoModelForCausalLM.from_pretrained(model_name, 
                                                                torch_dtype=torch.float16, 
                                                                attn_implementation="flash_attention_2",
                                                                cache_dir=CACHE_DIR, device_map='auto')
            

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token    
        self.tokenizer.add_special_tokens(
			{'additional_special_tokens': [LOC_FINDER_TOKEN]},
			replace_additional_special_tokens=False
		)

    def set_template_and_label_set(self, template=BaseTemplate, label_set=Dict[str, int]):
        self.template = template
        self.label_set = label_set
        
        # Get label token indices
        label_token_indices = []
        first_tokenized, first_loc = get_location(template, self.tokenizer, defaultdict(str), label_set[0])
        for classname in label_set:
            tokenized_i, loc_i = get_location(template, self.tokenizer, defaultdict(str), classname)
            assert loc_i == first_loc, \
                "Check your label_set or template, because given fixed input_dict, locations MUST be the same for all the classnames"
            label_token_indices.append(tokenized_i[loc_i + 1])
        self.label_token_indices = label_token_indices
    
    def set_train(self):
        self.encoder.train()

    def set_eval(self):
        self.encoder.eval()

    def forward(self, examples): 

        B = examples['label'].shape[0]
        examples = [{key: int(value) if key in ['idx', 'label'] else value for key, value in zip(examples.keys(), data)} for data in zip(*examples.values())]
        locations = []
        sentences = []
        for i in range(B):
            example = examples[i]
            sentence = self.template(example, self.label_set[example['label']])
            example_tokenized, loc = get_location(self.template, self.tokenizer, example, self.label_set[example['label']])
            sentences.append(sentence)
            locations.append(loc)

        toks = self.tokenizer(sentences, return_tensors='pt', padding=True)

        # Align tensor devices with the model
        model_device = next(self.encoder.parameters()).device
        toks = {key: value.to(model_device) for key, value in toks.items()}

        output = self.encoder(input_ids=toks['input_ids'], attention_mask=toks['attention_mask'])
        logits = output.logits # (bs, PADDED_SEQLEN, |V|)
        logits = logits[torch.arange(logits.shape[0]), locations][:, self.label_token_indices]

        return logits

        

