import torch
from torch import nn
from open_flamingo import Flamingo


class PrecomputedCLIPFLamingo(nn.Module):
    """
    Wrapper for OpenFlamingo model to use precomputed vision features
    """
    def __init__(self, of_model: Flamingo):
        super().__init__()
        self.of_model = of_model
        # remove the clip model
        del self.of_model.vision_encoder
        torch.cuda.empty_cache()

    def forward(
        self,
        vision_x_feat: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        clear_conditioned_layers: bool = True,
        past_key_values=None,
    ):
        # run perceiver on top of features and save the cachec
        # see Flamingo.cache_media and Flamingo._encode_vision_x for more details
        vision_x_feat = self.of_model.perceiver(vision_x_feat)
        for layer in self.of_model.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x_feat)

        self.of_model._condition_media_locations(input_ids=lang_x)
        self.of_model.lang_encoder._use_cached_vision_x = True

        return self.of_model.forward(
            vision_x=None,
            lang_x=lang_x,
            attention_mask=attention_mask,
            labels=labels,
            clear_conditioned_layers=clear_conditioned_layers,
            past_key_values=past_key_values,
        )