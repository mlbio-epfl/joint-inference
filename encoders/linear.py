import torch.nn as nn
import torch.nn.functional as F

from misc.constants import PIXELS_PHI


class LinearEncoder(nn.Module):
    def __init__(self, input_dim, num_classes, wn, emb_norm=False, unit_wnorm=False):
        super(LinearEncoder, self).__init__()
        self.emb_norm = emb_norm
        self.unit_wnorm = unit_wnorm
        wn = wn or unit_wnorm

        self.encoder = nn.Linear(input_dim, num_classes)
        self.wn = wn
        if self.wn:
            self.encoder = nn.utils.weight_norm(self.encoder)

    def forward(self, input_dict):
        x = input_dict[PIXELS_PHI]

        if self.emb_norm:
            x = F.normalize(x, p=2, dim=-1)

        if self.unit_wnorm:
            self.encoder.weight_g.data.fill_(1.0)

        return self.encoder(x)
