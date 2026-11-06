import torch.nn as nn

from .extractor import Channel_Attention_Transformer_Extractor, ResidualBlock
from .refinement import NormalizationRefinement
from .update import LSTMMultiUpdateBlock


class DLNR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        context_dims = args.hidden_dims
        self.extractor = Channel_Attention_Transformer_Extractor()  #
        self.update_block = LSTMMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        self.bias_convs = nn.ModuleList(
            [nn.Conv2d(context_dims[i], args.hidden_dims[i] * 4, 3, padding=3 // 2) for i in
             range(self.args.n_gru_layers)])

        self.volume_conv = nn.Sequential(
            ResidualBlock(128, 128, 'instance', stride=1),
            nn.Conv2d(128, 256, 3, padding=1))

        self.normalizationRefinement = NormalizationRefinement()

    # fastforks: the test-mode forward is `inference.forward`; `freeze_bn`, `initialize_flow` (training) dropped,
    # `upsample_flow` lives in inference.py.
