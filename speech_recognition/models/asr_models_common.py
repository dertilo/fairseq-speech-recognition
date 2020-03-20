from torch import nn as nn

DEFAULT_DEC_TRANSFORMER_CONFIG = ((256, 2, 1024, True, 0.2, 0.2, 0.2),) * 2
DEFAULT_DEC_CONV_CONFIG = ((256, 3, True),) * 2


def Linear(in_features, out_features, bias=True, dropout=0):
    """Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    # m.weight.data.uniform_(-0.1, 0.1)
    # if bias:
    #     m.bias.data.uniform_(-0.1, 0.1)
    return m
