import argparse
import math

import torch
from torch import nn as nn

from fairseq import utils
from fairseq.models import FairseqIncrementalDecoder
from fairseq.modules import TransformerDecoderLayer, LinearizedConvolution
from speech_recognition.models.asr_models_common import Linear, \
    DEFAULT_DEC_TRANSFORMER_CONFIG, DEFAULT_DEC_CONV_CONFIG


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    # nn.init.uniform_(m.weight, -0.1, 0.1)
    # nn.init.constant_(m.weight[padding_idx], 0)
    return m

def LinearizedConv1d(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    """Weight-normalized Conv1d layer optimized for decoding"""
    m = LinearizedConvolution(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    nn.init.normal_(m.weight, mean=0, std=std)
    nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m, dim=2)

def prepare_transformer_decoder_params(
    input_dim,
    num_heads,
    ffn_dim,
    normalize_before,
    dropout,
    attention_dropout,
    relu_dropout,
):
    args = argparse.Namespace()
    args.decoder_embed_dim = input_dim
    args.decoder_attention_heads = num_heads
    args.attention_dropout = attention_dropout
    args.dropout = dropout
    args.activation_dropout = relu_dropout
    args.decoder_normalize_before = normalize_before
    args.decoder_ffn_embed_dim = ffn_dim
    return args

def add_decoder_args(parser):
    parser.add_argument(
        "--tgt-embed-dim",
        type=int,
        metavar="N",
        help="embedding dimension of the decoder target tokens",
    )
    parser.add_argument(
        "--transformer-dec-config",
        type=str,
        metavar="EXPR",
        help="""
        a tuple containing the configuration of the decoder transformer layers
        configurations:
        [(input_dim,
          num_heads,
          ffn_dim,
          normalize_before,
          dropout,
          attention_dropout,
          relu_dropout), ...]
        """,
    )
    parser.add_argument(
        "--conv-dec-config",
        type=str,
        metavar="EXPR",
        help="""
        an array of tuples for the decoder 1-D convolution config
            [(out_channels, conv_kernel_size, use_layer_norm), ...]""",
    )

class ConvTransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs.
            Default: ``False``
        left_pad (bool, optional): whether the input is left-padded. Default:
            ``False``
    """

    def __init__(
        self,
        dictionary,
        embed_dim=512,
        transformer_config=DEFAULT_DEC_TRANSFORMER_CONFIG,
        conv_config=DEFAULT_DEC_CONV_CONFIG,
        encoder_output_dim=512,
    ):

        super().__init__(dictionary)
        vocab_size = len(dictionary)
        self.padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(vocab_size, embed_dim, self.padding_idx)

        self.conv_layers = nn.ModuleList()
        for i in range(len(conv_config)):
            out_channels, kernel_size, layer_norm = conv_config[i]
            if i == 0:
                conv_layer = LinearizedConv1d(
                    embed_dim, out_channels, kernel_size, padding=kernel_size - 1
                )
            else:
                conv_layer = LinearizedConv1d(
                    conv_config[i - 1][0],
                    out_channels,
                    kernel_size,
                    padding=kernel_size - 1,
                )
            self.conv_layers.append(conv_layer)
            if layer_norm:
                self.conv_layers.append(nn.LayerNorm(out_channels))
            self.conv_layers.append(nn.ReLU())

        self.layers = nn.ModuleList()
        if conv_config[-1][0] != transformer_config[0][0]:
            self.layers.append(Linear(conv_config[-1][0], transformer_config[0][0]))
        self.layers.append(
            TransformerDecoderLayer(
                prepare_transformer_decoder_params(*transformer_config[0])
            )
        )

        for i in range(1, len(transformer_config)):
            if transformer_config[i - 1][0] != transformer_config[i][0]:
                self.layers.append(
                    Linear(transformer_config[i - 1][0], transformer_config[i][0])
                )
            self.layers.append(
                TransformerDecoderLayer(
                    prepare_transformer_decoder_params(*transformer_config[i])
                )
            )
        self.fc_out = Linear(transformer_config[-1][0], vocab_size)

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
        Returns:
            tuple:
                - the last decoder layer's output of shape `(batch, tgt_len,
                  vocab)`
                - the last decoder layer's attention weights of shape `(batch,
                  tgt_len, src_len)`
        """
        target_padding_mask = (
            (prev_output_tokens == self.padding_idx).to(prev_output_tokens.device)
            if incremental_state is None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)

        # B x T x C -> T x B x C
        x = self._transpose_if_training(x, incremental_state)

        for layer in self.conv_layers:
            if isinstance(layer, LinearizedConvolution):
                x = layer(x, incremental_state)
            else:
                x = layer(x)

        # B x T x C -> T x B x C
        x = self._transpose_if_inference(x, incremental_state)

        # decoder layers
        for layer in self.layers:
            if isinstance(layer, TransformerDecoderLayer):
                x, _, _ = layer(
                    x,
                    (encoder_out["encoder_out"] if encoder_out is not None else None),
                    (
                        encoder_out["encoder_padding_mask"].t()
                        if encoder_out["encoder_padding_mask"] is not None
                        else None
                    ),
                    incremental_state,
                    self_attn_mask=(
                        self.buffered_future_mask(x)
                        if incremental_state is None
                        else None
                    ),
                    self_attn_padding_mask=(
                        target_padding_mask if incremental_state is None else None
                    ),
                )
            else:
                x = layer(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        x = self.fc_out(x)

        return x, None

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            not hasattr(self, "_future_mask")
            or self._future_mask is None
            or self._future_mask.device != tensor.device
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(tensor.new(dim, dim)), 1
            )
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1
            )
        return self._future_mask[:dim, :dim]

    def _transpose_if_training(self, x, incremental_state):
        if incremental_state is None:
            x = x.transpose(0, 1)
        return x

    def _transpose_if_inference(self, x, incremental_state):
        if incremental_state:
            x = x.transpose(0, 1)
        return x