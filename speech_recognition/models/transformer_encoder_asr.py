import torch

import torch.nn as nn
from typing import Iterable

from fairseq.models.transformer import TransformerEncoder
from fairseq.modules import PositionalEmbedding, TransformerEncoderLayer, LayerNorm, \
    VGGBlock
from speech_recognition.models.vggtransformer_refactored import DEFAULT_ENC_VGGBLOCK_CONFIG


class ASRTransformerEncoder(TransformerEncoder):

    def __init__(self, args):
        nn.Module.__init__(self) # is this necessary?
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout = args.dropout
        self.build_vgg(args.encoder_embed_dim)

        self.encoder_layerdrop = args.encoder_layerdrop

        self.max_source_positions = args.max_source_positions

        self.layer_wise_attention = getattr(args, "layer_wise_attention", False)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [TransformerEncoderLayer(args) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(self.embed_dim)
        else:
            self.layer_norm = None

        self.embed_positions = None



    def build_vgg(self,
                  encoder_embed_dim=512,
                  input_feat_per_channel=80,
                  vggblock_config=DEFAULT_ENC_VGGBLOCK_CONFIG,
                  in_channels=1,
                  ):
        self.num_vggblocks = 0
        if vggblock_config is not None:
            if not isinstance(vggblock_config, Iterable):
                raise ValueError("vggblock_config is not iterable")
            self.num_vggblocks = len(vggblock_config)

        self.conv_layers = nn.ModuleList()
        self.in_channels = in_channels
        self.input_dim = input_feat_per_channel

        if vggblock_config is not None:
            for _, config in enumerate(vggblock_config):
                (
                    out_channels,
                    conv_kernel_size,
                    pooling_kernel_size,
                    num_conv_layers,
                    layer_norm,
                ) = config
                self.conv_layers.append(
                    VGGBlock(
                        in_channels,
                        out_channels,
                        conv_kernel_size,
                        pooling_kernel_size,
                        num_conv_layers,
                        input_dim=input_feat_per_channel,
                        layer_norm=layer_norm,
                    )
                )
                in_channels = out_channels
                input_feat_per_channel = self.conv_layers[-1].output_dim

        vgg_embed_dim = self.conv_layers[-1].total_output_dim
        self.linear = nn.Linear(vgg_embed_dim, encoder_embed_dim)

    def forward_embedding(self, src_tokens):
        """
        src_tokens: padded tensor (B, T, C * feat)
        src_lengths: tensor of original lengths of input utterances (B,)
        """
        bsz, max_seq_len, _ = src_tokens.size()
        x = src_tokens.view(bsz, max_seq_len, self.in_channels, self.input_dim)
        x = x.transpose(1, 2).contiguous()
        # (B, C, T, feat)

        for layer_idx in range(len(self.conv_layers)):
            x = self.conv_layers[layer_idx](x)

        bsz, _, output_seq_len, _ = x.size()

        # (B, C, T, feat) -> (B, T, C, feat) -> (T, B, C, feat) -> (T, B, C * feat)
        x = x.transpose(1, 2).transpose(0, 1)
        x = x.contiguous().view(output_seq_len, bsz, -1)
        return x,None