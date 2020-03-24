# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.models import (
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from speech_recognition.models.asr_models_common import DEFAULT_DEC_CONV_CONFIG
from speech_recognition.models.conv_transformer_decoder import ConvTransformerDecoder, \
    add_decoder_args
from speech_recognition.models.vgg_transformer_encoder import add_encoder_args, \
    VGGTransformerEncoder, DEFAULT_ENC_VGGBLOCK_CONFIG, DEFAULT_ENC_TRANSFORMER_CONFIG


@register_model("asr_vggtransformer")
class VGGTransformerModel(FairseqEncoderDecoderModel):
    """
    Transformers with convolutional context for ASR
    https://arxiv.org/abs/1904.11660
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        add_encoder_args(parser)
        add_decoder_args(parser)


    @classmethod
    def build_encoder(cls, args, task):
        return VGGTransformerEncoder(
            input_feat_per_channel=args.input_feat_per_channel,
            vggblock_config=eval(args.vggblock_enc_config),
            transformer_config=eval(args.transformer_enc_config),
            encoder_output_dim=args.enc_output_dim,
            in_channels=args.in_channels,
        )

    @classmethod
    def build_decoder(cls, args, task):
        return ConvTransformerDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.tgt_embed_dim,
            transformer_config=eval(args.transformer_dec_config),
            conv_config=eval(args.conv_dec_config),
            encoder_output_dim=args.enc_output_dim,
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted
        # (in case there are any new ones)
        base_architecture(args)

        encoder = cls.build_encoder(args, task)
        decoder = cls.build_decoder(args, task)
        return cls(encoder, decoder)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = super().get_normalized_probs(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

# seq2seq models
def base_architecture(args):
    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 40)
    args.vggblock_enc_config = getattr(
        args, "vggblock_enc_config", DEFAULT_ENC_VGGBLOCK_CONFIG
    )
    args.transformer_enc_config = getattr(
        args, "transformer_enc_config", DEFAULT_ENC_TRANSFORMER_CONFIG
    )
    args.enc_output_dim = getattr(args, "enc_output_dim", 512)
    args.in_channels = getattr(args, "in_channels", 1)
    args.tgt_embed_dim = getattr(args, "tgt_embed_dim", 128)
    args.transformer_dec_config = getattr(
        args, "transformer_dec_config", DEFAULT_ENC_TRANSFORMER_CONFIG
    )
    args.conv_dec_config = getattr(args, "conv_dec_config", DEFAULT_DEC_CONV_CONFIG)
    args.transformer_context = getattr(args, "transformer_context", "None")


@register_model_architecture("asr_vggtransformer", "vggtransformer_1")
def vggtransformer_1(args):
    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
    args.vggblock_enc_config = getattr(
        args, "vggblock_enc_config", "[(64, 3, 2, 2, True), (128, 3, 2, 2, True)]"
    )
    args.transformer_enc_config = getattr(
        args,
        "transformer_enc_config",
        "((1024, 16, 4096, True, 0.15, 0.15, 0.15),) * 14",
    )
    args.enc_output_dim = getattr(args, "enc_output_dim", 1024)
    args.tgt_embed_dim = getattr(args, "tgt_embed_dim", 128)
    args.conv_dec_config = getattr(args, "conv_dec_config", "((256, 3, True),) * 4")
    args.transformer_dec_config = getattr(
        args,
        "transformer_dec_config",
        "((1024, 16, 4096, True, 0.15, 0.15, 0.15),) * 4",
    )


@register_model_architecture("asr_vggtransformer", "vggtransformer_2")
def vggtransformer_2(args):
    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
    args.vggblock_enc_config = getattr(
        args, "vggblock_enc_config", "[(64, 3, 2, 2, True), (128, 3, 2, 2, True)]"
    )
    args.transformer_enc_config = getattr(
        args,
        "transformer_enc_config",
        "((1024, 16, 4096, True, 0.15, 0.15, 0.15),) * 16",
    )
    args.enc_output_dim = getattr(args, "enc_output_dim", 1024)
    args.tgt_embed_dim = getattr(args, "tgt_embed_dim", 512)
    args.conv_dec_config = getattr(args, "conv_dec_config", "((256, 3, True),) * 4")
    args.transformer_dec_config = getattr(
        args,
        "transformer_dec_config",
        "((1024, 16, 4096, True, 0.15, 0.15, 0.15),) * 6",
    )


@register_model_architecture("asr_vggtransformer", "vggtransformer_small")
def vggtransformer_small(args):
    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
    args.vggblock_enc_config = getattr(
        args, "vggblock_enc_config", "[(64, 3, 2, 2, True), (128, 3, 2, 2, True)]"
    )
    args.transformer_enc_config = getattr(
        args, "transformer_enc_config", "((512, 8, 2048, True, 0.15, 0.15, 0.15),) * 6"
    )

    args.enc_output_dim = getattr(args, "enc_output_dim", 512)
    args.tgt_embed_dim = getattr(args, "tgt_embed_dim", 512)
    args.conv_dec_config = getattr(args, "conv_dec_config", "((256, 3, True),) * 4")
    args.transformer_dec_config = getattr(
        args, "transformer_dec_config", "((512, 8, 2048, True, 0.15, 0.15, 0.15),) * 4"
    )


@register_model_architecture("asr_vggtransformer", "vggtransformer_smaller")
def vggtransformer_smaller(args):
    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
    args.vggblock_enc_config = getattr(
        args, "vggblock_enc_config", "[(64, 3, 2, 2, True), (128, 3, 2, 2, True)]"
    )
    args.transformer_enc_config = getattr(
        args, "transformer_enc_config", "((512, 8, 2048, True, 0.15, 0.15, 0.15),) * 3"
    )

    args.enc_output_dim = getattr(args, "enc_output_dim", 512)
    args.tgt_embed_dim = getattr(args, "tgt_embed_dim", 512)
    args.conv_dec_config = getattr(args, "conv_dec_config", "((256, 3, True),) * 4")
    args.transformer_dec_config = getattr(
        args, "transformer_dec_config", "((512, 8, 2048, True, 0.15, 0.15, 0.15),) * 2"
    )


@register_model_architecture("asr_vggtransformer", "vggtransformer_base")
def vggtransformer_base(args):
    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
    args.vggblock_enc_config = getattr(
        args, "vggblock_enc_config", "[(64, 3, 2, 2, True), (128, 3, 2, 2, True)]"
    )
    args.transformer_enc_config = getattr(
        args, "transformer_enc_config", "((512, 8, 2048, True, 0.15, 0.15, 0.15),) * 12"
    )

    args.enc_output_dim = getattr(args, "enc_output_dim", 512)
    args.tgt_embed_dim = getattr(args, "tgt_embed_dim", 512)
    args.conv_dec_config = getattr(args, "conv_dec_config", "((256, 3, True),) * 4")
    args.transformer_dec_config = getattr(
        args, "transformer_dec_config", "((512, 8, 2048, True, 0.15, 0.15, 0.15),) * 6"
    )
    # Size estimations:
    # Encoder:
    #   - vggblock param: 64*1*3*3 + 64*64*3*3 + 128*64*3*3  + 128*128*3 = 258K
    #   Transformer:
    #   - input dimension adapter: 2560 x 512 -> 1.31M
    #   - transformer_layers (x12) --> 37.74M
    #       * MultiheadAttention: 512*512*3 (in_proj) + 512*512 (out_proj) = 1.048M
    #       * FFN weight: 512*2048*2 = 2.097M
    #   - output dimension adapter: 512 x 512 -> 0.26 M
    # Decoder:
    #   - LinearizedConv1d: 512 * 256 * 3 + 256 * 256 * 3 * 3
    #   - transformer_layer: (x6) --> 25.16M
    #        * MultiheadAttention (self-attention): 512*512*3 + 512*512 = 1.048M
    #        * MultiheadAttention (encoder-attention): 512*512*3 + 512*512 = 1.048M
    #        * FFN: 512*2048*2 = 2.097M
    # Final FC:
    #   - FC: 512*5000 = 256K (assuming vocab size 5K)
    # In total:
    #       ~65 M

@register_model_architecture("asr_vggtransformer", "vggtransformer_66")
def vggtransformer_66(args):
    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
    args.vggblock_enc_config = getattr(
        args, "vggblock_enc_config", "[(64, 3, 2, 2, True), (128, 3, 2, 2, True)]"
    )
    args.transformer_enc_config = getattr(
        args, "transformer_enc_config", "((512, 8, 2048, True, 0.15, 0.15, 0.15),) * 6"
    )

    args.enc_output_dim = getattr(args, "enc_output_dim", 512)
    args.tgt_embed_dim = getattr(args, "tgt_embed_dim", 512)
    args.conv_dec_config = getattr(args, "conv_dec_config", "((256, 3, True),) * 4")
    args.transformer_dec_config = getattr(
        args, "transformer_dec_config", "((512, 8, 2048, True, 0.15, 0.15, 0.15),) * 6"
    )