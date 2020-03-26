# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from argparse import Namespace
import torch.nn as nn
from fairseq import checkpoint_utils
from fairseq.models import (
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import TransformerDecoder, Embedding
from fairseq.models.transformer_lm import base_lm_architecture, \
    DEFAULT_MAX_TARGET_POSITIONS
from fairseq.models.wav2vec import Wav2VecModel
from speech_recognition.models.asr_models_common import DEFAULT_DEC_CONV_CONFIG
from speech_recognition.models.conv_transformer_decoder import ConvTransformerDecoder, \
    add_decoder_args
from speech_recognition.models.vgg_transformer_encoder import add_encoder_args, \
    VGGTransformerEncoder, DEFAULT_ENC_VGGBLOCK_CONFIG, DEFAULT_ENC_TRANSFORMER_CONFIG
import os
HOME = os.environ['HOME']

class ASRTransformerEncoder(VGGTransformerEncoder):

    def __init__(self, input_feat_per_channel,
                 vggblock_config=DEFAULT_ENC_VGGBLOCK_CONFIG,
                 transformer_config=DEFAULT_ENC_TRANSFORMER_CONFIG,
                 encoder_output_dim=512, in_channels=1, transformer_context=None,
                 transformer_sampling=None):
        super().__init__(input_feat_per_channel, vggblock_config, transformer_config,
                         encoder_output_dim, in_channels, transformer_context,
                         transformer_sampling)
        cp = checkpoint_utils.load_checkpoint_to_cpu(
            HOME + '/data/fairseq-data/wav2vec_models/checkpoint_last.pt')
        model = Wav2VecModel.build_model(cp['args'], task=None)
        model.load_state_dict(cp['model'])
        # model.eval()
        self.wav2vec_model = model

    def forward(self, x, src_lengths, **kwargs):

        z = self.wav2vec_model.feature_extractor(x)
        c = self.wav2vec_model.feature_aggregator(z).squeeze().t()

        d =  super().forward(c, src_lengths, **kwargs)
        epm = d.get('encoder_padding_mask', None)
        epm = epm.t() if epm is not None else None
        return EncoderOut(
            encoder_out=d['encoder_out'],  # T x B x C
            encoder_padding_mask=epm,  # B x T
            encoder_embedding=None,  # B x T x C
            encoder_states=None,  # List[T x B x C]
        )

class ASRTransformerDecoderWrapper(TransformerDecoder):
    def __init__(self, dictionary):
        args = Namespace()
        base_lm_architecture(args)
        args.decoder_layerdrop=0
        args.max_target_positions = getattr(args, 'tokens_per_sample',DEFAULT_MAX_TARGET_POSITIONS)

        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        emb = Embedding(num_embeddings, args.decoder_embed_dim, padding_idx)
        super().__init__(args, dictionary, emb, False)

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
        return ASRTransformerEncoder(
            input_feat_per_channel=args.input_feat_per_channel,
            vggblock_config=eval(args.vggblock_enc_config),
            transformer_config=eval(args.transformer_enc_config),
            encoder_output_dim=args.enc_output_dim,
            in_channels=args.in_channels,
        )

    @classmethod
    def build_decoder(cls, args, task):
        decoder = ASRTransformerDecoderWrapper(task.target_dictionary)
        print('loading language model checkpoint')
        checkpoint_file = '/home/users/t/tilo-himmelsbach/data/fairseq-data/checkpoints/lm_librispeech/checkpoint20.pt'
        # checkpoint_file = '/tmp/checkpoint20.pt'
        state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint_file, {'no_encoder_attn':False})
        decoder_layers = {k.replace('decoder.',''):v for k,v in state['model'].items()}
        decoder.load_state_dict(decoder_layers,strict=False)

        # decoder = ConvTransformerDecoder(dictionary=task.target_dictionary,
        #                                  embed_dim=args.tgt_embed_dim,
        #                                  transformer_config=eval(
        #                                      args.transformer_dec_config),
        #                                  conv_config=eval(args.conv_dec_config),
        #                                  encoder_output_dim=args.enc_output_dim, )
        return decoder

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
    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 512)
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