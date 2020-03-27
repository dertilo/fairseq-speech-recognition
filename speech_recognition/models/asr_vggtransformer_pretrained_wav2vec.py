import os
from argparse import Namespace

import torch

from fairseq import checkpoint_utils
from fairseq.models import register_model, FairseqEncoderDecoderModel, \
    register_model_architecture
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import TransformerDecoder, Embedding
from fairseq.models.transformer_lm import base_lm_architecture, \
    DEFAULT_MAX_TARGET_POSITIONS
from fairseq.models.wav2vec import Wav2VecModel
from speech_recognition.models.conv_transformer_decoder import add_decoder_args
from speech_recognition.models.vgg_transformer_encoder import VGGTransformerEncoder, \
    DEFAULT_ENC_VGGBLOCK_CONFIG, DEFAULT_ENC_TRANSFORMER_CONFIG, add_encoder_args
from speech_recognition.models.vggtransformer import base_architecture

HOME = os.environ['HOME']


def freeze_module_params(m):
    if m is not None:
        for p in m.parameters():
            p.requires_grad = False


class ASRTransformerEncoder(VGGTransformerEncoder):

    def __init__(self, input_feat_per_channel,
                 vggblock_config=DEFAULT_ENC_VGGBLOCK_CONFIG,
                 transformer_config=DEFAULT_ENC_TRANSFORMER_CONFIG,
                 encoder_output_dim=512, in_channels=1, transformer_context=None,
                 transformer_sampling=None):
        super().__init__(input_feat_per_channel, vggblock_config, transformer_config,
                         encoder_output_dim, in_channels, transformer_context,
                         transformer_sampling)
        wav2vec_checkpoint = HOME + '/data/fairseq-data/wav2vec_models/checkpoint_last.pt'
        # wav2vec_checkpoint = '/tmp/checkpoint_last.pt'
        cp = checkpoint_utils.load_checkpoint_to_cpu(wav2vec_checkpoint)
        model = Wav2VecModel.build_model(cp['args'], task=None)
        model.load_state_dict(cp['model'])
        freeze_module_params(model)
        self.wav2vec_model = model

    def forward(self, src_tokens, src_lengths, **kwargs):
        self.wav2vec_model.eval()
        with torch.no_grad():
            z = self.wav2vec_model.feature_extractor(src_tokens.squeeze())
            c = self.wav2vec_model.feature_aggregator(z).permute(0,2,1)
        subsample_factor = src_tokens.shape[1]/c.shape[1]
        src_lengths = torch.ceil(src_lengths /subsample_factor).type(torch.int64)
        src_lengths = torch.min(torch.tensor(c.shape[1]).to(src_lengths.device),src_lengths)
        d =  super().forward(c, src_lengths, **kwargs)
        epm = d.get('encoder_padding_mask', None)
        epm = epm.t() if epm is not None else None
        return EncoderOut(
            encoder_out=d['encoder_out'],  # T x B x C
            encoder_padding_mask=epm,  # B x T
            encoder_embedding=None,  # B x T x C
            encoder_states=None,  # List[T x B x C]
        )

    def reorder_encoder_out(self, encoder_out:EncoderOut, new_order):
        if encoder_out.encoder_padding_mask is not None:
            epm = encoder_out.encoder_padding_mask.index_select(1, new_order)
        else:
            epm = encoder_out.encoder_padding_mask

        return EncoderOut(
            encoder_out=encoder_out.encoder_out.index_select(
            1, new_order
        ),  # T x B x C
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


@register_model("asr_vggtransformer_pretrained_wav2vec")
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
        # print('loading language model checkpoint')
        # checkpoint_file = '/home/users/t/tilo-himmelsbach/data/fairseq-data/checkpoints/lm_librispeech/checkpoint20.pt'
        # # checkpoint_file = '/tmp/checkpoint20.pt'
        # state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint_file, {'no_encoder_attn':False})
        # decoder_layers = {k.replace('decoder.',''):v for k,v in state['model'].items()}
        # decoder.load_state_dict(decoder_layers,strict=False)

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


@register_model_architecture("asr_vggtransformer_pretrained_wav2vec", "vggtransformer_66_wav2vec")
def vggtransformer_66_wav2vec(args):
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