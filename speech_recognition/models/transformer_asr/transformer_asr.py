# from torch.nn import Embedding
#
# from fairseq import utils
# from fairseq.models import register_model, register_model_architecture
# from fairseq.models.transformer import TransformerModel, base_architecture, \
#     DEFAULT_MAX_SOURCE_POSITIONS, DEFAULT_MAX_TARGET_POSITIONS
# from speech_recognition.models.transformer_encoder_asr import ASRTransformerEncoder
#
#
# @register_model("asr_transformer")
# class ASRTransformerModel(TransformerModel):
#
#     @classmethod
#     def build_model(cls, args, task):
#         """Build a new model instance."""
#
#         # make sure all arguments are present in older models
#         base_architecture(args)
#
#         if args.encoder_layers_to_keep:
#             args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
#         if args.decoder_layers_to_keep:
#             args.decoder_layers = len(args.decoder_layers_to_keep.split(","))
#
#         if getattr(args, "max_source_positions", None) is None:
#             args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
#         if getattr(args, "max_target_positions", None) is None:
#             args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS
#
#         tgt_dict = task.target_dictionary
#
#         def build_embedding(dictionary, embed_dim, path=None):
#             num_embeddings = len(dictionary)
#             padding_idx = dictionary.pad()
#             emb = Embedding(num_embeddings, embed_dim, padding_idx)
#             # if provided, load from preloaded dictionaries
#             if path:
#                 embed_dict = utils.parse_embedding(path)
#                 utils.load_embedding(embed_dict, dictionary, emb)
#             return emb
#
#         decoder_embed_tokens = build_embedding(
#             tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
#         )
#
#         encoder = cls.build_encoder(args)
#         decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
#         return cls(args, encoder, decoder)
#
#     @classmethod
#     def build_encoder(cls, args):
#         return ASRTransformerEncoder(args)
#
#     @classmethod
#     def build_decoder(cls, args, tgt_dict, embed_tokens):
#         return super().build_decoder(args, tgt_dict, embed_tokens)
#
# @register_model_architecture("asr_transformer", "asr_transformer")
# def base_architecture(args):
#     # args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
#     args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
#     args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
#     args.encoder_layers = getattr(args, "encoder_layers", 6)
#     args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
#     args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
#     args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
#     args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
#     args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
#     args.decoder_ffn_embed_dim = getattr(
#         args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
#     )
#     args.decoder_layers = getattr(args, "decoder_layers", 6)
#     args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
#     args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
#     args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
#     args.attention_dropout = getattr(args, "attention_dropout", 0.0)
#     args.activation_dropout = getattr(args, "activation_dropout", 0.0)
#     args.activation_fn = getattr(args, "activation_fn", "relu")
#     args.dropout = getattr(args, "dropout", 0.1)
#     args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
#     args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
#     args.share_decoder_input_output_embed = getattr(
#         args, "share_decoder_input_output_embed", False
#     )
#     args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
#     args.no_token_positional_embeddings = getattr(
#         args, "no_token_positional_embeddings", False
#     )
#     args.adaptive_input = getattr(args, "adaptive_input", False)
#     args.no_cross_attention = getattr(args, "no_cross_attention", False)
#     args.cross_self_attention = getattr(args, "cross_self_attention", False)
#     args.layer_wise_attention = getattr(args, "layer_wise_attention", False)
#
#     args.decoder_output_dim = getattr(
#         args, "decoder_output_dim", args.decoder_embed_dim
#     )
#     args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
#
#     args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
#     args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
