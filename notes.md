# colab
sync it to colab

    rclone sync -P --exclude .git/** --exclude .idea/** --exclude build/** --exclude *.pyc --max-size 100k /home/tilo/code/SPEECH/fairseq-speech-recognition remote:fairseq-speech-recognition
    
run training

    python train.py /content/librispeech/fairseq_preprocessed_librispeech --log-format simple --save-dir checkpoints/debug --tensorboard-logdir /mydrive/tensorboard_logdir/debug --max-epoch 8 --task speech_recognition --arch vggtransformer_enc_small  --optimizer adam --lr 0.0001 --clip-norm 10.0   --max-tokens 5000 --log-interval 100 --criterion ctc_loss --user-dir /mydrive/fairseq-speech-recognition/speech_recognition

#### adam 0.0001 in colab lead to acc=58 by end ot 4 epochs
* seems not to be working
    
    !cd /mydrive/fairseq && python train.py /content/librispeech/fairseq_preprocessed_librispeech --log-format simple --save-dir checkpoints/librispeech_adam_0001 --tensorboard-logdir /mydrive/tensorboard_logdir/asr_librispeech_adam_0001 --max-epoch 8 --task speech_recognition --arch vggtransformer_small  --optimizer adam --lr 0.0001 --clip-norm 10.0   --max-tokens 5000 --log-interval 100 --criterion cross_entropy_acc --user-dir examples/speech_recognition/

# HPC

## wav2vec

after having run: `./examples/speech_recognition/datasets/prepare-librispeech.sh`

    python scripts/wav2vec_manifest.py ../data/asr_data/LibriSpeech/train_960/ --dest wav2vec_manifest --ext flac
    
old

    python train.py wav2vec_manifest --save-dir wav2vec_models --tensorboard-logdir tensorboard_logdir/wav2vec_libri --log-interval 100 --num-workers 6 --fp16 --max-update 400000 --save-interval 1 --no-epoch-checkpoints --arch wav2vec --task audio_pretraining --lr 1e-06 --min-lr 1e-09 --optimizer adam --max-lr 0.005 --lr-scheduler cosine --conv-feature-layers "[(512, 10, 5), (512, 8, 4), (512, 4, 2), (512, 4, 2), (512, 4, 2), (512, 1, 1), (512, 1, 1)]" --conv-aggregator-layers "[(512, 2, 1), (512, 3, 1), (512, 4, 1), (512, 5, 1), (512, 6, 1), (512, 7, 1), (512, 8, 1), (512, 9, 1), (512, 10, 1), (512, 11, 1), (512, 12, 1), (512, 13, 1)]" --skip-connections-agg --residual-scale 0.5 --log-compression --warmup-updates 500 --warmup-init-lr 1e-07 --criterion binary_cross_entropy --num-negatives 10 --max-sample-size 150000 --max-tokens 1500000 --skip-invalid-size-inputs-valid-test

#### train
    python train.py $HOME/data/fairseq-data/wav2vec_manifest --save-dir $HOME/data/fairseq-data/wav2vec_model/librispeech --tensorboard-logdir $HOME/data/tensorboard_logdir/wav2vec_librispeech --num-workers 6 --fp16 --max-update 400000 --save-interval 1     --arch wav2vec --task audio_pretraining --lr 1e-06 --min-lr 1e-09 --optimizer adam --max-lr 0.005 --lr-scheduler cosine     --conv-feature-layers "[(512, 10, 5), (512, 8, 4), (512, 4, 2), (512, 4, 2), (512, 4, 2), (512, 1, 1), (512, 1, 1)]"     --conv-aggregator-layers "[(512, 2, 1), (512, 3, 1), (512, 4, 1), (512, 5, 1), (512, 6, 1), (512, 7, 1), (512, 8, 1), (512, 9, 1), (512, 10, 1), (512, 11, 1), (512, 12, 1), (512, 13, 1)]"     --skip-connections-agg --residual-scale 0.5 --log-compression --warmup-updates 500 --warmup-init-lr 1e-07 --criterion binary_cross_entropy --num-negatives 10     --max-sample-size 150000 --max-tokens 1500000 --skip-invalid-size-inputs-valid-test
    
#### train quantized

python train.py wav2vec_manifest --tensorboard-logdir $HOME/data/tensorboard_logdir/wav2vec_vq --save-dir wav2vec_models_vq --num-workers 6 --fp16 --max-update 400000 \
--save-interval 1 --no-epoch-checkpoints --arch wav2vec --task audio_pretraining --lr 1e-06 --min-lr 1e-09 \
--optimizer adam --max-lr 1e-05 --lr-scheduler cosine \
--conv-feature-layers "[(512, 10, 5), (512, 8, 4), (512, 4, 2), (512, 4, 2), (512, 4, 2), (512, 1, 1), (512, 1, 1), (512, 1, 1)]" \
--conv-aggregator-layers "[(512, 2, 1), (512, 3, 1), (512, 4, 1), (512, 5, 1), (512, 6, 1), (512, 7, 1), (512, 8, 1), (512, 9, 1), (512, 10, 1), (512, 11, 1), (512, 12, 1), (512, 13, 1)]" \
--activation gelu --offset auto --skip-connections-agg --residual-scale 0.5 \
--log-keys "['prob_perplexity','code_perplexity','temp']" --vq-type gumbel --vq-groups 2 --vq-depth 2 \
--combine-groups --vq-vars 320 --vq-temp "(2,0.5,0.999995)" --prediction-steps 12 --warmup-updates 1000 \
--warmup-init-lr 1e-07 --criterion binary_cross_entropy --num-negatives 10 --max-sample-size 150000 \
--max-tokens 300000 --cross-sample-negatives 0 --update-freq 1 --seed 2 --skip-invalid-size-inputs-valid-test

## ASR
#### librispeech_960 on hpc 

    python train.py $HOME/data/asr_data/fairseq_librispeech --save-dir $HOME/data/fairseq-data/checkpoints/librispeech_6 --tensorboard-logdir $HOME/data/tensorboard_logdir/asr_librispeech_6 --max-epoch 8 --task speech_recognition --arch vggtransformer_66  --optimizer adam --lr 0.0001 --clip-norm 10.0   --max-tokens 5000 --log-interval 100 --criterion cross_entropy_acc --user-dir examples/speech_recognition/
debug    
    python train.py $HOME/data/asr_data/fairseq_librispeech --save-dir $HOME/data/fairseq-data/checkpoints/debug --tensorboard-logdir $HOME/data/tensorboard_logdir/debug --max-epoch 8 --task speech_recognition --arch vggtransformer_66  --optimizer adam --lr 0.0001 --clip-norm 10.0   --max-tokens 5000 --log-interval 100 --criterion cross_entropy_acc --user-dir $HOME/SPEECH/fairseq-speech-recognition/speech_recognition


## pretrain transformer decoder (LANGUAGE MODEL)

#### preprocess
TEXT=examples/language_model/wikitext-103
TEXT=$HOME/data/asr_data/fairseq_librispeech

fairseq-preprocess     --only-source    --srcdict $TEXT/dict.txt     --trainpref $TEXT/train.bpe.txt --validpref $TEXT/valid.bpe.txt --testpref $TEXT/valid.bpe.txt     --destdir data-bin/librispeech_lm     --workers 20
 
 
#### train    

    
    fairseq-train --task language_modeling \
      data-bin/librispeech_lm \
      --tensorboard-logdir $HOME/data/tensorboard_logdir/lm_librispeech \
      --log-interval 1000 \
      --save-dir checkpoints/lm_librispeech \
      --arch transformer_lm --share-decoder-input-output-embed \
      --dropout 0.1 \
      --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
      --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
      --tokens-per-sample 512 --sample-break-mode none \
      --max-tokens 2048 --update-freq 16 \
      --fp16 \
      --max-update 50000