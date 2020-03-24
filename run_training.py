from fairseq_cli.train import cli_main
import sys

if __name__ == '__main__':

    # s="/some_where/fairseq/train.py /home/tilo/data/asr_data/fairseq_preprocessed_librispeech --max-epoch 8 --task speech_recognition --arch vggtransformer_enc_small  --optimizer adadelta --lr 1.0 --adadelta-eps 1e-8 --adadelta-rho 0.95 --clip-norm 10.0   --max-tokens 5000 --log-interval 100 --criterion cross_entropy_acc --user-dir examples/speech_recognition/"
    #
    s="train.py /home/tilo/data/asr_data/fairseq_preprocessed_librispeech --num-workers 0 --train-subset valid --log-format simple --max-epoch 8 --task speech_recognition --arch vggtransformer_66 --optimizer adam --lr 0.0001 --clip-norm 10.0 --batch-size 2 --log-interval 100 --criterion cross_entropy_acc --user-dir /home/tilo/code/SPEECH/fairseq-speech-recognition/speech_recognition"
    # s="train.py /home/tilo/data/asr_data/fairseq_preprocessed_librispeech --num-workers 0 --train-subset valid --log-format simple --max-epoch 8 --task speech_recognition --arch vggtransformer_smaller --optimizer adam --lr 0.0001 --clip-norm 10.0 --batch-size 2 --log-interval 100 --criterion cross_entropy_acc --user-dir /home/tilo/code/SPEECH/fairseq-speech-recognition/speech_recognition"
    # s="train.py /home/tilo/data/asr_data/fairseq_preprocessed_librispeech --num-workers 0 --train-subset valid --log-format simple --max-epoch 8 --task speech_recognition --arch vggtransformer_enc_small --optimizer adam --lr 0.0001 --clip-norm 10.0 --batch-size 2 --log-interval 100 --criterion ctc_loss --user-dir /home/tilo/code/SPEECH/fairseq-speech-recognition/speech_recognition"
    x = s.split(' ')
    print(x)
    sys.argv= x

    cli_main()