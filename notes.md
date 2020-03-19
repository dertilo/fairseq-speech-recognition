
sync it to colab

    rclone sync -P --exclude .git/** --exclude .idea/** --exclude build/** --exclude *.pyc --max-size 100k /home/tilo/code/SPEECH/fairseq-speech-recognition remote:fairseq-speech-recognition