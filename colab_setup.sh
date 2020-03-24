#bin/bash

export CUDA_HOME=/usr/local/cuda-10.1
git clone https://github.com/NVIDIA/apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex

cd /mydrive/fairseq
pip install -e .
pip install torchaudio
pip install tensorboardX