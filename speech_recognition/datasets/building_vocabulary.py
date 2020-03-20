from collections import Counter

from tqdm import tqdm
from util import data_io
import os

HOME = os.environ["HOME"]


def build_vocabulary(
    text_g, vocab_file="fairseq_dict.txt", min_freq=1000,
):
    counter = Counter((c for t in tqdm(text_g) for c in t.replace(" ", "_")))
    vocab = counter.most_common(200)
    assert len(vocab) > 0
    data_io.write_lines(
        vocab_file, ["%s %d" % (c, f) for c, f in vocab if f > min_freq],
    )


if __name__ == "__main__":
    corpus_file = HOME + "/data/asr_data/fairseq_librispeech/data/lang_char/input.txt"
    build_vocabulary(corpus_file)
