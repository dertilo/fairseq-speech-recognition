#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

from collections import namedtuple
from itertools import chain
import argparse
import os
import json
import multiprocessing
from pathlib import Path

import torchaudio
from typing import NamedTuple, List

from tqdm import tqdm
from util import data_io

from examples.speech_recognition.datasets.building_vocabulary import build_vocabulary
from fairseq.data import Dictionary

MILLISECONDS_TO_SECONDS = 0.001
HOME = os.environ["HOME"]

tgt_dict = None


def process_sample(aud_path, lable, utt_id, dictionary_file):
    global tgt_dict
    if tgt_dict is None:
        tgt_dict = Dictionary.load(dictionary_file)
    input = {}
    output = {}
    si, ei = torchaudio.info(aud_path)
    input["length_ms"] = int(
        si.length / si.channels / si.rate / MILLISECONDS_TO_SECONDS
    )
    input["path"] = aud_path

    token = " ".join(lable.replace(" ", "_"))
    ids = tgt_dict.encode_line(token, append_eos=False)
    output["text"] = lable
    output["token"] = token
    output["tokenid"] = ", ".join(map(str, [t.tolist() for t in ids]))
    return utt_id, {"input": input, "output": output}


def build_samples(labels, audio_dirs, audio_format):

    for path, _, files in chain.from_iterable(os.walk(path) for path in audio_dirs):
        for f in files:
            if f.endswith(audio_format):
                utt_id = os.path.splitext(f)[0]
                if utt_id not in labels:
                    continue
                yield os.path.join(path, f), utt_id


def fun(args):
    return process_sample(*args)


def create_prepared_data_json(
    id2text, dictionary_file, output_json, audio_dirs, audio_format, num_processes
):
    g = (
        (aud_path, id2text[utt_id], utt_id, dictionary_file)
        for aud_path, utt_id in tqdm(build_samples(id2text, audio_dirs, audio_format))
    )

    with multiprocessing.Pool(processes=num_processes) as pool:
        utts = {k: v for k, v in pool.imap_unordered(fun, g)}
        with open(output_json, "w") as f:
            json.dump({"utts": utts}, f, indent=4)


DataSet = namedtuple("Dataset", "name audio_folders")


def build_id_text_generator(paths: List[str]):
    for p in paths:
        for file in Path(p).rglob("*.trans.txt"):
            for line in data_io.read_lines(str(file)):
                eid, text = line.split(" ", 1)
                yield eid, text


if __name__ == "__main__":

    parser = argparse.ArgumentParser(allow_abbrev=False)
    # fmt: off
    parser.add_argument("--asr_path", default=HOME + "/data/asr_data", type=str)
    parser.add_argument("--train_dirs", default=['dev-clean'], nargs='+',type=str)
    parser.add_argument("--num_processes", default=8, type=int)
    parser.add_argument("--overwrite", default=True)
    # fmt: on
    args = parser.parse_args()

    asr_path = args.asr_path
    raw_data_path = asr_path + "/LibriSpeech"
    output_path = asr_path + "/fairseq_preprocessed_librispeech"
    dictionary_file = output_path + "/dict.txt"
    audio_format = "flac"

    print(args.train_dirs)

    train_set = DataSet(
        "train", [os.path.join(raw_data_path, t) for t in args.train_dirs],
    )
    valid_set = DataSet(
        "valid", [os.path.join(raw_data_path, p) for p in ["dev-clean", "dev-other"]],
    )
    test_set_clean = DataSet("test-clean", [os.path.join(raw_data_path, "test-clean")],)
    test_set_other = DataSet("test-other", [os.path.join(raw_data_path, "test-other")],)
    if not os.path.isdir(output_path) or args.overwrite:
        os.makedirs(output_path, exist_ok=True)
        build_vocabulary(
            (t for _, t in build_id_text_generator(train_set.audio_folders)),
            dictionary_file,
        )

    for ds in [train_set, valid_set, test_set_clean, test_set_other]:
        output_file = os.path.join(output_path, ds.name + ".json")

        id_texts_g = build_id_text_generator(ds.audio_folders)
        id2text = {k: v for k, v in id_texts_g}

        create_prepared_data_json(
            id2text,
            dictionary_file,
            output_file,
            ds.audio_folders,
            audio_format,
            args.num_processes,
        )
