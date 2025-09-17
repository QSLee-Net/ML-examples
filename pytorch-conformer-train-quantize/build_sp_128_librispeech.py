#
# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

#!/usr/bin/env python3

import argparse
import os
import sys
import tempfile
from pathlib import Path

import torchaudio
from torchaudio.datasets import LIBRISPEECH

import sentencepiece as spm


def normalize_text(text: str, lowercase: bool = False) -> str:
    t = text.strip()
    if lowercase:
        t = t.lower()
    return t


def build_corpus(root: str, subset: str, lowercase: bool, limit: int | None) -> str:
    dataset = LIBRISPEECH(root=root, url=subset, download=True)
    n = len(dataset)
    if limit is not None:
        n = min(n, limit)

    tmp_fd, tmp_path = tempfile.mkstemp(prefix="librispeech_corpus_", suffix=".txt")
    os.close(tmp_fd)

    with open(tmp_path, "w", encoding="utf-8") as f:
        for idx in range(n):
            try:
                _, _, transcript, *_ = dataset[idx]
            except Exception as ex:
                print(f"Warning: failed to read sample {idx}: {ex}", file=sys.stderr)
                continue
            line = normalize_text(transcript, lowercase)
            if line:
                f.write(line + "\n")
    return tmp_path


def train_sentencepiece(
    corpus_path: str,
    output_dir: str,
    vocab_size: int,
    model_type: str,
    character_coverage: float,
    model_prefix: str,
    pad_id: int,
    disable_bos_eos: bool,
    seed_sentencepiece: int | None,
    input_sentence_size: int,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_prefix_path = str(Path(output_dir) / model_prefix)

    sp_args = [
        f"--input={corpus_path}",
        f"--model_prefix={model_prefix_path}",
        f"--vocab_size={vocab_size}",
        f"--model_type={model_type}",
        f"--character_coverage={character_coverage}",
        "--unk_id=0",
        "--input_sentence_size="+str(input_sentence_size),
        "--shuffle_input_sentence=true",
        "--hard_vocab_limit=true",
        "--num_threads=32",
    ]

    if disable_bos_eos:
        sp_args += ["--bos_id=-1", "--eos_id=-1"]
    else:
        sp_args += ["--bos_id=1", "--eos_id=2"]

    if pad_id is None or pad_id < 0:
        sp_args += ["--pad_id=-1"]
    else:
        sp_args += [f"--pad_id={pad_id}"]

    if seed_sentencepiece is not None:
        sp_args += [f"--seed_sentencepiece_size={seed_sentencepiece}"]

    spm.SentencePieceTrainer.Train(" ".join(sp_args))

    model_path = model_prefix_path + ".model"
    vocab_path = model_prefix_path + ".vocab"
    return model_path, vocab_path


def main():
    parser = argparse.ArgumentParser(description="Train a 128-token SentencePiece tokenizer on LibriSpeech using torchaudio.")
    parser.add_argument("--root", type=str, default="./data", help="Directory to store/lookup LibriSpeech.")
    parser.add_argument("--subset", type=str, default="train-clean-100",
                        choices=[
                            "train-clean-100", "train-clean-360", "train-other-500",
                            "dev-clean", "dev-other", "test-clean", "test-other"
                        ],
                        help="LibriSpeech subset to use.")
    parser.add_argument("--output_dir", type=str, default="./tokenizer_out", help="Where to write the tokenizer files.")
    parser.add_argument("--vocab_size", type=int, default=128, help="Total vocabulary size.")
    parser.add_argument("--model_type", type=str, default="unigram", choices=["unigram", "bpe"],
                        help="SentencePiece model type. 'unigram' tends to match English lists like yours.")
    parser.add_argument("--character_coverage", type=float, default=1.0,
                        help="Fraction of characters covered by the model (1.0 is fine for English).")
    parser.add_argument("--model_prefix", type=str, default="librispeech_sp",
                        help="Prefix (filename stem) for the trained model.")
    parser.add_argument("--lowercase", action="store_true", help="Lowercase transcripts before training.")
    parser.add_argument("--pad_id", type=int, default=-1, help="Pad ID; set to -1 to disable (default).")
    parser.add_argument("--disable_bos_eos", action="store_true", help="Disable BOS/EOS (recommended).")
    parser.add_argument("--enable_bos_eos", action="store_true", help="Enable BOS/EOS tokens.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of samples for a quick run.")
    parser.add_argument("--seed_sentencepiece_size", type=int, default=None,
                        help="Advanced: initial seed size for SentencePiece's sentence sampling (optional).")
    parser.add_argument("--input_sentence_size", type=int, default=1000000,
                        help="Number of sentences to sample during training.")

    args = parser.parse_args()

    disable_bos_eos = True
    if args.enable_bos_eos:
        disable_bos_eos = False
    if args.disable_bos_eos:
        disable_bos_eos = True

    print(f"Preparing corpus from LibriSpeech subset='{args.subset}'...")
    corpus_path = build_corpus(root=args.root, subset=args.subset, lowercase=args.lowercase, limit=args.limit)

    print(f"Training SentencePiece... corpus_path {corpus_path}")
    model_path, vocab_path = train_sentencepiece(
        corpus_path=corpus_path,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        model_prefix=args.model_prefix,
        pad_id=args.pad_id,
        disable_bos_eos=disable_bos_eos,
        seed_sentencepiece=args.seed_sentencepiece_size,
        input_sentence_size=args.input_sentence_size,
    )

    print("Done!")
    print(f"Model  : {model_path}")
    print(f"Vocab  : {vocab_path}")


if __name__ == "__main__":
    main()