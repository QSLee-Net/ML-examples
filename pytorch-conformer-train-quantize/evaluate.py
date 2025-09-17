#
# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchaudio.datasets import LIBRISPEECH
import sentencepiece as spm

# ---------------------------------------------------------------------
#  Import the components we need from the original training script
# ---------------------------------------------------------------------
import train                                  # assumes train.py is in the same directory
from train import (
    create_model,
    AudioPreprocessor,
    collate_eval_factory,
    evaluate,
)

# ---------------------------------------------------------------------
#  Argument parsing
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Conformer evaluation only")
    p.add_argument("--root", type=str, default="/shared/LIBRISPEECH",
                   help="LibriSpeech root directory")
    p.add_argument("--set", type=str, default="test-clean",
                   help="Comma-separated LibriSpeech subset names (e.g. test-clean,test-other)")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Batch size for evaluation")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader worker processes")
    p.add_argument("--sp-model", type=str,
                   default="tokenizer_out/librispeech_sp.model",
                   help="Path to the SentencePiece *.model used at training time")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a trained checkpoint (*.pt)")
    return p.parse_args()

# ---------------------------------------------------------------------
#  Main evaluation routine
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # -------- SentencePiece ----------
    sp = spm.SentencePieceProcessor()
    sp.load(args.sp_model)

    # expose the tokenizer inside the imported `train` module so that
    # train.int_to_text() works correctly during decoding
    train.sp = sp

    vocab_size = sp.get_piece_size() + 1      # +1 for CTC blank

    # -------- Model ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(vocab_size).to(device)

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)           # handles raw state_dict vs wrapper

    # tolerate DataParallel prefix differences
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys in checkpoint: {missing}")
    if unexpected:
        print(f"[warn] unexpected keys in checkpoint: {unexpected}")

    print(f"=> loaded weights from {ckpt_path}")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel …")
        model = torch.nn.DataParallel(model)

    loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    # -------- Datasets & evaluation ----------
    subsets = [s.strip() for s in args.set.split(",") if s.strip()]
    preproc = AudioPreprocessor(training=False)
    for subset in subsets:
        ds = LIBRISPEECH(args.root, url=subset, download=True)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_eval_factory(preproc),
            num_workers=args.num_workers,
        )

        wer, val_loss = evaluate(model, loader, device, loss_fn)
        print(f"\n── Results on {subset} ──")
        print(f"  • WER:  {wer:6.2f} %")
        print(f"  • CTC loss: {val_loss:8.4f}")

if __name__ == "__main__":
    main()