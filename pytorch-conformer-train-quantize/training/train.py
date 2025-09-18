#
# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import sys
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import math

import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
from torchaudio.datasets import LIBRISPEECH

import sentencepiece as spm

# -----------------------------------------------------------------------------
# Make Conformer repo importable (assumes it is cloned at ./conformer/)
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "conformer"))
from conformer import model as conformer_model  # type: ignore


# -----------------------------------------------------------------------------
#  SentencePiece helpers
#    • blank = 0  (CTC)
#    • SentencePiece piece-ids are shifted by +1
# -----------------------------------------------------------------------------

BLANK = 0                                  # CTC blank id

def text_to_int(text: str) -> torch.Tensor:
    ids = sp.encode(text.lower(), out_type=int)
    # shift by +1 so blank stays at 0
    ids = [i + 1 for i in ids]
    return torch.tensor(ids, dtype=torch.long)

def int_to_text(token_ids: list[int]) -> str:
    # Collapse repeats & remove blank (0)
    pieces, prev = [], None
    for t in token_ids:
        if t == BLANK:
            prev = None
            continue
        if t == prev:              # ← add this
            continue
        raw = t - 1
        pieces.append(raw)
        prev = t
    return sp.decode(pieces)


# -----------------------------------------------------------------------------
# Data pre-processing  (LibriSpeech “heavy” SpecAugment + 3-way speed-perturb)
# -----------------------------------------------------------------------------
class AudioPreprocessor:
    """Mel-spectrogram front-end with heavy SpecAugment (f=27, t=10, p=0.05)."""

    def __init__(
            self,
            sample_rate: int = 16_000,
            n_mels: int = 80,
            n_fft: int = 512,
            hop_length: int = 160,
            freq_mask_param: int = 27,
            time_mask_ratio: float = 0.05,
            time_mask_count: int = 10,
            speeds: tuple[float, ...] = (0.9, 1.0, 1.1),
            training: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.speeds = speeds
        self.training = training
        self.time_mask_ratio = time_mask_ratio
        self.time_mask_count = time_mask_count

        # Feature extraction
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            center=False
        )

        # SpecAugment ops (iid_masks=True → independent per example)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param, iid_masks=False)
        self.time_mask = T.TimeMasking(time_mask_param=1, iid_masks=False, p=1.0)  # param set on-the-fly

    # -------------- helpers
    def _speed_perturb(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return waveform
        factor = random.choice(self.speeds)
        if factor == 1.0:
            return waveform
        new_sr = int(self.sample_rate * factor)
        wav = F.resample(waveform, self.sample_rate, new_sr)
        return F.resample(wav, new_sr, self.sample_rate)

    # -------------- main entry
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = self._speed_perturb(waveform)
        spec = self.mel_spec(waveform).squeeze(0).add_(1e-6).log().transpose(0, 1)
        if self.training:
            spec = self.freq_mask(spec)
            # dynamic time-mask length (ratio × sequence length)
            full_len = spec.size(0)
            tm_param = max(1, int(self.time_mask_ratio * full_len))
            self.time_mask.time_mask_param = tm_param
            for _ in range(self.time_mask_count):
                spec = self.time_mask(spec)
        return spec

# -----------------------------------------------------------------------------
# Model factory (architecture *not* exposed — fixed to Conformer‑S)
# -----------------------------------------------------------------------------

def create_model(vocab_size) -> nn.Module:
    return conformer_model.Conformer(
        num_classes=vocab_size,
        input_dim=80,
        encoder_dim=144,
        num_encoder_layers=16,
        num_attention_heads=4,
        feed_forward_expansion_factor=4,
        conv_expansion_factor=2,
        input_dropout_p=0.1,
        feed_forward_dropout_p=0.1,
        attention_dropout_p=0.1,
        conv_dropout_p=0.1,
        conv_kernel_size=31,
        half_step_residual=True,
    )


# -----------------------------------------------------------------------------
# Training utilities
# -----------------------------------------------------------------------------
def greedy_decode(logits: torch.Tensor, lens: torch.Tensor) -> list[str]:
    best = logits.argmax(-1)                 # (B, max_T)
    transcripts = []
    for i, seq in enumerate(best):
        valid = seq[: int(lens[i])]          # trim padding
        transcripts.append(int_to_text(valid.tolist()))
    return transcripts


def _collate_common(batch, preprocessor, include_text: bool):
    feats, feat_lens, tgts, tgt_lens, txts = [], [], [], [], []
    for waveform, _, transcript, *_ in batch:
        # 1) feature + length
        f = preprocessor(waveform)
        feats.append(f); feat_lens.append(len(f))
        # 2) tokenized target + length
        t = text_to_int(transcript)
        tgts.append(t); tgt_lens.append(len(t))
        # 3) optionally store the raw transcript
        if include_text:
            txts.append(transcript.lower())

    # pad to a batch
    feats   = pad_sequence(feats,   batch_first=True)
    tgts    = pad_sequence(tgts,    batch_first=True)
    feat_lens = torch.tensor(feat_lens)
    tgt_lens  = torch.tensor(tgt_lens)

    if include_text:
        return feats, feat_lens, tgts, tgt_lens, txts
    else:
        return feats, feat_lens, tgts, tgt_lens


def collate_factory(preprocessor: AudioPreprocessor):
    return lambda batch: _collate_common(batch, preprocessor, include_text=False)


def collate_eval_factory(preprocessor: AudioPreprocessor):
    return lambda batch: _collate_common(batch, preprocessor, include_text=True)


@torch.no_grad()
def evaluate(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        loss_fn: nn.Module,
) -> tuple[float, float]:
    """Return (WER %, mean validation loss)."""
    model.eval()
    total_edits = total_words = 0
    val_loss_sum = 0.0
    val_steps = 0
    first_batch_done = False
    for feats, feat_lens, tgts, tgt_lens, refs in tqdm(loader, leave=False):
        feats     = feats.to(device, non_blocking=True)
        feat_lens = feat_lens.to(device, non_blocking=True)
        tgts      = tgts.to(device, non_blocking=True)
        tgt_lens  = tgt_lens.to(device, non_blocking=True)

        logits, logit_lens = model(feats, feat_lens)

        vloss = loss_fn(
            logits.transpose(0, 1),
            tgts, logit_lens, tgt_lens
        )
        val_loss_sum += vloss.item()
        val_steps += 1

        hyps = greedy_decode(logits.cpu(), logit_lens.cpu())

        if not first_batch_done:
            print("\n── Decoding debug (first batch) ──")
            for ref, hyp in list(zip(refs, hyps)):
                print(f"REF: {ref.lower()}", f"\nHYP: {hyp}")
            first_batch_done = True

        for ref, hyp in zip(refs, hyps):
            total_edits += F.edit_distance(ref.split(), hyp.split())
            total_words += len(ref.split())

    wer = 100 * total_edits / total_words if total_words else 0.0
    val_loss = val_loss_sum / max(val_steps, 1)
    return wer, val_loss

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Conformer on LibriSpeech")
    # === dataset ===
    parser.add_argument("--root", type=str, default="/shared/LIBRISPEECH", help="LibriSpeech root directory")
    parser.add_argument(
        "--train-sets",
        type=str,
        default="train-clean-100,train-clean-360,train-other-500",
        help="Comma‑separated subset names for training",
    )
    parser.add_argument("--valid-set", type=str, default="dev-clean", help="Validation subset name")
    # === training hyper‑params ===
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Mini-batch per gradient update")
    parser.add_argument("--lr", type=float, default=4e-4,
                        help="Peak LR after warm-up")
    parser.add_argument("--save-dir", type=str, default="/shared/conformer/checkpoints",
                        help="Where to write epoch checkpoints")
    parser.add_argument(
        "--betas",
        type=str,
        default="0.9,0.999",
        help="Adam betas as a comma‑separated pair",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay")
    parser.add_argument("--warmup-epochs", type=float, default=1.0, help="Noam LR warm‑up steps")
    parser.add_argument("--grad-clip", type=float, default=5.0, help="Gradient clipping threshold (L2 norm)")
    # --- gradient-accumulation ---
    parser.add_argument(
        "--accum-steps",
        type=int,
        default=1,
        help="How many mini-batches to accumulate gradients over before "
             "performing an optimizer update (≃ effective batch-size multiplier)",
    )
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers for training")
    parser.add_argument("--val-num-workers", type=int, default=4, help="DataLoader workers for validation")
    # === data augmentation / front‑end ===
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--time-mask-ratio", type=float, default=0.05)
    parser.add_argument(
        "--speeds",
        type=str,
        default="0.9,1.0,1.1",
        help="Comma‑separated speed perturb factors",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="If set, disables SpecAugment & speed perturb during training"
    )
    parser.add_argument(
        "--sp-model",
        type=str,
        default="tokenizer_out/librispeech_sp.model",
        help="Path to a SentencePiece model (*.model)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint (*.pt) to load model weights from")

    parser.add_argument("--freq-mask-param", type=int, default=27)
    parser.add_argument("--time-mask-count", type=int, default=10)

    return parser.parse_args()

# -----------------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print(args)

    train_sets = [s.strip() for s in args.train_sets.split(",") if s.strip()]
    speed_factors = tuple(float(s) for s in args.speeds.split(","))
    beta1, beta2 = (float(x) for x in args.betas.split(","))

    # =============== datasets ===============
    train_set = torch.utils.data.ConcatDataset(
        [LIBRISPEECH(args.root, url=u, download=True) for u in train_sets]
    )
    valid_set = LIBRISPEECH(args.root, url=args.valid_set, download=True)

    # =============== preprocessors ===============
    preprocessor_train = AudioPreprocessor(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        freq_mask_param=args.freq_mask_param,
        time_mask_ratio=args.time_mask_ratio,
        time_mask_count=args.time_mask_count,
        speeds=speed_factors,
        training=not args.no_augment,
    )

    preprocessor_eval = AudioPreprocessor(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        freq_mask_param=args.freq_mask_param,      # match train spec
        time_mask_ratio=args.time_mask_ratio,
        time_mask_count=args.time_mask_count,      # harmless when training=False
        speeds=speed_factors,
        training=False,
    )

    global sp
    sp = spm.SentencePieceProcessor()
    sp.load(args.sp_model)

    print(f"Tokens count of SentencePieceProcessor :{sp.get_piece_size()}")
    vocab_size = sp.get_piece_size() + 1

    # helper that returns a function capturing the model path
    def build_spm_loader(sp_model_path: str):
        def _init_spm_worker(_worker_id: int):
            """
            Runs once in **each** worker process.

            Creates an independent SentencePieceProcessor and stores
            it in a module-level variable that is *local to the worker*.
            """
            global sp
            sp = spm.SentencePieceProcessor()   # brand-new handle
            sp.load(sp_model_path)
        return _init_spm_worker

    # =============== loaders ===============
    spm_worker_init = build_spm_loader(args.sp_model)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_factory(preprocessor_train),
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=spm_worker_init,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_eval_factory(preprocessor_eval),
        num_workers=args.val_num_workers,
        worker_init_fn=spm_worker_init,
    )

    # =============== model & optim ===============
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # +1 for CTC blank
    model = create_model(vocab_size).to(device)

    # ----- load model weights from checkpoint, if requested -----
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt.get("model", ckpt)
        # handle DataParallel vs single-GPU
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state)
        else:
            model.load_state_dict(state)
        print(f"=> Loaded model weights from {args.checkpoint}")


    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel …")
        model = torch.nn.DataParallel(model)

    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(beta1, beta2),
        weight_decay=args.weight_decay,
        eps=1e-9
    )

    # ---- LR schedule: warm-up → linear decay (counts optimiser-updates) ----
    steps_per_epoch   = len(train_loader)
    updates_per_epoch = math.ceil(steps_per_epoch / args.accum_steps)
    total_updates     = updates_per_epoch * args.epochs
    warm_updates      = int(args.warmup_epochs * updates_per_epoch)

    def lr_lambda_linear(step: int) -> float:
        if step < warm_updates:
            return step / warm_updates
        return max((total_updates - step) / (total_updates - warm_updates), 0.)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_linear)

    # =============== training loop ===============
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

        optimizer.zero_grad(set_to_none=True)

        for i, (feats_i, feat_lens_i, tgts_i, tgt_lens_i) in enumerate(pbar, 1):
            feats     = feats_i.to(device, non_blocking=True)
            feat_lens = feat_lens_i.to(device, non_blocking=True)
            tgts      = tgts_i.to(device, non_blocking=True)
            tgt_lens  = tgt_lens_i.to(device, non_blocking=True)

            logits, logit_lens = model(feats, feat_lens)
            loss = ctc_loss(logits.transpose(0, 1), tgts, logit_lens, tgt_lens)

            running_loss += loss.item()                          # accumulate raw loss
            (loss / args.accum_steps).backward()                 # scale for accumulation

            # perform optimiser update every `accum_steps` mini-batches
            if i % args.accum_steps == 0 or i == steps_per_epoch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr":   f"{current_lr:.3e}",
                "upd":  f"{(i-1)//args.accum_steps+1:>4d}/{updates_per_epoch}",
            })

        # ---------- aggregate epoch metrics ----------
        train_loss = running_loss / steps_per_epoch
        wer_val, val_loss = evaluate(model, valid_loader, device, ctc_loss)

        print(
            f"\nEpoch {epoch} -- "
            f"train loss (avg through epoch): {train_loss:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"{args.valid_set} WER: {wer_val:.2f}% | "
            f"last LR {current_lr:.3e}"
        )

        ckpt_path = Path(args.save_dir) / f"epoch_{epoch:03d}.pt"
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(
            {
                "wer_val": wer_val,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch,
                "model": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            },
            ckpt_path,
        )


if __name__ == "__main__":
    main()