"""
train_xlsr.py — train the XLS-R + SLS detector (frozen backbone, trainable head).

Runs in the detector env (.venv):

    python src/train_xlsr.py                         # trains on data/dataset.csv (existing)
    python src/train_xlsr.py --manifest data/corpus.csv --out models/sonave_xlsr_corpus

Only the small SLS head trains; XLS-R-300M stays frozen (fits 8 GB, fast). This is
the "better brain" lever — same data as v1, stronger backbone — so the In-the-Wild
number isolates the effect of the backbone+head upgrade.

Labels: fake = 1, real = 0.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import model_sls  # noqa: E402


class ClipSet:
    def __init__(self, df: pd.DataFrame, augment: bool = False):
        self.rows = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import librosa
        r = self.rows.iloc[i]
        wav, _ = librosa.load(str(config.ROOT / r["path"]), sr=model_sls.SR, mono=True)
        wav = model_sls.fit_length(wav, train=True)
        if self.augment and np.random.random() < 0.5:
            # Degrade only HALF the clips (both real AND fake), so the model learns
            # fakeness through the channel — "noisy != fake" — while still seeing
            # plenty of clean audio to keep its sharp clean-audio detection.
            import augment as aug
            wav = model_sls.fit_length(aug.augment(wav), train=True)
        y = 1 if r["label"] == "fake" else 0
        return wav.astype(np.float32), y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(config.DATA / "dataset.csv"))
    ap.add_argument("--out", default=str(config.ROOT / "models" / "sonave_xlsr"))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)   # head-only -> higher LR is fine
    ap.add_argument("--augment", action="store_true",
                    help="apply real-call degradation to training clips")
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    df = pd.read_csv(args.manifest)
    train_df = df[df["split"] == "train"].copy()
    n_fake = int((train_df.label == "fake").sum())
    n_real = int((train_df.label == "real").sum())
    print(f"train: {len(train_df)} clips ({n_real} real / {n_fake} fake)")

    model = model_sls.SLSDetector().to(device)

    def collate(batch):
        wavs = [b[0] for b in batch]
        ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
        inp = model_sls.make_inputs(wavs, device)
        return inp, ys

    print(f"augment (real-call degradation): {args.augment}")
    loader = DataLoader(ClipSet(train_df, augment=args.augment), batch_size=args.batch,
                        shuffle=True, collate_fn=collate, num_workers=0)

    # class weights (mild imbalance)
    w = torch.tensor([len(train_df) / (2 * n_real), len(train_df) / (2 * n_fake)],
                     dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.epochs):
        model.head.train()
        run, correct, seen = 0.0, 0, 0
        for step, (inp, ys) in enumerate(loader):
            ys = ys.to(device)
            logits = model(**inp)
            loss = loss_fn(logits, ys)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item() * len(ys)
            correct += (logits.argmax(-1) == ys).sum().item()
            seen += len(ys)
            if step % 30 == 0:
                print(f"  epoch {epoch+1} step {step}/{len(loader)} "
                      f"loss {loss.item():.3f}", flush=True)
        print(f"epoch {epoch+1}: loss {run/seen:.3f} train_acc {correct/seen:.3f}",
              flush=True)

    out = Path(args.out)
    model.save(out)
    print(f"\nSaved SLS head -> {out}")
    print("Next: python src/eval_xlsr.py --model", out)


if __name__ == "__main__":
    main()
