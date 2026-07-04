"""
train_detector.py — fine-tune our own detector to catch MODERN fake voices.

Runs in the detector env (.venv):

    python src/train_detector.py                 # sensible defaults
    python src/train_detector.py --epochs 5 --batch 8 --lr 1e-5

Idea: start from the ASVspoof-trained model (already ~98% on old attacks) and
continue training on a mix that ADDS modern XTTS-v2 clones. Keeping the old
ASVspoof spoof clips in the mix prevents it from forgetting old attacks while it
learns the new ones. Trains only the transformer + head (the conv feature encoder
is frozen — standard, and keeps us inside 8 GB).

Trains on split=="train" only. The test split (held-out speakers + unseen modern
clones) is untouched here and judged later by eval_detector.py.

Label convention (matches the base model's id2label {0:fake, 1:real}):
    fake -> 0,  real -> 1.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

DSET_CSV = config.DATA / "dataset.csv"
OUT_DIR = config.ROOT / "models" / "sonave_v1"   # v1: trained on XTTS + YourTTS
MAX_SEC = 4.0
MAX_LEN = int(MAX_SEC * config.SAMPLE_RATE)   # 64000 samples


class ClipDataset:
    """Loads a wav, crops/pads to a fixed 4 s window. Random crop for train."""

    def __init__(self, df: pd.DataFrame, train: bool):
        self.rows = df.reset_index(drop=True)
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import librosa
        r = self.rows.iloc[i]
        wav, _ = librosa.load(str(config.ROOT / r["path"]),
                              sr=config.SAMPLE_RATE, mono=True)
        wav = self._fit(wav)
        y = 0 if r["label"] == "fake" else 1
        return wav.astype(np.float32), y

    def _fit(self, wav: np.ndarray) -> np.ndarray:
        if len(wav) >= MAX_LEN:
            if self.train:
                start = np.random.randint(0, len(wav) - MAX_LEN + 1)
            else:
                start = (len(wav) - MAX_LEN) // 2
            return wav[start:start + MAX_LEN]
        return np.pad(wav, (0, MAX_LEN - len(wav)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    df = pd.read_csv(DSET_CSV)
    train_df = df[df["split"] == "train"].copy()
    print(f"train clips: {len(train_df)}  "
          f"({(train_df.label=='real').sum()} real / "
          f"{(train_df.label=='fake').sum()} fake; "
          f"modern={ (train_df.kind=='modern').sum()}, old={(train_df.kind=='old').sum()})")

    extractor = AutoFeatureExtractor.from_pretrained(config.DETECTOR_HF_MODEL)
    model = AutoModelForAudioClassification.from_pretrained(
        config.DETECTOR_HF_MODEL).to(device)

    # Freeze the conv feature encoder; fine-tune transformer + classifier head.
    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()

    def collate(batch):
        wavs = [b[0] for b in batch]
        ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
        inp = extractor(wavs, sampling_rate=config.SAMPLE_RATE,
                        return_tensors="pt", padding=True)
        return inp, ys

    loader = DataLoader(ClipDataset(train_df, train=True), batch_size=args.batch,
                        shuffle=True, collate_fn=collate, num_workers=0)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    model.train()
    for epoch in range(args.epochs):
        running, correct, seen = 0.0, 0, 0
        for step, (inp, ys) in enumerate(loader):
            inp = {k: v.to(device) for k, v in inp.items()}
            ys = ys.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                out = model(**inp, labels=ys)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * len(ys)
            correct += (out.logits.argmax(-1) == ys).sum().item()
            seen += len(ys)
            if step % 20 == 0:
                print(f"  epoch {epoch+1} step {step}/{len(loader)} "
                      f"loss {loss.item():.3f}", flush=True)
        print(f"epoch {epoch+1}: loss {running/seen:.3f}  "
              f"train_acc {correct/seen:.3f}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    extractor.save_pretrained(OUT_DIR)
    print(f"\nSaved fine-tuned detector -> {OUT_DIR}")
    print("Next: python src/eval_detector.py")


if __name__ == "__main__":
    main()
