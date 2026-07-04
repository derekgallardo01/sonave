"""
model_sls.py — the stronger detector: XLS-R-300M backbone + SLS-style head.

Why: our v0/v1 used wav2vec2-base (95M), which capped generalization at ~60% catch
on In-the-Wild. The field's strong recipe is XLS-R-300M (multilingual SSL) with a
head that reads ALL transformer layers, not just the last — the "Selective Layer
Summarization" (SLS) idea. Different layers of XLS-R capture different spoof cues;
summarizing across them generalizes far better to unseen fakes.

Compute fit (8 GB): the 300M backbone is FROZEN (no_grad, no optimizer state); only
the small head trains. That keeps memory low and training fast.

Label convention here: fake = 1, real = 0. score_paths() returns P(fake) in [0,1].
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

BACKBONE = "facebook/wav2vec2-xls-r-300m"
SR = 16_000
MAX_LEN = 4 * SR   # 4 s crop, matches the rest of the pipeline


class SLSHead(nn.Module):
    """Weighted sum over layers -> attentive statistics pooling over time -> 2-class."""

    def __init__(self, n_layers: int, dim: int, hidden: int = 256, p_drop: float = 0.3):
        super().__init__()
        self.layer_w = nn.Parameter(torch.zeros(n_layers))   # softmax across layers
        self.attn = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.cls = nn.Sequential(
            nn.Linear(dim * 2, hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, 2),
        )

    def forward(self, hs: torch.Tensor) -> torch.Tensor:   # hs: [B, L, T, D]
        w = torch.softmax(self.layer_w, dim=0).view(1, -1, 1, 1)
        x = (hs * w).sum(dim=1)                             # [B, T, D]
        a = torch.softmax(self.attn(x), dim=1)             # [B, T, 1]
        mean = (a * x).sum(dim=1)                           # [B, D]
        var = (a * (x - mean.unsqueeze(1)) ** 2).sum(dim=1).clamp_min(1e-6)
        feat = torch.cat([mean, var.sqrt()], dim=-1)        # [B, 2D]
        return self.cls(feat)                               # [B, 2]


class SLSDetector(nn.Module):
    def __init__(self, backbone: str = BACKBONE):
        super().__init__()
        from transformers import Wav2Vec2Model
        self.backbone_name = backbone
        self.backbone = Wav2Vec2Model.from_pretrained(backbone)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        n_layers = self.backbone.config.num_hidden_layers + 1   # +1 for embeddings
        self.head = SLSHead(n_layers, self.backbone.config.hidden_size)

    def forward(self, input_values, attention_mask=None) -> torch.Tensor:
        with torch.no_grad():
            out = self.backbone(input_values, attention_mask=attention_mask,
                                output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=1)          # [B, L, T, D]
        return self.head(hs)

    # --- persistence: only the tiny head is saved ---
    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.head.state_dict(), out_dir / "head.pt")
        (out_dir / "meta.json").write_text(
            json.dumps({"backbone": self.backbone_name}), encoding="utf-8")

    @classmethod
    def load(cls, out_dir: Path, device: str = "cuda"):
        meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
        m = cls(meta["backbone"]).to(device)
        m.head.load_state_dict(torch.load(out_dir / "head.pt", map_location=device))
        m.eval()
        return m


# --- shared audio + scoring helpers -----------------------------------------
_EXTRACTOR = None


def _extractor():
    global _EXTRACTOR
    if _EXTRACTOR is None:
        from transformers import AutoFeatureExtractor
        _EXTRACTOR = AutoFeatureExtractor.from_pretrained(BACKBONE)
    return _EXTRACTOR


def fit_length(wav: np.ndarray, train: bool) -> np.ndarray:
    """Crop/pad to MAX_LEN; random crop for training, center crop for eval."""
    if len(wav) >= MAX_LEN:
        start = np.random.randint(0, len(wav) - MAX_LEN + 1) if train else (len(wav) - MAX_LEN) // 2
        return wav[start:start + MAX_LEN]
    return np.pad(wav, (0, MAX_LEN - len(wav)))


def make_inputs(wavs: list[np.ndarray], device: str):
    inp = _extractor()(wavs, sampling_rate=SR, return_tensors="pt", padding=True)
    return {k: v.to(device) for k, v in inp.items()}


@torch.no_grad()
def score_paths(model: "SLSDetector", paths, device: str = "cuda",
                batch: int = 8) -> np.ndarray:
    """Return P(fake) in [0,1] for each wav path."""
    import librosa
    model.eval()
    scores = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        wavs = [fit_length(librosa.load(str(p), sr=SR, mono=True)[0], train=False)
                for p in chunk]
        inp = make_inputs(wavs, device)
        probs = torch.softmax(model(**inp), dim=-1)[:, 1]   # P(fake=1)
        scores.extend(probs.detach().cpu().numpy().tolist())
    return np.array(scores)
