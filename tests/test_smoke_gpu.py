"""End-to-end smoke test of the REAL detector (loads the model, scores audio).

Opt-in: auto-skipped unless the trained model AND a sample clip are present locally
(they're gitignored), so the default suite stays fast/offline. Run the full thing on
a machine with the model:  pytest tests/test_smoke_gpu.py -v
"""
import glob
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_HAS_MODEL = bool(glob.glob(str(ROOT / "models" / "*" / "head.pt")))


def _sample(kind):
    pats = {"real": ["data/captured/*.wav", "data/real/*.wav"],
            "fake": ["data/captured_fake/*.wav", "data/fake/**/*.wav"]}[kind]
    for p in pats:
        hits = glob.glob(str(ROOT / p), recursive=True)
        if hits:
            return hits[0]
    return None


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not _HAS_MODEL,
                       reason="no trained model under models/*/head.pt (gitignored)"),
]


def test_scores_real_and_fake_end_to_end():
    real, fake = _sample("real"), _sample("fake")
    if not (real and fake):
        pytest.skip("no sample real/fake clips on disk")
    import detector
    for path in (real, fake):
        res = detector.score_bytes(Path(path).read_bytes())
        assert 0.0 <= res["p_fake"] <= 1.0
        assert res["verdict"] in ("real", "suspect", "fake")
        assert res["model_version"] == detector.MODEL_VERSION  # whatever model is configured


def test_score_clip_windows_a_long_file():
    fake = _sample("fake")
    if not fake:
        pytest.skip("no sample clip on disk")
    import detector
    res = detector.score_clip(Path(fake).read_bytes())
    assert res["n_windows"] >= 0
    if res["n_windows"]:
        assert 0.0 <= res["p_fake"] <= 1.0
