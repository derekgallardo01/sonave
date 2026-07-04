"""
Sonave Phase 0 — central config.

One place for every path, knob, and default so the scripts stay boring and
reproducible. Import this from every src/ script; don't hardcode paths elsewhere.
"""
from __future__ import annotations

from pathlib import Path

# --- Repo layout -------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
REAL_DIR = DATA / "real"                 # genuine human speech (WAV, 16 kHz mono)
FAKE_XTTS_DIR = DATA / "fake" / "xtts"   # locally generated voice clones
FAKE_ITW_DIR = DATA / "fake" / "itw"     # In-the-Wild fake slice
FAKE_ASV_DIR = DATA / "fake" / "asv"     # ASVspoof 2019 LA eval spoof slice
COMPRESSED_DIR = DATA / "compressed"     # Opus round-tripped copies, per bitrate
MANIFEST = DATA / "manifest.csv"         # the single source of truth for samples

RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"
METRICS_CSV = RESULTS / "metrics.csv"
FINDINGS = RESULTS / "findings.md"

# Where big downloads land (kept out of the repo via .gitignore).
DOWNLOADS = DATA / "_downloads"

# --- Audio -------------------------------------------------------------------
# The detector and the whole pipeline standardize on 16 kHz mono. LibriSpeech and
# In-the-Wild are already 16 kHz; XTTS output (24 kHz) gets resampled down.
SAMPLE_RATE = 16_000
CHANNELS = 1

# --- Compression sweep -------------------------------------------------------
# Google Meet / WebRTC voice is Opus, mono, ~16-40 kbps. We sweep three points
# plus an uncompressed control. "control" is a no-op copy so the pipeline's own
# fidelity can be sanity-checked against the clean baseline.
OPUS_BITRATES = ["16k", "24k", "32k"]
CONTROL = "control"                      # uncompressed passthrough
CONDITIONS = [CONTROL] + OPUS_BITRATES   # everything evaluate.py iterates over

# --- Dataset sizing ----------------------------------------------------------
# Balanced ~300 real / ~300 fake, split across two tracks:
#   controlled  = LibriSpeech real  vs  XTTS clones of the same speakers
#   benchmark   = In-the-Wild real  vs  In-the-Wild fake
N_LIBRI_REAL = 150      # real, controlled track
N_XTTS_FAKE = 150       # fake, controlled track (cloned from the libri reals)
N_ITW_REAL = 150        # real, benchmark track
N_ITW_FAKE = 150        # fake, benchmark track

# In-distribution track: ASVspoof 2019 LA eval — the detector's home turf, where
# clean EER should be LOW. This is the track that can actually test the thesis
# (a strong clean baseline that compression can degrade), which the OOD XTTS/ITW
# tracks could not. See results/findings.md for why.
N_ASV_REAL = 150        # bonafide, in-distribution track
N_ASV_FAKE = 150        # spoof, in-distribution track

TRACK_CONTROLLED = "controlled"
TRACK_BENCHMARK = "benchmark"
TRACK_INDIST = "asvspoof"

# --- Detector ----------------------------------------------------------------
# Primary: a wav2vec2 anti-spoofing classifier trained on ASVspoof (SSL front-end).
# score_wav() returns P(fake) in [0, 1] (higher = more likely fake).
#
# Detector chosen by empirical shootout (scratchpad/shootout.py) on our own clips:
# the obvious first pick, MelodyMachine/Deepfake-audio-detection-V2, turned out to
# be overfit to its training data — it scored clean LibriSpeech AND XTTS clones at
# P(fake)=0.000 (EER ~46% controlled / ~34% benchmark), i.e. no real generalization.
# The ASVspoof-trained model below separated our out-of-distribution clips far
# better (clean EER ~7.5% controlled / ~15% benchmark), giving a credible baseline
# to measure the compression drop from. Both use id2label {0:'fake', 1:'real'}.
DETECTOR_HF_MODEL = "Bisher/wav2vec2_ASV_deepfake_audio_detection"

# --- Reproducibility ---------------------------------------------------------
SEED = 1337


def compressed_path(original_rel_path: str, condition: str) -> Path:
    """
    Map an original clip + a condition to its audio file for scoring.

    The uncompressed CONTROL condition simply points back at the original file
    (no copy), so the pipeline can prove it adds no degradation of its own. Every
    Opus bitrate lives under data/compressed/<bitrate>/<basename>.wav. Basenames
    are globally unique (libri_/xtts_/itw_ prefixes), so a flat per-bitrate dir
    is unambiguous. Both compress.py and evaluate.py call this — single source of
    truth for the layout.
    """
    if condition == CONTROL:
        return ROOT / original_rel_path
    return COMPRESSED_DIR / condition / Path(original_rel_path).name


def ensure_dirs() -> None:
    """Create every directory the pipeline writes into. Safe to call repeatedly."""
    for d in (
        REAL_DIR,
        FAKE_XTTS_DIR,
        FAKE_ITW_DIR,
        FAKE_ASV_DIR,
        COMPRESSED_DIR,
        DOWNLOADS,
        RESULTS,
        PLOTS,
    ):
        d.mkdir(parents=True, exist_ok=True)
    for cond in OPUS_BITRATES:
        (COMPRESSED_DIR / cond).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print("Sonave Phase 0 config")
    print(f"  ROOT         = {ROOT}")
    print(f"  SAMPLE_RATE  = {SAMPLE_RATE} Hz, {CHANNELS} ch")
    print(f"  CONDITIONS   = {CONDITIONS}")
    print(f"  detector     = {DETECTOR_HF_MODEL}")
    print(f"  target set   = {N_LIBRI_REAL + N_ITW_REAL} real / "
          f"{N_XTTS_FAKE + N_ITW_FAKE} fake")
    print("  dirs ensured OK")
