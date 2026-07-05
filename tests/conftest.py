"""Shared test fixtures. Loads the two FastAPI apps (railway/app.py and service/app.py)
as distinct modules (both are named `app`, so we load by path to avoid collision) and
keeps everything offline — no GPU, no network, no model downloads."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "service", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def load_module(name: str, relpath: str):
    """Import a .py file under a unique module name (avoids app.py name collisions)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def railway_mod(monkeypatch):
    """The Railway capture service, loaded fresh with a clean env (no scorer)."""
    monkeypatch.setenv("SONAVE_SCORER_URL", "")
    monkeypatch.setenv("SONAVE_RECALL_API_KEY", "test-key")
    mod = load_module("rwapp", "railway/app.py")
    mod.QUALITY.clear()
    mod.VERDICTS.clear()
    mod.ROLL.clear()
    return mod


def pcm16(samples):
    """float [-1,1] list/array -> S16LE bytes, as Recall/WS delivers."""
    import array
    a = array.array("h", [max(-32768, min(32767, int(x * 32767))) for x in samples])
    return a.tobytes()
