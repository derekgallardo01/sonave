"""fit_length — the crop/pad that every clip passes through before the backbone.
Off-by-one here silently corrupts every score, so pin its exact behavior."""
import numpy as np

import model_sls


def test_pads_short_clips_to_max_len():
    out = model_sls.fit_length(np.ones(1000, dtype=np.float32), train=False)
    assert len(out) == model_sls.MAX_LEN
    assert np.all(out[:1000] == 1.0) and np.all(out[1000:] == 0.0)  # zero-padded tail


def test_exact_length_passthrough():
    w = np.arange(model_sls.MAX_LEN, dtype=np.float32)
    out = model_sls.fit_length(w, train=False)
    assert len(out) == model_sls.MAX_LEN


def test_eval_uses_center_crop():
    w = np.arange(model_sls.MAX_LEN * 2, dtype=np.float32)
    out = model_sls.fit_length(w, train=False)
    start = (len(w) - model_sls.MAX_LEN) // 2
    assert len(out) == model_sls.MAX_LEN
    assert out[0] == start  # deterministic center crop


def test_train_crop_in_bounds():
    w = np.arange(model_sls.MAX_LEN * 3, dtype=np.float32)
    for _ in range(20):
        out = model_sls.fit_length(w, train=True)
        assert len(out) == model_sls.MAX_LEN
        assert 0 <= out[0] <= len(w) - model_sls.MAX_LEN
