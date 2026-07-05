"""The tri-state verdict policy and result shaping in service/detector.py — the
thresholds the whole product's real/suspect/fake calls hinge on. Pure functions;
the model itself is never loaded."""
import detector


def test_verdict_tristate_boundaries():
    assert detector.verdict(0.0) == "real"
    assert detector.verdict(detector.TAU_REAL - 1e-6) == "real"
    assert detector.verdict(detector.TAU_REAL) == "suspect"
    assert detector.verdict(detector.TAU_FAKE - 1e-6) == "suspect"
    assert detector.verdict(detector.TAU_FAKE) == "fake"
    assert detector.verdict(1.0) == "fake"


def test_result_shape_and_fields():
    r = detector._result(0.9)
    assert r["verdict"] == "fake"
    assert r["p_fake"] == 0.9
    assert 0.0 <= r["confidence"] <= 1.0
    assert r["model_version"] == detector.MODEL_VERSION


def test_confidence_is_bounded_at_extremes():
    assert 0.0 <= detector._result(0.0)["confidence"] <= 1.0
    assert 0.0 <= detector._result(1.0)["confidence"] <= 1.0


def test_thresholds_are_ordered():
    assert 0.0 < detector.TAU_REAL < detector.TAU_FAKE <= 1.0
