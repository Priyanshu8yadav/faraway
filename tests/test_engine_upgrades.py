"""
Synthetic verification of the upgraded motion + forecasting layers.

Runs WITHOUT the YOLO weights or demo videos — it feeds hand-built frames and
signals so every claim about the new methods is checkable on any machine.

    python -m pytest tests/test_engine_upgrades.py -v
    (or just:  python tests/test_engine_upgrades.py)
"""
import sys
from pathlib import Path

# Make the repo root importable no matter what directory pytest/python is run
# from — otherwise `import dashboard` fails with ModuleNotFoundError.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2

from dashboard.ca_crs_engine import ZoneProcessor, V_MAX
from dashboard.forecasting import ZoneForecaster


def _bare_processor():
    """A ZoneProcessor with no video source (bypass __init__, set flow state)."""
    p = object.__new__(ZoneProcessor)
    p._prev_gray = None
    return p


def _texture(seed=0, size=256):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (size, size), dtype=np.uint8)
    # blur so Farneback has smooth gradients to track
    return cv2.GaussianBlur(img, (7, 7), 0)


def _shift(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


def test_static_scene_reads_zero_speed():
    p = _bare_processor()
    g = _texture()
    p._flow_features(g)              # prime prev_gray
    s, c = p._flow_features(g)       # identical frame → no motion
    assert s < 0.02, f"static scene should read ~0 speed, got {s}"


def test_uniform_motion_low_conflict():
    """Everyone moving the same way → speed>0, conflict≈0 (aligned flow)."""
    p = _bare_processor()
    g0 = _texture(seed=1)
    g1 = _shift(g0, dx=6, dy=0)      # whole scene drifts right by 6 px
    p._flow_features(g0)
    s, c = p._flow_features(g1)
    assert s > 0.1, f"moving scene should register speed, got {s}"
    assert c < 0.25, f"aligned flow should be LOW conflict, got {c}"


def test_opposing_flow_high_conflict():
    """Left half →right, right half →left: true counter-flow → conflict≈high.
    This is the exact case the old IQR metric got wrong."""
    p = _bare_processor()
    g0 = _texture(seed=2)
    left  = _shift(g0[:, :128], dx=6, dy=0)
    right = _shift(g0[:, 128:], dx=-6, dy=0)
    g1 = np.hstack([left, right])
    p._flow_features(g0)
    s, c = p._flow_features(g1)
    assert c > 0.5, f"opposing flow should be HIGH conflict, got {c}"


def test_conflict_beats_old_iqr_on_5050_split():
    """Sanity: circular variance is high for balanced counter-flow — the split
    where the paper says 'fraction opposing' degenerates."""
    p = _bare_processor()
    g0 = _texture(seed=3)
    left  = _shift(g0[:, :128], dx=5, dy=0)
    right = _shift(g0[:, 128:], dx=-5, dy=0)
    g1 = np.hstack([left, right])
    p._flow_features(g0)
    _, c = p._flow_features(g1)
    assert c > 0.5


def test_forecaster_emits_eta_when_rising_toward_danger():
    """A steadily rising CRS should yield a finite ETA before it hits 0.70."""
    fc = ZoneForecaster()
    result = None
    # ramp 0.30 → 0.66 over 30 ticks at a realistic 8 FPS (dt=0.125s)
    for crs in np.linspace(0.30, 0.66, 30):
        result = fc.update(float(crs), dt=0.125)
    assert result.direction == "rising", result.direction
    assert result.eta_seconds is not None, "expected an ETA for a rising signal"
    assert result.eta_seconds > 0
    print(f"  rising-signal ETA to DANGER = {result.eta_seconds}s "
          f"(level={result.level}, trend={result.trend_per_sec}/s)")


def test_forecaster_no_eta_when_falling():
    fc = ZoneForecaster()
    result = None
    for crs in np.linspace(0.66, 0.20, 30):   # recovering
        result = fc.update(float(crs), dt=0.125)
    assert result.direction in ("falling", "flat")
    assert result.eta_seconds is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
