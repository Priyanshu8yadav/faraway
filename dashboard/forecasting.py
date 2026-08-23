"""
forecasting.py
──────────────
Short-horizon risk forecasting for CA-CRS⁺ (Module 7 — Predictive Layer).

The base CA-CRS⁺ engine is *reactive*: it scores the current frame. This module
adds the *predictive* capability the framework is named for. It runs Holt's
linear-trend method (double exponential smoothing) on each zone's live CRS
signal to produce:

    • a smoothed level  ℓ_t   (de-noised current risk)
    • a trend           b_t   (risk velocity, per tick)
    • an h-step forecast ŷ_{t+h} = ℓ_t + h · b_t
    • a time-to-DANGER ETA: seconds until the projected risk crosses 0.70

Why Holt and not a neural net?
  - It is a *named, citable* forecasting method (Holt 1957), not a black box.
  - It needs no training data, no GPU, and runs in microseconds per tick — so
    it never touches the demo's frame budget.
  - Level + trend are individually interpretable, which matters when a judge
    asks "why does it say DANGER in 7 seconds?" The answer is a number you can
    point at (current risk 0.55, rising 0.021/tick, threshold 0.70 → ~7 ticks).

Honesty note (keep this in the paper / Q&A):
  This is a *local linear extrapolation of the recent risk trajectory*. It is
  accurate for the short horizon over which crowd risk is locally smooth
  (seconds), and it is explicitly NOT a claim of long-range crowd prediction.
  The ETA is a projection conditional on the current trend persisting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Optional


# DANGER threshold — kept in sync with ca_crs_engine.THRESHOLD_DANGER
DANGER_THRESHOLD = 0.70

# Number of past ticks used to estimate the tick interval (seconds/tick).
_DT_WINDOW = 20

# We only trust a trend after this many observations (warm-up).
_MIN_OBS_FOR_TREND = 5

# Forecast horizon cap (seconds) — we never project further than this. Beyond a
# few seconds a linear extrapolation of crowd risk is not defensible.
_MAX_ETA_SECONDS = 60.0


@dataclass
class ForecastResult:
    """Output of one forecasting tick for a single zone."""
    level:          float = 0.0      # ℓ_t  — smoothed current risk
    trend_per_sec:  float = 0.0      # b_t normalised to per-second (risk velocity)
    forecast_curve: list  = field(default_factory=list)  # ŷ over the next horizon
    forecast_times: list  = field(default_factory=list)  # seconds-ahead for each point
    eta_seconds:    Optional[float] = None   # time until projected CRS ≥ DANGER
    direction:      str = "flat"     # "rising" | "falling" | "flat"
    projected_peak: float = 0.0      # max of the forecast curve (clipped 0–1)


class ZoneForecaster:
    """Holt's linear-trend forecaster for one zone's CRS signal.

    Usage (per tick, per zone):
        fc = forecaster.update(current_crs)
        # fc.eta_seconds is None if the zone is not on track to cross DANGER
    """

    def __init__(
        self,
        alpha: float = 0.4,          # level smoothing (higher = more reactive)
        beta:  float = 0.3,          # trend smoothing (higher = trend adapts faster)
        horizon_seconds: float = 15.0,
        forecast_points: int = 15,
        danger_threshold: float = DANGER_THRESHOLD,
    ):
        self.alpha = alpha
        self.beta = beta
        self.horizon_seconds = horizon_seconds
        self.forecast_points = forecast_points
        self.danger = danger_threshold

        self._level: Optional[float] = None
        self._trend: float = 0.0
        self._n_obs: int = 0

        # Rolling record of real wall-clock intervals between ticks, so the ETA
        # is reported in *seconds* rather than in ticks (ticks vary with FPS).
        import time as _time
        self._time = _time
        self._last_t: Optional[float] = None
        self._dts: deque = deque(maxlen=_DT_WINDOW)

    # ── tick interval estimate ────────────────────────────────────────────────
    def _dt_seconds(self) -> float:
        now = self._time.time()
        if self._last_t is not None:
            dt = now - self._last_t
            # guard against pauses / first-frame spikes
            if 0.0 < dt < 5.0:
                self._dts.append(dt)
        self._last_t = now
        if not self._dts:
            return 0.125  # sensible default (~8 FPS) before we have data
        return sum(self._dts) / len(self._dts)

    # ── main update ───────────────────────────────────────────────────────────
    def update(self, crs: float, dt: Optional[float] = None) -> ForecastResult:
        """Advance the forecast by one tick.

        dt: seconds since the previous tick. If None (normal live use) it is
        measured from the wall clock. Pass an explicit value when replaying a
        recorded run or unit-testing, where wall-clock timing is meaningless.
        """
        crs = float(max(0.0, min(1.0, crs)))
        if dt is None:
            dt = self._dt_seconds()
        else:
            dt = float(dt) if dt and dt > 0 else 0.125

        # Holt initialisation on first observation.
        if self._level is None:
            self._level = crs
            self._trend = 0.0
            self._n_obs = 1
            return ForecastResult(level=crs, direction="flat")

        prev_level = self._level
        # Level: blend new obs with the previous level projected forward by trend.
        self._level = self.alpha * crs + (1 - self.alpha) * (prev_level + self._trend)
        # Trend: blend the newly observed slope with the previous trend.
        self._trend = self.beta * (self._level - prev_level) + (1 - self.beta) * self._trend
        self._n_obs += 1

        # trend is "per tick"; convert to per-second for a stable, FPS-independent read
        trend_per_sec = self._trend / dt if dt > 0 else 0.0

        # ── build the forecast curve (seconds-ahead) ──────────────────────────
        curve, times = [], []
        step = self.horizon_seconds / self.forecast_points
        for i in range(1, self.forecast_points + 1):
            t_ahead = i * step
            yhat = self._level + trend_per_sec * t_ahead
            curve.append(max(0.0, min(1.0, yhat)))
            times.append(round(t_ahead, 2))

        # ── time-to-DANGER ETA ────────────────────────────────────────────────
        eta = None
        if self._n_obs >= _MIN_OBS_FOR_TREND and self._level < self.danger:
            if trend_per_sec > 1e-4:  # only if genuinely rising
                eta_raw = (self.danger - self._level) / trend_per_sec
                if 0.0 < eta_raw <= _MAX_ETA_SECONDS:
                    eta = round(eta_raw, 1)

        # ── direction label ───────────────────────────────────────────────────
        if trend_per_sec > 0.003:
            direction = "rising"
        elif trend_per_sec < -0.003:
            direction = "falling"
        else:
            direction = "flat"

        return ForecastResult(
            level=round(self._level, 4),
            trend_per_sec=round(trend_per_sec, 4),
            forecast_curve=curve,
            forecast_times=times,
            eta_seconds=eta,
            direction=direction,
            projected_peak=round(max(curve) if curve else self._level, 4),
        )

    def reset(self):
        self._level = None
        self._trend = 0.0
        self._n_obs = 0
        self._last_t = None
        self._dts.clear()
