"""
ca_crs_engine.py
────────────────
Core CA-CRS⁺ risk scoring engine — paper-aligned (Modules 2–6).

Module 2  — Detection & Occlusion Correction: κ(ρ) = 1 + α₀·ρ^γb
Module 3  — Non-linear risk score + causal attribution r_f
Module 4  — Cause-to-gate mapping + ripple-effect prevention (threshold > 0.50)
Module 5  — Logarithmic marshal demand: D_mar = ⌈α_l · log₁₀(1 + N_k)⌉
Module 6  — Urgency-weighted GRS: w_k=2 for DANGER, w_k=1 otherwise

Empirically tuned values (calibration 2026-06-08, n=1,601 images, R²=0.615):
  κ: α₀=8.2442, γ=1.1333
  Risk: w1=0.40, w2=0.30, w3=0.30, γ_exp=0.01, λ=8.4915
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from dashboard.threaded_stream import ThreadedStream
from dashboard.forecasting import ZoneForecaster, ForecastResult

# ── Module 2: Calibrated κ(ρ) constants ──────────────────────────────────────
KAPPA_ALPHA0  = 8.2442
KAPPA_GAMMA   = 1.1333

# ── Module 3: Non-linear risk score weights ───────────────────────────────────
RISK_W1        = 0.40     # density weight
RISK_W2        = 0.30     # speed/gridlock weight
RISK_W3        = 0.30     # conflict weight
RISK_GAMMA_EXP = 0.01     # exponential crush amplitude
RISK_LAMBDA    = 8.4915   # exponential crush sharpness

# Causal dominance threshold (Module 3 spec)
CAUSAL_DOMINANCE_THRESHOLD = 0.40

# ── Zone capacity constants (Module 2 fix) ────────────────────────────────────
RHO_MAX   = 7.0     # persons/m² — saturation density
AREA_M2   = 50.0    # assumed zone area in m²
CAPACITY  = int(RHO_MAX * AREA_M2)  # = 350 heads — used as ρ denominator

# ── Speed saturation ──────────────────────────────────────────────────────────
V_MAX = 15.0   # px/frame speed saturation (from config.yaml physics.v_max)

# ── Optical-flow constants (Module 2 motion, paper eqs. 1–2) ──────────────────
# Dense Farneback flow is computed on the YOLO-sized grayscale frame. A pixel is
# considered "moving" if its flow magnitude exceeds this threshold (px/frame);
# static background is excluded so speed/conflict reflect the crowd, not noise.
FLOW_MOVING_THRESH = 0.6   # px/frame — below this a pixel is treated as static

# ── Risk thresholds ───────────────────────────────────────────────────────────
THRESHOLD_WARN   = 0.35
THRESHOLD_DANGER = 0.70

# ── Module 4: Cause-to-gate mapping ──────────────────────────────────────────
# Gate command is driven by dominant causal factor, not overall status.
# When no single factor dominates (MIXED), issue HOLD to prevent reckless action.
CAUSE_GATE_MAP = {
    "Density":  ("OPEN",     "#ef4444"),  # relieve density → open exit
    "Speed":    ("CLOSE",    "#f59e0b"),  # stop rapid inflow → close entry
    "Conflict": ("REDIRECT", "#6366f1"),  # separate flows → redirect
    "MIXED":    ("HOLD",     "#94a3b8"),  # no clear cause → safe default
}

# ── Module 5: Dynamic marshal multipliers by risk state ──────────────────────
# D_mar^(k) = ⌈α_{l_k} · log₁₀(1 + N_k)⌉
ALPHA_BY_STATUS = {
    "SAFE":    0,
    "WARNING": 4,
    "DANGER":  10,
}


# ── ZoneState dataclass ───────────────────────────────────────────────────────
@dataclass
class ZoneState:
    zone_id:  str
    label:    str
    color:    str   = "#6b7280"
    risk:     float = 0.0
    density:  float = 0.0      # D_norm
    speed:    float = 0.0      # S_norm
    conflict: float = 0.0      # C_norm

    # Module 3: causal attribution
    causal_ratios:  dict = field(default_factory=lambda: {"Density": 0.0, "Speed": 0.0, "Conflict": 0.0})
    dominant_cause: str  = "Density"

    head_count:      int   = 0
    corrected_count: int   = 0
    kappa:           float = 1.0
    count_mode:      str   = "dmcount"   # "dmcount" | "yolo"
    gate_cmd:        str   = "HOLD"
    gate_color:      str   = "#22c55e"
    status:          str   = "SAFE"
    fps:             float = 0.0
    peak_risk:       float = 0.0   # max CRS seen this session

    risk_history:     list[float] = field(default_factory=list)
    density_history:  list[float] = field(default_factory=list)
    speed_history:    list[float] = field(default_factory=list)
    conflict_history: list[float] = field(default_factory=list)
    frame_rgb:        Optional[np.ndarray] = None

    # ── Module 7: Predictive layer (Holt linear-trend forecast) ───────────────
    forecast_level:     float = 0.0    # ℓ_t smoothed current risk
    forecast_trend:     float = 0.0    # risk velocity, per second
    forecast_curve:     list  = field(default_factory=list)  # projected CRS
    forecast_times:     list  = field(default_factory=list)  # seconds-ahead
    forecast_direction: str   = "flat"  # rising | falling | flat
    eta_danger_seconds: Optional[float] = None   # projected time to DANGER, or None


# ── Module 2: κ(ρ) occlusion correction ──────────────────────────────────────
def kappa(rho: float) -> float:
    """κ(ρ) = 1 + α₀·ρ^γ  — empirically calibrated on 1,601 images (R²=0.615).
    ρ = detected_heads / zone_capacity  (per-zone density ratio, NOT dataset max).
    """
    return 1.0 + KAPPA_ALPHA0 * (max(rho, 0.0) ** KAPPA_GAMMA)


# ── Module 3: Non-linear risk score ──────────────────────────────────────────
def compute_risk(D: float, S: float, C: float) -> float:
    """CA-CRS⁺ non-linear risk score.
    CRS = w₁·D + Φ(D,S) + w₃·C
    Φ(D,S) = w₂·S·(1−D) + γ_exp·exp(λ·(D−S))
    The exponential term captures the gridlock paradox: high D + low S = crush.
    """
    phi = RISK_W2 * S * (1.0 - D) + RISK_GAMMA_EXP * np.exp(RISK_LAMBDA * (D - S))
    return float(np.clip(RISK_W1 * D + phi + RISK_W3 * C, 0.0, 1.0))


# ── Module 3: Causal attribution ─────────────────────────────────────────────

# Density threshold above which C_norm is considered unreliable.
# At ≥6 persons/m² optical flow completely breaks down (paper, Module 3 note).
C_UNRELIABLE_DENSITY_NORM = 6.0 / RHO_MAX   # = 0.857 (6/7)

def compute_causal_ratios(D: float, S: float, C: float) -> tuple[dict, str]:
    """r_f^(k) = (w_f · X_f) / Σ_j(w_j · X_j)

    Dominant cause = argmax(r_f) if and only if r_f > CAUSAL_DOMINANCE_THRESHOLD.
    If no factor exceeds the threshold, returns 'MIXED' — which maps to HOLD.
    This prevents the system from acting on a spurious signal when the crowd
    behaviour is genuinely ambiguous.
    """
    weighted = {
        "Density":  RISK_W1 * D,
        "Speed":    RISK_W2 * S,
        "Conflict": RISK_W3 * C,
    }
    denom = sum(weighted.values()) + 1e-9
    ratios = {k: round(v / denom, 4) for k, v in weighted.items()}
    dominant = max(ratios, key=ratios.get)
    # Paper rule: dominant cause only declared if strictly > 0.40
    # Otherwise label as MIXED — operator gets HOLD, preventing a wrong intervention
    if ratios[dominant] <= CAUSAL_DOMINANCE_THRESHOLD:
        dominant = "MIXED"
    return ratios, dominant


# ── Status classifier ─────────────────────────────────────────────────────────
def classify(risk: float) -> tuple[str, str]:
    if risk >= THRESHOLD_DANGER:
        return "DANGER",  "#ef4444"
    if risk >= THRESHOLD_WARN:
        return "WARNING", "#f59e0b"
    return "SAFE", "#22c55e"


# ── Module 5: Per-zone logarithmic marshal demand ─────────────────────────────
def marshal_demand(zone_states: list[ZoneState]) -> tuple[int, str]:
    """D_mar = Σ_k ⌈α_{l_k} · log₁₀(1 + N_k)⌉
    α is 0 / 4 / 10 for SAFE / WARNING / DANGER respectively.
    Returns (total_marshals, overall_status_label).
    """
    total = 0
    for s in zone_states:
        alpha = ALPHA_BY_STATUS[s.status]
        n_k   = s.corrected_count
        total += math.ceil(alpha * math.log10(1 + n_k)) if n_k > 0 else 0

    # Overall label driven by worst zone
    statuses = [s.status for s in zone_states]
    if "DANGER" in statuses:
        label = "CRITICAL"
    elif "WARNING" in statuses:
        label = "STRAINED"
    else:
        label = "ADEQUATE"
    return total, label


# ── Module 6: Urgency-weighted GRS ───────────────────────────────────────────
def compute_grs(zone_states: list[ZoneState]) -> float:
    """GRS = Σ(w_k · CRS_k) / Σ(w_k)
    w_k = 2 for DANGER zones, 1 otherwise.
    Ensures localized critical escalations are correctly reflected globally.
    """
    if not zone_states:
        return 0.0
    weights = [2.0 if z.status == "DANGER" else 1.0 for z in zone_states]
    weighted_sum = sum(w * z.risk for w, z in zip(weights, zone_states))
    return float(np.clip(weighted_sum / sum(weights), 0.0, 1.0))


# ── ZoneProcessor ─────────────────────────────────────────────────────────────
class ZoneProcessor:
    """Per-zone video processor: threaded frame read → YOLO → CA-CRS⁺ pipeline."""

    def __init__(
        self,
        zone_id:      str,
        label:        str,
        video_source: str,
        model=None,
        conf:         float = 0.25,
        imgsz:        int   = 640,
        target_fps:   int   = 8,
        counter=None,                 # shared DMCountCounter (density-map mode)
        count_mode:   str   = "dmcount",   # "dmcount" | "yolo"
        count_min_interval: float = 0.6,   # min seconds between heavy DM-Count runs
        capacity:     int   = CAPACITY,    # safe occupancy of this zone (D_norm denom)
    ):
        self.zone_id = zone_id
        self.label   = label
        self.source  = video_source

        self._stream = ThreadedStream(video_source, queue_size=2)

        self.model   = model
        self.conf    = conf
        self.imgsz   = imgsz

        # ── Counting engine (density-map by default) ─────────────────────────
        self.counter    = counter
        self.count_mode = count_mode
        self.count_min_interval = count_min_interval
        self.capacity   = int(capacity)
        self._last_count   = 0
        self._last_heat:  Optional[np.ndarray] = None   # precomputed color heatmap
        self._last_alpha: Optional[np.ndarray] = None   # precomputed blend weights
        self._last_count_t = 0.0

        src_fps = self._stream.fps
        self._frame_skip  = max(1, int(src_fps / target_fps))
        self._frame_count = 0

        # Previous grayscale frame for dense optical flow (Module 2 motion).
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_time = time.time()

        # Module 7: per-zone predictive forecaster over the CRS signal.
        self._forecaster = ZoneForecaster(danger_threshold=THRESHOLD_DANGER)

        self.state = ZoneState(zone_id=zone_id, label=label)

    # ── Motion features from dense optical flow (paper eqs. 1–2) ──────────────
    def _flow_features(self, gray: np.ndarray) -> tuple[float, float]:
        """Dense Farneback optical flow → (S_norm, C_raw).

        This replaces the old centroid-index speed and IQR conflict, both of
        which were wrong:
          - the old speed matched detections *by array position* across frames,
            i.e. it measured the distance between unrelated heads (pure noise);
          - the old conflict was horizontal spread, so a wide orderly crowd
            scored as 'conflict'.

        S_norm  (paper eq. 1): mean flow magnitude over MOVING pixels, normalised
                 by V_MAX. Static background is excluded so a still crowd reads 0.

        C_raw   (paper eq. 2): magnitude-weighted circular variance of flow
                 DIRECTION,  C = 1 − |Σ mᵢ·e^{jθᵢ}| / Σ mᵢ.
                 Aligned flow  ⇒ C→0;  balanced counter-flow ⇒ C→1. Unlike the
                 old IQR proxy this is maximal for true opposing flows and does
                 NOT vanish at a 50/50 split.
        """
        prev = self._prev_gray
        self._prev_gray = gray
        if prev is None or prev.shape != gray.shape:
            return 0.0, 0.0

        # Dense flow (Farneback). Cheap on the already-downscaled YOLO frame.
        flow = cv2.calcOpticalFlowFarneback(
            prev, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        fx, fy = flow[..., 0], flow[..., 1]
        mag = np.sqrt(fx * fx + fy * fy)

        moving = mag > FLOW_MOVING_THRESH
        n_moving = int(moving.sum())
        if n_moving == 0:
            return 0.0, 0.0

        # ── S_norm: mean speed over moving pixels, saturated at V_MAX ─────────
        s_norm = float(np.clip(mag[moving].mean() / V_MAX, 0.0, 1.0))

        # ── C_raw: magnitude-weighted circular variance of direction ─────────
        m = mag[moving]
        theta = np.arctan2(fy[moving], fx[moving])
        resultant = np.abs(np.sum(m * np.exp(1j * theta)))
        c_raw = float(np.clip(1.0 - resultant / (m.sum() + 1e-9), 0.0, 1.0))

        return s_norm, c_raw

    # ── Main frame processing ─────────────────────────────────────────────────
    def process_frame(self, do_count: bool = True) -> ZoneState:
        """do_count=False → skip the heavy DM-Count inference this tick and
        reuse the last density result (round-robin scheduling). Motion, risk,
        forecast and the display still update every tick."""
        t0 = time.time()

        ok, frame = self._stream.read()
        if not ok or frame is None:
            return self.state

        self._frame_count += 1
        if self._frame_count % self._frame_skip != 0:
            return self.state

        h0, w0 = frame.shape[:2]
        boxes   = np.zeros((0, 4))   # only populated in YOLO mode
        overlay = None               # density heatmap frame in DM-Count mode

        if self.count_mode == "dmcount" and self.counter is not None:
            # ── DM-Count density regression (paper counting front end) ────────
            # Regresses a continuous density map and integrates it → true count.
            # No occlusion correction needed: unlike detection it does not
            # under-count in dense crowds. Heavy, so throttled per zone.
            now = time.time()
            # Run the heavy inference only when it is this zone's turn (do_count)
            # AND the min interval elapsed — but always run once at startup so the
            # first frame has a result.
            need = self._last_heat is None or (
                do_count and (now - self._last_count_t) >= self.count_min_interval
            )
            if need:
                try:
                    count, heat, alpha = self.counter.infer(frame)
                    self._last_count = count
                    self._last_heat, self._last_alpha = heat, alpha
                    self._last_count_t = now
                except Exception:
                    count = self._last_count
            else:
                count = self._last_count
            # Re-blend the last heatmap onto the CURRENT frame every tick → the
            # video stays live and smooth even between heavy inferences.
            overlay = self.counter.compose(frame, self._last_heat, self._last_alpha)
            head_count      = count      # regressed count (no separate "raw")
            corrected_count = count      # nothing to correct
            kap             = 1.0        # κ not applied in density-regression mode
        else:
            # ── YOLO detection + κ(ρ) occlusion correction (fallback) ─────────
            try:
                scale   = self.imgsz / max(h0, w0)
                frame_s = cv2.resize(frame, (int(w0 * scale), int(h0 * scale)))
                results = self.model.predict(
                    frame_s, conf=self.conf, imgsz=self.imgsz, verbose=False
                )[0]
                if results.boxes and len(results.boxes):
                    boxes = results.boxes.xyxy.cpu().numpy() / scale
            except Exception:
                boxes = np.zeros((0, 4))
            head_count      = len(boxes)
            rho             = head_count / self.capacity
            kap             = kappa(rho)
            corrected_count = int(head_count * kap)

        # ── Module 3: Normalized feature inputs ──────────────────────────────
        D_norm = float(np.clip(corrected_count / self.capacity, 0.0, 1.0))

        # Motion features from dense optical flow (paper eqs. 1–2). Computed on a
        # flow-dedicated grayscale so it is independent of the counting engine.
        flow_scale = 640.0 / max(w0, 1)
        gray = cv2.cvtColor(
            cv2.resize(frame, (int(w0 * flow_scale), int(h0 * flow_scale))),
            cv2.COLOR_BGR2GRAY,
        )
        S_norm, C_raw = self._flow_features(gray)

        # C_norm reliability gate (paper, Module 3):
        # At density ≥6 persons/m² (D_norm ≥ 0.857), optical flow breaks down
        # entirely due to body occlusion — C_norm becomes meaningless noise.
        # Zero it out so only Density drives attribution in crush conditions.
        C_norm = 0.0 if D_norm >= C_UNRELIABLE_DENSITY_NORM else C_raw

        risk_val = compute_risk(D_norm, S_norm, C_norm)
        status, color = classify(risk_val)

        # ── Module 3: Causal attribution r_f ─────────────────────────────────
        causal_ratios, dominant_cause = compute_causal_ratios(D_norm, S_norm, C_norm)

        # ── Module 4: Cause-to-gate mapping ──────────────────────────────────
        # MIXED dominant cause (no factor crossed 0.40) → always HOLD
        # to avoid acting on an ambiguous signal.
        if status == "SAFE" or dominant_cause == "MIXED":
            gate_cmd, gate_color = "HOLD", "#22c55e"
        else:
            gate_cmd, gate_color = CAUSE_GATE_MAP[dominant_cause]

        # ── Annotate frame ────────────────────────────────────────────────────
        # DM-Count mode draws the HUD over the density heatmap; YOLO mode over
        # the raw frame with boxes.
        base_vis = overlay if (overlay is not None) else frame
        frame_ann = self._annotate(
            base_vis.copy(), boxes, risk_val, status, color,
            head_count, corrected_count, dominant_cause, causal_ratios,
            mode=self.count_mode,
        )
        frame_rgb = cv2.cvtColor(frame_ann, cv2.COLOR_BGR2RGB)

        elapsed = time.time() - t0
        fps     = 1.0 / max(elapsed, 0.001)

        # ── Update state ──────────────────────────────────────────────────────
        s = self.state
        s.risk            = round(risk_val, 4)
        s.density         = round(D_norm, 4)
        s.speed           = round(S_norm, 4)
        s.conflict        = round(C_norm, 4)
        s.causal_ratios   = causal_ratios
        s.dominant_cause  = dominant_cause
        s.head_count      = head_count
        s.corrected_count = corrected_count
        s.kappa           = round(kap, 3)
        s.count_mode      = self.count_mode
        s.status          = status
        s.color           = color
        s.gate_cmd        = gate_cmd
        s.gate_color      = gate_color
        s.fps             = round(fps, 1)
        s.frame_rgb       = frame_rgb
        s.peak_risk       = max(s.peak_risk, risk_val)  # session peak

        MAXHIST = 150
        s.risk_history     = (s.risk_history     + [risk_val])[-MAXHIST:]
        s.density_history  = (s.density_history  + [D_norm  ])[-MAXHIST:]
        s.speed_history    = (s.speed_history    + [S_norm  ])[-MAXHIST:]
        s.conflict_history = (s.conflict_history + [C_norm  ])[-MAXHIST:]

        # ── Module 7: Predictive layer — forecast + time-to-DANGER ETA ────────
        fc: ForecastResult = self._forecaster.update(risk_val)
        s.forecast_level     = fc.level
        s.forecast_trend     = fc.trend_per_sec
        s.forecast_curve     = fc.forecast_curve
        s.forecast_times     = fc.forecast_times
        s.forecast_direction = fc.direction
        s.eta_danger_seconds = fc.eta_seconds

        return s

    # ── Frame annotation ──────────────────────────────────────────────────────
    def _annotate(
        self, frame, boxes, risk, status, color,
        raw_count, corr_count, dominant_cause, causal_ratios, mode="dmcount"
    ):
        h, w = frame.shape[:2]
        c = {"SAFE": (34, 197, 94), "WARNING": (245, 158, 11), "DANGER": (239, 68, 68)}[status]

        # Coloured border
        cv2.rectangle(frame, (0, 0), (w, h), c, 14)

        # Bounding boxes (YOLO mode only; DM-Count shows the heatmap instead)
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)

        # HUD overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        font = cv2.FONT_HERSHEY_DUPLEX
        dom_pct = causal_ratios.get(dominant_cause, 0.0) * 100
        if mode == "dmcount":
            count_txt = f"People: {corr_count}  (DM-Count density)"
        else:
            count_txt = f"Heads: {raw_count} -> {corr_count} (kappa)"
        cv2.putText(frame, count_txt, (12, 28), font, 0.65, (255, 255, 255), 1)
        cv2.putText(frame, f"CRS: {risk:.3f}  [{status}]",
                    (12, 58), font, 0.85, c, 2)
        cv2.putText(frame, f"Cause: {dominant_cause} ({dom_pct:.0f}%)",
                    (12, 88), font, 0.55, (180, 180, 180), 1)
        return frame

    def stop(self):
        self._stream.stop()
