"""
dmcount_counter.py
──────────────────
Live crowd counter built on DM-Count (density-map regression). Replaces
detection-based counting for the CA-CRS⁺ demo: instead of drawing boxes on
visible heads (which under-counts badly in dense crowds), it regresses a
continuous density map and integrates it to a count — the method the paper uses.

Returns, per frame:
  • count      : integer people estimate (∑ density)
  • density    : HxW float density map, resized to the frame
  • overlay    : the frame with a translucent JET density heatmap blended on top
                 (this is the demo's headline visual)

No torchvision dependency (preprocessing done by hand) so it runs anywhere
torch is installed. Device auto-selects MPS (Apple Silicon) → CUDA → CPU.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from dashboard.dmcount_model import build_dmcount

# ImageNet normalization — MUST match DM-Count's test-time transform.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DMCountCounter:
    """Thread-safe DM-Count wrapper. One model instance shared across zones."""

    def __init__(self, weights_path: str, longest_side: int = 1024,
                 device: Optional[torch.device] = None):
        self.weights_path = str(weights_path)
        self.longest_side = int(longest_side)
        self.device = device or _pick_device()
        self.model = build_dmcount(self.weights_path, self.device)
        self._lock = threading.Lock()   # serialize inference (esp. on MPS)

    # ── preprocessing ─────────────────────────────────────────────────────────
    def _preprocess(self, frame_bgr: np.ndarray):
        h0, w0 = frame_bgr.shape[:2]
        scale = 1.0
        if self.longest_side and max(h0, w0) > self.longest_side:
            scale = self.longest_side / max(h0, w0)
        w, h = int(round(w0 * scale)), int(round(h0 * scale))
        img = cv2.resize(frame_bgr, (w, h))
        # pad to a multiple of 16 (VGG has 4 /2 pools)
        W, H = (w + 15) // 16 * 16, (h + 15) // 16 * 16
        padded = cv2.copyMakeBorder(img, 0, H - h, 0, W - w, cv2.BORDER_CONSTANT, value=0)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - _MEAN) / _STD
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return tensor, (h, w), (h0, w0)

    # ── inference ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def infer(self, frame_bgr: np.ndarray, overlay_alpha: float = 0.55):
        """Run DM-Count. Returns (count:int, heat_bgr, alpha_hw).

        heat/alpha are precomputed once here (the expensive blur+colormap) so the
        overlay can be re-applied cheaply onto every subsequent live frame via
        compose() — that keeps the video smooth between heavy inferences.
        """
        with self._lock:
            tensor, (h, w), (h0, w0) = self._preprocess(frame_bgr)
            density, _ = self.model(tensor)                    # 1x1xHxW
            count = float(density.sum().item())
            dmap = density[0, 0].detach().to("cpu").numpy()    # padded density

        dmap = dmap[:h, :w]
        dmap_full = cv2.resize(dmap, (w0, h0), interpolation=cv2.INTER_CUBIC)
        dmap_full = np.clip(dmap_full, 0, None)
        heat, alpha = self._heat_and_alpha((h0, w0), dmap_full, overlay_alpha)
        return int(round(count)), heat, alpha

    # ── heatmap precompute (expensive: blur + colormap, done once per count) ──
    @staticmethod
    def _heat_and_alpha(shape_hw, density, overlay_alpha=0.55):
        h, w = shape_hw
        d = np.clip(density, 0, None).astype(np.float32)
        if float(d.max()) <= 1e-8:
            return None, None
        sigma = max(h, w) / 90.0                     # size-relative smoothing
        d = cv2.GaussianBlur(d, (0, 0), sigmaX=sigma, sigmaY=sigma)
        d = d / (d.max() + 1e-8)
        field = np.power(d, 0.65)                    # lift mid densities
        norm = (field * 255.0).astype(np.uint8)
        cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
        heat = cv2.applyColorMap(norm, cmap)
        alpha = (field * float(overlay_alpha)).astype(np.float32)   # HxW blend weight
        return heat, alpha

    # ── cheap per-frame compose (no blur/colormap — just a weighted blend) ────
    @staticmethod
    def compose(frame_bgr, heat, alpha):
        """Blend a precomputed heat/alpha onto a fresh frame. ~2-5 ms."""
        if heat is None or alpha is None:
            return frame_bgr
        if heat.shape[:2] != frame_bgr.shape[:2]:
            heat = cv2.resize(heat, (frame_bgr.shape[1], frame_bgr.shape[0]))
            alpha = cv2.resize(alpha, (frame_bgr.shape[1], frame_bgr.shape[0]))
        a = alpha[..., None]
        out = frame_bgr.astype(np.float32) * (1 - a) + heat.astype(np.float32) * a
        return np.clip(out, 0, 255).astype(np.uint8)

    def set_longest_side(self, longest: int):
        self.longest_side = int(longest)
