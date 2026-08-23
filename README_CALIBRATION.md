# CA-CRS⁺ — Canonical Calibration (read before trusting any number here)

This note exists because the auto-generated Round-1 pipeline left contradictory
calibration files in this repo. It records what is real and how to reproduce it.

## The occlusion model κ(ρ)
**Canonical result — use this:**

    κ(ρ) = 1 + 8.2442 · ρ^1.1333        ρ = gt_count / max_gt_count
    R² = 0.6152   (n = 1269 images after quality filter, from 1601)
    β (bounding-box overlap) predictor: R² = −0.04  → rejected as useless

Source of truth:
- `outputs/calibration/kappa_params_density.json`  ← canonical fitted params
- `outputs/calibration/sahi_calibration_dataset.csv` ← raw per-image data
- `outputs/calibration/kappa_params_sahi.json`      ← same fit, with β control

**Reproduce it live (no saved params are read):**

    python verify_kappa.py

Expected output: α₀≈8.244, γ≈1.133, R²≈0.615, and the β control at R²<0.

## What was quarantined and why
Moved to `_QUARANTINE_broken_round1/` — do NOT use, do NOT cite:
- `calibration_dataset.csv` — its "detected" column is a copy of ground truth
  (99.7% of rows have detected == gt_count), so it contains no occlusion signal.
- `kappa_params.json` — degenerate fit on that fake data (R² = −0.011, params
  stuck at optimizer bounds). This is NOT the real κ result.
- `kappa_params_yolo.json` — old YOLO-person baseline fit (R² = −0.024).
- `run_all.log` — logs the meaningless "R²=1.0000" from the broken fit above.

The broken calibration lived in `run_all.py` Task C (it built the fake
detection table). That task is now disabled — see the guard at the top of
`task_calibrate_kappa`. Calibration is done by the SAHI head-detection scripts,
not by `run_all.py`.

## Sensitivity analysis (real, reproduces)
`outputs/sensitivity/` — genuine SALib Sobol run (12,288 Saltelli samples).
γ_exp (ST≈0.73) and λ (ST≈0.28) dominate; w₁/w₂/w₃ are inert (ST<0.03).

## Note on the research paper
The IEEE paper's counting accuracy numbers (density-map / point-based MAE) were
produced separately on Colab and are NOT in this repo. This repo is the Round-1
hackathon demo build only. Do not try to reproduce the paper's counting table
from these files — different models, different environment.
