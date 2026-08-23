<h1 align="center">🛡️ CA-CRS⁺</h1>
<p align="center"><b>Real-Time Causal Crowd-Risk Intelligence</b></p>
<p align="center">
Sees the crush coming · names the cause · fires the right gate — on one laptop, fully offline.
</p>

---

## What it is

Most crowd-safety systems are **reactive**: they raise one generic "high density" alarm and stop there. At Itaewon (2022) the crush began **20 minutes** before the first collapse — and no system alerted.

**CA-CRS⁺** is different. Running on standard RGB CCTV — no thermal cameras, no cloud — it:

- **Counts the crowd by density regression** (DM-Count), not head detection — so it stays accurate in dense crowds where detectors collapse.
- **Attributes the cause** of danger to Density, Speed, or Conflict, and refuses to act when the cause is ambiguous (`MIXED → HOLD`).
- **Catches the "gridlock" crush** — packed-and-motionless — that a linear density–speed model scores as safe.
- **Forecasts** the risk trajectory and shows a **time-to-DANGER countdown** — the alert nobody else generates.
- **Fires a named gate command** (open / close / redirect / hold) with a **ripple-effect check** so relieving one zone never triggers a second crush.
- **Sizes the response** — marshals scaled logarithmically to crowd volume.

> This repository is the **live demo system**. The methods are described in our IEEE-format paper (`CA-CRS+`); counting-accuracy benchmarks were produced separately on Colab.

---

## ✨ Highlights

| Capability | How |
|---|---|
| **Density-map counting** | DM-Count (VGG-19), occlusion-robust — regresses true count, not boxes |
| **Causal attribution** | `r_f = w_f·X_f / Σ w_j·X_j` — dominant factor decides the gate |
| **Non-linear crush score** | exponential term captures high-density / low-speed gridlock |
| **Predictive layer** | Holt linear-trend forecast → seconds-to-DANGER ETA |
| **Cause → gate mapping** | named Modbus TCP gate commands + ripple-effect prevention |
| **Resource-aware alert** | `D_mar = Σ⌈α·log₁₀(1+N)⌉` marshal demand |
| **Multi-view fusion** | urgency-weighted Global Risk Score across zones |
| **Runs offline** | Streamlit dashboard, Apple-silicon MPS / CUDA / CPU |

---

## 🚀 Quickstart

```bash
# 1 · clone
git clone https://github.com/Priyanshu8yadav/CA-CRS-Research.git
cd CA-CRS-Research

# 2 · environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3 · get the model weights  (see models/README.md)
#     • DM-Count  →  models/model_qnrf.pth
#     • YOLO fallback (optional) → models/yolov8n_crowd_head.pt

# 4 · run the live dashboard
streamlit run dashboard/app.py
```

Then in the sidebar: pick **Expert View**, point the three zones at your video files (or RTSP URLs), keep **Counting engine = DM-Count**, tune **Zone safe capacity** so scenes span SAFE→WARNING→DANGER, and press **▶ Start Watching**.

> **Tip (Apple silicon):** DM-Count auto-selects the **MPS** GPU. If it's slow, drop **DM-Count resolution** to 448. The Model Info panel shows the active compute device.

---

## 🧠 The science (one screen)

```
Occlusion-robust count       N   = ∫ density_map          (DM-Count, VGG-19)
Normalized density           D   = min(N / capacity, 1)
Speed / Conflict             S,C = dense optical flow  (magnitude, circular variance)

Non-linear risk (per zone)   CRS = w₁D + Φ(D,S) + w₃C
                             Φ   = w₂·S(1−D) + γ·e^{λ(D−S)}      ← crush term
Causal attribution           r_f = w_f·X_f / Σ w_j·X_j          → dominant cause
Forecast (predictive layer)  Holt level+trend → ETA = (0.70 − level) / trend
Gate command                 Density→OPEN · Speed→CLOSE · Conflict→REDIRECT · MIXED→HOLD
Ripple check                 REDIRECT → HOLD if downstream CRS > 0.50
Marshal demand               D_mar = Σ⌈α·log₁₀(1+N)⌉,  α ∈ {0,4,10}
Global Risk Score            GRS = Σ ωₖ·CRSₖ / Σ ωₖ,   ωₖ = 2 if DANGER else 1
```

---

## ✅ Reproduce our headline number (live)

The occlusion-correction calibration is regenerated from raw data — nothing is read from a saved-parameters file:

```bash
python verify_kappa.py
# → κ(ρ) = 1 + 8.24·ρ^1.13   R² = 0.615   (β predictor R² < 0, rejected)
```

See **[`README_CALIBRATION.md`](README_CALIBRATION.md)** for what is validated and how. Every figure in the paper is measured against ground truth or explicitly labeled a projection.

Run the tests (no weights required):

```bash
python tests/test_engine_upgrades.py     # 6/6 — optical-flow + forecasting checks
```

---

## 📁 Repository structure

```
dashboard/            live Streamlit app + CA-CRS⁺ engine
  app.py                 dashboard UI + live fragment loop
  ca_crs_engine.py       risk score · causal attribution · gates · GRS
  dmcount_model.py       DM-Count VGG-19 network
  dmcount_counter.py     density counter + heatmap overlay
  forecasting.py         Holt forecast + time-to-DANGER ETA
  threaded_stream.py     non-blocking multi-camera reader
calibration/          κ(ρ) occlusion calibration
sensitivity/          Sobol' parameter sensitivity
parsers/              dataset parsers (ShanghaiTech / UCF-QNRF / PETS2009)
verify_kappa.py       one-command live re-fit of the κ calibration
tests/                synthetic engine + forecasting tests
config.yaml           physics + threshold configuration
models/               weights (git-ignored — see models/README.md)
```

---

## 🛠️ Tech stack

Python · PyTorch (DM-Count) · OpenCV (optical flow) · Streamlit + Plotly (dashboard) · pymodbus (virtual PLC over Modbus TCP) · Ultralytics YOLO (optional fallback detector).

---

## 👥 Team

**Priyanshu Kumar Yadav** · **Abhinav Kumar Mishra** · **Sarvesh Tiwari** · **Nagendra Mishra**
Computer Science & Engineering, Rajkiya Engineering College, Mainpuri.

## 📄 License

Released under the [MIT License](LICENSE).
