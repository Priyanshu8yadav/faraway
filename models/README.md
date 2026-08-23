# Model weights

Weights are **not committed** (kept out of git for size). Download them into this
folder before running the dashboard.

## 1 · DM-Count density model (primary counting engine) — required

File: **`model_qnrf.pth`**  (~86 MB, VGG-19, trained on UCF-QNRF)

Source: the official DM-Count repository —
[cvlab-stonybrook/DM-Count](https://github.com/cvlab-stonybrook/DM-Count)
(`pretrained_models/model_qnrf.pth`). The `model_nwpu.pth` checkpoint also works.

Place it here:

```
models/model_qnrf.pth
```

## 2 · YOLO head detector (optional fallback) — only if you switch the engine to YOLO

File: **`yolov8n_crowd_head.pt`** (~6 MB) — published on this repo's **Releases** page.
The heavier `yolov8m_head_scut.pt` is an optional higher-accuracy alternative.

```
models/yolov8n_crowd_head.pt
```

---

The dashboard defaults to **DM-Count**, so `model_qnrf.pth` is all you need for the
live demo. The compute device (MPS / CUDA / CPU) is auto-selected and shown in the
sidebar's Model Info panel.
