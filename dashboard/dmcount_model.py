"""
dmcount_model.py
────────────────
VGG19-backbone density-regression network for DM-Count
(Wang et al., "Distribution Matching for Crowd Counting", NeurIPS 2020).

Architecture adapted from the official repo (cvlab-stonybrook/DM-Count) so the
published pretrained weights load directly. Two deliberate changes:
  • the ImageNet VGG init download is removed (we always load a DM-Count
    checkpoint, which overwrites every weight anyway) — avoids a network call;
  • deprecated F.upsample_bilinear → F.interpolate.

This is the counting front end used in the CA-CRS⁺ paper (density-map
regression), replacing detection-based counting which under-counts in dense
crowds.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F

# VGG-19 feature configuration (configuration "E")
_CFG_E = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
          512, 512, 512, 512, 'M', 512, 512, 512, 512]


def _make_layers(cfg, batch_norm=False):
    layers, in_c = [], 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv = nn.Conv2d(in_c, v, kernel_size=3, padding=1)
            layers += ([conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                       if batch_norm else [conv, nn.ReLU(inplace=True)])
            in_c = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    """DM-Count VGG-19 density regressor. Returns (density_map, normed_map)."""

    def __init__(self):
        super().__init__()
        self.features = _make_layers(_CFG_E)
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.density_layer = nn.Sequential(nn.Conv2d(128, 1, 1), nn.ReLU())

    def forward(self, x):
        x = self.features(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.reg_layer(x)
        mu = self.density_layer(x)
        B, C, H, W = mu.size()
        mu_sum = mu.view(B, -1).sum(1).view(B, 1, 1, 1)
        mu_normed = mu / (mu_sum + 1e-6)
        return mu, mu_normed


def build_dmcount(weights_path: str, device: torch.device) -> VGG:
    """Construct the network and load a DM-Count checkpoint."""
    model = VGG()
    state = torch.load(weights_path, map_location=device)
    # checkpoints are plain state_dicts
    model.load_state_dict(state)
    model.to(device).eval()
    return model
