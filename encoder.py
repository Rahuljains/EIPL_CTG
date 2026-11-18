"""Track E: a small 1-D CNN pre-trained to fill in masked spans of the FHR.

Self-supervised, so it needs no labels: about 40% of the fetal heart rate is
hidden in random 30-90 second spans and the network reconstructs it from the
surrounding trace and the uterine contraction channel. The mean-pooled encoder
feature map is the 128-d window embedding the Layer 2 likelihood then uses.

Pre-training uses JNU-CTG only. No CTU-UHB signal and no label of any kind ever
reaches this model, so a test labour cannot leak into the representation.

Author: Rahul Jain
Date:   17 Nov 2026
Usage:  from ctg.encoder import MaskedCTGAutoencoder, MaskedWindowDataset
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from . import config as C


# ----------------------------------------------------------------- normalise
def normalise_fhr(fhr: np.ndarray) -> np.ndarray:
    """Centre and scale the FHR; the same transform is used for JNU and CTU."""
    return (fhr - C.FHR_NORM_CENTRE) / C.FHR_NORM_SCALE


def denormalise_fhr(value: np.ndarray) -> np.ndarray:
    """Invert normalise_fhr, so a loss can be reported in bpm."""
    return value * C.FHR_NORM_SCALE + C.FHR_NORM_CENTRE


def normalise_uc(uc: np.ndarray) -> np.ndarray:
    """Scale and clip the UC channel."""
    return np.clip(uc / C.UC_NORM_SCALE, 0.0, C.UC_NORM_CLIP)


def make_mask(length: int, rng: np.random.Generator,
              mask_frac: float = C.MASK_FRAC,
              span_range: Tuple[int, int] = C.MASK_SPAN_S) -> np.ndarray:
    """Return a boolean mask hiding about mask_frac of the trace in random spans."""
    mask = np.zeros(length, dtype=bool)
    target = int(round(mask_frac * length))
    guard = 0
    while mask.sum() < target and guard < 100:
        span = int(rng.integers(span_range[0], span_range[1] + 1))
        start = int(rng.integers(0, max(1, length - span)))
        mask[start:start + span] = True
        guard += 1
    return mask


def interpolation_baseline(fhr: np.ndarray, mask: np.ndarray,
                           observed: np.ndarray) -> float:
    """Mean absolute error of linear interpolation across each masked span, in bpm.

    This is the number the encoder has to beat: if a straight line through the
    gap predicts the hidden trace as well as the network does, the learned
    representation has added nothing.
    """
    visible = observed & ~mask
    if visible.sum() < 2:
        return float("nan")
    index = np.arange(len(fhr))
    filled = np.interp(index, index[visible], fhr[visible])
    target = mask & observed
    if target.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(filled[target] - fhr[target])))


class MaskedWindowDataset(Dataset):
    """Serves (input channels, target FHR, loss mask) triples for the masked task."""

    def __init__(self, windows: np.ndarray, valid: np.ndarray,
                 seed: int = C.SEED, fixed_masks: bool = False) -> None:
        self.windows = windows                  # (N, 2, 900) float16: FHR, UC
        self.valid = valid                      # (N, 900) bool: originally observed
        self.seed = seed
        self.fixed_masks = fixed_masks          # True for validation: same mask each epoch

    def __len__(self) -> int:
        """Number of windows."""
        return len(self.windows)

    def __getitem__(self, index: int):
        """Build one masked example."""
        rng = (np.random.default_rng(self.seed + index) if self.fixed_masks
               else np.random.default_rng())
        fhr_raw = self.windows[index, 0].astype(np.float32)
        uc_raw = self.windows[index, 1].astype(np.float32)
        observed = self.valid[index].astype(bool) & np.isfinite(fhr_raw)

        fhr = normalise_fhr(np.nan_to_num(fhr_raw, nan=C.FHR_NORM_CENTRE))
        uc = normalise_uc(np.nan_to_num(uc_raw, nan=0.0))
        fhr = np.where(observed, fhr, 0.0)

        mask = make_mask(len(fhr), rng)
        available = observed & ~mask

        channels = np.stack([np.where(mask, 0.0, fhr),          # FHR, masked spans zeroed
                             uc,                                 # UC stays visible
                             available.astype(np.float32)])      # availability channel

        loss_mask = mask & observed                              # masked AND truly observed
        return (torch.from_numpy(channels.astype(np.float32)),
                torch.from_numpy(fhr.astype(np.float32)),
                torch.from_numpy(loss_mask.astype(np.float32)))


class MaskedCTGAutoencoder(nn.Module):
    """1-D CNN encoder/decoder. The mean-pooled encoder output is the embedding."""

    def __init__(self, embedding_dim: int = C.EMB_DIM) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(3, 32, kernel_size=7, padding=3),
                                  nn.BatchNorm1d(32), nn.GELU())
        self.encoder = nn.Sequential(
            self._block(32, 64, 7), self._block(64, 64, 7),
            self._block(64, 128, 5), self._block(128, embedding_dim, 5))
        self.decoder = nn.Sequential(
            self._up(embedding_dim, 128, 5), self._up(128, 64, 5),
            self._up(64, 64, 5), self._up(64, 32, 5))
        self.head = nn.Conv1d(32, 1, kernel_size=5, padding=2)

    @staticmethod
    def _block(in_channels: int, out_channels: int, kernel: int) -> nn.Sequential:
        """Strided convolution, halving the time axis."""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel, stride=2,
                      padding=kernel // 2),
            nn.BatchNorm1d(out_channels), nn.GELU())

    @staticmethod
    def _up(in_channels: int, out_channels: int, kernel: int) -> nn.Sequential:
        """Upsample by 2 then convolve."""
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel,
                      padding=kernel // 2),
            nn.BatchNorm1d(out_channels), nn.GELU())

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the encoder feature map (batch, embedding_dim, time/16)."""
        return self.encoder(self.stem(x))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 128-d window embedding: the feature map mean-pooled over time."""
        return self.encode(x).mean(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct the normalised FHR at the input length."""
        features = self.encode(x)
        decoded = self.head(self.decoder(features))
        if decoded.shape[-1] != x.shape[-1]:
            decoded = nn.functional.interpolate(decoded, size=x.shape[-1],
                                                mode="linear", align_corners=False)
        return decoded.squeeze(1)


def masked_l1(prediction: torch.Tensor, target: torch.Tensor,
              mask: torch.Tensor) -> torch.Tensor:
    """L1 loss over the positions that were masked and were truly observed."""
    total = mask.sum()
    if total == 0:
        return prediction.sum() * 0.0
    return (torch.abs(prediction - target) * mask).sum() / total


@torch.no_grad()
def embedding_std(model: MaskedCTGAutoencoder, batch: torch.Tensor) -> float:
    """Mean over dimensions of the embedding standard deviation (collapse check)."""
    model.eval()
    embeddings = model.embed(batch)
    return float(embeddings.std(dim=0).mean().item())


def build_ctu_inputs(fhr_1hz: np.ndarray, uc_1hz: np.ndarray,
                     valid_1hz: np.ndarray) -> np.ndarray:
    """Build the 3-channel input for a CTU-UHB window at inference time.

    Nothing is masked here: `avail` is simply the observed mask, which is the
    unmasked case the embedding script needs.
    """
    observed = valid_1hz.astype(bool) & np.isfinite(fhr_1hz)
    fhr = normalise_fhr(np.nan_to_num(fhr_1hz, nan=C.FHR_NORM_CENTRE))
    fhr = np.where(observed, fhr, 0.0)
    uc = normalise_uc(np.nan_to_num(uc_1hz, nan=0.0))
    return np.stack([fhr, uc, observed.astype(np.float32)]).astype(np.float32)
