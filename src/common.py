"""
Shared utilities for every pipeline (P0..P4): segment boundaries, loading the
feature cache into K-segment tensors, train-fold-only standardisation, FAST
calibration, metrics and seeding. Keeping these in one place is what makes
the pipeline comparison fair.
"""
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

N_CACHE_SEGMENTS = 8          # features are cached at 8 segments; K must divide 8
FAST_CLIP_LEN = 32            # frames per FAST-VQA clip
CLIP_FRAMES_PER_SEG = 8       # CLIP frames per cached segment
MOTION_STEPS = 32             # motion sequence length per segment (resampled)

# per-frame animation statistics stored by extract_features.py (order matters)
FRAME_STAT_NAMES = [
    "grad_mean", "grad_std", "strong_edge_frac", "log_laplacian_var", "edge_coherence",     # edge & contour (5)
    "flat_patch_frac", "flat_patch_lum_std", "flat_patch_rgb_disp", "nbr_patch_color_diff",  # flat colour (4)
    "sat_mean", "sat_std", "val_mean", "val_std", "hue_entropy",                            # colour (5)
]
# per consecutive-frame-pair statistics stored by extract_features.py
MOTION_STAT_NAMES = ["mean_abs_diff", "diff_std", "changed_pixel_frac", "ssim_change"]
TEMPORAL_STAT_NAMES = [
    "edge_disagree_mean", "edge_disagree_std", "sat_change_mean",
    "bright_change_mean", "bright_hf_variation", "edge_density_change",
]
ANIM_FEATURE_NAMES = FRAME_STAT_NAMES + TEMPORAL_STAT_NAMES      # 20
FI = {n: i for i, n in enumerate(FRAME_STAT_NAMES)}


def seg_bounds(n_frames, k):
    """Frame boundaries of k equal temporal segments: [b0=0, b1, ..., bk=n_frames]."""
    return [int(np.floor(s * n_frames / k + 0.5)) for s in range(k + 1)]


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# --------------------------------------------------------------------------- data
def load_metadata(work_dir):
    meta = pd.read_csv(Path(work_dir) / "metadata.csv")
    meta = meta[meta["available"].astype(bool)].sort_values("name").reset_index(drop=True)
    return meta


def load_splits(work_dir):
    return json.loads((Path(work_dir) / "splits.json").read_text())


def _resample(seq, steps):
    """seq: (L, C) -> (steps, C) by linear interpolation over normalised time."""
    L = seq.shape[0]
    if L == 0:
        return np.zeros((steps, seq.shape[1]), np.float32)
    if L == 1:
        return np.repeat(seq, steps, axis=0).astype(np.float32)
    x_old = np.linspace(0.0, 1.0, L)
    x_new = np.linspace(0.0, 1.0, steps)
    return np.stack([np.interp(x_new, x_old, seq[:, c]) for c in range(seq.shape[1])], 1).astype(np.float32)


def _segment_features(z, k):
    """Turn one cached video (npz dict) into K-segment inputs."""
    n = int(z["n_frames"])
    g = N_CACHE_SEGMENTS // k

    fast = z["fast_seg_feat"].astype(np.float32).reshape(k, g, -1).mean(1)                 # (K, 768)
    clip = z["clip_feat"].astype(np.float32)                                               # (8*8, 512)
    clip = clip / (np.linalg.norm(clip, axis=-1, keepdims=True) + 1e-8)
    clip = clip.reshape(k, g * CLIP_FRAMES_PER_SEG, -1).mean(1)                            # (K, 512)

    fr = z["frame_stats"].astype(np.float32)          # (N, 14)
    pm = z["pair_motion"].astype(np.float32)          # (N-1, 4) pair j = frames (j, j+1)
    pe = z["pair_edge_xor"].astype(np.float32)        # (N-1,)

    b = seg_bounds(n, k)
    motion, anim = [], []
    for s in range(k):
        b0, b1 = b[s], b[s + 1]
        p0, p1 = max(b0 - 1, 0), max(b1 - 1, 0)       # pairs whose 2nd frame is in [b0, b1)
        motion.append(_resample(pm[p0:p1], MOTION_STEPS).T)                                # (4, 32)

        fs = fr[b0:b1] if b1 > b0 else fr[min(b0, n - 1):min(b0, n - 1) + 1]
        spatial = fs.mean(0)                                                               # (14,)
        ext = fr[max(b0 - 1, 0):max(b1, b0 + 1)]       # include previous frame for changes
        ep = pe[p0:p1]

        def absdiff_mean(col):
            v = ext[:, FI[col]]
            return float(np.abs(np.diff(v)).mean()) if len(v) > 1 else 0.0

        v = ext[:, FI["val_mean"]]
        temporal = np.array([
            float(ep.mean()) if len(ep) else 0.0,
            float(ep.std()) if len(ep) else 0.0,
            absdiff_mean("sat_mean"),
            absdiff_mean("val_mean"),
            float(np.abs(np.diff(v, n=2)).mean()) if len(v) > 2 else 0.0,
            absdiff_mean("strong_edge_frac"),
        ], np.float32)
        anim.append(np.concatenate([spatial, temporal]))
    return fast, clip, np.stack(motion), np.stack(anim)


def load_feature_arrays(work_dir, meta, k=4):
    """Load all cached videos into arrays ordered like `meta`."""
    assert N_CACHE_SEGMENTS % k == 0, f"K={k} must divide {N_CACHE_SEGMENTS}"
    fdir = Path(work_dir) / "features"
    F, C, M, A, QF, missing = [], [], [], [], [], []
    for name in meta["name"]:
        p = fdir / f"{name}.npz"
        if not p.exists():
            missing.append(name)
            continue
        with np.load(p) as z:
            f, c, m, a = _segment_features(z, k)
            F.append(f); C.append(c); M.append(m); A.append(a)
            QF.append(float(z["fast_whole_raw"]))
    if missing:
        raise FileNotFoundError(f"{len(missing)} videos have no cached features, e.g. {missing[:5]}. "
                                f"Run extract_features.py first.")
    return {
        "fast": np.stack(F), "clip": np.stack(C), "motion": np.stack(M), "anim": np.stack(A),
        "q_fast_raw": np.asarray(QF, np.float64), "mos": meta["mos"].to_numpy(np.float64),
    }


def source_indices(meta, sources):
    return np.where(meta["source"].isin(sources).to_numpy())[0]


# --------------------------------------------------------------------------- standardisation
class Standardizer:
    """z-score per feature channel, statistics from training videos only."""

    def __init__(self, axis):
        self.axis = axis
        self.mean = None
        self.std = None

    def fit(self, x):
        self.mean = x.mean(axis=self.axis, keepdims=True)
        self.std = x.std(axis=self.axis, keepdims=True) + 1e-6
        return self

    def transform(self, x):
        return ((x - self.mean) / self.std).astype(np.float32)

    def state(self):
        return {"mean": self.mean, "std": self.std}


# --------------------------------------------------------------------------- FAST calibration
def _logistic4(x, a, b, c, d):
    return a + (b - a) / (1.0 + np.exp(-(x - c) / d))


class FastCalibrator:
    """Maps the raw FAST-VQA score to the MOS scale; fitted on training videos only."""

    def __init__(self, kind="linear"):
        self.kind = kind
        self.params = None

    def fit(self, s, mos):
        p = np.polyfit(s, mos, 1)
        self.params = ("linear", p)
        if self.kind == "logistic":
            try:
                p0 = [mos.min(), mos.max(), float(np.median(s)), float(np.std(s) + 1e-6)]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pl, _ = curve_fit(_logistic4, s, mos, p0=p0, maxfev=20000)
                self.params = ("logistic", pl)
            except Exception:
                print("  [calib] logistic fit failed -> linear")
        return self

    def predict(self, s):
        kind, p = self.params
        return np.polyval(p, s) if kind == "linear" else _logistic4(s, *p)


# --------------------------------------------------------------------------- metrics
def _logistic5(x, b1, b2, b3, b4, b5):
    return b1 * (0.5 - 1.0 / (1.0 + np.exp(b2 * (x - b3)))) + b4 * x + b5


def logistic5_map(pred, mos):
    """Standard VQEG 5-parameter monotonic mapping used before PLCC."""
    try:
        p0 = [float(mos.max() - mos.min()), 1.0 / (float(np.std(pred)) + 1e-6),
              float(np.mean(pred)), 0.0, float(np.mean(mos))]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, _ = curve_fit(_logistic5, pred, mos, p0=p0, maxfev=20000)
        out = _logistic5(pred, *p)
        if not np.all(np.isfinite(out)):
            raise RuntimeError
        return out
    except Exception:
        return pred


def compute_metrics(pred, mos):
    pred = np.asarray(pred, np.float64)
    mos = np.asarray(mos, np.float64)
    if len(pred) < 3 or np.std(pred) < 1e-12:
        return {"SROCC": np.nan, "KRCC": np.nan, "PLCC": np.nan, "PLCC_raw": np.nan,
                "RMSE": float(np.sqrt(np.mean((pred - mos) ** 2))), "n": len(pred)}
    mapped = logistic5_map(pred, mos)
    return {
        "SROCC": float(stats.spearmanr(pred, mos)[0]),
        "KRCC": float(stats.kendalltau(pred, mos)[0]),
        "PLCC": float(stats.pearsonr(mapped, mos)[0]),
        "PLCC_raw": float(stats.pearsonr(pred, mos)[0]),
        "RMSE": float(np.sqrt(np.mean((pred - mos) ** 2))),      # on the MOS scale, no mapping
        "n": int(len(pred)),
    }


def metrics_by_distortion(df, pred_col="pred", mos_col="mos"):
    out = {"ALL": compute_metrics(df[pred_col], df[mos_col])}
    for d, g in df.groupby("distortion"):
        out[d] = compute_metrics(g[pred_col], g[mos_col])
    return out
