"""
Step 2 - One-time feature cache (all frozen parts of the pipeline).

For every video it decodes the frames ONCE (background thread) and, on the GPU:
  * FAST-VQA-B (frozen): 8 segment clips (32 frames each) -> pooled 768-D features
    + 4 official whole-video clips -> raw FAST-VQA quality score
  * CLIP ViT-B/16 (frozen): 8 frames per segment (64 total) -> 512-D embeddings
  * Motion statistics for EVERY consecutive frame pair (4 per pair)
  * Animation statistics for EVERY frame (14) + edge-map disagreement per pair

Raw per-frame / per-pair series are stored, so any pipeline can re-segment them
(K = 1, 2, 4 or 8) without re-extracting. Output: <work_dir>/features/<name>.npz
Already-extracted videos are skipped, so the script can be stopped and resumed.

Usage:
  python extract_features.py --dataset_root D:/BeDAMQA --work_dir D:/avqa_work \
      --fast_repo D:/FAST-VQA-and-FasterVQA --fast_weights D:/weights/FAST_VQA_B_1_4.pth
  (smoke test on 2 sources first:  --sources 1 2 )
"""
import argparse
import contextlib
import io
import json
import math
import os
import queue
import sys
import threading
import time
import types
import zlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common import (CLIP_FRAMES_PER_SEG, FAST_CLIP_LEN, FRAME_STAT_NAMES, MOTION_STAT_NAMES,
                    N_CACHE_SEGMENTS, load_metadata, seg_bounds)

# ---- constants (fixed for ALL videos; recorded in extraction_config.json) ----
FAST_MEAN = (123.675, 116.28, 103.53)          # official FAST-VQA normalisation (RGB, 0-255)
FAST_STD = (58.395, 57.12, 57.375)
FAST_SIG_MEAN, FAST_SIG_STD = -0.110198185, 0.04178565   # official FAST-VQA-B rescale (vqa.py)
FAST_WHOLE_CLIPS = 4                           # official test sampling: 4 clips x 32 frames
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_SIZE = 224
EDGE_THRESH = 0.10        # Sobel magnitude (image in [0,1]) above which a pixel is a strong edge
FLAT_THRESH = 0.02        # mean gradient magnitude below which a 16x16 patch is "flat"
CHANGE_THRESH = 0.02      # |frame difference| above which a pixel counts as changed (~5/255)
PATCH = 16
HUE_BINS = 36


# =============================================================================== FAST-VQA
def load_fast_vqa(repo, weights, device):
    """Import only fastvqa.models from the official repo (skips fastvqa.datasets, which
    needs decord/skvideo that we do not use) and load FAST-VQA-B weights."""
    repo = Path(repo).expanduser().resolve()
    pkg_dir = repo / "fastvqa"
    if not (pkg_dir / "models").is_dir():
        sys.exit(f"[ERROR] {repo} does not look like the FAST-VQA repo (no fastvqa/models).")
    try:                                  # only used by the X-CLIP variant, never by FAST-VQA-B
        import clip  # noqa: F401  (only checks it is importable)
    except Exception:
        sys.modules["clip"] = types.ModuleType("clip")
    pkg = types.ModuleType("fastvqa")
    pkg.__path__ = [str(pkg_dir)]
    sys.modules["fastvqa"] = pkg
    from fastvqa.models import DiViDeAddEvaluator
    import yaml

    with open(repo / "options" / "fast" / "fast-b.yml", "r") as f:
        opt = yaml.safe_load(f)
    model = DiViDeAddEvaluator(**opt["model"]["args"]).to(device)
    try:
        ckpt = torch.load(weights, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(weights, map_location=device)
    sd = ckpt.get("state_dict", ckpt)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    frag_args = opt["data"]["val-kv1k"]["args"]["sample_types"]["fragments"]
    return model, frag_args


def official_whole_indices(n, clip_len=FAST_CLIP_LEN, num_clips=FAST_WHOLE_CLIPS, frame_interval=1):
    """Exact copy of fastvqa SampleFrames._get_test_clips as called in the official vqa.py."""
    ori = clip_len * frame_interval
    avg = (n - ori + 1) / float(num_clips)
    if n > ori - 1:
        offs = (np.arange(num_clips) * avg + avg / 2.0).astype(np.int32)
    else:
        offs = np.zeros((num_clips,), np.int32)
    inds = offs[:, None] + np.arange(clip_len)[None, :] * frame_interval
    return np.mod(inds, n)


def segment_indices(n, k, per_seg):
    """per_seg frame indices spread uniformly (bin centres) inside each of k segments."""
    b = seg_bounds(n, k)
    out = []
    for s in range(k):
        b0, b1 = b[s], b[s + 1]
        L = max(b1 - b0, 1)
        idx = b0 + np.floor((np.arange(per_seg) + 0.5) * L / per_seg).astype(np.int64)
        out.append(np.clip(idx, 0, n - 1))
    return np.stack(out)


class FragmentPlan:
    """Same grid-mini-patch sampling as fastvqa get_spatial_fragments (7x7 grid of 32x32
    patches, one random offset set per 32-frame clip, upsample if the video is < 224)."""

    def __init__(self, H, W, n_clips, gen, device, fh=7, fw=7, fsh=32, fsw=32):
        size_h, size_w = fh * fsh, fw * fsw
        ratio = min(H / size_h, W / size_w)
        self.scale = (1.0 / ratio) if ratio < 1 else 1.0
        Hs, Ws = (int(math.floor(H * self.scale)), int(math.floor(W * self.scale))) if self.scale != 1.0 else (H, W)
        hgrids = torch.tensor([min(Hs // fh * i, Hs - fsh) for i in range(fh)])
        wgrids = torch.tensor([min(Ws // fw * i, Ws - fsw) for i in range(fw)])
        hl, wl = Hs // fh, Ws // fw
        rows, cols = [], []
        for _ in range(n_clips):
            rh = torch.randint(hl - fsh, (fh, fw), generator=gen) if hl > fsh else torch.zeros((fh, fw), dtype=torch.long)
            rw = torch.randint(wl - fsw, (fh, fw), generator=gen) if wl > fsw else torch.zeros((fh, fw), dtype=torch.long)
            hs = hgrids[:, None] + rh
            ws = wgrids[None, :] + rw
            rows.append(hs[:, :, None] + torch.arange(fsh))      # (fh, fw, fsh)
            cols.append(ws[:, :, None] + torch.arange(fsw))      # (fh, fw, fsw)
        self.rows = torch.stack(rows).to(device)
        self.cols = torch.stack(cols).to(device)
        self.out_hw = (fh * fsh, fw * fsw)

    def crop(self, frame, c):
        """frame: (H', W', 3) uint8 -> (224, 224, 3) fragment image for clip c."""
        p = frame[self.rows[c][:, :, :, None], self.cols[c][:, :, None, :]]   # (fh, fw, fsh, fsw, 3)
        return p.permute(0, 2, 1, 3, 4).reshape(self.out_hw[0], self.out_hw[1], 3)


# =============================================================================== stats (GPU)
class StatComputer:
    def __init__(self, device):
        sob = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device) / 4.0
        self.kx = sob.view(1, 1, 3, 3)
        self.ky = sob.t().contiguous().view(1, 1, 3, 3)
        self.klap = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=device).view(1, 1, 3, 3)
        x = torch.arange(11, device=device, dtype=torch.float32) - 5
        g = torch.exp(-x ** 2 / (2 * 1.5 ** 2))
        g = g / g.sum()
        self.gh, self.gv = g.view(1, 1, 1, 11), g.view(1, 1, 11, 1)

    def frame_stats(self, rgb):
        """rgb: (B,3,h,w) in [0,1] -> (B,14) stats, luminance (B,1,h,w), edge map (B,1,h,w)."""
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        yp = F.pad(y, (1, 1, 1, 1), mode="replicate")
        gx, gy = F.conv2d(yp, self.kx), F.conv2d(yp, self.ky)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
        edge = mag > EDGE_THRESH

        grad_mean = mag.mean((1, 2, 3))
        grad_std = mag.flatten(1).std(1)
        edge_frac = edge.float().mean((1, 2, 3))
        lap = F.conv2d(yp, self.klap)
        log_lap = torch.log10(lap.flatten(1).var(1) + 1e-8)
        jxx = F.avg_pool2d(gx * gx, 7, 1, 3, count_include_pad=False)
        jyy = F.avg_pool2d(gy * gy, 7, 1, 3, count_include_pad=False)
        jxy = F.avg_pool2d(gx * gy, 7, 1, 3, count_include_pad=False)
        coh = torch.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2 + 1e-12).sum((1, 2, 3)) / ((jxx + jyy).sum((1, 2, 3)) + 1e-8)

        H, W = y.shape[-2:]
        hp, wp = (H // PATCH) * PATCH, (W // PATCH) * PATCH
        yc, mc, rc = y[..., :hp, :wp], mag[..., :hp, :wp], rgb[..., :hp, :wp]
        flat = (F.avg_pool2d(mc, PATCH) < FLAT_THRESH).float()                   # (B,1,ph,pw)
        py, py2 = F.avg_pool2d(yc, PATCH), F.avg_pool2d(yc * yc, PATCH)
        p_ystd = torch.sqrt((py2 - py * py).clamp(min=0))
        prgb, prgb2 = F.avg_pool2d(rc, PATCH), F.avg_pool2d(rc * rc, PATCH)
        p_cstd = torch.sqrt((prgb2 - prgb * prgb).clamp(min=0)).mean(1, keepdim=True)
        nflat = flat.sum((1, 2, 3)).clamp(min=1)
        flat_frac = flat.mean((1, 2, 3))
        flat_lum = (p_ystd * flat).sum((1, 2, 3)) / nflat
        flat_rgb = (p_cstd * flat).sum((1, 2, 3)) / nflat
        dh = (prgb[:, :, 1:, :] - prgb[:, :, :-1, :]).abs().mean((1, 2, 3))
        dw = (prgb[:, :, :, 1:] - prgb[:, :, :, :-1]).abs().mean((1, 2, 3))
        nbr = 0.5 * (dh + dw)

        maxc, amax = rgb.max(1)
        minc = rgb.min(1)[0]
        delta = maxc - minc
        sat = torch.where(maxc > 1e-6, delta / maxc.clamp(min=1e-6), torch.zeros_like(maxc))
        val = maxc
        d = delta.clamp(min=1e-6)
        R, G, Bc = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        hue = torch.where(amax == 0, ((G - Bc) / d) % 6.0,
                          torch.where(amax == 1, (Bc - R) / d + 2.0, (R - G) / d + 4.0)) / 6.0
        hue = torch.where(delta > 1e-6, hue, torch.zeros_like(hue))
        valid = ((sat > 0.1) & (val > 0.1)).float().flatten(1)
        bins = (hue * HUE_BINS).long().clamp(0, HUE_BINS - 1).flatten(1)
        hist = torch.zeros(rgb.shape[0], HUE_BINS, device=rgb.device).scatter_add_(1, bins, valid)
        p = hist / hist.sum(1, keepdim=True).clamp(min=1)
        ent = -(p * torch.log2(p.clamp(min=1e-12))).sum(1)

        st = torch.stack([grad_mean, grad_std, edge_frac, log_lap, coh,
                          flat_frac, flat_lum, flat_rgb, nbr,
                          sat.mean((1, 2)), sat.flatten(1).std(1), val.mean((1, 2)), val.flatten(1).std(1), ent], 1)
        return st, y, edge

    def _blur(self, x):
        return F.conv2d(F.conv2d(x, self.gh), self.gv)

    def pair_stats(self, y, e):
        """y, e: (B+1,1,h,w) consecutive frames -> (B,5): 4 motion stats + edge disagreement."""
        a, b = y[1:], y[:-1]
        dabs = (a - b).abs()
        mad = dabs.mean((1, 2, 3))
        dstd = dabs.flatten(1).std(1)
        chg = (dabs > CHANGE_THRESH).float().mean((1, 2, 3))
        mu1, mu2 = self._blur(a), self._blur(b)
        s11 = self._blur(a * a) - mu1 * mu1
        s22 = self._blur(b * b) - mu2 * mu2
        s12 = self._blur(a * b) - mu1 * mu2
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s11 + s22 + C2))).mean((1, 2, 3))
        exor = (e[1:] != e[:-1]).float().mean((1, 2, 3))
        return torch.stack([mad, dstd, chg, 1.0 - ssim, exor], 1)


# =============================================================================== decoding
class ReaderThread(threading.Thread):
    """Decodes a video in the background (OpenCV releases the GIL) into uint8 BGR chunks."""

    def __init__(self, path, chunk, maxq=4):
        super().__init__(daemon=True)
        self.path, self.chunk = str(path), chunk
        self.q = queue.Queue(maxsize=maxq)
        self.stop_evt = threading.Event()
        self.start()

    def _put(self, item):
        while not self.stop_evt.is_set():
            try:
                self.q.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def run(self):
        cap = None
        try:
            cap = cv2.VideoCapture(self.path)
            if not cap.isOpened():
                raise RuntimeError(f"OpenCV cannot open {self.path}")
            buf = []
            while not self.stop_evt.is_set():
                ok, fr = cap.read()
                if not ok:
                    break
                buf.append(fr)
                if len(buf) == self.chunk:
                    if not self._put(np.stack(buf)):
                        return
                    buf = []
            if buf:
                self._put(np.stack(buf))
            self._put(None)
        except Exception as ex:          # forwarded to the consumer
            self._put(ex)
        finally:
            if cap is not None:
                cap.release()

    def close(self):
        """Stop decoding early (used after an error) and free queued frames."""
        self.stop_evt.set()
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                break

    def __iter__(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item


def probe(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if n <= 0:                                  # unreliable container -> count
        n = 0
        while cap.grab():
            n += 1
    cap.release()
    return n, fps


# =============================================================================== per video
def clip_preprocess(frames_u8, mean, std):
    """(b,H,W,3) uint8 RGB -> (b,3,224,224): resize short side to 224 (bicubic) + centre crop
    + CLIP normalisation, i.e. the official CLIP transform done on the GPU."""
    x = frames_u8.permute(0, 3, 1, 2).float().div_(255.0)
    H, W = x.shape[-2:]
    s = CLIP_SIZE / min(H, W)
    nh, nw = max(CLIP_SIZE, int(H * s)), max(CLIP_SIZE, int(W * s))
    x = F.interpolate(x, size=(nh, nw), mode="bicubic", antialias=True, align_corners=False).clamp_(0, 1)
    top, left = int(round((nh - CLIP_SIZE) / 2.0)), int(round((nw - CLIP_SIZE) / 2.0))
    x = x[..., top:top + CLIP_SIZE, left:left + CLIP_SIZE]
    return (x - mean) / std


def stream_video(reader, n, name, ctx):
    dev, args = ctx["device"], ctx["args"]
    seg_idx = segment_indices(n, N_CACHE_SEGMENTS, FAST_CLIP_LEN)              # (8, 32)
    whole_idx = official_whole_indices(n)                                      # (4, 32)
    clips_idx = np.concatenate([seg_idx, whole_idx], 0)                        # (12, 32)
    clipf_idx = segment_indices(n, N_CACHE_SEGMENTS, CLIP_FRAMES_PER_SEG).reshape(-1)   # (64,)
    fast_map, clip_map = defaultdict(list), defaultdict(list)
    for c in range(clips_idx.shape[0]):
        for p in range(clips_idx.shape[1]):
            fast_map[int(clips_idx[c, p])].append((c, p))
    for k, f in enumerate(clipf_idx):
        clip_map[int(f)].append(k)

    gen = torch.Generator().manual_seed(int(args.seed) * 1_000_003 + zlib.crc32(name.encode()))
    plan = None
    fast_buf = clip_buf = None
    fstats, pstats = [], []
    prev_y = prev_e = None
    f0 = 0
    for chunk in reader:
        t = torch.from_numpy(chunk).to(dev).flip(-1)             # (B,H,W,3) uint8 RGB
        B, H, W = t.shape[:3]
        if plan is None:
            plan = FragmentPlan(H, W, clips_idx.shape[0], gen, dev)
            fast_buf = torch.zeros((clips_idx.shape[0], FAST_CLIP_LEN, *plan.out_hw, 3), dtype=torch.uint8, device=dev)
            clip_buf = torch.zeros((len(clipf_idx), 3, CLIP_SIZE, CLIP_SIZE), dtype=torch.float32, device=dev)

        # FAST-VQA fragments (cropped at native resolution, as in the official sampler)
        need = [j for j in range(B) if (f0 + j) in fast_map]
        if need:
            src = t[need]
            if plan.scale != 1.0:
                up = F.interpolate(src.permute(0, 3, 1, 2).float() / 255.0, scale_factor=plan.scale, mode="bilinear")
                src = (up * 255.0).to(torch.uint8).permute(0, 2, 3, 1)
            for jj, j in enumerate(need):
                for c, p in fast_map[f0 + j]:
                    fast_buf[c, p] = plan.crop(src[jj], c)

        # CLIP frames
        needc = [j for j in range(B) if (f0 + j) in clip_map]
        if needc:
            x = clip_preprocess(t[needc], ctx["clip_mean"], ctx["clip_std"])
            for jj, j in enumerate(needc):
                for k in clip_map[f0 + j]:
                    clip_buf[k] = x[jj]

        # motion + animation statistics on every frame (fixed analysis resolution)
        rgb = t.permute(0, 3, 1, 2).float().div_(255.0)
        if args.analysis_short_side > 0 and min(H, W) > args.analysis_short_side:
            s = args.analysis_short_side / min(H, W)
            rgb = F.interpolate(rgb, size=(int(round(H * s)), int(round(W * s))), mode="bilinear",
                                antialias=True, align_corners=False).clamp_(0, 1)
        st, y, e = ctx["stats"].frame_stats(rgb)
        fstats.append(st)
        if prev_y is not None:
            y, e_cat = torch.cat([prev_y, y]), torch.cat([prev_e, e])
        else:
            e_cat = e
        if y.shape[0] > 1:
            pstats.append(ctx["stats"].pair_stats(y, e_cat))
        prev_y, prev_e = y[-1:], e_cat[-1:]
        f0 += B
        del t, rgb

    if f0 == 0:
        raise RuntimeError("no frames decoded")
    return {"count": f0, "hw": (H, W), "analysis_hw": tuple(prev_y.shape[-2:]),
            "fast_buf": fast_buf, "clip_buf": clip_buf,
            "fstats": torch.cat(fstats), "pstats": torch.cat(pstats) if pstats else torch.zeros((0, 5), device=dev)}


@torch.no_grad()
def finalize(state, n, fps, out_path, ctx):
    dev, args = ctx["device"], ctx["args"]
    amp = dict(device_type="cuda", dtype=torch.float16, enabled=(args.amp and dev.type == "cuda"))

    # FAST-VQA-B: 8 segment clips + 4 official whole-video clips
    feats, scores = [], []
    fb = state["fast_buf"]
    for i in range(0, fb.shape[0], args.fast_batch):
        x = fb[i:i + args.fast_batch].float()
        x = ((x - ctx["fast_mean"]) / ctx["fast_std"]).permute(0, 4, 1, 2, 3).contiguous()   # (b,3,32,224,224)
        with torch.autocast(**amp), contextlib.redirect_stdout(io.StringIO()):   # FAST-VQA prints window sizes
            s, f = ctx["fast"]({"fragments": x}, return_pooled_feats=True)
        scores.append(s.float().mean((1, 2, 3, 4)))
        feats.append(f["fragments"].float())
    scores = torch.cat(scores).cpu().numpy()
    feats = torch.cat(feats).cpu().numpy()
    whole = scores[N_CACHE_SEGMENTS:]
    raw = float(whole.mean())

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev.type == "cuda")):
        emb = ctx["clip"].encode_image(state["clip_buf"])
    emb = emb.float().cpu().numpy()

    ps = state["pstats"].cpu().numpy()
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh,
                 fast_seg_feat=feats[:N_CACHE_SEGMENTS].astype(np.float32),
                 fast_seg_score=scores[:N_CACHE_SEGMENTS].astype(np.float32),
                 fast_whole_clip_scores=whole.astype(np.float32),
                 fast_whole_raw=np.float64(raw),
                 fast_whole_sigmoid=np.float64(1.0 / (1.0 + np.exp(-(raw - FAST_SIG_MEAN) / FAST_SIG_STD))),
                 clip_feat=emb.astype(np.float16),
                 frame_stats=state["fstats"].cpu().numpy().astype(np.float32),
                 pair_motion=ps[:, :4].astype(np.float32),
                 pair_edge_xor=ps[:, 4].astype(np.float32),
                 n_frames=np.int64(n), fps=np.float64(fps),
                 height=np.int64(state["hw"][0]), width=np.int64(state["hw"][1]),
                 analysis_hw=np.array(state["analysis_hw"], np.int64))
    os.replace(tmp, out_path)


# =============================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--fast_repo", required=True, help="Clone of github.com/VQAssessment/FAST-VQA-and-FasterVQA")
    ap.add_argument("--fast_weights", required=True, help="FAST_VQA_B_1_4.pth")
    ap.add_argument("--clip_model", default="ViT-B-16")
    ap.add_argument("--clip_pretrained", default="openai", help="'openai' or a local open_clip checkpoint path")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--analysis_short_side", type=int, default=540,
                    help="Short side used for motion/animation stats (0 = native). Keep fixed across runs.")
    ap.add_argument("--chunk", type=int, default=16, help="Frames per GPU chunk (lower if out of memory)")
    ap.add_argument("--fast_batch", type=int, default=4, help="FAST-VQA clips per forward pass (lower if OOM)")
    ap.add_argument("--amp", action="store_true", help="fp16 for FAST-VQA (faster; default fp32 like official)")
    ap.add_argument("--sources", type=int, nargs="*", default=None, help="Only these source IDs (smoke test)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="Fragment-offset seed (fixed -> reproducible features)")
    args = ap.parse_args()

    root = Path(args.dataset_root).expanduser().resolve()
    work = Path(args.work_dir).expanduser().resolve()
    out_dir = work / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)

    cfg = {k: v for k, v in vars(args).items() if k not in ("sources", "overwrite")}
    cfg.update(EDGE_THRESH=EDGE_THRESH, FLAT_THRESH=FLAT_THRESH, CHANGE_THRESH=CHANGE_THRESH, PATCH=PATCH,
               HUE_BINS=HUE_BINS, N_CACHE_SEGMENTS=N_CACHE_SEGMENTS, FAST_CLIP_LEN=FAST_CLIP_LEN,
               CLIP_FRAMES_PER_SEG=CLIP_FRAMES_PER_SEG, frame_stats=FRAME_STAT_NAMES, pair_motion=MOTION_STAT_NAMES)
    cfg_path = work / "extraction_config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        for key in ("analysis_short_side", "seed", "clip_model", "clip_pretrained", "EDGE_THRESH", "FLAT_THRESH", "CHANGE_THRESH"):
            if old.get(key) != cfg.get(key) and not args.overwrite:
                sys.exit(f"[ERROR] '{key}' differs from the existing cache ({old.get(key)} vs {cfg.get(key)}). "
                         f"Mixed settings would make features inconsistent. Use --overwrite to re-extract everything.")
    cfg_path.write_text(json.dumps(cfg, indent=2, default=str))

    meta = load_metadata(work)
    if args.sources:
        meta = meta[meta["source"].isin(args.sources)]
    todo = [r for r in meta.itertuples() if args.overwrite or not (out_dir / f"{r.name}.npz").exists()]
    print(f"{len(meta)} videos selected, {len(meta) - len(todo)} already cached, {len(todo)} to extract.")
    if not todo:
        return

    print("Loading FAST-VQA-B ...")
    fast, _ = load_fast_vqa(args.fast_repo, args.fast_weights, dev)
    print("Loading CLIP ...")
    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_pretrained, device=dev)
    clip_model.eval()
    ctx = {
        "device": dev, "args": args, "fast": fast, "clip": clip_model, "stats": StatComputer(dev),
        "fast_mean": torch.tensor(FAST_MEAN, device=dev), "fast_std": torch.tensor(FAST_STD, device=dev),
        "clip_mean": torch.tensor(CLIP_MEAN, device=dev).view(1, 3, 1, 1),
        "clip_std": torch.tensor(CLIP_STD, device=dev).view(1, 3, 1, 1),
    }

    fail_log = work / "extraction_failures.txt"
    t_start = time.time()
    next_reader = ReaderThread(root / todo[0].rel_path, args.chunk)
    for i, r in enumerate(todo):
        t0 = time.time()
        path = root / r.rel_path
        reader = next_reader
        next_reader = None
        try:
            n, fps = probe(path)
            with torch.no_grad():
                state = stream_video(reader, n, r.name, ctx)
            if i + 1 < len(todo):                       # decode next video while the GPU finishes this one
                next_reader = ReaderThread(root / todo[i + 1].rel_path, args.chunk)
            if state["count"] != n:                     # container frame count was wrong -> resample with true count
                print(f"  [info] {r.name}: header said {n} frames, decoded {state['count']} -> re-sampling")
                n = state["count"]
                del state
                reader = ReaderThread(path, args.chunk)
                with torch.no_grad():
                    state = stream_video(reader, n, r.name, ctx)
            finalize(state, n, fps, out_dir / f"{r.name}.npz", ctx)
            del state
            el = time.time() - t_start
            eta = el / (i + 1) * (len(todo) - i - 1)
            print(f"[{i + 1}/{len(todo)}] {r.name}: {n} frames @ {fps:.2f} fps  ({time.time() - t0:.1f}s, ETA {eta / 60:.1f} min)")
        except Exception as ex:
            msg = f"{r.name}\t{type(ex).__name__}: {ex}"
            print(f"[FAIL] {msg}")
            with open(fail_log, "a") as fh:
                fh.write(msg + "\n")
            reader.close()
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            if next_reader is None and i + 1 < len(todo):
                next_reader = ReaderThread(root / todo[i + 1].rel_path, args.chunk)
    print(f"Done in {(time.time() - t_start) / 60:.1f} min. Failures (if any) in {fail_log}")


if __name__ == "__main__":
    main()
