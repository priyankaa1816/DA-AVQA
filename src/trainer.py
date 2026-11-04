"""
Shared training / evaluation protocol for ALL pipelines (P1, P2, P3, P4 ...).

Each pipeline file only defines its model and calls `run(args, model_fn, pipeline)`.
Everything else - splits, standardisation, FAST calibration, optimiser, early stopping,
metrics, output files - lives here, so the pipelines differ ONLY in the architecture.

Model contract:
    pred, extras = model(fast, clip, motion, anim, q_fast)
        fast (B,K,768)  clip (B,K,512)  motion (B,K,4,32)  anim (B,K,20)  q_fast (B,)
    extras must contain:
        "delta"  (B,)     learned correction dQ
        "alpha"  (B,K)    segment attention weights
        "branch" (B,K,4)  importance of [fast, clip, motion, anim] per segment
    optional:
        "subscores" (B,4) per-branch quality sub-scores (late-fusion pipelines)
"""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

from common import (FastCalibrator, Standardizer, compute_metrics, load_feature_arrays, load_metadata,
                    load_splits, metrics_by_distortion, set_seed, source_indices)

BRANCH_NAMES = ["fast", "clip", "motion", "anim"]
METRIC_KEYS = ["SROCC", "KRCC", "PLCC", "PLCC_raw", "RMSE"]


# =============================================================================== args
def build_parser(default_tag, default_segments=4):
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--mode", choices=["cv", "final"], default="cv")
    ap.add_argument("--tag", default=default_tag, help="Results go to <work_dir>/results/<tag>/<mode>")
    ap.add_argument("--segments", type=int, default=default_segments, help="K temporal segments (must divide 8)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--min_epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--huber_delta", type=float, default=1.0)
    ap.add_argument("--rank_weight", type=float, default=0.0, help="lambda for the pairwise ranking loss (0 = off)")
    ap.add_argument("--clip_grad", type=float, default=1.0)
    ap.add_argument("--calib", choices=["linear", "logistic"], default="linear", help="FAST score -> MOS calibration")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


# =============================================================================== helpers
def rank_loss(pred, y):
    """Eq. 22: mean log(1 + exp(-sgn(yi - yj)(pi - pj))) over pairs with different MOS."""
    dp = pred[:, None] - pred[None, :]
    dy = y[:, None] - y[None, :]
    mask = torch.triu(torch.ones_like(dy, dtype=torch.bool), 1) & (dy.abs() > 1e-6)
    if not mask.any():
        return pred.sum() * 0.0
    return F.softplus(-torch.sign(dy[mask]) * dp[mask]).mean()


def prepare_fold(data, tr, calib_kind, device):
    """Fit standardisation + FAST calibration on the training videos only."""
    ms = Standardizer(axis=(0, 1, 3)).fit(data["motion"][tr])   # per motion channel
    an = Standardizer(axis=(0, 1)).fit(data["anim"][tr])        # per animation feature
    cal = FastCalibrator(calib_kind).fit(data["q_fast_raw"][tr], data["mos"][tr])

    def t(x):
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)

    T = {"fast": t(data["fast"]), "clip": t(data["clip"]), "motion": t(ms.transform(data["motion"])),
         "anim": t(an.transform(data["anim"])), "qfast": t(cal.predict(data["q_fast_raw"])), "mos": t(data["mos"])}
    norm = {"motion": ms.state(), "anim": an.state(), "calib": cal.params}
    return T, norm


def _inputs(T, idx):
    return T["fast"][idx], T["clip"][idx], T["motion"][idx], T["anim"][idx], T["qfast"][idx]


@torch.no_grad()
def predict(model, T, idx):
    model.eval()
    idx_t = torch.as_tensor(idx, device=T["mos"].device)
    pred, ex = model(*_inputs(T, idx_t))
    out = {"pred": pred.cpu().numpy(), "qfast": T["qfast"][idx_t].cpu().numpy(),
           "delta": ex["delta"].cpu().numpy(), "alpha": ex["alpha"].cpu().numpy(),
           "branch": ex["branch"].mean(1).cpu().numpy()}
    if "subscores" in ex:
        out["subscores"] = ex["subscores"].cpu().numpy()
    return out


def quick_val_score(pred, mos):
    if np.std(pred) < 1e-12:
        return -np.inf
    s = stats.spearmanr(pred, mos)[0] + stats.pearsonr(pred, mos)[0]
    return float(s / 2) if np.isfinite(s) else -np.inf


def split_fingerprint(work):
    return hashlib.sha256((Path(work) / "splits.json").read_bytes()).hexdigest()[:12]


def cache_fingerprint(work, meta):
    """Hash of metadata + every cached .npz file. Independent of K, so it is the SAME for
    P1/P2 (K=4) and P3 (K=8) when everyone uses the same feature cache."""
    work = Path(work)
    h = hashlib.sha256()
    h.update(meta[["name", "source", "distortion", "level", "mos"]].to_csv(index=False).encode())
    for name in meta["name"]:
        h.update(name.encode())
        h.update((work / "features" / f"{name}.npz").read_bytes())
    return h.hexdigest()[:12]


def input_fingerprint(data):
    """Hash of the K-segment model inputs (differs between K=4 and K=8 by design)."""
    h = hashlib.sha256()
    for k in ("fast", "clip", "motion", "anim", "q_fast_raw", "mos"):
        h.update(np.ascontiguousarray(data[k]).tobytes())
    return h.hexdigest()[:12]


# =============================================================================== training
def train_one(model_fn, T, tr, va, seed, args, device):
    set_seed(seed)
    model = model_fn().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    tr_t = torch.as_tensor(tr, device=device)
    mos_va = T["mos"][torch.as_tensor(va, device=device)].cpu().numpy()
    gen = torch.Generator().manual_seed(seed)

    best_score, best_state, best_ep, bad = -np.inf, copy.deepcopy(model.state_dict()), -1, 0
    for ep in range(args.epochs):
        model.train()
        perm = tr_t[torch.randperm(len(tr), generator=gen).to(device)]
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            if len(b) < 2:
                continue
            pred, _ = model(*_inputs(T, b))
            y = T["mos"][b]
            loss = F.huber_loss(pred, y, delta=args.huber_delta)
            if args.rank_weight > 0:
                loss = loss + args.rank_weight * rank_loss(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()

        score = quick_val_score(predict(model, T, va)["pred"], mos_va)
        if score > best_score + 1e-4:
            best_score, best_state, best_ep, bad = score, copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= args.patience and ep + 1 >= args.min_epochs:
                break
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_ep + 1, "val_score": best_score, "epochs_run": ep + 1}


# =============================================================================== main loop
def run(args, model_fn, pipeline):
    work = Path(args.work_dir).expanduser().resolve()
    out = work / "results" / args.tag / args.mode
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    meta = load_metadata(work)
    splits = load_splits(work)
    print(f"[{pipeline}] loading cached features for {len(meta)} videos (K={args.segments}) ...")
    data = load_feature_arrays(work, meta, k=args.segments)
    split_id, cache_id, data_id = split_fingerprint(work), cache_fingerprint(work, meta), input_fingerprint(data)
    n_params = sum(p.numel() for p in model_fn().parameters() if p.requires_grad)
    print(f"[{pipeline}] split id {split_id} | cache id {cache_id} | data id (K={args.segments}) {data_id} "
          f"| trainable params {n_params:,}")
    print("           (every pipeline must print the SAME split id and cache id; "
          "data id must match between pipelines with the same K)")

    if args.mode == "cv":
        runs = splits["folds"]
    else:
        print("\n*** FINAL MODE: evaluates on the locked test sources. Run this once, after all choices are made. ***\n")
        runs = [dict(splits["final"], fold="final")]

    run_rows, pred_frames = [], []
    t_all = time.time()
    for fold in runs:
        tr_s, va_s, te_s = set(fold["train_sources"]), set(fold["val_sources"]), set(fold["test_sources"])
        assert not (tr_s & va_s or tr_s & te_s or va_s & te_s), "source overlap between splits!"
        tr = source_indices(meta, fold["train_sources"])
        va = source_indices(meta, fold["val_sources"])
        te = source_indices(meta, fold["test_sources"])
        T, norm = prepare_fold(data, tr, args.calib, device)
        for seed in args.seeds:
            t0 = time.time()
            model, info = train_one(model_fn, T, tr, va, seed, args, device)
            r = predict(model, T, te)
            mos_te = data["mos"][te]
            m_model = compute_metrics(r["pred"], mos_te)
            m_fast = compute_metrics(r["qfast"], mos_te)

            row = {"pipeline": pipeline, "fold": fold["fold"], "seed": seed, "n_train": len(tr), "n_val": len(va),
                   "n_test": len(te), **info, "time_s": round(time.time() - t0, 1)}
            row.update({k: m_model[k] for k in METRIC_KEYS})
            row.update({f"fast_{k}": m_fast[k] for k in METRIC_KEYS})
            run_rows.append(row)

            df = meta.iloc[te][["name", "source", "distortion", "level", "mos"]].copy()
            df["pred"], df["q_fast_cal"], df["delta_q"] = r["pred"], r["qfast"], r["delta"]
            for k in range(r["alpha"].shape[1]):
                df[f"alpha_{k + 1}"] = r["alpha"][:, k]
            for j, bname in enumerate(BRANCH_NAMES):
                df[f"branch_{bname}"] = r["branch"][:, j]
            if "subscores" in r:
                for j, bname in enumerate(BRANCH_NAMES):
                    df[f"sub_{bname}"] = r["subscores"][:, j]
            df["fold"], df["seed"] = fold["fold"], seed
            pred_frames.append(df)

            torch.save({"pipeline": pipeline, "model": model.state_dict(), "norm": norm, "args": vars(args),
                        "fold": fold, "seed": seed}, out / "checkpoints" / f"fold{fold['fold']}_seed{seed}.pt")
            print(f"fold {fold['fold']} seed {seed}: SROCC {m_model['SROCC']:.4f}  PLCC {m_model['PLCC']:.4f}  "
                  f"KRCC {m_model['KRCC']:.4f}  RMSE {m_model['RMSE']:.4f} | FAST-only SROCC {m_fast['SROCC']:.4f} "
                  f"| best ep {info['best_epoch']} ({row['time_s']}s)")

    runs_df = pd.DataFrame(run_rows)
    preds = pd.concat(pred_frames, ignore_index=True)
    runs_df.to_csv(out / "runs.csv", index=False)
    preds.to_csv(out / "predictions.csv", index=False)

    # ---- mean +- std over fold x seed runs
    summary = {"pipeline": pipeline, "split_id": split_id, "cache_id": cache_id, "data_id": data_id,
               "segments": args.segments, "trainable_params": n_params,
               "args": vars(args), "n_runs": len(runs_df), "per_run": {}}
    print(f"\n===== {pipeline} [{args.mode}]  mean ± std over {len(runs_df)} runs =====")
    for prefix, label in [("", pipeline), ("fast_", "FAST-only (calibrated)")]:
        cells = []
        for k in METRIC_KEYS:
            mu, sd = runs_df[prefix + k].mean(), runs_df[prefix + k].std(ddof=0)
            summary["per_run"][f"{label}_{k}"] = {"mean": float(mu), "std": float(sd)}
            cells.append(f"{k} {mu:.4f}±{sd:.4f}")
        print(f"{label:24s} " + "  ".join(cells))

    # ---- pooled test predictions per seed, overall + per distortion
    rows = []
    for seed, g in preds.groupby("seed"):
        for col, label in [("pred", pipeline), ("q_fast_cal", "FAST-only")]:
            for dist, m in metrics_by_distortion(g, pred_col=col).items():
                rows.append({"model": label, "seed": seed, "distortion": dist, **{k: m[k] for k in METRIC_KEYS}})
    pooled = pd.DataFrame(rows).groupby(["model", "distortion"])[METRIC_KEYS].mean().reset_index()
    pooled.to_csv(out / "per_distortion.csv", index=False)
    summary["pooled_by_distortion"] = pooled.to_dict(orient="records")
    print("\nPooled test predictions per distortion (mean over seeds):")
    print(pooled.pivot(index="distortion", columns="model", values="SROCC").round(4).add_prefix("SROCC ").to_string())

    # ---- explainability: branch importance / segment attention / dQ per distortion
    bcols = [f"branch_{b}" for b in BRANCH_NAMES]
    acols = [c for c in preds.columns if c.startswith("alpha_")]
    scols = [c for c in preds.columns if c.startswith("sub_")]
    expl = preds.groupby("distortion")[bcols + acols + scols + ["delta_q"]].mean()
    expl.to_csv(out / "branch_attention_by_distortion.csv")
    print("\nMean branch importance / segment attention / ΔQ per distortion:")
    print(expl.round(3).to_string())

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nSaved results to {out}   (total {(time.time() - t_all) / 60:.1f} min)")
