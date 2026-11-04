"""
DA-AVQA - full run of the selected pipeline (P1) on the complete BeDAMQA dataset.

Put this file (and report_p1.py) in your existing code folder next to common.py, trainer.py,
blocks.py and train_p1.py. Nothing in those files is changed, and the feature cache you already
extracted is reused as-is.

Stages
  data      metadata.csv + splits.json        -> prepare_data.py runs ONLY if they are missing
  features  cached FAST-VQA / CLIP / motion / animation features
            -> reused; extract_features.py runs ONLY if some videos have no features yet
               (it skips the videos that are already done)
  cv        4 source-grouped folds x N seeds on the 12 dev sources
            -> an out-of-fold prediction for every dev video (main performance estimate)
  final     train on 10 dev sources (val 2), evaluate ONCE on the locked test sources 12/14/15
            -> N seeds + seed ensemble
  deploy    train on all 15 sources for a fixed number of epochs (median best epoch of 'final')
            -> checkpoints for later use (e.g. cross-dataset on CGVQM+D); it has no test numbers
  report    tables (CSV / Markdown / LaTeX) + figures (PNG, 600 dpi) -> report_p1.py

Training protocol, hyper-parameters, standardisation and FAST calibration are identical to the
pipeline comparison (same defaults as trainer.py). The only additions are per-epoch learning
curves and inference timing, which do not change what gets trained.

Usage (all features already extracted):
  python run_full_p1.py --work_dir "D:/avqa_work"

Usage (some videos still need features):
  python run_full_p1.py --work_dir "D:/avqa_work" --dataset_root "D:/BeDAMQA" \
      --fast_repo "D:/FAST-VQA-and-FasterVQA" --fast_weights "D:/weights/FAST_VQA_B_1_4.pth"

Only regenerate the report (no GPU, no training):
  python run_full_p1.py --work_dir "D:/avqa_work" --stages report

Outputs: <work_dir>/results/P1_full/{cv,final,deploy,report}/
"""
import argparse
import copy
import hashlib
import json
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common import (compute_metrics, load_feature_arrays, load_metadata, load_splits,  # noqa: E402
                    set_seed, source_indices)
from trainer import BRANCH_NAMES, METRIC_KEYS, predict, prepare_fold, quick_val_score, rank_loss  # noqa: E402
from train_p1 import P1Model  # noqa: E402

PIPELINE = "P1"
EXPECTED_VIDEOS = 374
TRAIN_STAGES = ("cv", "final", "deploy")


# =============================================================================== arguments
def build_parser():
    ap = argparse.ArgumentParser(description="Full DA-AVQA run of pipeline P1 (input -> trained models -> report)")
    ap.add_argument("--work_dir", required=True, help="Folder with metadata.csv, splits.json and features/")
    ap.add_argument("--tag", default="P1_full", help="Results go to <work_dir>/results/<tag>/")
    ap.add_argument("--stages", nargs="+", default=["cv", "final", "deploy", "report"],
                    choices=["cv", "final", "deploy", "report"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="Seeds 0-2 are the ones used in the pipeline comparison")
    # only needed if metadata / features are missing
    ap.add_argument("--dataset_root", default=None, help="Folder with MOSScores.xlsx and the distortion folders")
    ap.add_argument("--fast_repo", default=None, help="Clone of FAST-VQA-and-FasterVQA")
    ap.add_argument("--fast_weights", default=None, help="FAST_VQA_B_1_4.pth")
    ap.add_argument("--extract_args", default="", help='Extra args for extract_features.py, e.g. "--chunk 8 --fast_batch 2"')
    # training - defaults identical to trainer.py (do not change if you want the P1 you selected)
    ap.add_argument("--segments", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--min_epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--huber_delta", type=float, default=1.0)
    ap.add_argument("--rank_weight", type=float, default=0.0)
    ap.add_argument("--clip_grad", type=float, default=1.0)
    ap.add_argument("--calib", choices=["linear", "logistic"], default="linear")
    ap.add_argument("--deploy_epochs", type=int, default=0, help="0 = median best epoch of the final runs")
    ap.add_argument("--n_boot", type=int, default=1000, help="Bootstrap resamples for confidence intervals")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


# =============================================================================== data + features
def run_cmd(cmd):
    cmd = [str(c) for c in cmd]
    print("\n$ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(CODE_DIR))


def ensure_data(args, work):
    if (work / "metadata.csv").exists() and (work / "splits.json").exists():
        print("[data] metadata.csv and splits.json found -> reusing them (splits unchanged)")
        return
    if not args.dataset_root:
        sys.exit("[data] metadata.csv / splits.json are missing. Pass --dataset_root so prepare_data.py can build them.")
    run_cmd([sys.executable, CODE_DIR / "prepare_data.py", "--dataset_root", args.dataset_root, "--work_dir", work])


def guess_missing(work, meta):
    feat_dir = work / "features"
    have = {p.stem for p in feat_dir.rglob("*.npz")} if feat_dir.exists() else set()
    if not have:
        return None
    return [str(n) for n in meta["name"] if str(n) not in have]


def ensure_features(args, work, meta):
    try:
        data = load_feature_arrays(work, meta, k=args.segments)
        print(f"[features] cache complete -> reusing the features of all {len(meta)} videos (no extraction)")
        return data
    except Exception as e:
        print(f"[features] cache incomplete: {type(e).__name__}: {e}")
    missing = guess_missing(work, meta)
    if missing is not None:
        more = " ..." if len(missing) > 10 else ""
        print(f"[features] about {len(missing)} of {len(meta)} videos have no feature file yet: {missing[:10]}{more}")
    if not (args.dataset_root and args.fast_repo and args.fast_weights):
        sys.exit("[features] pass --dataset_root, --fast_repo and --fast_weights so the missing videos can be "
                 "extracted (videos that already have features are skipped).")
    run_cmd([sys.executable, CODE_DIR / "extract_features.py", "--dataset_root", args.dataset_root,
             "--work_dir", work, "--fast_repo", args.fast_repo, "--fast_weights", args.fast_weights]
            + shlex.split(args.extract_args))
    data = load_feature_arrays(work, meta, k=args.segments)
    print(f"[features] cache complete after extraction ({len(meta)} videos)")
    return data


def data_ids(work, data):
    """Same hashing as trainer.fingerprint: identical IDs = identical splits / model inputs."""
    split_id = hashlib.sha256((work / "splits.json").read_bytes()).hexdigest()[:12]
    h = hashlib.sha256()
    for k in ("fast", "clip", "motion", "anim", "q_fast_raw", "mos"):
        h.update(np.ascontiguousarray(data[k]).tobytes())
    return split_id, h.hexdigest()[:12]


def env_info(device):
    info = {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(),
            "device": str(device)}
    if device.type == "cuda" and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(device.index or 0)
    return info


# =============================================================================== training
def _inputs(T, idx):
    return T["fast"][idx], T["clip"][idx], T["motion"][idx], T["anim"][idx], T["qfast"][idx]


def _batch_loss(model, T, b, args):
    pred, _ = model(*_inputs(T, b))
    y = T["mos"][b]
    loss = F.huber_loss(pred, y, delta=args.huber_delta)
    if args.rank_weight > 0:
        loss = loss + args.rank_weight * rank_loss(pred, y)
    return loss


def _curve_metrics(pred, mos):
    """Cheap per-epoch metrics for the learning curves (no logistic fit)."""
    rmse = float(np.sqrt(np.mean((pred - mos) ** 2)))
    if np.std(pred) < 1e-12:
        return np.nan, np.nan, rmse
    return float(stats.spearmanr(pred, mos)[0]), float(stats.pearsonr(pred, mos)[0]), rmse


def _epoch(model, opt, T, tr_t, gen, args, device):
    model.train()
    perm = tr_t[torch.randperm(len(tr_t), generator=gen).to(device)]
    losses = []
    for i in range(0, len(perm), args.batch):
        b = perm[i:i + args.batch]
        if len(b) < 2:
            continue
        loss = _batch_loss(model, T, b, args)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        opt.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses)) if losses else np.nan


def train_with_history(model_fn, T, tr, va, seed, args, device):
    """Same procedure as trainer.train_one (seeding, batching, optimiser, early stopping on the
    validation sources), plus a per-epoch log used for the learning-curve figure."""
    set_seed(seed)
    model = model_fn().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    tr_t = torch.as_tensor(tr, device=device)
    mos_tr = T["mos"][tr_t].cpu().numpy()
    mos_va = T["mos"][torch.as_tensor(va, device=device)].cpu().numpy()
    gen = torch.Generator().manual_seed(seed)

    best_score, best_state, best_ep, bad = -np.inf, copy.deepcopy(model.state_dict()), -1, 0
    hist = []
    for ep in range(args.epochs):
        tr_loss = _epoch(model, opt, T, tr_t, gen, args, device)
        p_va = predict(model, T, va)["pred"]
        score = quick_val_score(p_va, mos_va)
        p_tr = predict(model, T, tr)["pred"]
        ts, tp, trm = _curve_metrics(p_tr, mos_tr)
        vs, vp, vrm = _curve_metrics(p_va, mos_va)
        hist.append({"epoch": ep + 1, "train_loss": tr_loss, "train_SROCC": ts, "train_PLCC_raw": tp, "train_RMSE": trm,
                     "val_SROCC": vs, "val_PLCC_raw": vp, "val_RMSE": vrm, "val_score": score})
        if score > best_score + 1e-4:
            best_score, best_state, best_ep, bad = score, copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= args.patience and ep + 1 >= args.min_epochs:
                break
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_ep + 1, "val_score": best_score, "epochs_run": ep + 1}, hist


def train_fixed(model_fn, T, tr, n_epochs, seed, args, device):
    """Deploy model: all videos, no validation set, fixed number of epochs."""
    set_seed(seed)
    model = model_fn().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    tr_t = torch.as_tensor(tr, device=device)
    mos_tr = T["mos"][tr_t].cpu().numpy()
    gen = torch.Generator().manual_seed(seed)
    hist = []
    for ep in range(n_epochs):
        tr_loss = _epoch(model, opt, T, tr_t, gen, args, device)
        ts, tp, trm = _curve_metrics(predict(model, T, tr)["pred"], mos_tr)
        hist.append({"epoch": ep + 1, "train_loss": tr_loss, "train_SROCC": ts, "train_PLCC_raw": tp, "train_RMSE": trm})
    return model, hist


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# =============================================================================== stages
def run_protocol(stage, folds, data, meta, args, device, out_root):
    out = out_root / stage
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    model_fn = lambda: P1Model(dropout=args.dropout)  # noqa: E731
    rows, frames, hists = [], [], []
    t_stage = time.time()
    n_total, n_done = len(folds) * len(args.seeds), 0

    for fold in folds:
        tr_s, va_s, te_s = set(fold["train_sources"]), set(fold["val_sources"]), set(fold["test_sources"])
        assert not (tr_s & va_s or tr_s & te_s or va_s & te_s), "source overlap between splits!"
        tr = source_indices(meta, fold["train_sources"])
        va = source_indices(meta, fold["val_sources"])
        te = source_indices(meta, fold["test_sources"])
        T, norm = prepare_fold(data, tr, args.calib, device)       # standardisation + calibration on train only

        for seed in args.seeds:
            t0 = time.time()
            model, info, hist = train_with_history(model_fn, T, tr, va, seed, args, device)
            t_train = time.time() - t0
            _sync(device)
            t1 = time.perf_counter()
            r = predict(model, T, te)
            _sync(device)
            infer_ms = 1000.0 * (time.perf_counter() - t1) / max(len(te), 1)

            mos_te = data["mos"][te]
            m_model = compute_metrics(r["pred"], mos_te)
            m_fast = compute_metrics(r["qfast"], mos_te)
            row = {"pipeline": PIPELINE, "stage": stage, "fold": fold["fold"], "seed": seed, "n_train": len(tr),
                   "n_val": len(va), "n_test": len(te), **info, "time_s": round(t_train, 1),
                   "infer_ms_per_video": round(infer_ms, 4)}
            row.update({k: m_model[k] for k in METRIC_KEYS})
            row.update({f"fast_{k}": m_fast[k] for k in METRIC_KEYS})
            rows.append(row)

            df = meta.iloc[te][["name", "source", "distortion", "level", "mos"]].copy()
            df["pred"], df["q_fast_cal"], df["delta_q"] = r["pred"], r["qfast"], r["delta"]
            for k in range(r["alpha"].shape[1]):
                df[f"alpha_{k + 1}"] = r["alpha"][:, k]
            for j, bname in enumerate(BRANCH_NAMES):
                df[f"branch_{bname}"] = r["branch"][:, j]
            df["fold"], df["seed"] = fold["fold"], seed
            frames.append(df)

            h = pd.DataFrame(hist)
            h["fold"], h["seed"] = fold["fold"], seed
            hists.append(h)

            torch.save({"pipeline": PIPELINE, "model": model.state_dict(), "norm": norm, "args": vars(args),
                        "fold": fold, "seed": seed}, out / "checkpoints" / f"fold{fold['fold']}_seed{seed}.pt")
            n_done += 1
            print(f"[{stage} {n_done}/{n_total}] fold {fold['fold']} seed {seed}: "
                  f"SROCC {m_model['SROCC']:.4f}  PLCC {m_model['PLCC']:.4f}  RMSE {m_model['RMSE']:.4f} | "
                  f"FAST-only SROCC {m_fast['SROCC']:.4f} | best ep {info['best_epoch']}/{info['epochs_run']} "
                  f"({row['time_s']}s)", flush=True)

    runs = pd.DataFrame(rows)
    runs.to_csv(out / "runs.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(out / "predictions.csv", index=False)
    pd.concat(hists, ignore_index=True).to_csv(out / "history.csv", index=False)

    print(f"\n===== {PIPELINE} [{stage}]  mean ± std over {len(runs)} runs =====")
    for prefix, label in [("", PIPELINE), ("fast_", "FAST-only (calibrated)")]:
        cells = [f"{k} {runs[prefix + k].mean():.4f}±{runs[prefix + k].std(ddof=0):.4f}" for k in METRIC_KEYS]
        print(f"{label:24s} " + "  ".join(cells))
    print(f"[{stage}] saved to {out}  ({(time.time() - t_stage) / 60:.1f} min)\n")
    return runs


def run_deploy(data, meta, args, device, out_root):
    out = out_root / "deploy"
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    if args.deploy_epochs > 0:
        n_ep, src = args.deploy_epochs, "--deploy_epochs"
    else:
        for stage in ("final", "cv"):
            f = out_root / stage / "runs.csv"
            if f.exists():
                n_ep = int(round(pd.read_csv(f)["best_epoch"].median()))
                src = f"median best epoch of the '{stage}' runs"
                break
        else:
            sys.exit("[deploy] run the 'final' (or 'cv') stage first, or pass --deploy_epochs")
    n_ep = max(n_ep, 1)
    print(f"[deploy] training on ALL {len(meta)} videos for {n_ep} epochs ({src}), seeds {args.seeds}")
    print("[deploy] these models have no held-out test data; use the cv/final numbers for reporting.")

    all_idx = np.arange(len(meta))
    T, norm = prepare_fold(data, all_idx, args.calib, device)
    model_fn = lambda: P1Model(dropout=args.dropout)  # noqa: E731
    fits, hists = [], []
    for seed in args.seeds:
        t0 = time.time()
        model, hist = train_fixed(model_fn, T, all_idx, n_ep, seed, args, device)
        r = predict(model, T, all_idx)
        m = compute_metrics(r["pred"], data["mos"])
        fits.append({"seed": seed, "epochs": n_ep, "time_s": round(time.time() - t0, 1),
                     **{f"train_fit_{k}": m[k] for k in METRIC_KEYS}})
        h = pd.DataFrame(hist)
        h["seed"] = seed
        hists.append(h)
        torch.save({"pipeline": PIPELINE, "model": model.state_dict(), "norm": norm, "args": vars(args),
                    "fold": {"fold": "deploy", "train_sources": sorted(meta["source"].unique().tolist())},
                    "seed": seed, "epochs": n_ep}, out / "checkpoints" / f"deploy_seed{seed}.pt")
        print(f"[deploy] seed {seed}: training-set fit SROCC {m['SROCC']:.4f} ({fits[-1]['time_s']}s)")
    pd.DataFrame(fits).to_csv(out / "train_fit.csv", index=False)
    pd.concat(hists, ignore_index=True).to_csv(out / "history.csv", index=False)
    (out / "info.json").write_text(json.dumps({"epochs": n_ep, "epochs_source": src, "seeds": args.seeds,
                                               "n_videos": len(meta),
                                               "note": "trained on every video; training-set fit only, not a "
                                                       "generalisation estimate"}, indent=2))
    print(f"[deploy] saved to {out}\n")


# =============================================================================== main
def main():
    args = build_parser().parse_args()
    work = Path(args.work_dir).expanduser().resolve()
    out_root = work / "results" / args.tag
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    t_all = time.time()
    print(f"DA-AVQA full run | pipeline {PIPELINE} | work_dir {work} | device {device} | stages {args.stages}")

    ensure_data(args, work)
    meta = load_metadata(work)
    splits = load_splits(work)
    print(f"[data] {len(meta)} videos, {meta['source'].nunique()} sources")
    print(meta.groupby("distortion").size().rename("videos").to_string())
    if len(meta) != EXPECTED_VIDEOS:
        print(f"[data] WARNING: expected {EXPECTED_VIDEOS} distorted videos (BeDAMQA minus 02_Blur_3), "
              f"found {len(meta)}. If metadata.csv was built from a subset, rebuild it with "
              f"prepare_data.py on the full dataset first.")

    summary_path = out_root / "run_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    if any(s in args.stages for s in TRAIN_STAGES):
        data = ensure_features(args, work, meta)
        split_id, data_id = data_ids(work, data)
        n_params = sum(p.numel() for p in P1Model(dropout=args.dropout).parameters() if p.requires_grad)
        print(f"[{PIPELINE}] split id {split_id} | data id (K={args.segments}) {data_id} | "
              f"trainable params {n_params:,}\n")
        summary.update({"pipeline": PIPELINE, "tag": args.tag, "split_id": split_id, "data_id": data_id,
                        "trainable_params": n_params, "n_videos": len(meta), "args": vars(args),
                        "env": env_info(device)})

        timings = summary.setdefault("stage_minutes", {})
        if "cv" in args.stages:
            t0 = time.time()
            run_protocol("cv", splits["folds"], data, meta, args, device, out_root)
            timings["cv"] = round((time.time() - t0) / 60, 2)
        if "final" in args.stages:
            print("*** FINAL: evaluates once on the locked test sources (P1 is now the chosen pipeline) ***")
            t0 = time.time()
            run_protocol("final", [dict(splits["final"], fold="final")], data, meta, args, device, out_root)
            timings["final"] = round((time.time() - t0) / 60, 2)
        if "deploy" in args.stages:
            t0 = time.time()
            run_deploy(data, meta, args, device, out_root)
            timings["deploy"] = round((time.time() - t0) / 60, 2)
        summary_path.write_text(json.dumps(summary, indent=2, default=str))

    if "report" in args.stages:
        import report_p1
        report_p1.make_report(work, tag=args.tag, n_boot=args.n_boot)

    print(f"\nDone in {(time.time() - t_all) / 60:.1f} min. Results: {out_root}")


if __name__ == "__main__":
    main()
