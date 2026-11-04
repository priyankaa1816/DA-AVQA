"""
Report tables + figures for the full P1 run (run_full_p1.py).

Re-run on its own any time (no GPU, no retraining):
  python report_p1.py --work_dir "D:/avqa_work"

Reads   <work_dir>/results/P1_full/{cv,final,deploy}/ + metadata.csv + splits.json
Writes  <work_dir>/results/P1_full/report/
          REPORT.md              all tables + figures with captions in one file
          tables/*.csv|.md|.tex  every table (LaTeX needs \\usepackage{booktabs,graphicx})
          figures/*.png          600 dpi, bold labels
          figures/captions.md    suggested caption for every figure
"""
import argparse
import json
import re
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
from common import compute_metrics, load_metadata, load_splits  # noqa: E402

DPI = 600
plt.rcParams.update({
    "font.size": 9, "font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold",
    "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
})

P1_LABEL, FAST_LABEL = "P1 (proposed)", "FAST-VQA only"
MODEL_COLS = [("pred", P1_LABEL, "#1f4e9c"), ("q_fast_cal", FAST_LABEL, "#e3892b")]   # (column, label, colour)
SHORT = {"pred": "P1", "q_fast_cal": "FAST"}
DIST_ORDER = ["H.264", "H.265", "Frame freeze", "Frame hanging", "Blur"]
DIST_COLORS = {"H.264": "#1f77b4", "H.265": "#2ca02c", "Frame freeze": "#d62728",
               "Frame hanging": "#9467bd", "Blur": "#8c564b"}
BRANCH_LABELS = {"fast": "FAST-VQA", "clip": "CLIP", "motion": "Motion", "anim": "Animation"}
BRANCH_COLORS = ["#1f4e9c", "#6aa84f", "#cc4125", "#8e7cc3"]
CLIP_SECONDS = 10.0   # BeDAMQA clip length, used only for segment labels

# Numbers reported in the RF-AVQA paper (IEEE TCE 2026, Table I) on BeDAMQA. Different protocol:
# computed on the whole dataset, no source-held-out split -> context only, not a head-to-head result.
PAPER_REFERENCE = [
    # method, type, SROCC per distortion (H.264, H.265, freeze, hanging, blur), overall SROCC, overall PLCC
    ("RF-AVQA [paper]", "Unsupervised NR VQA", [0.943, 0.688, 0.931, 0.907, 0.614], 0.863, 0.879),
    ("3DMSCN [as reported in RF-AVQA]", "Supervised NR VQA", [0.757, 0.630, 0.741, 0.761, 0.536], 0.753, 0.809),
    ("NIQE [as reported in RF-AVQA]", "Unsupervised NR IQA", [0.868, 0.329, 0.661, 0.879, 0.332], 0.749, 0.765),
]


# =============================================================================== small helpers
def pretty_dist(d):
    k = "".join(ch for ch in str(d).lower() if ch.isalnum())
    if "264" in k or "avc" in k:
        return "H.264"
    if "265" in k or "hevc" in k:
        return "H.265"
    if "freez" in k:
        return "Frame freeze"
    if "hang" in k:
        return "Frame hanging"
    if "blur" in k:
        return "Blur"
    return str(d)


def dists_in(df):
    present = list(pd.unique(df["dist"]))
    return [d for d in DIST_ORDER if d in present] + sorted(d for d in present if d not in DIST_ORDER)


def dcolor(d):
    return DIST_COLORS.get(d, "#7f7f7f")


def natural_key(v):
    m = re.search(r"[-+]?\d*\.?\d+", str(v))
    return (0, float(m.group()), str(v)) if m else (1, 0.0, str(v))


def level_label(v):
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def pm(mu, sd, nd=3):
    if mu is None or not np.isfinite(mu):
        return "–"
    return f"{mu:.{nd}f} ± {sd:.{nd}f}" if np.isfinite(sd) else f"{mu:.{nd}f}"


def fmt_ci(v, lo, hi, nd=3):
    if not np.isfinite(v):
        return "–"
    return f"{v:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return "–"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def fmt_src(srcs):
    return ", ".join(str(s) for s in sorted(srcs, key=natural_key))


def metrics(df, col):
    return compute_metrics(df[col].to_numpy(float), df["mos"].to_numpy(float))


def srocc(a, b):
    if np.std(a) < 1e-12:
        return np.nan
    return float(stats.spearmanr(a, b)[0])


def wilcoxon_p(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 2 or np.allclose(d, 0):
        return np.nan
    try:
        return float(stats.wilcoxon(d).pvalue)
    except Exception:
        return np.nan


def per_seed_metrics(preds, col):
    """Pool the test predictions of each seed (CV: all folds -> every dev video once), then metrics
    overall and per distortion. Mean/std over seeds of these numbers = seed variability."""
    rows = []
    for seed, g in preds.groupby("seed"):
        rows.append({"seed": seed, "dist": "Overall", **metrics(g, col)})
        for d, gd in g.groupby("dist"):
            rows.append({"seed": seed, "dist": d, **metrics(gd, col)})
    return pd.DataFrame(rows)


def seed_avg(preds):
    """One row per video: predictions averaged over seeds (= seed ensemble)."""
    alpha = [c for c in preds.columns if c.startswith("alpha_")]
    branch = [c for c in preds.columns if c.startswith("branch_")]
    num = ["pred", "q_fast_cal", "delta_q"] + alpha + branch
    g = preds.groupby("name", sort=False)
    out = g[["source", "distortion", "dist", "level", "mos", "fold"]].first()
    out = out.join(g[num].mean()).join(g["pred"].std(ddof=0).rename("pred_seed_std"))
    out["n_seeds"] = g.size()
    return out.reset_index()


def _log5(x, b1, b2, b3, b4, b5):
    with np.errstate(over="ignore", invalid="ignore"):
        return b1 * (0.5 - 1.0 / (1.0 + np.exp(b2 * (x - b3)))) + b4 * x + b5


def fit_curve(x, y):
    """VQEG 5-parameter logistic for drawing the fitted curve (linear fallback)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    try:
        p0 = [float(y.max() - y.min()), 1.0 / (float(np.std(x)) + 1e-6), float(np.mean(x)), 0.0, float(np.mean(y))]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, _ = curve_fit(_log5, x, y, p0=p0, maxfev=20000)
        if not np.all(np.isfinite(_log5(x, *p))):
            raise RuntimeError
        return lambda t: _log5(t, *p)
    except Exception:
        c = np.polyfit(x, y, 1)
        return lambda t: np.polyval(c, t)


# --------------------------------------------------------------------------- table writers
def _cell(v):
    if isinstance(v, (float, np.floating)):
        return "–" if not np.isfinite(v) else f"{v:.4f}"
    return str(v)


def df_to_md(df):
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_cell(v).replace("|", "/") for v in r) + " |")
    return "\n".join(lines)


def _tex(s):
    s = str(s)
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#", "$": r"\$",
           "{": r"\{", "}": r"\}"}
    s = "".join(rep.get(ch, ch) for ch in s)
    for a, b in [("±", r"$\pm$"), ("Δ", r"$\Delta$"), ("≤", r"$\leq$"), ("×", r"$\times$"), ("−", "-"),
                 ("μ", r"$\mu$"), ("σ", r"$\sigma$"), ("α", r"$\alpha$")]:
        s = s.replace(a, b)
    return s


def df_to_tex(df, caption, label):
    align = "l" + "c" * (df.shape[1] - 1)
    body = [" & ".join(_tex(c) for c in df.columns) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        body.append(" & ".join(_tex(_cell(v)) for v in r) + r" \\")
    return "\n".join([
        r"% needs \usepackage{booktabs,graphicx}",
        r"\begin{table}[t]", r"\centering", rf"\caption{{{_tex(caption)}}}", rf"\label{{tab:{label}}}",
        r"\resizebox{\linewidth}{!}{%", rf"\begin{{tabular}}{{{align}}}", r"\toprule", *body,
        r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""])


# =============================================================================== report object
class Report:
    def __init__(self, work, tag, n_boot):
        self.work = work
        self.root = work / "results" / tag
        self.out = self.root / "report"
        self.fig_dir = self.out / "figures"
        self.tab_dir = self.out / "tables"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.tab_dir.mkdir(parents=True, exist_ok=True)
        self.n_boot = n_boot
        self.tables, self.figures, self.boot_sets, self.headline = [], [], [], []

    # ------------------------------------------------------------------ io
    def safe(self, fn, *a, **k):
        try:
            return fn(*a, **k)
        except Exception as e:
            print(f"  [report] WARNING: {fn.__name__} failed ({type(e).__name__}: {e}) - skipped")
            traceback.print_exc(limit=3)
            plt.close("all")

    def save_table(self, raw, disp, name, caption):
        n = f"table{len(self.tables) + 1:02d}_{name}"
        raw.to_csv(self.tab_dir / f"{n}.csv", index=False)
        (self.tab_dir / f"{n}.md").write_text(f"**{caption}**\n\n{df_to_md(disp)}\n", encoding="utf-8")
        (self.tab_dir / f"{n}.tex").write_text(df_to_tex(disp, caption, n), encoding="utf-8")
        self.tables.append((n, caption, disp))
        print(f"  table  {n}")

    def save_fig(self, fig, name, caption):
        n = f"fig{len(self.figures) + 1:02d}_{name}"
        fig.savefig(self.fig_dir / f"{n}.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        self.figures.append((n, caption))
        print(f"  figure {n}")

    # ------------------------------------------------------------------ bootstrap
    def boot_ci(self, df, col, seed=0):
        """95% CI by resampling videos with replacement (same resamples for every model: seed fixed)."""
        pred, mos = df[col].to_numpy(float), df["mos"].to_numpy(float)
        rng = np.random.default_rng(seed)
        vals = {k: [] for k in ("SROCC", "KRCC", "PLCC", "RMSE")}
        for _ in range(self.n_boot):
            i = rng.integers(0, len(mos), len(mos))
            m = compute_metrics(pred[i], mos[i])
            for k in vals:
                vals[k].append(m[k])
        return {k: np.nanpercentile(np.asarray(v, float), [2.5, 97.5]) for k, v in vals.items()}

    def paired_boot(self, df, seed=0):
        a, b, y = df["pred"].to_numpy(float), df["q_fast_cal"].to_numpy(float), df["mos"].to_numpy(float)
        obs = srocc(a, y) - srocc(b, y)
        rng = np.random.default_rng(seed)
        diffs = np.empty(self.n_boot)
        for k in range(self.n_boot):
            i = rng.integers(0, len(y), len(y))
            diffs[k] = srocc(a[i], y[i]) - srocc(b[i], y[i])
        lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
        return diffs, obs, lo, hi, float(np.nanmean(diffs <= 0))

    # ================================================================== tables
    def table_dataset(self, meta):
        rows = []
        for d in dists_in(meta) + ["Overall"]:
            g = meta if d == "Overall" else meta[meta["dist"] == d]
            levels = "–" if d == "Overall" else ", ".join(level_label(v) for v in sorted(g["level"].unique(), key=natural_key))
            rows.append({"Distortion": d, "Videos": len(g), "Sources": g["source"].nunique(), "Levels": levels,
                         "MOS mean": g["mos"].mean(), "MOS std": g["mos"].std(), "MOS min": g["mos"].min(),
                         "MOS max": g["mos"].max()})
        raw = pd.DataFrame(rows)
        disp = raw.copy()
        for c in ("MOS mean", "MOS std", "MOS min", "MOS max"):
            disp[c] = disp[c].map(lambda v: f"{v:.2f}")
        self.save_table(raw, disp, "dataset", "BeDAMQA videos used (distorted only) and MOS statistics per distortion.")

    def table_splits(self, meta, splits):
        src = meta["source"].astype(str)

        def n(srcs):
            return int(src.isin([str(s) for s in srcs]).sum())

        rows = []
        for f in splits["folds"]:
            rows.append({"Run": f"CV fold {f['fold']}", "Train sources": fmt_src(f["train_sources"]),
                         "Val sources": fmt_src(f["val_sources"]), "Test sources": fmt_src(f["test_sources"]),
                         "Videos (train/val/test)": f"{n(f['train_sources'])}/{n(f['val_sources'])}/{n(f['test_sources'])}"})
        fin = splits["final"]
        rows.append({"Run": "Final (locked test)", "Train sources": fmt_src(fin["train_sources"]),
                     "Val sources": fmt_src(fin["val_sources"]), "Test sources": fmt_src(fin["test_sources"]),
                     "Videos (train/val/test)": f"{n(fin['train_sources'])}/{n(fin['val_sources'])}/{n(fin['test_sources'])}"})
        rows.append({"Run": "Deploy", "Train sources": "all", "Val sources": "– (fixed epochs)", "Test sources": "–",
                     "Videos (train/val/test)": f"{len(meta)}/0/0"})
        df = pd.DataFrame(rows)
        self.save_table(df, df, "splits", "Source-grouped split protocol: no source content appears in more than "
                                          "one of train / validation / test within a run.")

    def table_overall(self, cv, fin):
        raw, disp = [], []
        for key, st, per_lab, pool_lab in [
            ("cv", cv, "CV per run (fold × seed)", "CV out-of-fold, seed-averaged"),
            ("final", fin, "Final test per seed", "Final test, seed ensemble"),
        ]:
            if st is None:
                continue
            runs, avg = st["runs"], st["avg"]
            for col, lab, _ in MODEL_COLS:
                pre = "" if col == "pred" else "fast_"
                r = {"stage": key, "kind": "per_run", "model": lab, "n_runs": len(runs),
                     "n_videos": float(runs["n_test"].mean())}
                d = {"Evaluation": f"{per_lab}, n = {len(runs)}", "Model": lab,
                     "N videos": f"{runs['n_test'].mean():.0f} per run"}
                for k in ("SROCC", "KRCC", "PLCC", "RMSE"):
                    mu, sd = runs[pre + k].mean(), runs[pre + k].std(ddof=0)
                    r[f"{k}_mean"], r[f"{k}_std"] = mu, sd
                    d[k] = pm(mu, sd)
                raw.append(r)
                disp.append(d)
            for col, lab, _ in MODEL_COLS:
                m, ci = metrics(avg, col), self.boot_ci(avg, col)
                r = {"stage": key, "kind": "pooled", "model": lab, "n_runs": len(runs), "n_videos": len(avg)}
                d = {"Evaluation": f"{pool_lab} (95% CI)", "Model": lab, "N videos": str(len(avg))}
                for k in ("SROCC", "KRCC", "PLCC", "RMSE"):
                    r[k], r[f"{k}_ci_lo"], r[f"{k}_ci_hi"] = m[k], ci[k][0], ci[k][1]
                    d[k] = fmt_ci(m[k], ci[k][0], ci[k][1])
                raw.append(r)
                disp.append(d)
            p1 = runs["SROCC"].mean()
            self.headline.append(
                f"- **{per_lab}**: P1 SROCC {pm(p1, runs['SROCC'].std(ddof=0))}, PLCC "
                f"{pm(runs['PLCC'].mean(), runs['PLCC'].std(ddof=0))} vs FAST-VQA only SROCC "
                f"{pm(runs['fast_SROCC'].mean(), runs['fast_SROCC'].std(ddof=0))} ({len(runs)} runs).")
        self.save_table(pd.DataFrame(raw), pd.DataFrame(disp), "overall",
                        "Overall performance. Per-run rows: mean ± std over runs. Pooled rows: metrics on "
                        "seed-averaged predictions with 95% bootstrap CI. PLCC after VQEG 5-parameter logistic "
                        "mapping; RMSE on the MOS scale without mapping.")

    def table_per_dist(self, st, name, caption):
        preds = st["preds"]
        tabs = {col: per_seed_metrics(preds, col) for col, _, _ in MODEL_COLS}
        raw, disp = [], []
        for d in dists_in(preds) + ["Overall"]:
            n_vid = int(st["avg"].shape[0] if d == "Overall" else (st["avg"]["dist"] == d).sum())
            r, dd = {"Distortion": d, "N": n_vid}, {"Distortion": d, "N": n_vid}
            for col, _, _ in MODEL_COLS:
                t = tabs[col][tabs[col]["dist"] == d]
                for k in ("SROCC", "PLCC", "RMSE"):
                    mu, sd = t[k].mean(), t[k].std(ddof=0)
                    r[f"{SHORT[col]}_{k}_mean"], r[f"{SHORT[col]}_{k}_std"] = mu, sd
                    dd[f"{SHORT[col]} {k}"] = pm(mu, sd)
            raw.append(r)
            disp.append(dd)
        self.save_table(pd.DataFrame(raw), pd.DataFrame(disp), name, caption)

    def table_per_dist_pooled(self, avg, name, caption):
        rows = []
        for d in dists_in(avg) + ["Overall"]:
            g = avg if d == "Overall" else avg[avg["dist"] == d]
            r = {"Distortion": d, "N": len(g)}
            for col, _, _ in MODEL_COLS:
                m = metrics(g, col)
                for k in ("SROCC", "KRCC", "PLCC", "RMSE"):
                    r[f"{SHORT[col]} {k}"] = m[k]
            rows.append(r)
        df = pd.DataFrame(rows)
        disp = df.copy()
        for c in disp.columns[2:]:
            disp[c] = disp[c].map(lambda v: "–" if not np.isfinite(v) else f"{v:.3f}")
        self.save_table(df, disp, name, caption)

    def table_folds(self, cv, fin, splits):
        test_of = {str(f["fold"]): f["test_sources"] for f in splits["folds"]}
        sets = [(str(f), g) for f, g in cv["runs"].groupby("fold")]
        if fin is not None:
            sets.append(("final", fin["runs"]))
        rows, disp = [], []
        for f, g in sets:
            srcs = splits["final"]["test_sources"] if f == "final" else test_of.get(f, [])
            r = {"Split": "Final (locked)" if f == "final" else f"CV fold {f}", "Test sources": fmt_src(srcs),
                 "N test": int(g["n_test"].iloc[0]), "Seeds": len(g)}
            d = dict(r)
            for col, lab, _ in MODEL_COLS:
                pre = "" if col == "pred" else "fast_"
                for k in ("SROCC", "PLCC", "RMSE"):
                    r[f"{SHORT[col]}_{k}_mean"], r[f"{SHORT[col]}_{k}_std"] = g[pre + k].mean(), g[pre + k].std(ddof=0)
                    d[f"{SHORT[col]} {k}"] = pm(g[pre + k].mean(), g[pre + k].std(ddof=0))
            r["best_epoch_mean"] = g["best_epoch"].mean()
            d["Best epoch"] = f"{g['best_epoch'].mean():.0f}"
            rows.append(r)
            disp.append(d)
        self.save_table(pd.DataFrame(rows), pd.DataFrame(disp), "per_split",
                        "Performance per evaluation split (mean ± std over seeds). Large differences between folds "
                        "indicate content dependence.")

    def table_sources(self, cv, fin):
        sets = [(s, "CV test", g) for s, g in cv["avg"].groupby("source")]
        if fin is not None:
            sets += [(s, "Locked test", g) for s, g in fin["avg"].groupby("source")]
        sets.sort(key=lambda t: natural_key(t[0]))
        rows = []
        for s, kind, g in sets:
            r = {"Source": str(s), "Evaluated as": kind, "N": len(g)}
            for col, _, _ in MODEL_COLS:
                m = metrics(g, col)
                r[f"{SHORT[col]} SROCC"], r[f"{SHORT[col]} PLCC"], r[f"{SHORT[col]} RMSE"] = m["SROCC"], m["PLCC"], m["RMSE"]
            rows.append(r)
        df = pd.DataFrame(rows)
        disp = df.copy()
        for c in disp.columns[3:]:
            disp[c] = disp[c].map(lambda v: "–" if not np.isfinite(v) else f"{v:.3f}")
        self.save_table(df, disp, "per_source",
                        "Performance per source content (seed-averaged predictions; CV sources from their CV test "
                        "fold, sources 12/14/15 from the final model).")

    def explain_frames(self, preds):
        bcols = [c for c in preds.columns if c.startswith("branch_")]
        acols = sorted([c for c in preds.columns if c.startswith("alpha_")], key=lambda c: int(c.split("_")[1]))
        rows = []
        for d in dists_in(preds) + ["Overall"]:
            g = preds if d == "Overall" else preds[preds["dist"] == d]
            r = {"Distortion": d}
            r.update({BRANCH_LABELS.get(c[7:], c[7:]): g[c].mean() for c in bcols})
            r.update({f"Seg {c.split('_')[1]}": g[c].mean() for c in acols})
            r["mean ΔQ"] = g["delta_q"].mean()
            r["mean |ΔQ|"] = g["delta_q"].abs().mean()
            rows.append(r)
        return pd.DataFrame(rows), [BRANCH_LABELS.get(c[7:], c[7:]) for c in bcols], \
            [f"Seg {c.split('_')[1]}" for c in acols]

    def table_explain(self, cv):
        df, _, _ = self.explain_frames(cv["preds"])
        disp = df.copy()
        for c in disp.columns[1:]:
            disp[c] = disp[c].map(lambda v: f"{v:.3f}")
        self.save_table(df, disp, "explainability",
                        "What P1 attends to (CV test predictions, all seeds): mean element-wise gate per branch "
                        "(0–1), segment attention weights (sum to 1), and the learned correction ΔQ added to the "
                        "calibrated FAST-VQA score.")

    def table_significance(self, cv, fin):
        rows = []
        for slabel, st in [("CV", cv), ("Final test", fin)]:
            if st is None:
                continue
            runs = st["runs"]
            d = (runs["SROCC"] - runs["fast_SROCC"]).to_numpy(float)
            rows.append({"Data": f"{slabel}: per-run SROCC (n = {len(d)} runs)", "Test": "Wilcoxon signed-rank",
                         "ΔSROCC (P1 − FAST)": pm(np.nanmean(d), np.nanstd(d)), "95% CI": "–",
                         "P1 wins": f"{int((d > 0).sum())}/{len(d)}", "p-value": fmt_p(wilcoxon_p(d))})
            avg = st["avg"]
            diffs, obs, lo, hi, p0 = self.paired_boot(avg)
            kind = "out-of-fold" if slabel == "CV" else "seed ensemble"
            self.boot_sets.append((f"{slabel} ({kind}, N = {len(avg)})", diffs, obs, lo, hi))
            rows.append({"Data": f"{slabel}: pooled {kind} (N = {len(avg)})", "Test": f"paired bootstrap ({self.n_boot}×)",
                         "ΔSROCC (P1 − FAST)": f"{obs:+.3f}", "95% CI": f"[{lo:+.3f}, {hi:+.3f}]", "P1 wins": "–",
                         "p-value": f"P(Δ≤0) = {p0:.3f}"})
        if cv is not None:
            for dname in dists_in(cv["avg"]):
                sub = cv["avg"][cv["avg"]["dist"] == dname]
                _, obs, lo, hi, p0 = self.paired_boot(sub)
                rows.append({"Data": f"CV out-of-fold, {dname} (N = {len(sub)})", "Test": f"paired bootstrap ({self.n_boot}×)",
                             "ΔSROCC (P1 − FAST)": f"{obs:+.3f}", "95% CI": f"[{lo:+.3f}, {hi:+.3f}]", "P1 wins": "–",
                             "p-value": f"P(Δ≤0) = {p0:.3f}"})
        df = pd.DataFrame(rows)
        self.save_table(df, df, "significance",
                        "Statistical comparison of P1 against the calibrated FAST-VQA score it builds on. Note: with "
                        "5 runs the smallest possible two-sided Wilcoxon p-value is 0.0625; bootstrap resamples "
                        "videos, which ignores the correlation between videos of the same source.")

    def table_complexity(self, summary, cv, fin, deploy):
        items = []

        def add(k, v):
            items.append({"Item": k, "Value": v})

        tp = summary.get("trainable_params")
        add("Trainable parameters (P1 head)", f"{tp:,}" if isinstance(tp, int) else "–")
        add("Frozen backbones (features cached once)", "FAST-VQA-B, CLIP ViT-B/16 + hand-crafted motion / animation statistics")
        add("Temporal segments K", str(summary.get("args", {}).get("segments", "–")))
        for lab, st in [("CV", cv), ("Final", fin)]:
            if st is None:
                continue
            r = st["runs"]
            add(f"{lab}: runs", str(len(r)))
            add(f"{lab}: training time per run (s)", pm(r["time_s"].mean(), r["time_s"].std(ddof=0), 1))
            add(f"{lab}: epochs run / best epoch", f"{r['epochs_run'].mean():.0f} / {r['best_epoch'].mean():.0f}")
            if "infer_ms_per_video" in r:
                add(f"{lab}: inference per video, head only (ms)", f"{r['infer_ms_per_video'].mean():.3f}")
        if deploy:
            add("Deploy models", f"{len(deploy.get('seeds', []))} seeds × {deploy.get('epochs')} epochs on "
                                 f"{deploy.get('n_videos')} videos ({deploy.get('epochs_source')})")
        env = summary.get("env", {})
        add("Hardware", env.get("gpu", env.get("device", "–")))
        add("Software", f"Python {env.get('python', '?')}, PyTorch {env.get('torch', '?')}")
        add("Split id / data id", f"{summary.get('split_id', '–')} / {summary.get('data_id', '–')}")
        for stg, mins in summary.get("stage_minutes", {}).items():
            add(f"Wall time: {stg} stage (min)", f"{mins}")
        df = pd.DataFrame(items)
        self.save_table(df, df, "complexity", "Model size and compute. Inference time excludes feature extraction, "
                                              "which is done once per video by the frozen backbones.")

    def table_reference(self, cv, fin):
        cols = DIST_ORDER
        rows = []

        def ours(lab, avg, col, protocol):
            r = {"Method": lab, "Type": "Learned NR VQA (ours)" if col == "pred" else "Calibrated FAST-VQA",
                 "Protocol": protocol}
            for d in cols:
                g = avg[avg["dist"] == d]
                r[f"{d} SROCC"] = metrics(g, col)["SROCC"] if len(g) >= 3 else np.nan
            m = metrics(avg, col)
            r["Overall SROCC"], r["Overall PLCC"] = m["SROCC"], m["PLCC"]
            rows.append(r)

        for st, prot in [(cv, "held-out sources, 4-fold CV (out-of-fold)"), (fin, "locked test sources 12/14/15")]:
            if st is None:
                continue
            ours(P1_LABEL, st["avg"], "pred", prot)
            ours(FAST_LABEL, st["avg"], "q_fast_cal", prot)
        for name, typ, per, so, pl in PAPER_REFERENCE:
            r = {"Method": name, "Type": typ, "Protocol": "whole dataset (reported in RF-AVQA paper)"}
            r.update({f"{d} SROCC": v for d, v in zip(cols, per)})
            r["Overall SROCC"], r["Overall PLCC"] = so, pl
            rows.append(r)
        df = pd.DataFrame(rows)
        disp = df.copy()
        for c in disp.columns[3:]:
            disp[c] = disp[c].map(lambda v: "–" if not np.isfinite(v) else f"{v:.3f}")
        self.save_table(df, disp, "reference_context",
                        "Context only: published numbers on BeDAMQA use a different protocol (whole dataset, no "
                        "held-out source contents), so they are not a direct head-to-head comparison.")

    # ================================================================== figures
    def scatter_panel(self, ax, df, col, title, by_dist=True, color=None, s=12):
        if by_dist:
            for d in dists_in(df):
                sub = df[df["dist"] == d]
                ax.scatter(sub[col], sub["mos"], s=s, alpha=0.75, color=dcolor(d), edgecolors="none", label=d)
        else:
            ax.scatter(df[col], df["mos"], s=s, alpha=0.75, color=color, edgecolors="none")
        x = df[col].to_numpy(float)
        if len(df) >= 5 and np.std(x) > 1e-12:
            xs = np.linspace(x.min(), x.max(), 200)
            ax.plot(xs, fit_curve(x, df["mos"])(xs), color="black", lw=1.4, label="Logistic fit")
        m = metrics(df, col)
        ax.text(0.03, 0.97, f"SROCC = {m['SROCC']:.3f}\nPLCC  = {m['PLCC']:.3f}\nRMSE  = {m['RMSE']:.3f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=7.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.9))
        ax.set_title(title)
        ax.set_xlabel("Predicted MOS")
        ax.set_ylabel("Subjective MOS")

    def fig_dataset(self, meta):
        ds = dists_in(meta)
        data = [meta.loc[meta["dist"] == d, "mos"].to_numpy(float) for d in ds]
        fig, ax = plt.subplots(figsize=(5.6, 3.0))
        bp = ax.boxplot(data, positions=range(len(ds)), widths=0.55, patch_artist=True, showfliers=False)
        for patch, d in zip(bp["boxes"], ds):
            patch.set_facecolor(dcolor(d))
            patch.set_alpha(0.45)
        for med in bp["medians"]:
            med.set_color("black")
        rng = np.random.default_rng(0)
        for i, (d, v) in enumerate(zip(ds, data)):
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=8, color=dcolor(d), alpha=0.85, edgecolors="none")
        ax.set_xticks(range(len(ds)))
        ax.set_xticklabels(ds, rotation=20, ha="right")
        ax.set_ylabel("Subjective MOS")
        ax.set_title(f"MOS distribution per distortion (N = {len(meta)})")
        self.save_fig(fig, "dataset_mos_distribution",
                      "MOS distribution of the BeDAMQA distorted videos per distortion type (box: quartiles; dots: videos).")

    def fig_scatter_pair(self, avg, stage_label, name, caption):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
        for ax, (col, lab, _) in zip(axes, MODEL_COLS[::-1]):
            self.scatter_panel(ax, avg, col, lab)
        axes[1].set_ylabel("")
        h, lab = axes[0].get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=len(lab), bbox_to_anchor=(0.5, -0.08), frameon=False)
        fig.suptitle(stage_label, fontweight="bold", y=1.02)
        fig.tight_layout()
        self.save_fig(fig, name, caption)

    def fig_scatter_per_dist(self, avg, stage_label, name, caption):
        panels = dists_in(avg) + ["Overall"]
        nrow = int(np.ceil(len(panels) / 3))
        fig, axes = plt.subplots(nrow, 3, figsize=(7.6, 2.6 * nrow), squeeze=False)
        for ax, p in zip(axes.flat, panels):
            if p == "Overall":
                self.scatter_panel(ax, avg, "pred", "All distortions", s=7)
            else:
                self.scatter_panel(ax, avg[avg["dist"] == p], "pred", p, by_dist=False, color=dcolor(p))
        for ax in list(axes.flat)[len(panels):]:
            ax.axis("off")
        fig.suptitle(stage_label, fontweight="bold", y=1.01)
        fig.tight_layout()
        self.save_fig(fig, name, caption)

    def fig_per_dist_bars(self, stage_sets):
        fig, axes = plt.subplots(len(stage_sets), 2, figsize=(7.6, 2.9 * len(stage_sets)), squeeze=False)
        for r, (slabel, preds) in enumerate(stage_sets):
            tabs = {col: per_seed_metrics(preds, col) for col, _, _ in MODEL_COLS}
            ds = dists_in(preds) + ["Overall"]
            x, w = np.arange(len(ds)), 0.38
            for c, met in enumerate(("SROCC", "PLCC")):
                ax, lows = axes[r, c], [0.0]
                for j, (col, lab, colr) in enumerate(MODEL_COLS):
                    t = tabs[col].groupby("dist")[met].agg(["mean", "std"]).reindex(ds)
                    sd = t["std"].fillna(0).to_numpy()
                    ax.bar(x + (j - 0.5) * w, t["mean"].to_numpy(), w, yerr=sd, color=colr, label=lab, capsize=2,
                           edgecolor="black", linewidth=0.4, error_kw=dict(lw=0.8))
                    lows.append(np.nanmin(t["mean"].to_numpy() - sd))
                ax.set_xticks(x)
                ax.set_xticklabels(ds, rotation=25, ha="right")
                ax.set_ylim(min(0.0, np.nanmin(lows) - 0.05), 1.0)
                ax.set_ylabel(met)
                ax.set_title(f"{slabel}: {met}")
        h, lab = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, lab, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03), frameon=False)
        fig.tight_layout()
        self.save_fig(fig, "per_distortion_bars",
                      "SROCC and PLCC per distortion type for P1 and the calibrated FAST-VQA score. Bars: mean over "
                      "seeds of the pooled test predictions; error bars: std over seeds.")

    def fig_split_bars(self, cv, fin, splits):
        test_of = {str(f["fold"]): f["test_sources"] for f in splits["folds"]}
        groups = [(f"Fold {f}\n(test {fmt_src(test_of.get(str(f), []))})", g) for f, g in cv["runs"].groupby("fold")]
        if fin is not None:
            groups.append((f"Final\n(test {fmt_src(splits['final']['test_sources'])})", fin["runs"]))
        x, w = np.arange(len(groups)), 0.38
        fig, ax = plt.subplots(figsize=(max(5.0, 1.25 * len(groups) + 1.5), 3.0))
        for j, (col, lab, colr) in enumerate(MODEL_COLS):
            pre = "" if col == "pred" else "fast_"
            mu = [g[pre + "SROCC"].mean() for _, g in groups]
            sd = [g[pre + "SROCC"].std(ddof=0) for _, g in groups]
            ax.bar(x + (j - 0.5) * w, mu, w, yerr=sd, color=colr, label=lab, capsize=2, edgecolor="black",
                   linewidth=0.4, error_kw=dict(lw=0.8))
        ax.set_xticks(x)
        ax.set_xticklabels([g[0] for g in groups], fontsize=7)
        ax.set_ylabel("SROCC")
        ax.set_ylim(0, 1)
        ax.set_title("Performance per evaluation split (mean ± std over seeds)")
        ax.legend(loc="lower right")
        self.save_fig(fig, "per_split_srocc",
                      "SROCC per cross-validation fold and on the locked final test set. Each fold tests on source "
                      "contents never seen in training.")

    def fig_run_distribution(self, run_sets):
        mets = ("SROCC", "PLCC")
        fig, axes = plt.subplots(len(mets), len(run_sets), figsize=(3.5 * len(run_sets), 2.8 * len(mets)), squeeze=False)
        for c, (slabel, runs) in enumerate(run_sets):
            for r, met in enumerate(mets):
                ax = axes[r, c]
                a, b = runs[f"fast_{met}"].to_numpy(float), runs[met].to_numpy(float)
                for ai, bi in zip(a, b):
                    ax.plot([0, 1], [ai, bi], color="0.7", lw=0.7, zorder=1)
                bp = ax.boxplot([a, b], positions=[0, 1], widths=0.45, patch_artist=True, showfliers=False)
                for patch, colr in zip(bp["boxes"], [MODEL_COLS[1][2], MODEL_COLS[0][2]]):
                    patch.set_facecolor(colr)
                    patch.set_alpha(0.5)
                for med in bp["medians"]:
                    med.set_color("black")
                ax.scatter(np.zeros_like(a), a, s=10, color=MODEL_COLS[1][2], zorder=3)
                ax.scatter(np.ones_like(b), b, s=10, color=MODEL_COLS[0][2], zorder=3)
                ax.set_xticks([0, 1])
                ax.set_xticklabels([FAST_LABEL, P1_LABEL])
                ax.set_ylabel(met)
                ax.set_title(f"{slabel} (n = {len(runs)} runs)")
        fig.tight_layout()
        self.save_fig(fig, "paired_runs",
                      "Per-run SROCC / PLCC of P1 and the calibrated FAST-VQA score; grey lines connect the same "
                      "fold and seed (paired comparison).")

    def fig_training_curves(self, hist, runs):
        h = hist.replace([np.inf, -np.inf], np.nan)
        n_runs = h.groupby(["fold", "seed"]).ngroups
        min_alive = max(1, int(np.ceil(0.2 * n_runs)))

        def agg(col):
            pv = h.pivot_table(index="epoch", columns=["fold", "seed"], values=col)
            keep = pv.notna().sum(axis=1) >= min_alive
            return pv.index[keep].to_numpy(), pv.mean(axis=1)[keep].to_numpy(), pv.std(axis=1)[keep].fillna(0).to_numpy()

        best = float(runs["best_epoch"].median())
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
        panels = [
            (axes[0], [("train_loss", "Train", MODEL_COLS[0][2])], "Training loss (Huber)", "Loss"),
            (axes[1], [("train_SROCC", "Train", MODEL_COLS[0][2]), ("val_SROCC", "Validation", "#cc4125")],
             "SROCC during training", "SROCC"),
            (axes[2], [("train_RMSE", "Train", MODEL_COLS[0][2]), ("val_RMSE", "Validation", "#cc4125")],
             "RMSE during training", "RMSE (MOS units)"),
        ]
        for ax, lines, title, ylab in panels:
            for col, lab, colr in lines:
                if col not in h:
                    continue
                ep, mu, sd = agg(col)
                ax.plot(ep, mu, color=colr, lw=1.4, label=lab)
                ax.fill_between(ep, mu - sd, mu + sd, color=colr, alpha=0.18)
            ax.axvline(best, color="black", ls="--", lw=0.9, label=f"median best epoch ({best:.0f})")
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylab)
            ax.legend(loc="best")
        fig.tight_layout()
        self.save_fig(fig, "training_curves",
                      f"Learning curves over the {n_runs} CV runs (mean ± std across runs still training at that "
                      f"epoch; shown while at least {min_alive} runs are active). The model kept is the one at the "
                      "best validation epoch (early stopping on validation sources).")

    def fig_mos_vs_level(self, avg, stage_label):
        ds = dists_in(avg)
        nrow = int(np.ceil((len(ds) + 1) / 3))
        fig, axes = plt.subplots(nrow, 3, figsize=(7.8, 2.6 * nrow), squeeze=False)
        series = [("mos", "Subjective MOS", "black", "o"), ("pred", P1_LABEL, MODEL_COLS[0][2], "s"),
                  ("q_fast_cal", FAST_LABEL, MODEL_COLS[1][2], "^")]
        for ax, d in zip(axes.flat, ds):
            sub = avg[avg["dist"] == d]
            levels = sorted(sub["level"].unique(), key=natural_key)
            x = np.arange(len(levels))
            g = sub.groupby("level")
            for col, lab, colr, mk in series:
                ax.errorbar(x, g[col].mean().reindex(levels).to_numpy(), yerr=g[col].std().reindex(levels).fillna(0).to_numpy(),
                            color=colr, marker=mk, ms=4, lw=1.3, capsize=2, label=lab)
            ax.set_xticks(x)
            ax.set_xticklabels([level_label(v) for v in levels], rotation=30, ha="right")
            ax.set_title(d)
            ax.set_xlabel("Distortion level")
            ax.set_ylabel("MOS")
        rest = list(axes.flat)[len(ds):]
        h, lab = axes.flat[0].get_legend_handles_labels()
        for ax in rest:
            ax.axis("off")
        rest[0].legend(h, lab, loc="center", fontsize=8)
        fig.suptitle(stage_label, fontweight="bold", y=1.01)
        fig.tight_layout()
        self.save_fig(fig, "mos_vs_level",
                      "Mean subjective MOS and mean predicted MOS per distortion level (error bars: std across source "
                      "contents). A good model follows the black curve at every level.")

    def fig_error_by_level(self, avg, stage_label):
        ds = dists_in(avg)
        panels = ds + ["Overall"]
        nrow = int(np.ceil(len(panels) / 3))
        fig, axes = plt.subplots(nrow, 3, figsize=(7.8, 2.6 * nrow), squeeze=False)
        a = avg.assign(err_pred=(avg["pred"] - avg["mos"]).abs(), err_q_fast_cal=(avg["q_fast_cal"] - avg["mos"]).abs())
        for ax, p in zip(axes.flat, panels):
            if p == "Overall":
                x, w = np.arange(len(ds)), 0.38
                for j, (col, lab, colr) in enumerate(MODEL_COLS):
                    ax.bar(x + (j - 0.5) * w, [a.loc[a["dist"] == d, f"err_{col}"].mean() for d in ds], w,
                           color=colr, label=lab, edgecolor="black", linewidth=0.4)
                ax.set_xticks(x)
                ax.set_xticklabels(ds, rotation=30, ha="right")
                ax.set_title("Mean absolute error per distortion")
                ax.set_ylabel("MAE (MOS units)")
                ax.legend(loc="upper left", fontsize=6.5)
                continue
            sub = a[a["dist"] == p]
            levels = sorted(sub["level"].unique(), key=natural_key)
            x = np.arange(len(levels))
            g = sub.groupby("level")
            for col, lab, colr in MODEL_COLS:
                ax.plot(x, g[f"err_{col}"].mean().reindex(levels).to_numpy(), marker="o", ms=4, color=colr, lw=1.3, label=lab)
            ax.set_xticks(x)
            ax.set_xticklabels([level_label(v) for v in levels], rotation=30, ha="right")
            ax.set_title(p)
            ax.set_xlabel("Distortion level")
            ax.set_ylabel("MAE (MOS units)")
        for ax in list(axes.flat)[len(panels):]:
            ax.axis("off")
        fig.suptitle(stage_label, fontweight="bold", y=1.01)
        fig.tight_layout()
        self.save_fig(fig, "error_by_level",
                      "Mean absolute prediction error per distortion level, P1 vs calibrated FAST-VQA.")

    def fig_residuals(self, sets):
        fig, axes = plt.subplots(1, len(sets), figsize=(3.8 * len(sets), 3.0), squeeze=False)
        for ax, (slabel, avg) in zip(axes[0], sets):
            res = [(avg[col] - avg["mos"]).to_numpy(float) for col, _, _ in MODEL_COLS]
            lim = float(np.nanmax(np.abs(np.concatenate(res)))) + 1e-6
            bins = np.linspace(-lim, lim, 31)
            for r, (col, lab, colr) in zip(res, MODEL_COLS):
                ax.hist(r, bins=bins, alpha=0.55, color=colr, edgecolor="black", linewidth=0.3,
                        label=f"{lab} (μ = {r.mean():+.2f}, σ = {r.std():.2f})")
            ax.axvline(0, color="black", lw=0.8, ls="--")
            ax.set_xlabel("Prediction − MOS")
            ax.set_ylabel("Number of videos")
            ax.set_title(slabel)
            ax.legend(loc="upper left")
        fig.tight_layout()
        self.save_fig(fig, "residual_histogram", "Distribution of prediction errors (prediction − MOS).")

    def fig_per_source(self, cv, fin):
        sets = [(s, False, g) for s, g in cv["avg"].groupby("source")]
        if fin is not None:
            sets += [(s, True, g) for s, g in fin["avg"].groupby("source")]
        sets.sort(key=lambda t: natural_key(t[0]))
        x, w = np.arange(len(sets)), 0.38
        fig, ax = plt.subplots(figsize=(max(6.0, 0.48 * len(sets) + 1.5), 3.0))
        for j, (col, lab, colr) in enumerate(MODEL_COLS):
            vals = [metrics(g, col)["SROCC"] for _, _, g in sets]
            bars = ax.bar(x + (j - 0.5) * w, vals, w, color=colr, label=lab, edgecolor="black", linewidth=0.4)
            for b, (_, is_test, _) in zip(bars, sets):
                if is_test:
                    b.set_hatch("////")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s, _, _ in sets])
        ax.set_xlabel("Source content")
        ax.set_ylabel("SROCC")
        lo = min([0.0] + [metrics(g, c)["SROCC"] for _, _, g in sets for c, _, _ in MODEL_COLS
                          if np.isfinite(metrics(g, c)["SROCC"])])
        ax.set_ylim(min(0.0, lo - 0.05), 1.0)
        h, lab = ax.get_legend_handles_labels()
        if fin is not None:
            h.append(Patch(facecolor="white", edgecolor="black", hatch="////"))
            lab.append("Locked test source (final model)")
        ax.legend(h, lab, loc="lower right", fontsize=7)
        ax.set_title("SROCC per source content (seed-averaged predictions)")
        self.save_fig(fig, "per_source_srocc",
                      "SROCC per source content. Every source is evaluated by a model that never saw it in training "
                      "(CV folds for the 12 dev sources; the final model for the locked test sources).")

    def heatmap(self, mat, name, title, caption, cmap="viridis", cbar_label=""):
        vals = mat.to_numpy(float)
        fig, ax = plt.subplots(figsize=(1.15 * mat.shape[1] + 2.4, 0.45 * mat.shape[0] + 1.3))
        im = ax.imshow(vals, cmap=cmap, aspect="auto")
        ax.set_xticks(range(mat.shape[1]))
        ax.set_xticklabels(mat.columns)
        ax.set_yticks(range(mat.shape[0]))
        ax.set_yticklabels(mat.index)
        ax.grid(False)
        mid = (np.nanmin(vals) + np.nanmax(vals)) / 2
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                ax.text(j, i, f"{vals[i, j]:.3f}", ha="center", va="center", fontsize=7.5, fontweight="bold",
                        color="white" if vals[i, j] < mid else "black")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(cbar_label, fontweight="bold")
        ax.set_title(title)
        self.save_fig(fig, name, caption)

    def fig_explain(self, cv):
        df, bcols, acols = self.explain_frames(cv["preds"])
        df = df.set_index("Distortion")
        self.heatmap(df[bcols], "branch_gate_heatmap", "Mean branch gate per distortion",
                     "Mean element-wise gate value (sigmoid, 0–1) of each branch per distortion type (CV test "
                     "predictions). Higher = the branch's features are passed through more strongly.",
                     cbar_label="Mean gate")
        k = len(acols)
        att = df[acols].copy()
        att.columns = [f"Seg {i + 1}\n{i * CLIP_SECONDS / k:g}–{(i + 1) * CLIP_SECONDS / k:g} s" for i in range(k)]
        self.heatmap(att, "segment_attention_heatmap", "Mean segment attention per distortion",
                     "Mean temporal segment attention α_k per distortion type (weights sum to 1; 1/K = uniform).",
                     cmap="magma", cbar_label="Attention weight")

    def fig_gate_violin(self, avg):
        cols = [c for c in avg.columns if c.startswith("branch_")]
        data = [avg[c].to_numpy(float) for c in cols]
        fig, ax = plt.subplots(figsize=(4.8, 3.0))
        vp = ax.violinplot(data, positions=range(len(cols)), showmeans=True, showextrema=True)
        for body, colr in zip(vp["bodies"], BRANCH_COLORS):
            body.set_facecolor(colr)
            body.set_edgecolor("black")
            body.set_alpha(0.55)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([BRANCH_LABELS.get(c[7:], c[7:]) for c in cols])
        ax.set_ylabel("Mean gate value per video")
        ax.set_title("Distribution of branch gates (CV videos)")
        self.save_fig(fig, "branch_gate_violin", "Per-video distribution of the mean gate of each branch.")

    def fig_delta_q(self, avg):
        ds = dists_in(avg)
        data = [avg.loc[avg["dist"] == d, "delta_q"].to_numpy(float) for d in ds]
        fig, ax = plt.subplots(figsize=(5.6, 3.0))
        bp = ax.boxplot(data, positions=range(len(ds)), widths=0.55, patch_artist=True, showfliers=False)
        for patch, d in zip(bp["boxes"], ds):
            patch.set_facecolor(dcolor(d))
            patch.set_alpha(0.45)
        for med in bp["medians"]:
            med.set_color("black")
        rng = np.random.default_rng(0)
        for i, (d, v) in enumerate(zip(ds, data)):
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=7, color=dcolor(d), alpha=0.85, edgecolors="none")
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xticks(range(len(ds)))
        ax.set_xticklabels(ds, rotation=20, ha="right")
        ax.set_ylabel("ΔQ (MOS units)")
        ax.set_title("Learned correction added to FAST-VQA score")
        self.save_fig(fig, "delta_q_by_distortion",
                      "Distribution of the learned residual ΔQ (P1 prediction − calibrated FAST-VQA score) per "
                      "distortion type. Large |ΔQ| shows where the motion / animation / CLIP branches change the "
                      "FAST-VQA estimate most.")

    def fig_bootstrap(self):
        if not self.boot_sets:
            return
        fig, axes = plt.subplots(1, len(self.boot_sets), figsize=(3.8 * len(self.boot_sets), 2.9), squeeze=False)
        for ax, (lab, diffs, obs, lo, hi) in zip(axes[0], self.boot_sets):
            ax.hist(diffs[np.isfinite(diffs)], bins=40, color=MODEL_COLS[0][2], alpha=0.6, edgecolor="black", linewidth=0.3)
            ax.axvline(obs, color="black", lw=1.4, label=f"observed {obs:+.3f}")
            ax.axvline(lo, color="black", lw=0.9, ls=":", label=f"95% CI [{lo:+.3f}, {hi:+.3f}]")
            ax.axvline(hi, color="black", lw=0.9, ls=":")
            ax.axvline(0, color="#cc4125", lw=1.0, ls="--", label="no difference")
            ax.set_xlabel("ΔSROCC (P1 − FAST-VQA only)")
            ax.set_ylabel("Bootstrap resamples")
            ax.set_title(lab, fontsize=9)
            ax.legend(loc="upper left", fontsize=6.5)
        fig.tight_layout()
        self.save_fig(fig, "bootstrap_delta_srocc",
                      f"Paired bootstrap ({self.n_boot} resamples of videos) of the SROCC gain of P1 over the "
                      "calibrated FAST-VQA score.")

    def fig_deploy_curve(self, hist):
        fig, ax = plt.subplots(figsize=(4.6, 2.9))
        pv = hist.pivot_table(index="epoch", columns="seed", values="train_loss")
        mu, sd = pv.mean(axis=1), pv.std(axis=1).fillna(0)
        ax.plot(pv.index, mu, color=MODEL_COLS[0][2], lw=1.4)
        ax.fill_between(pv.index, mu - sd, mu + sd, color=MODEL_COLS[0][2], alpha=0.2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training loss (Huber)")
        ax.set_title(f"Deploy models: all videos, {pv.shape[1]} seeds")
        self.save_fig(fig, "deploy_training_loss",
                      "Training loss of the deploy models (trained on all videos for a fixed number of epochs).")

    # ================================================================== markdown report
    def write_markdown(self, summary):
        lines = ["# DA-AVQA – P1 full-run report", ""]
        if summary:
            tp = summary.get("trainable_params")
            tp_s = f"{tp:,}" if isinstance(tp, int) else "?"
            lines += [f"Pipeline **{summary.get('pipeline', 'P1')}** | videos: {summary.get('n_videos', '?')} | "
                      f"trainable parameters: {tp_s}  ",
                      f"split id `{summary.get('split_id', '?')}` | data id `{summary.get('data_id', '?')}` | "
                      f"seeds {summary.get('args', {}).get('seeds', '?')}", ""]
        lines += ["## Headline", ""] + self.headline + [""]
        lines += ["## Tables", ""]
        for n, cap, disp in self.tables:
            lines += [f"### {n}", "", f"*{cap}*", "", df_to_md(disp), ""]
        lines += ["## Figures", ""]
        for n, cap in self.figures:
            lines += [f"### {n}", "", f"![{n}](figures/{n}.png)", "", f"*{cap}*", ""]
        (self.out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
        (self.fig_dir / "captions.md").write_text(
            "\n\n".join(f"**{n}.png** – {cap}" for n, cap in self.figures) + "\n", encoding="utf-8")


# =============================================================================== driver
def load_stage(root, stage):
    d = root / stage
    if not (d / "runs.csv").exists():
        return None
    runs = pd.read_csv(d / "runs.csv")
    preds = pd.read_csv(d / "predictions.csv")
    preds["dist"] = preds["distortion"].map(pretty_dist)
    hist = pd.read_csv(d / "history.csv") if (d / "history.csv").exists() else None
    return {"runs": runs, "preds": preds, "hist": hist, "avg": seed_avg(preds)}


def make_report(work, tag="P1_full", n_boot=1000):
    warnings.filterwarnings("ignore")
    work = Path(work).expanduser().resolve()
    R = Report(work, tag, n_boot)
    cv, fin = load_stage(R.root, "cv"), load_stage(R.root, "final")
    if cv is None:
        raise SystemExit(f"[report] no CV results in {R.root / 'cv'} - run run_full_p1.py first")
    meta = load_metadata(work).copy()
    meta["dist"] = meta["distortion"].map(pretty_dist)
    splits = load_splits(work)
    summary = json.loads((R.root / "run_summary.json").read_text()) if (R.root / "run_summary.json").exists() else {}
    deploy = json.loads((R.root / "deploy" / "info.json").read_text()) if (R.root / "deploy" / "info.json").exists() else None
    n_seeds_cv = cv["preds"]["seed"].nunique()
    print(f"[report] writing to {R.out}  (bootstrap {n_boot}x, may take a minute)")

    # ---------------- tables
    R.safe(R.table_dataset, meta)
    R.safe(R.table_splits, meta, splits)
    R.safe(R.table_overall, cv, fin)
    R.safe(R.table_per_dist, cv, "per_distortion_cv",
           f"Per-distortion performance, 4-fold source-grouped CV (out-of-fold predictions pooled per seed; mean ± "
           f"std over {n_seeds_cv} seeds).")
    if fin is not None:
        R.safe(R.table_per_dist, fin, "per_distortion_final_seeds",
               "Per-distortion performance on the locked test sources (mean ± std over seeds).")
        R.safe(R.table_per_dist_pooled, fin["avg"], "per_distortion_final_ensemble",
               "Per-distortion performance on the locked test sources, seed-ensemble prediction (mean of the "
               "seed models).")
    R.safe(R.table_folds, cv, fin, splits)
    R.safe(R.table_sources, cv, fin)
    R.safe(R.table_explain, cv)
    R.safe(R.table_significance, cv, fin)
    R.safe(R.table_complexity, summary, cv, fin, deploy)
    R.safe(R.table_reference, cv, fin)

    # ---------------- figures
    R.safe(R.fig_dataset, meta)
    R.safe(R.fig_scatter_pair, cv["avg"], f"4-fold CV, out-of-fold predictions (N = {len(cv['avg'])})",
           "scatter_cv", "Predicted vs subjective MOS on held-out source contents (4-fold CV, seed-averaged), "
                         "calibrated FAST-VQA (left) and P1 (right), with VQEG logistic fit.")
    if fin is not None:
        R.safe(R.fig_scatter_pair, fin["avg"], f"Locked test sources, seed ensemble (N = {len(fin['avg'])})",
               "scatter_final_test", "Predicted vs subjective MOS on the locked test sources (seed ensemble).")
    R.safe(R.fig_scatter_per_dist, cv["avg"], "P1 per distortion (4-fold CV, out-of-fold)", "scatter_per_distortion_cv",
           "P1 predictions vs MOS per distortion type (CV out-of-fold, seed-averaged).")
    stage_sets = [("4-fold CV", cv["preds"])] + ([("Locked test", fin["preds"])] if fin is not None else [])
    R.safe(R.fig_per_dist_bars, stage_sets)
    R.safe(R.fig_split_bars, cv, fin, splits)
    run_sets = [("4-fold CV", cv["runs"])] + ([("Locked test", fin["runs"])] if fin is not None else [])
    R.safe(R.fig_run_distribution, run_sets)
    if cv["hist"] is not None:
        R.safe(R.fig_training_curves, cv["hist"], cv["runs"])
    R.safe(R.fig_mos_vs_level, cv["avg"], "4-fold CV, out-of-fold predictions")
    R.safe(R.fig_error_by_level, cv["avg"], "4-fold CV, out-of-fold predictions")
    res_sets = [("4-fold CV (out-of-fold)", cv["avg"])] + ([("Locked test (ensemble)", fin["avg"])] if fin is not None else [])
    R.safe(R.fig_residuals, res_sets)
    R.safe(R.fig_per_source, cv, fin)
    R.safe(R.fig_explain, cv)
    R.safe(R.fig_gate_violin, cv["avg"])
    R.safe(R.fig_delta_q, cv["avg"])
    R.safe(R.fig_bootstrap)
    dh = R.root / "deploy" / "history.csv"
    if dh.exists():
        R.safe(R.fig_deploy_curve, pd.read_csv(dh))

    R.write_markdown(summary)
    print(f"[report] {len(R.tables)} tables, {len(R.figures)} figures -> {R.out}  (open REPORT.md)")


def main():
    ap = argparse.ArgumentParser(description="Tables + figures for the full P1 run")
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--tag", default="P1_full")
    ap.add_argument("--n_boot", type=int, default=1000)
    a = ap.parse_args()
    make_report(a.work_dir, tag=a.tag, n_boot=a.n_boot)


if __name__ == "__main__":
    main()
