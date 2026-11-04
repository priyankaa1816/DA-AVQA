"""
Step 1 - Dataset audit + fixed source-disjoint splits.

Reads MOSScores.xlsx, finds every distorted .mp4 under the dataset root
(the Reference folder is ignored, it has no MOS), joins them, reports
anything missing (02_Blur_3 is expected), and writes:

    <work_dir>/metadata.csv   one row per MOS entry (available=False if file missing)
    <work_dir>/splits.json    locked test sources + 4 grouped CV folds + final split

splits.json is written ONCE and reused by every pipeline (P0..P4) so all
comparisons use identical data. It is never overwritten unless --force.

Usage:
    python prepare_data.py --dataset_root "D:/BeDAMQA" --work_dir "D:/avqa_work"
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Name code in file names -> canonical distortion (matches the folder names)
CODE_TO_DISTORTION = {
    "blur": "Blur",
    "fhfh": "FrameHang",
    "frfr": "FrameFreeze",
    "h264": "H264",
    "h265": "H265",
}
NAME_RE = re.compile(r"(\d{1,3})_([A-Za-z0-9]+)_(\d+)")


def clean_name(raw):
    """'01_Blur_3.mp4\'\'\'' -> '01_Blur_3' (also handles spaces / quotes / no extension)."""
    m = NAME_RE.search(str(raw))
    if m is None:
        return None
    src, code, lvl = m.groups()
    return f"{int(src):02d}_{code}_{lvl}"


def find_mos_file(root: Path):
    cands = [p for p in root.glob("*.xls*") if "mos" in p.name.lower()]
    if not cands:
        cands = list(root.glob("*.xls*"))
    if not cands:
        sys.exit(f"[ERROR] No MOS spreadsheet found in {root}. Pass --mos_file.")
    return cands[0]


def read_mos(mos_file: Path):
    df = pd.read_excel(mos_file, header=0)
    df = df.iloc[:, :2]
    df.columns = ["raw_name", "mos"]
    df = df.dropna(subset=["raw_name"])
    df["name"] = df["raw_name"].map(clean_name)
    bad = df[df["name"].isna()]
    if len(bad):
        print(f"[WARN] {len(bad)} MOS rows with unparseable names were skipped:\n{bad}")
    df = df.dropna(subset=["name"]).copy()
    df["mos"] = pd.to_numeric(df["mos"], errors="coerce")
    if df["mos"].isna().any():
        sys.exit(f"[ERROR] Non-numeric MOS values:\n{df[df['mos'].isna()]}")
    dup = df[df["name"].duplicated(keep=False)]
    if len(dup):
        sys.exit(f"[ERROR] Duplicate names in MOS sheet:\n{dup}")
    return df[["name", "mos"]]


def scan_videos(root: Path):
    found = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() != ".mp4":
            continue
        rel = p.relative_to(root)
        if any(part.lower() == "reference" for part in rel.parts[:-1]):
            continue
        name = clean_name(p.stem)
        if name is None:
            print(f"[WARN] Skipping file with unexpected name: {rel}")
            continue
        if name in found:
            sys.exit(f"[ERROR] Duplicate video name {name}: {found[name]} and {rel}")
        found[name] = rel
    return found


def make_splits(sources, seed, n_test, n_folds, n_val, test_sources=None):
    rng = np.random.RandomState(seed)
    sources = sorted(int(s) for s in sources)
    if test_sources:
        test = sorted(int(s) for s in test_sources)
        unknown = set(test) - set(sources)
        if unknown:
            sys.exit(f"[ERROR] --test_sources not in dataset: {sorted(unknown)}")
    else:
        test = sorted(rng.choice(sources, size=n_test, replace=False).tolist())
    dev = [s for s in sources if s not in test]
    if len(dev) % n_folds != 0:
        print(f"[WARN] {len(dev)} dev sources are not divisible by {n_folds} folds; folds will be uneven.")
    perm = rng.permutation(dev).tolist()
    groups = [sorted(perm[i::n_folds]) for i in range(n_folds)]

    folds = []
    for f, g in enumerate(groups):
        rest = [s for s in dev if s not in g]
        frng = np.random.RandomState(seed + 1000 + f)
        val = sorted(frng.choice(rest, size=n_val, replace=False).tolist())
        train = [s for s in rest if s not in val]
        folds.append({"fold": f, "train_sources": train, "val_sources": val, "test_sources": g})

    frng = np.random.RandomState(seed + 9999)
    fval = sorted(frng.choice(dev, size=n_val, replace=False).tolist())
    final = {"train_sources": [s for s in dev if s not in fval], "val_sources": fval, "test_sources": test}
    return {
        "seed": seed,
        "locked_test_sources": test,
        "dev_sources": dev,
        "folds": folds,
        "final": final,
        "note": "Do not touch locked_test_sources until the final evaluation.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True, help="Folder with MOSScores.xlsx + Blur/H264/... folders")
    ap.add_argument("--work_dir", required=True, help="Where metadata.csv / splits.json / features go")
    ap.add_argument("--mos_file", default=None)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n_test_sources", type=int, default=3)
    ap.add_argument("--n_folds", type=int, default=4)
    ap.add_argument("--n_val_sources", type=int, default=2)
    ap.add_argument("--test_sources", type=int, nargs="*", default=None, help="Optional: fix the locked test sources")
    ap.add_argument("--force", action="store_true", help="Overwrite an existing splits.json (breaks comparability!)")
    args = ap.parse_args()

    root = Path(args.dataset_root).expanduser().resolve()
    work = Path(args.work_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)

    mos_file = Path(args.mos_file) if args.mos_file else find_mos_file(root)
    print(f"MOS file : {mos_file}")
    mos = read_mos(mos_file)
    files = scan_videos(root)
    print(f"MOS rows : {len(mos)}   |   distorted .mp4 files found: {len(files)}")

    rows = []
    for name, m in zip(mos["name"], mos["mos"]):
        src, code, lvl = name.split("_")
        rel = files.get(name)
        dist_code = CODE_TO_DISTORTION.get(code.lower(), code)
        folder = rel.parts[0] if rel is not None and len(rel.parts) > 1 else None
        if folder is not None and folder.lower() != dist_code.lower():
            print(f"[WARN] {name}: name code '{code}' -> {dist_code}, but file is in folder '{folder}'. Using folder.")
        rows.append({
            "name": name,
            "source": int(src),
            "code": code,
            "distortion": folder if folder is not None else dist_code,
            "level": int(lvl),
            "mos": float(m),
            "rel_path": rel.as_posix() if rel is not None else "",
            "available": rel is not None,
        })
    meta = pd.DataFrame(rows).sort_values("name").reset_index(drop=True)

    missing = meta.loc[~meta["available"], "name"].tolist()
    extra = sorted(set(files) - set(meta["name"]))
    print(f"Missing videos (MOS but no file): {missing if missing else 'none'}")
    if extra:
        print(f"[WARN] Files without MOS (ignored): {extra}")
    avail = meta[meta["available"]]
    print(f"Usable videos: {len(avail)}")
    print("\nVideos per distortion:\n", avail.groupby("distortion").size().to_string())
    print("\nVideos per source:\n", avail.groupby("source").size().to_string())
    print(f"\nMOS range: {avail['mos'].min():.3f} - {avail['mos'].max():.3f}")

    meta.to_csv(work / "metadata.csv", index=False)
    print(f"\nWrote {work / 'metadata.csv'}")

    sp_path = work / "splits.json"
    if sp_path.exists() and not args.force:
        print(f"splits.json already exists -> kept unchanged ({sp_path}). Use --force only if you really mean it.")
        splits = json.loads(sp_path.read_text())
    else:
        # splits are built from ALL sources in the MOS sheet, independent of which files exist
        splits = make_splits(meta["source"].unique(), args.seed, args.n_test_sources,
                             args.n_folds, args.n_val_sources, args.test_sources)
        sp_path.write_text(json.dumps(splits, indent=2))
        print(f"Wrote {sp_path}")

    def count(srcs):
        return int(avail["source"].isin(srcs).sum())

    print(f"\nLocked test sources: {splits['locked_test_sources']}  ({count(splits['locked_test_sources'])} videos)")
    for f in splits["folds"]:
        print(f"Fold {f['fold']}: train {f['train_sources']} ({count(f['train_sources'])}) | "
              f"val {f['val_sources']} ({count(f['val_sources'])}) | test {f['test_sources']} ({count(f['test_sources'])})")
    fi = splits["final"]
    print(f"Final : train {fi['train_sources']} ({count(fi['train_sources'])}) | val {fi['val_sources']} "
          f"({count(fi['val_sources'])}) | test {fi['test_sources']} ({count(fi['test_sources'])})")


if __name__ == "__main__":
    main()
