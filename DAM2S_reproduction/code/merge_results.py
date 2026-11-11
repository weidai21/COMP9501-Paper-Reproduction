"""Merge several run directories (results JSON + predictions + projections) into one.

Usage: python -I merge_results.py <out_dir> <run_dir> [<run_dir> ...]
For every method the configurations of all runs are pooled ("all") and the best one is kept
("best"), together with its target predictions and (for DAM2S) its learned projections.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def main(out_dir, run_dirs):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged, preds, src = {}, {}, {}
    for d in map(Path, run_dirs):
        res = json.load(open(d / "or_results.json"))
        pr = dict(np.load(d / "or_preds.npz")) if (d / "or_preds.npz").exists() else {}
        for name, r in res.items():
            if name not in merged:
                merged[name] = dict(best=r["best"], all=list(r["all"]))
                preds[name], src[name] = pr.get(name), d
                continue
            merged[name]["all"] += r["all"]
            if r["best"]["acc"] > merged[name]["best"]["acc"]:
                merged[name]["best"] = r["best"]
                preds[name], src[name] = pr.get(name), d
    json.dump(merged, open(out / "or_results.json", "w"), indent=1)
    np.savez(out / "or_preds.npz", **{k: v for k, v in preds.items() if v is not None})
    for name, d in src.items():
        f = d / f"or_proj_{name}.npz"
        if f.exists():
            shutil.copy(f, out / f.name)
    for name, r in merged.items():
        print(f"{name:18s} {100 * r['best']['acc']:6.2f}  {r['best']['params']}  ({len(r['all'])} configs, from {src[name].name})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
