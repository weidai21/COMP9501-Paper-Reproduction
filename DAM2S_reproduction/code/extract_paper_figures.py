"""Crop the original figure / tables of the TPAMI'18 DAM2S paper at high resolution.

Usage: python -I extract_paper_figures.py <paper.pdf> <out_dir>
Regions are given in PDF points (1/72 inch), found with `pdftotext -bbox`.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

DPI = 600
# name: (page, x0, y0, x1, y1) in points
REGIONS = {
    "paper_fig1_with_caption": (2, 24, 36, 284, 154),
    "paper_fig1": (2, 24, 36, 284, 127),
    "paper_table1_objectives": (4, 316, 43, 514, 143),
    "paper_table2_accuracy": (5, 22, 43, 544, 125),
    "paper_table3_time": (6, 22, 43, 284, 107),
}


def autotrim(img, pad=20, thr=245):
    a = np.asarray(img.convert("L"))
    ys, xs = np.where(a < thr)
    if len(ys) == 0:
        return img
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, a.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, a.shape[1])
    return img.crop((x0, y0, x1, y1))


def main(pdf, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s = DPI / 72.0
    with tempfile.TemporaryDirectory() as td:
        pages = {}
        for name, (pg, x0, y0, x1, y1) in REGIONS.items():
            if pg not in pages:
                prefix = Path(td) / f"p{pg}"
                subprocess.run(["pdftoppm", "-r", str(DPI), "-f", str(pg), "-l", str(pg), "-png",
                                str(pdf), str(prefix)], check=True)
                pages[pg] = Image.open(next(Path(td).glob(f"p{pg}*.png"))).convert("RGB")
            crop = pages[pg].crop((int(x0 * s), int(y0 * s), int(x1 * s), int(y1 * s)))
            crop = autotrim(crop)
            crop.save(out_dir / f"{name}.png", dpi=(DPI, DPI))
            print(f"{name}: {crop.size}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
