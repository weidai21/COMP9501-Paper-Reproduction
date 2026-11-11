"""Gradient + LBP KDES visual features for the RGB images (setting of the CVPR'14 preliminary
version of the paper, where KDES rather than DeCAF6 is used for the visual view).

Usage: python -I kdes_rgb.py <or_split.json> <out.npz>
Grey-level images in [0,1], longer side resized to 200 px, the same dense patches / KPCA
descriptors as kdes.py, and an EMK vocabulary (1000 words, 1x1/2x2/3x3 pyramid) learned on
patches of both domains (a different vocabulary from the depth one).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kdes import KDES, PATCH, emk_features, learn_emk  # noqa: E402


def load_gray(path, min_side=64, max_side=200):
    g = np.asarray(Image.open(path).convert("L")).astype(np.float64) / 255.0
    h, w = g.shape
    s = min(max_side / max(h, w), max(1.0, min_side / min(h, w)))
    if s != 1.0:
        t = torch.from_numpy(g)[None, None].float()
        t = F.interpolate(t, size=(max(int(round(h * s)), PATCH), max(int(round(w * s)), PATCH)),
                          mode="bilinear", align_corners=False)
        g = t[0, 0].double().numpy()
    return g


def main(split_json, out_npz):
    split = json.load(open(split_json))
    items = split["source"] + split["target"]
    kd = KDES()
    rng = np.random.default_rng(0)
    g_desc, g_cent, l_desc, l_cent = [], [], [], []
    for i, s in enumerate(items):
        img = load_gray(s["rgb"])
        d, c = kd.gradient(img)
        g_desc.append(d); g_cent.append(c)
        d, c = kd.lbp(img)
        l_desc.append(d); l_cent.append(c)
        if i % 500 == 0:
            print(f"  patch descriptors {i}/{len(items)}", flush=True)
    out = []
    for name, descs, cents in (("gradient", g_desc, g_cent), ("lbp", l_desc, l_cent)):
        words, gamma, Gi = learn_emk(descs, rng)
        out.append(emk_features(descs, cents, words, gamma, Gi))
        print(f"  {name} KDES object features: {out[-1].shape}", flush=True)
    X = np.concatenate(out, 1)
    ns = len(split["source"])
    np.savez_compressed(out_npz, Xs=X[:ns], Xt=X[ns:],
                        ys=np.array([s["label"] for s in split["source"]]),
                        yt=np.array([s["label"] for s in split["target"]]))
    print("RGB KDES:", X[:ns].shape, X[ns:].shape)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
