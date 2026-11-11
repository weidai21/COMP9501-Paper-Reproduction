"""Depth kernel descriptors (Gradient KDES + LBP KDES) with EMK object-level features.

Python re-implementation of Bo, Ren & Fox, "Kernel descriptors for visual recognition"
(NIPS'10) / "Depth kernel descriptors for object recognition" (IROS'11), which the paper
follows for the depth images of the source domain:
  * dense 16x16 patches with stride 8,
  * gradient KDES: k_o (25 orientation bases, gamma_o=5) x k_p (5x5 position bases, gamma_p=3),
    LBP KDES: k_b (all 256 binary patterns, gamma_b=0.5) x k_p; both reduced to 200-d by
    kernel PCA on the joint basis (Kronecker structure, so it is exact and cheap),
  * object-level features by efficient match kernels (EMK) with 1000 basis vectors
    (k-means) and a 1x1, 2x2, 3x3 spatial pyramid -> 14,000-d per descriptor type,
  * gradient and LBP object features are concatenated (28,000-d).

Usage: python -I kdes.py <or_split.json> <out.npz>
"""
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from sklearn.cluster import MiniBatchKMeans

PATCH, STRIDE, DIM = 16, 8, 200
EPS_G, EPS_S = 0.8, 0.2  # low-contrast constants of the gradient / LBP descriptors
N_WORDS = 1000
PYRAMID = (1, 2, 3)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_depth(path, min_side=64, max_side=200):
    """uint16 depth (mm) -> float depth in decimetres with holes filled by nearest valid pixel."""
    d = np.asarray(Image.open(path)).astype(np.float64)
    valid = d > 0
    if not valid.any():
        d[:] = 1000.0
    elif not valid.all():
        idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
        d = d[tuple(idx)]
    d = d / 100.0
    h, w = d.shape
    s = 1.0
    if min(h, w) < min_side:
        s = min_side / min(h, w)
    if max(h, w) * s > max_side:
        s = max_side / max(h, w)
    if s != 1.0:
        t = torch.from_numpy(d)[None, None].float()
        t = F.interpolate(t, size=(max(int(round(h * s)), PATCH), max(int(round(w * s)), PATCH)),
                          mode="bilinear", align_corners=False)
        d = t[0, 0].double().numpy()
    return d


def kpca_pairs(lam_a, lam_b, dim):
    """Top-`dim` eigenpairs of a Kronecker product kernel K_a (x) K_b."""
    prod = np.outer(lam_a, lam_b)
    order = np.argsort(prod.ravel())[::-1][:dim]
    ia, ib = np.unravel_index(order, prod.shape)
    return ia, ib, prod.ravel()[order]


class KDES:
    def __init__(self):
        # position kernel on a 5x5 grid of basis positions in [0,1]^2
        g = np.linspace(0, 1, 5)
        pb = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
        lp, up = np.linalg.eigh(np.exp(-3.0 * ((pb[:, None] - pb[None]) ** 2).sum(-1)))
        c = (np.arange(PATCH) + 0.5) / PATCH
        pos = np.stack(np.meshgrid(c, c, indexing="ij"), -1).reshape(-1, 2)
        kp = np.exp(-3.0 * ((pos[:, None] - pb[None]) ** 2).sum(-1))  # (256 pixels, 25 bases)
        self.pos_filters = (kp @ up).T.reshape(25, PATCH, PATCH)  # projections on eigvecs of K_p
        self.lp = lp
        # orientation kernel: 25 basis orientations on the unit circle
        th = 2 * np.pi * np.arange(25) / 25
        self.ob = np.stack([np.cos(th), np.sin(th)], 1)
        lo, uo = np.linalg.eigh(np.exp(-5.0 * ((self.ob[:, None] - self.ob[None]) ** 2).sum(-1)))
        self.lo, self.uo = lo, uo
        self.g_io, self.g_ip, lam = kpca_pairs(lo, lp, DIM)
        self.g_scale = 1.0 / np.sqrt(lam)
        # LBP kernel: all 256 8-bit patterns
        bits = ((np.arange(256)[:, None] >> np.arange(8)[None]) & 1).astype(np.float64)
        lb, ub = np.linalg.eigh(np.exp(-0.5 * ((bits[:, None] - bits[None]) ** 2).sum(-1)))
        self.lb, self.ub = lb, ub
        self.l_ib, self.l_ip, lam = kpca_pairs(lb, lp, DIM)
        self.l_scale = 1.0 / np.sqrt(lam)
        self.lbp_table = (ub * lb[None, :])  # (256 patterns, 256 eigen-components): U_b' k_b(b, .)

    def _pool(self, chan, weight, ia, ip, scale, eps):
        """sum_{z in P} w~(z) chan_ia(z) posfilter_ip(z - P) for dense patches; chan: (C,H,W)."""
        H, W = weight.shape
        uniq = np.unique(ia)
        x = torch.from_numpy(chan[uniq] * weight[None]).float().to(DEV)[None]  # (1,C,H,W)
        filt = torch.from_numpy(self.pos_filters).float().to(DEV)
        # grouped conv: every selected channel is correlated with all 25 position filters
        w = filt[None].repeat(len(uniq), 1, 1, 1).reshape(len(uniq) * 25, 1, PATCH, PATCH)
        out = F.conv2d(x, w, stride=STRIDE, groups=len(uniq))[0]  # (C*25, h, w)
        lut = {c: k for k, c in enumerate(uniq)}
        sel = torch.tensor([lut[a] * 25 + b for a, b in zip(ia, ip)], device=DEV)
        feat = out[sel]  # (DIM, h, w)
        ones = torch.ones(1, 1, PATCH, PATCH, device=DEV)
        energy = F.conv2d(torch.from_numpy(weight ** 2).float().to(DEV)[None, None], ones, stride=STRIDE)[0, 0]
        feat = feat / torch.sqrt(energy + eps)[None]
        feat = feat * torch.from_numpy(scale).float().to(DEV)[:, None, None]
        h, w_ = feat.shape[1:]
        cy = (torch.arange(h, device=DEV) * STRIDE + PATCH / 2) / H
        cx = (torch.arange(w_, device=DEV) * STRIDE + PATCH / 2) / W
        centers = torch.stack(torch.meshgrid(cy, cx, indexing="ij"), -1).reshape(-1, 2)
        return feat.reshape(DIM, -1).T.cpu().numpy(), centers.cpu().numpy()

    def gradient(self, img):
        gy, gx = np.gradient(img)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        th = np.arctan2(gy, gx)
        u = np.stack([np.cos(th), np.sin(th)], -1)
        ko = np.exp(-5.0 * ((u[..., None, :] - self.ob) ** 2).sum(-1))  # (H,W,25)
        chan = np.moveaxis(ko @ self.uo, -1, 0)  # projections on eigvecs of K_o
        return self._pool(chan, mag, self.g_io, self.g_ip, self.g_scale, EPS_G)

    def lbp(self, img):
        H, W = img.shape
        p = np.pad(img, 1, mode="edge")
        offs = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        nb = np.stack([p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W] for dy, dx in offs])
        code = ((nb > img[None]).astype(np.int64) << np.arange(8)[:, None, None]).sum(0)
        std = np.concatenate([nb, img[None]]).std(0)
        chan = np.moveaxis(self.lbp_table[code], -1, 0)  # (256,H,W)
        return self._pool(chan, std, self.l_ib, self.l_ip, self.l_scale, EPS_S)


def emk_features(descs, centers, words, gamma, G_isqrt):
    """Average of the finite-dim kernel map over each spatial-pyramid cell."""
    Z = torch.from_numpy(words).float().to(DEV)
    Gi = torch.from_numpy(G_isqrt).float().to(DEV)
    feats = []
    for d, c in zip(descs, centers):
        d = torch.from_numpy(d).float().to(DEV)
        phi = torch.exp(-gamma * torch.cdist(d, Z) ** 2) @ Gi  # (n_patch, N_WORDS)
        cells = []
        for L in PYRAMID:
            cid = np.minimum((c * L).astype(int), L - 1)
            cid = torch.from_numpy(cid[:, 0] * L + cid[:, 1]).to(DEV)
            acc = torch.zeros(L * L, N_WORDS, device=DEV).index_add_(0, cid, phi)
            cnt = torch.bincount(cid, minlength=L * L).clamp(min=1)[:, None]
            cells.append(acc / cnt)
        feats.append(torch.cat(cells).reshape(-1).cpu().numpy())
    return np.stack(feats).astype(np.float32)


def learn_emk(descs, rng, n_sample=200000):
    allp = np.concatenate(descs)
    sub = allp[rng.choice(len(allp), min(n_sample, len(allp)), replace=False)]
    km = MiniBatchKMeans(N_WORDS, batch_size=8192, n_init=3, random_state=0).fit(sub)
    words = km.cluster_centers_
    t = torch.from_numpy(sub[:20000]).float().to(DEV)
    nn2 = (torch.cdist(t, torch.from_numpy(words).float().to(DEV)) ** 2).min(1).values
    gamma = 1.0 / nn2.mean().item()
    G = np.exp(-gamma * ((words[:, None] - words[None]) ** 2).sum(-1))
    lam, U = np.linalg.eigh(G)
    lam = np.maximum(lam, 1e-6 * lam.max())
    return words, gamma, (U / np.sqrt(lam)) @ U.T


def main(split_json, out_npz):
    split = json.load(open(split_json))
    kd = KDES()
    rng = np.random.default_rng(0)
    g_desc, g_cent, l_desc, l_cent = [], [], [], []
    for i, s in enumerate(split["source"]):
        img = load_depth(s["depth"])
        d, c = kd.gradient(img)
        g_desc.append(d); g_cent.append(c)
        d, c = kd.lbp(img)
        l_desc.append(d); l_cent.append(c)
        if i % 500 == 0:
            print(f"  patch descriptors {i}/{len(split['source'])}", flush=True)
    out = []
    for name, descs, cents in (("gradient", g_desc, g_cent), ("lbp", l_desc, l_cent)):
        words, gamma, Gi = learn_emk(descs, rng)
        out.append(emk_features(descs, cents, words, gamma, Gi))
        print(f"  {name} KDES object features: {out[-1].shape}, gamma={gamma:.4g}", flush=True)
    Z = np.concatenate(out, 1)
    np.savez_compressed(out_npz, Z=Z, ys=np.array([s["label"] for s in split["source"]]))
    print("depth KDES:", Z.shape)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
