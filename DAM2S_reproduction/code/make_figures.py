"""All figures / table images for the slides.

Usage: python -I make_figures.py <project_root>
Reads features/or_split.json, results/or_results.json, results/or_preds.npz,
results/or_proj_*.npz and writes PNG (300 dpi) + PDF files into figures/.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

# palette / chrome (dataviz reference palette, light mode)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#ffffff"
BOX_FILL, BOX_EDGE = "#eef4fc", "#86b6ef"
GRAY_BAR = "#c3c2b7"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "figure.facecolor": SURF,
    "axes.facecolor": SURF, "savefig.facecolor": SURF, "legend.frameon": False,
    "mathtext.fontset": "dejavusans",
})

METHODS = ["SVM_A", "SVM+", "KCCA", "SVM2K", "KMM", "DAM", "SGF", "LMK", "TCA", "SA", "DIP",
           "DAM2S_A", "DAM2S_B", "DAM2S_C"]
PAPER_OR = [27.52, 29.04, 31.52, 34.21, 28.88, 27.37, 38.93, 39.50, 37.79, 44.55, 45.13,
            53.17, 54.33, 56.89]
PAPER_TIME_CDHAR = [20.7, 168.4, 398.2]
CATS = ["ball", "calculator", "cereal_box", "coffee_mug", "flashlight", "keyboard", "lightbulb",
        "mushroom", "soda_can", "tomato"]


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("  wrote", name)


def style_axes(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def depth_vis(path):
    """Depth for display: near = dark, far = light (rank-equalised), missing depth = black."""
    d = np.asarray(Image.open(path)).astype(float)
    v = d > 0
    out = np.zeros_like(d)
    if v.any():
        r = np.argsort(np.argsort(d[v], kind="stable"), kind="stable")
        out[v] = 0.12 + 0.83 * r / max(len(r) - 1, 1)
    return out


def square(im, fill=255):
    """Pad an image (PIL or 2-D float array) to a square tile."""
    a = np.asarray(im)
    h, w = a.shape[:2]
    s = max(h, w)
    pad = [((s - h) // 2, s - h - (s - h) // 2), ((s - w) // 2, s - w - (s - w) // 2)]
    if a.ndim == 3:
        pad.append((0, 0))
    return np.pad(a, pad, constant_values=fill if a.dtype == np.uint8 else 1.0)


# ----------------------------------------------------------------------------- Fig. 1
def fig1(root, out):
    rgbd = root / "data/rgbd_full/extracted/rgbd-dataset"
    c256 = root / "data/caltech256/extracted/256_ObjectCategories"
    src = [("Keyboard", rgbd / "keyboard/keyboard_1/keyboard_1_1_60"),
           ("Ball", rgbd / "ball/ball_3/ball_3_1_60"),
           ("Coffee mug", rgbd / "coffee_mug/coffee_mug_1/coffee_mug_1_1_60")]
    tgt = [c256 / "045.computer-keyboard/045_0004.jpg", c256 / "147.mushroom/147_0007.jpg",
           c256 / "041.coffee-mug/041_0013.jpg"]
    W, H = 13.0, 4.6
    fig = plt.figure(figsize=(W, H))
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set_xlim(0, W)
    canvas.set_ylim(0, H)
    canvas.axis("off")

    def box(x, y, w, h):
        canvas.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                        fc=BOX_FILL, ec=BOX_EDGE, lw=1.4))

    def img(im, x, y, w, h, cmap=None):
        ax = fig.add_axes([x / W, y / H, w / W, h / H])
        arr = np.asarray(im)
        ax.imshow(arr, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None, aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#ffffff")
            s.set_linewidth(1.5)

    box(0.15, 0.25, 5.6, 4.15)
    xs = [0.45, 2.15, 4.15]
    for (name, stem), x in zip(src, xs):
        img(Image.open(f"{stem}_crop.png").convert("RGB"), x, 2.55, 1.35, 1.35)
        img(depth_vis(f"{stem}_depthcrop.png"), x, 0.85, 1.35, 1.35, cmap="gray")
        canvas.add_patch(FancyArrowPatch((x + 0.675, 2.27), (x + 0.675, 2.5), arrowstyle="<|-|>",
                                         mutation_scale=9, color=INK2, lw=1.2))
        canvas.text(x + 0.675, 4.02, name, ha="center", va="center", fontsize=10.5, color=INK2)
    canvas.text(3.82, 3.22, "···", ha="center", va="center", fontsize=20, color=INK2)
    canvas.text(3.82, 1.52, "···", ha="center", va="center", fontsize=20, color=INK2)
    canvas.text(0.3, 3.22, "RGB", rotation=90, ha="center", va="center", fontsize=10, color=MUTED)
    canvas.text(0.3, 1.52, "Depth", rotation=90, ha="center", va="center", fontsize=10, color=MUTED)
    canvas.text(2.95, 0.48, "Source domain: labeled RGB-D data", ha="center", va="center",
                fontsize=13, color=INK)

    for y, label, dy in ((2.95, "Different\ndistributions", 0.55), (1.75, "Different\nfeatures", -0.55)):
        canvas.add_patch(FancyArrowPatch((6.0, y), (7.2, y), arrowstyle="<|-|>", mutation_scale=22,
                                         color=BLUE, lw=3.0))
        canvas.text(6.6, y + dy, label, ha="center", va="center", fontsize=11.5, color=INK,
                    fontweight="bold")

    box(7.45, 1.0, 5.4, 2.7)
    for p, x in zip(tgt, [7.75, 9.35, 11.3]):
        img(Image.open(p).convert("RGB"), x, 1.75, 1.3, 1.3)
    canvas.text(10.97, 2.4, "···", ha="center", va="center", fontsize=20, color=INK2)
    canvas.text(10.15, 1.33, "Target domain: unlabeled RGB images", ha="center", va="center",
                fontsize=13, color=INK)
    canvas.text(10.15, 3.35, "Caltech-256", ha="center", va="center", fontsize=10.5, color=INK2)
    canvas.text(2.95, 4.25 + 0.0, "", ha="center")
    save(fig, out, "fig1_rgbd_to_rgb_reproduced")


def fig_dataset(root, split, out):
    by_cat = {c: [s for s in split["source"] if s["category"] == c] for c in CATS}
    tg = {c: [s for s in split["target"] if s["category"] == c] for c in CATS}
    fig, axes = plt.subplots(3, 10, figsize=(15, 5.2), gridspec_kw=dict(hspace=0.08, wspace=0.06))
    rng = np.random.default_rng(3)
    for j, c in enumerate(CATS):
        s = by_cat[c][len(by_cat[c]) // 3]
        t = tg[c][rng.integers(len(tg[c]))]
        axes[0, j].imshow(square(Image.open(s["rgb"]).convert("RGB")))
        axes[1, j].imshow(square(depth_vis(s["depth"])), cmap="gray", vmin=0, vmax=1)
        axes[2, j].imshow(square(Image.open(t["rgb"]).convert("RGB")))
        axes[0, j].set_title(c.replace("_", " "), fontsize=11, color=INK, fontweight="normal")
        for i in range(3):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            for sp in axes[i, j].spines.values():
                sp.set_visible(False)
    for i, lab in enumerate(["Source RGB\n(RGB-D Object)", "Source depth\n(RGB-D Object)",
                             "Target RGB\n(Caltech-256)"]):
        axes[i, 0].set_ylabel(lab, fontsize=10.5, color=INK2, rotation=90, labelpad=8)
    save(fig, out, "fig_dataset_samples")


def fig_framework(out):
    """Schematic of the DAM2S framework (Eq. 1 and Table 1 of the paper)."""
    W, H = 13.6, 5.6
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    def box(x, y, w, h, fc, ec, lw=1.3):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.1", fc=fc, ec=ec, lw=lw))

    def arrow(p, q, color=INK2, lw=1.4, style="-|>", ls="-"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13, color=color, lw=lw, linestyle=ls,
                                     shrinkA=2, shrinkB=2))

    def text(x, y, s, size=11, color=INK, weight="normal", ha="center"):
        ax.text(x, y, s, ha=ha, va="center", fontsize=size, color=color, fontweight=weight)

    # domains: target on top, source (RGB + depth) below
    box(0.15, 3.95, 3.55, 1.5, "#fdf1ec", "#f2a283")
    text(1.93, 5.17, "Target domain (unlabeled RGB)", 11.5, weight="bold")
    box(0.4, 4.1, 3.05, 0.8, "#ffffff", AXIS, 1.0)
    text(1.93, 4.67, "RGB image → DeCAF6", 10.5)
    text(1.93, 4.32, r"visual feature $\mathbf{x}^t$", 10.5, INK2)
    box(0.15, 0.45, 3.55, 3.25, BOX_FILL, BOX_EDGE)
    text(1.93, 3.42, "Source domain (labeled RGB-D)", 11.5, weight="bold")
    box(0.4, 2.15, 3.05, 0.8, "#ffffff", AXIS, 1.0)
    text(1.93, 2.72, "RGB image → DeCAF6", 10.5)
    text(1.93, 2.37, r"visual feature $\mathbf{x}^s$", 10.5, INK2)
    box(0.4, 0.65, 3.05, 0.8, "#ffffff", AXIS, 1.0)
    text(1.93, 1.22, "Depth image → KDES", 10.5)
    text(1.93, 0.87, r"depth feature $\mathbf{z}$ (training only)", 10.5, INK2)

    # shared visual projection with its domain regulariser, and the depth view
    box(4.35, 2.05, 2.75, 3.4, "#ffffff", BLUE, 1.6)
    text(5.72, 4.75, r"$\mathbf{P}'\phi(\mathbf{x})$", 14)
    text(5.72, 4.35, "shared visual projection", 9.5, INK2)
    ax.plot([4.55, 6.9], [3.98, 3.98], color=GRID, lw=1)
    text(5.72, 3.62, r"$\mu\,\Omega(\mathbf{P})$: domain mismatch", 10, weight="bold")
    text(5.72, 3.17, r"$\Omega^{mmd}$: MMD of $\mathbf{P}'\phi(\cdot)$  (A, B)", 9.5)
    text(5.72, 2.72, r"$\Omega^{ksa}=\frac{1}{2}\Vert\mathbf{P}-\mathbf{T}\Vert^2$  (C)", 9.5)
    text(5.72, 2.32, r"$\mathbf{T}$: kernel subspace alignment", 9, INK2)
    box(4.35, 0.65, 2.75, 0.8, "#ffffff", AXIS, 1.2)
    text(5.72, 1.22, r"$\mathbf{Q}'\psi(\mathbf{z})$ (A)  /  $\psi(\mathbf{z})$ (B, C)", 10.5)
    text(5.72, 0.87, "depth view", 9.5, INK2)
    arrow((3.45, 4.5), (4.35, 4.5), color=ORANGE, lw=1.8)
    arrow((3.45, 2.55), (4.35, 2.55), color=BLUE, lw=1.8)
    arrow((3.45, 1.05), (4.35, 1.05))

    # common subspace
    box(7.7, 0.45, 2.55, 5.0, BOX_FILL, BOX_EDGE)
    text(8.97, 5.17, "Common subspace", 11.5, weight="bold")
    rng = np.random.default_rng(1)
    for k, (cx, cy) in enumerate([(8.35, 4.15), (9.6, 3.95), (8.95, 2.3)]):
        pts = rng.normal(size=(8, 2)) * 0.17 + (cx, cy)
        ax.scatter(pts[:5, 0], pts[:5, 1], s=30, color=BLUE, marker="os^"[k], edgecolor=SURF, lw=0.8, zorder=3)
        ax.scatter(pts[5:, 0], pts[5:, 1], s=30, color=ORANGE, marker="os^"[k], edgecolor=SURF, lw=0.8, zorder=3)
    ax.plot([8.0, 9.95], [3.2, 3.45], color=INK2, lw=1.2)
    ax.plot([8.9, 9.1], [3.3, 4.85], color=INK2, lw=1.2)
    ax.scatter([8.1], [1.35], s=30, color=BLUE, edgecolor=SURF)
    text(8.25, 1.35, "source", 9, INK2, ha="left")
    ax.scatter([8.1], [1.0], s=30, color=ORANGE, edgecolor=SURF)
    text(8.25, 1.0, "target", 9, INK2, ha="left")
    text(8.97, 0.66, "shapes = classes", 9, INK2)
    arrow((7.1, 3.75), (7.7, 3.75))
    arrow((7.1, 1.05), (7.7, 1.05))

    # classifiers and the depth regulariser
    box(10.85, 3.85, 2.6, 1.6, "#ffffff", BLUE, 1.6)
    text(12.15, 5.05, r"visual SVMs $f^v_k$", 11.5, weight="bold")
    text(12.15, 4.6, r"$\mathbf{w}_k'\mathbf{P}'\phi(\mathbf{x})+b_k$", 11)
    text(12.15, 4.15, "→ predicts target images", 9.5, INK2)
    box(10.85, 2.0, 2.6, 1.55, "#f7f7f4", AXIS, 1.0)
    text(12.15, 3.25, r"$\lambda\,\Delta(\cdot,\cdot)$: use depth", 10, weight="bold")
    text(12.15, 2.82, r"$\Delta^{mfc}$: correlation (A)", 9.5)
    text(12.15, 2.4, r"$\Delta^{ecc}$: consistency (B, C)", 9.5)
    box(10.85, 0.45, 2.6, 1.25, "#ffffff", AXIS, 1.2)
    text(12.15, 1.3, r"depth SVMs $f^d_k$", 11.5, weight="bold")
    text(12.15, 0.85, "privileged view", 9.5, INK2)
    arrow((10.25, 4.65), (10.85, 4.65))
    arrow((10.25, 1.05), (10.85, 1.05))
    arrow((12.15, 3.55), (12.15, 3.85), color=MUTED, style="<|-|>")
    arrow((12.15, 1.7), (12.15, 2.0), color=MUTED, style="<|-|>")
    text(6.8, 0.1, "Alternating optimisation: SVM / SVM-2K duals with P fixed  ↔  P by a generalised "
         "eigen-problem (A, B) or in closed form (C), with line search", 9.5, INK2)
    save(fig, out, "fig_dam2s_framework")


def sweep(res, name):
    """The lambda sweep of a DAM2S variant at its best (m, kappa)."""
    bp = res[name]["best"]["params"]
    rs = [r for r in res[name]["all"] if r["params"]["m"] == bp["m"] and r["params"]["kappa"] == bp["kappa"]]
    return sorted(rs, key=lambda r: r["params"]["lam"])


# ----------------------------------------------------------------------------- tables
def draw_table(header, rows, col_w, title, out, name, bold=None, row_h=0.42, first_w=1.9,
               fontsize=12, mono_header=False):
    """Booktabs-style table; bold[r][c] marks cells to embolden."""
    W = first_w + sum(col_w) + 0.2
    Ht = row_h * (len(rows) + 1) + (0.75 if title else 0.25)
    fig = plt.figure(figsize=(W, Ht))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, Ht)
    ax.axis("off")
    top = Ht - (0.65 if title else 0.15)
    if title:
        ax.text(W / 2, Ht - 0.3, title, ha="center", va="center", fontsize=fontsize + 1, color=INK)
    xs = [0.1 + first_w + sum(col_w[:i]) + col_w[i] / 2 for i in range(len(col_w))]
    ax.plot([0.1, W - 0.1], [top, top], color=INK, lw=1.2)
    y = top - row_h / 2
    for x, h in zip(xs, header):
        ax.text(x, y, h, ha="center", va="center", fontsize=fontsize, color=INK,
                family="monospace" if mono_header and h.startswith("DAM2S") else None)
    ax.plot([0.1, W - 0.1], [top - row_h, top - row_h], color=INK, lw=0.7)
    for r, row in enumerate(rows):
        y = top - row_h * (r + 1.5)
        ax.text(0.15, y, row[0], ha="left", va="center", fontsize=fontsize, color=INK)
        for c, (x, v) in enumerate(zip(xs, row[1:])):
            b = bold is not None and bold[r][c]
            ax.text(x, y, v, ha="center", va="center", fontsize=fontsize, color=INK,
                    fontweight="bold" if b else "normal")
    yb = top - row_h * (len(rows) + 1)
    ax.plot([0.1, W - 0.1], [yb, yb], color=INK, lw=1.2)
    save(fig, out, name)


def table1(out):
    fig = plt.figure(figsize=(9.2, 3.3))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 9.2)
    ax.set_ylim(0, 3.3)
    ax.axis("off")
    ax.text(4.6, 3.05, "Summary of the objective functions of the three DAM2S algorithms",
            ha="center", va="center", fontsize=13, color=INK)
    x0, xs = 0.15, [1.9, 4.05, 6.15, 8.25]
    ax.plot([0.1, 9.1], [2.72, 2.72], color=INK, lw=1.2)
    for x, h in zip(xs[:3], ["DAM2S_A", "DAM2S_B", "DAM2S_C"]):
        ax.text(x + 1.05, 2.5, h, ha="center", va="center", fontsize=12, family="monospace", color=INK)
    ax.text(x0, 2.5, "Method", ha="left", va="center", fontsize=12, color=INK)
    ax.plot([0.1, 9.1], [2.28, 2.28], color=INK, lw=0.7)
    cells = [
        (r"$r(f^v,f^d)$", [(r"$\frac{1}{2}\Sigma_{k=1}^{K}\,(\Vert\mathbf{w}_k\Vert^2+\Vert\tilde{\mathbf{w}}_k\Vert^2)$", 0, 3)]),
        (r"$\ell(f^v,f^d)$", [(r"$\Sigma_{k=1}^{K}\,\Sigma_{i=1}^{n_s}\,(\xi^v_{i,k}+\xi^d_{i,k})$", 0, 3)]),
        (r"$\Omega(\mathbf{P})$", [(r"$\Omega^{mmd}(\mathbf{P})$", 0, 2), (r"$\Omega^{ksa}(\mathbf{P})$", 2, 3)]),
        (r"$\Delta(\cdot,\cdot)$", [(r"$\Delta^{mfc}(\mathbf{P},\mathbf{Q})$", 0, 1), (r"$\Delta^{ecc}(f^v,f^d)$", 1, 3)]),
    ]
    for (lab, entries), y in zip(cells, [2.0, 1.47, 0.95, 0.45]):
        ax.text(x0, y, lab, ha="left", va="center", fontsize=13, color=INK)
        for txt, a, b in entries:
            ax.text((xs[a] + xs[b]) / 2 + 0.0, y, txt, ha="center", va="center", fontsize=13, color=INK)
    ax.plot([0.1, 9.1], [0.18, 0.18], color=INK, lw=1.2)
    save(fig, out, "table1_objectives_retypeset")


def table2(res, out):
    ours = [100 * res[m]["best"]["acc"] if m in res else np.nan for m in METHODS]
    head = [m.replace("DAM2S_", "DAM2S_") for m in METHODS]
    rows = [["Paper (OR)"] + [f"{v:.2f}" for v in PAPER_OR],
            ["Ours (OR)"] + ["–" if np.isnan(v) else f"{v:.2f}" for v in ours]]
    bold = [[v == max(PAPER_OR) for v in PAPER_OR],
            [(not np.isnan(v)) and v == np.nanmax(ours) for v in ours]]
    draw_table(head, rows, [0.98] * 11 + [1.22] * 3,
               "Accuracy (%) of object recognition: RGB-D Object (source) → Caltech-256 (target)",
               out, "table2_or_paper_vs_ours", bold=bold, first_w=1.45, fontsize=11.5, mono_header=True)


def table3(res, out, times=None):
    ours = times or [np.mean([r["time"] for r in sweep(res, f"DAM2S_{v}")]) for v in "ABC"]
    rows = [["Paper (CD-HAR task)"] + [f"{v:.1f}" for v in PAPER_TIME_CDHAR],
            ["Ours (OR task)"] + [f"{v:.1f}" for v in ours]]
    draw_table(["DAM2S_A", "DAM2S_B", "DAM2S_C"], rows, [1.6] * 3,
               "Training time (seconds) of the three DAM2S methods", out, "table3_time_paper_vs_ours",
               first_w=2.4, mono_header=True)


# ----------------------------------------------------------------------------- charts
def chart_accuracy(res, out):
    ours = np.array([100 * res[m]["best"]["acc"] if m in res else np.nan for m in METHODS])
    paper = np.array(PAPER_OR)
    x = np.arange(len(METHODS))
    fig, ax = plt.subplots(figsize=(15, 4.8))
    style_axes(ax)
    ax.axvspan(10.5, 13.6, color="#f3f2ee", zorder=0, lw=0)
    ax.text(12.05, 66.5, "Proposed (DAM2S)", ha="center", va="bottom", fontsize=10.5, color=INK2)
    w = 0.38
    ax.bar(x - w / 2 - 0.01, paper, w, color=BLUE, label="Paper (TPAMI'18, Table 2)", zorder=2)
    ax.bar(x + w / 2 + 0.01, ours, w, color=ORANGE, label="Ours (reproduction)", zorder=2)
    for i in range(11, 14):
        ax.text(x[i] - w / 2, paper[i] + 0.8, f"{paper[i]:.1f}", ha="center", va="bottom", fontsize=9, color=INK2)
        if not np.isnan(ours[i]):
            ax.text(x[i] + w / 2, ours[i] + 0.8, f"{ours[i]:.1f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x, METHODS, fontsize=10)
    for t in ax.get_xticklabels()[11:]:
        t.set_fontweight("bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 70)
    ax.set_xlim(-0.6, 13.6)
    ax.legend(loc="upper left", ncol=2, fontsize=10.5)
    ax.set_title("Object recognition: RGB-D Object → Caltech-256 (10 classes)", loc="left")
    save(fig, out, "chart_table2_or_accuracy")


def chart_gain(res, out):
    """Improvement over the naive SVM_A, paper vs reproduction (absolute levels differ)."""
    ours = np.array([100 * res[m]["best"]["acc"] if m in res else np.nan for m in METHODS])
    paper = np.array(PAPER_OR)
    gp, go = paper[1:] - paper[0], ours[1:] - ours[0]
    names = METHODS[1:]
    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    style_axes(ax, "x")
    h = 0.38
    ax.barh(y + h / 2 + 0.01, gp, h, color=BLUE, label="Paper", zorder=2)
    ax.barh(y - h / 2 - 0.01, go, h, color=ORANGE, label="Ours", zorder=2)
    ax.axvline(0, color=AXIS, lw=1)
    for yi, a, b in zip(y, gp, go):
        ax.text(max(a, 0) + 0.4, yi + h / 2, f"{a:+.1f}", va="center", fontsize=8.5, color=INK2)
        if not np.isnan(b):
            ax.text(max(b, 0) + 0.4, yi - h / 2, f"{b:+.1f}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, names)
    for t in ax.get_yticklabels():
        if t.get_text().startswith("DAM2S"):
            t.set_fontweight("bold")
    ax.set_xlabel("Accuracy gain over SVM_A (percentage points)")
    ax.set_xlim(min(0, np.nanmin(np.r_[gp, go])) - 1, np.nanmax(np.r_[gp, go]) + 4)
    ax.legend(loc="upper right", fontsize=10.5)
    ax.set_title(f"Gain over SVM_A  (SVM_A: paper {paper[0]:.1f}%, ours {ours[0]:.1f}%)", loc="left")
    save(fig, out, "chart_gain_over_svm_a")


CVPR14_OR = {"SVM_A": 18.19, "SVM+": 18.59, "KCCA": 18.23, "SVM2K": 20.83, "KMM": 18.10, "DAM": 18.19,
             "SGF": 19.25, "LMK": 19.45, "TCA": 25.07, "SA": 21.09, "DIP": 25.47, "DA-M2S": 30.06}


def depth_gains(res):
    acc = lambda k: 100 * res[k]["best"]["acc"] if k in res else np.nan
    return [("SVM+ vs SVM_A", acc("SVM+") - acc("SVM_A")),
            ("SVM2K vs SVM_A", acc("SVM2K") - acc("SVM_A")),
            ("KCCA vs SVM_A", acc("KCCA") - acc("SVM_A")),
            ("DAM2S_B vs λ=0", acc("DAM2S_B") - acc("DAM2S_B_wo_depth")),
            ("DAM2S_C vs λ=0", acc("DAM2S_C") - acc("DAM2S_C_wo_depth"))]


def chart_depth_gain(res, res_k, out):
    g1, g2 = depth_gains(res), depth_gains(res_k)
    names = [n for n, _ in g1]
    y = np.arange(len(names))[::-1]
    h = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    style_axes(ax, "x")
    a = np.array([v for _, v in g1])
    b = np.array([v for _, v in g2])
    ax.barh(y + h / 2 + 0.01, a, h, color=BLUE, label="Visual = DeCAF6 (TPAMI'18 setting)", zorder=2)
    ax.barh(y - h / 2 - 0.01, b, h, color=ORANGE, label="Visual = RGB KDES (CVPR'14 setting)", zorder=2)
    ax.axvline(0, color=INK2, lw=1)
    for yi, u, v in zip(y, a, b):
        ax.text(u + (0.25 if u >= 0 else -0.25), yi + h / 2, f"{u:+.1f}", va="center",
                ha="left" if u >= 0 else "right", fontsize=9, color=INK2)
        ax.text(v + (0.25 if v >= 0 else -0.25), yi - h / 2, f"{v:+.1f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9, color=INK2)
    ax.set_yticks(y, names)
    allg = np.concatenate([a, b])
    ax.set_xlim(min(0.0, np.nanmin(allg)) - 0.6, np.nanmax(allg) + 1.2)
    ax.set_xlabel("Accuracy change from using depth (percentage points)")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_title("What does the depth view add?", loc="left")
    save(fig, out, "chart_depth_gain_two_settings")


def table_kdes_setting(res_k, out):
    meths = ["SVM_A", "SVM+", "KCCA", "SVM2K", "KMM", "DAM", "SGF", "LMK", "TCA", "SA", "DIP"]
    ours = [100 * res_k[m]["best"]["acc"] if m in res_k else np.nan for m in meths]
    ours_d = [100 * res_k[f"DAM2S_{v}"]["best"]["acc"] for v in "ABC"]
    head = meths + ["DA-M2S", "DAM2S_A", "DAM2S_B", "DAM2S_C"]
    paper = [f"{CVPR14_OR[m]:.2f}" for m in meths] + [f"{CVPR14_OR['DA-M2S']:.2f}", "–", "–", "–"]
    mine = [f"{v:.2f}" for v in ours] + ["–"] + [f"{v:.2f}" for v in ours_d]
    allv = ours + ours_d
    bold = [[h == "DA-M2S" for h in head],
            [v == np.nanmax(allv) for v in ours] + [False] + [v == np.nanmax(allv) for v in ours_d]]
    draw_table(head, [["CVPR'14 (KDES)"] + paper, ["Ours (KDES)"] + mine], [0.98] * 11 + [1.1] + [1.22] * 3,
               "OR accuracy (%) with RGB KDES visual features (setting of the CVPR'14 version)",
               out, "table_or_kdes_setting", bold=bold, first_w=1.75, fontsize=11.5, mono_header=True)


def chart_time(res, out, times=None):
    ours = times or [np.mean([r["time"] for r in sweep(res, f"DAM2S_{v}")]) for v in "ABC"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, vals, title in ((axes[0], PAPER_TIME_CDHAR, "Paper (CD-HAR task)"),
                            (axes[1], ours, "Ours (OR task)")):
        style_axes(ax)
        ax.bar(range(3), vals, 0.5, color=BLUE, zorder=2)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.02, f"{v:.1f} s", ha="center", va="bottom", fontsize=10, color=INK)
        ax.set_xticks(range(3), ["DAM2S_A", "DAM2S_B", "DAM2S_C"])
        ax.set_ylim(0, max(vals) * 1.18)
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("Training time (s)")
    save(fig, out, "chart_table3_training_time")


def chart_lambda(res, out):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    style_axes(ax, "both")
    cols = {"A": BLUE, "B": ORANGE, "C": AQUA}
    for v in "ABC":
        rs = sweep(res, f"DAM2S_{v}")
        lam = [r["params"]["lam"] for r in rs]
        acc = [100 * r["acc"] for r in rs]
        ax.plot(lam, acc, color=cols[v], lw=2, marker="o", ms=7, mec=SURF, mew=2,
                label=f"DAM2S_{v}", zorder=3)
    base = max((m for m in METHODS[:11] if m in res), key=lambda m: res[m]["best"]["acc"])
    yb = 100 * res[base]["best"]["acc"]
    ax.axhline(yb, color=MUTED, lw=1)
    ax.text(1.0, yb - 0.35, f"best baseline ({base}) {yb:.1f}%", fontsize=9.5, color=INK2, va="top", ha="right")
    ya = 100 * res["SVM_A"]["best"]["acc"]
    ax.axhline(ya, color=MUTED, lw=1)
    ax.text(1.0, ya + 0.35, f"SVM_A {ya:.1f}%", fontsize=9.5, color=INK2, va="bottom", ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("λ")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.16), fontsize=10)
    ax.set_title("Sensitivity to the trade-off parameter λ", loc="left")
    save(fig, out, "chart_lambda_sensitivity")


def ablation_rows(res):
    """(label, no-depth, init, full) for each variant; no-depth = same model with lambda = 0."""
    rows = []
    for v, wo in (("A", "DAM2S_B_wo_depth"), ("B", "DAM2S_B_wo_depth"), ("C", "DAM2S_C_wo_depth")):
        b = res[f"DAM2S_{v}"]["best"]
        rows.append((f"DAM2S_{v}", 100 * res[wo]["best"]["acc"] if wo in res else np.nan,
                     100 * b["acc_init"], 100 * b["acc"]))
    return rows


def chart_ablation(res, out):
    rows = ablation_rows(res)
    x = np.arange(len(rows))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    style_axes(ax)
    series = [("Without depth (λ = 0)", 1, GRAY_BAR), ("Initialisation only", 2, BLUE),
              ("Full model (alternating opt.)", 3, ORANGE)]
    for j, (lab, col, c) in enumerate(series):
        vals = [r[col] for r in rows]
        ax.bar(x + (j - 1) * (w + 0.02), vals, w, color=c, label=lab, zorder=2)
        for xi, v in zip(x, vals):
            if not np.isnan(v):
                ax.text(xi + (j - 1) * (w + 0.02), v + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=9,
                        color=INK2)
    lo = np.nanmin([r[1:] for r in rows])
    ax.set_ylim(np.floor(lo / 5) * 5 - 5, np.nanmax([r[1:] for r in rows]) + 4)
    ax.set_xticks(x, [r[0] for r in rows])
    ax.set_ylabel("Accuracy (%)")
    ax.legend(loc="upper left", fontsize=10, ncol=1)
    ax.set_title("Ablation: contribution of depth and of the alternating optimisation", loc="left")
    save(fig, out, "chart_ablation_depth_init")


def chart_convergence(res, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    cols = {"A": BLUE, "B": ORANGE, "C": AQUA}
    for v in "ABC":
        r = res[f"DAM2S_{v}"]["best"]
        obj = np.array(r["obj"])
        axes[0].plot(range(len(obj)), obj / obj[0], color=cols[v], lw=2, marker="o", ms=6, mec=SURF,
                     mew=1.5, label=f"DAM2S_{v}")
        acc0 = r.get("acc_init")
        accs = ([acc0] if acc0 is not None else []) + list(r["acc_iter"])
        it0 = 0 if acc0 is not None else 1
        axes[1].plot(range(it0, it0 + len(accs)), 100 * np.array(accs), color=cols[v], lw=2, marker="o",
                     ms=6, mec=SURF, mew=1.5, label=f"DAM2S_{v}")
    for ax in axes:
        style_axes(ax, "both")
        ax.set_xlabel("Iteration")
        ax.legend(fontsize=10)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axes[0].set_ylabel("Objective / initial objective")
    axes[1].set_ylabel("Target accuracy (%)")
    axes[0].set_title("Objective of the alternating optimisation", loc="left")
    axes[1].set_title("Target accuracy along the iterations", loc="left")
    save(fig, out, "chart_convergence")


def chart_per_class(res, preds, yt, out):
    best_v = max("ABC", key=lambda v: res[f"DAM2S_{v}"]["best"]["acc"])
    a = [np.mean(preds["SVM_A"][yt == k] == k) * 100 for k in range(10)]
    b = [np.mean(preds[f"DAM2S_{best_v}"][yt == k] == k) * 100 for k in range(10)]
    x = np.arange(10)
    fig, ax = plt.subplots(figsize=(12, 4.2))
    style_axes(ax)
    w = 0.38
    ax.bar(x - w / 2 - 0.01, a, w, color=BLUE, label="SVM_A (no adaptation, no depth)", zorder=2)
    ax.bar(x + w / 2 + 0.01, b, w, color=ORANGE, label=f"DAM2S_{best_v}", zorder=2)
    ax.set_xticks(x, [c.replace("_", " ") for c in CATS])
    ax.set_ylabel("Per-class accuracy (%)")
    ax.set_ylim(0, 119)
    ax.set_yticks(range(0, 101, 20))
    ax.legend(loc="upper left", ncol=2, fontsize=10.5)
    ax.set_title("Per-class accuracy on Caltech-256 (target)", loc="left")
    save(fig, out, "chart_per_class_accuracy")


def chart_confusion(res, preds, yt, out):
    best_v = max("ABC", key=lambda v: res[f"DAM2S_{v}"]["best"]["acc"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    for ax, m in zip(axes, ["SVM_A", f"DAM2S_{best_v}"]):
        Cm = np.zeros((10, 10))
        for t, p in zip(yt, preds[m]):
            Cm[t, p] += 1
        Cm = Cm / Cm.sum(1, keepdims=True) * 100
        ax.imshow(Cm, cmap=cmap, vmin=0, vmax=100)
        for i in range(10):
            for j in range(10):
                if Cm[i, j] >= 10:
                    ax.text(j, i, f"{Cm[i, j]:.0f}", ha="center", va="center", fontsize=8.5,
                            color="#ffffff" if Cm[i, j] > 55 else INK)
        ax.set_xticks(range(10), [c.replace("_", " ") for c in CATS], rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(10), [c.replace("_", " ") for c in CATS], fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_title(f"{m}  (row-normalised %)", loc="left")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("True class")
    save(fig, out, "chart_confusion_matrices")


def chart_tsne(root, res, out):
    from sklearn.manifold import TSNE
    rd = root / "results"
    best_v = max("ABC", key=lambda v: res[f"DAM2S_{v}"]["best"]["acc"])
    proj = np.load(rd / f"or_proj_DAM2S_{best_v}.npz")
    feats = np.load(root / "features/decaf6.npz")
    l2 = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)
    panels = [("Original DeCAF6 features", l2(feats["Xs"]), l2(feats["Xt"])),
              (f"Learned common space (DAM2S_{best_v})", proj["Xs"], proj["Xt"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, (title, Xs, Xt) in zip(axes, panels):
        Y = TSNE(2, perplexity=40, init="pca", random_state=0).fit_transform(np.vstack([Xs, Xt]))
        ns = len(Xs)
        ax.scatter(Y[:ns, 0], Y[:ns, 1], s=7, color=BLUE, alpha=0.55, lw=0, label="Source: RGB-D Object (RGB view)")
        ax.scatter(Y[ns:, 0], Y[ns:, 1], s=9, color=ORANGE, alpha=0.75, lw=0, label="Target: Caltech-256")
        ax.set_title(title, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.02), ncol=2, fontsize=10.5, markerscale=2.5)
    save(fig, out, "chart_tsne_domains")


def main(root):
    root = Path(root)
    out = root / "figures"
    out.mkdir(exist_ok=True)
    split = json.load(open(root / "features/or_split.json"))
    fig1(root, out)
    fig_dataset(root, split, out)
    fig_framework(out)
    table1(out)
    rp = root / "results/or_results.json"
    if not rp.exists():
        return
    res = json.load(open(rp))
    table2(res, out)
    chart_accuracy(res, out)
    chart_gain(res, out)
    if all(f"DAM2S_{v}" in res for v in "ABC"):
        tp = root / "results/timing.json"
        times = [json.load(open(tp))[f"DAM2S_{v}"]["mean"] for v in "ABC"] if tp.exists() else None
        table3(res, out, times)
        chart_time(res, out, times)
        chart_lambda(res, out)
        chart_convergence(res, out)
        if any(k.endswith("_wo_depth") for k in res):
            chart_ablation(res, out)
        pp = root / "results/or_preds.npz"
        if pp.exists():
            preds = dict(np.load(pp))
            yt = np.array([s["label"] for s in split["target"]])
            chart_per_class(res, preds, yt, out)
            chart_confusion(res, preds, yt, out)
        chart_tsne(root, res, out)
    rk = root / "results_kdes/or_results.json"
    if rk.exists():
        res_k = json.load(open(rk))
        if all(k in res_k for k in ("DAM2S_A", "DAM2S_B", "DAM2S_C")):
            table_kdes_setting(res_k, out)
            chart_depth_gain(res, res_k, out)


if __name__ == "__main__":
    main(sys.argv[1])
