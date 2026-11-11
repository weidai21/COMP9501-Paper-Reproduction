"""Build the source (RGB-D Object) / target (Caltech-256) file lists for the OR task.

Following the paper (and its CVPR'14 preliminary version):
  * 10 categories shared by RGB-D Object [Lai et al. ICRA'11] and Caltech-256,
  * RGB-D Object turntable videos are sub-sampled uniformly to 2,059 source RGB-D images.  The
    paper says "an interval of two seconds", but the ~210-frame videos of these 10 classes give
    only ~680 frames at 30 Hz, so by default every video is sampled uniformly with a budget
    proportional to its length so that exactly 2,059 images are kept (--interval keeps the
    fixed raw-frame interval variant),
  * every Caltech-256 image of the 10 categories is an (unlabeled) target sample.

Usage: python -I prepare_data.py <rgbd_root> <caltech256_root> <out_dir> [--total 2059 | --interval N]
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# RGB-D Object category -> Caltech-256 folder name (without the numeric prefix)
CATEGORIES = [
    ("ball", "soccer-ball"),
    ("calculator", "calculator"),
    ("cereal_box", "cereal-box"),
    ("coffee_mug", "coffee-mug"),
    ("flashlight", "flashlight"),
    ("keyboard", "computer-keyboard"),
    ("lightbulb", "lightbulb"),
    ("mushroom", "mushroom"),
    ("soda_can", "soda-can"),
    ("tomato", "tomato"),
]

FRAME_RE = re.compile(r"^(?P<cat>[a-z_]+)_(?P<inst>\d+)_(?P<video>\d+)_(?P<frame>\d+)_crop\.png$")


def find_dir(root, name):
    hits = [p for p in Path(root).rglob(name) if p.is_dir()]
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return sorted(hits, key=lambda p: len(p.parts))[0]


def source_list_total(rgbd_root, total):
    """Uniformly sample frames inside every turntable video so that exactly `total` RGB-D images
    are kept (the per-video budget is proportional to the video length, largest remainder)."""
    videos = {}
    for label, (cat, _) in enumerate(CATEGORIES):
        for f in find_dir(rgbd_root, cat).rglob("*_crop.png"):
            m = FRAME_RE.match(f.name)
            depth = f.with_name(f.name.replace("_crop.png", "_depthcrop.png"))
            if m is None or m["cat"] != cat or not depth.exists():
                continue
            key = (label, cat, int(m["inst"]), int(m["video"]))
            videos.setdefault(key, []).append((int(m["frame"]), f, depth))
    keys = sorted(videos)
    n_all = sum(len(videos[k]) for k in keys)
    quota = [total * len(videos[k]) / n_all for k in keys]
    budget = [int(q) for q in quota]
    for i in sorted(range(len(keys)), key=lambda i: quota[i] - budget[i], reverse=True)[:total - sum(budget)]:
        budget[i] += 1
    items = []
    for k, b in zip(keys, budget):
        frames = sorted(videos[k])
        pick = sorted(set(int(round(x)) for x in
                          [(j + 0.5) * len(frames) / b - 0.5 for j in range(b)]))
        for j in pick:
            frame, rgb, depth = frames[j]
            items.append(dict(rgb=str(rgb), depth=str(depth), label=k[0], category=k[1],
                              instance=k[2], video=k[3], frame=frame))
    return items


def source_list(rgbd_root, interval):
    items = []
    for label, (cat, _) in enumerate(CATEGORIES):
        cat_dir = find_dir(rgbd_root, cat)
        videos = defaultdict(list)
        for f in cat_dir.rglob("*_crop.png"):
            m = FRAME_RE.match(f.name)
            if m is None or m["cat"] != cat:
                continue
            depth = f.with_name(f.name.replace("_crop.png", "_depthcrop.png"))
            if not depth.exists():
                continue
            videos[(int(m["inst"]), int(m["video"]))].append((int(m["frame"]), f, depth))
        for (inst, video), frames in sorted(videos.items()):
            frames.sort()
            last = None
            for frame, rgb, depth in frames:
                # keep one frame every `interval` raw frames (2 s at 30 Hz)
                if last is None or frame - last >= interval:
                    items.append(dict(rgb=str(rgb), depth=str(depth), label=label, category=cat,
                                      instance=inst, video=video, frame=frame))
                    last = frame
    return items


def target_list(caltech_root):
    items = []
    folders = {p.name.split(".", 1)[1]: p for p in Path(caltech_root).rglob("*") if p.is_dir() and re.match(r"^\d{3}\.", p.name)}
    for label, (cat, c256) in enumerate(CATEGORIES):
        d = folders[c256]
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                items.append(dict(rgb=str(f), label=label, category=cat, caltech_folder=d.name))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rgbd_root")
    ap.add_argument("caltech_root")
    ap.add_argument("out_dir")
    ap.add_argument("--interval", type=int, default=0, help="raw-frame interval (0: use --total)")
    ap.add_argument("--total", type=int, default=2059, help="number of source RGB-D images")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    src = source_list(a.rgbd_root, a.interval) if a.interval > 0 else source_list_total(a.rgbd_root, a.total)
    tgt = target_list(a.caltech_root)
    json.dump(dict(categories=[c for c, _ in CATEGORIES], source=src, target=tgt),
              open(out / "or_split.json", "w"), indent=1)
    cs, ct = Counter(x["category"] for x in src), Counter(x["category"] for x in tgt)
    print(f"source (RGB-D Object): {len(src)} images, target (Caltech-256): {len(tgt)} images")
    for cat, _ in CATEGORIES:
        print(f"  {cat:12s} source {cs[cat]:4d}  target {ct[cat]:4d}")


if __name__ == "__main__":
    main()
