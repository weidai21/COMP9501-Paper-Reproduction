"""Training time of the three DAM2S methods (Table 3), measured on an otherwise idle machine:
each variant is trained `repeats` times with its best (m, lambda) from or_results.json.

Usage: python -I time_dam2s.py <decaf.npz> <kdes.npz> <results_dir> [repeats]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_or as R  # noqa: E402


def main(decaf, kdes, res_dir, repeats=3):
    res_dir = Path(res_dir)
    res = json.load(open(res_dir / "or_results.json"))
    D = R.load(decaf, kdes)
    out = {}
    for v in "ABC":
        p = res[f"DAM2S_{v}"]["best"]["params"]
        times, accs, iters = [], [], []
        for _ in range(repeats):
            t0 = time.time()  # includes the kernel-SA pre-computation of DAM2S_C
            model = R.run_dam2s(D, v, p["lam"], p["m"], p["mu"], p["kappa"], {})
            times.append(time.time() - t0)
            accs.append(R.evaluate(D, model.decision_target()))
            iters.append(len(model.history) - 1)
        out[f"DAM2S_{v}"] = dict(params=p, times=times, mean=float(np.mean(times)), acc=accs, iters=iters)
        print(f"DAM2S_{v}: {np.mean(times):.1f}s +- {np.std(times):.1f}  iters={iters}  acc={np.round(100 * np.array(accs), 2)}")
    json.dump(out, open(res_dir / "timing.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4], *(int(x) for x in sys.argv[4:5]))
