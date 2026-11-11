"""Object recognition experiment (RGB-D Object -> Caltech-256), Table 2 row "OR" + Table 3.

Usage: python -I run_or.py <decaf.npz> <kdes.npz> <out_dir> [--methods ...] [--quick]
Follows Sec. 4.2: Gaussian kernels (parameter = 1/mean squared distance), C = 1, all target
samples used as unlabeled training data, accuracy = mean of per-class accuracies,
mu = 1e5 (DAM2S_A/B) / 1e4 (DAM2S_C), lambda tuned in {1e-4,...,1}; baseline parameters are
tuned on the target set as in the paper, and every setting is logged.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling modules (scripts run with -I)
import baselines as B  # noqa: E402
from common import bandwidth, class_avg_acc, empirical_map, eigh_desc, onehot_pm, rbf  # noqa: E402
from dam2s import DAM2S  # noqa: E402

LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]


def l2n(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def blockwise_l2n(X):
    """gradient KDES | LBP KDES halves normalised separately."""
    half = X.shape[1] // 2
    return np.hstack([l2n(X[:, :half]), l2n(X[:, half:])]) / np.sqrt(2)


def load(decaf, kdes, visual_kdes=False):
    a, b = np.load(decaf), np.load(kdes)
    norm = blockwise_l2n if visual_kdes else l2n
    Xs, Xt = norm(a["Xs"].astype(np.float64)), norm(a["Xt"].astype(np.float64))
    Zs = blockwise_l2n(b["Z"].astype(np.float64))
    ys, yt = a["ys"], a["yt"]
    assert np.all(ys == b["ys"])
    K = int(ys.max()) + 1
    D = dict(Xs=Xs, Xt=Xt, Zs=Zs, ys=ys, yt=yt, K=K, ns=len(Xs), Y=onehot_pm(ys, K))
    gv = bandwidth(Xs, Xt)
    X = np.vstack([Xs, Xt])
    D["Kv_all"] = rbf(X, X, gv)
    ns = len(Xs)
    D["Kv_ss"], D["Kv_ts"], D["Kv_tt"] = D["Kv_all"][:ns, :ns], D["Kv_all"][ns:, :ns], D["Kv_all"][ns:, ns:]
    D["Kd_ss"] = rbf(Zs, Zs, bandwidth(Zs))
    D["Es"], D["Et"], D["lam_v"] = empirical_map(D["Kv_ss"], D["Kv_ts"])
    D["Ed"], _, D["lam_d"] = empirical_map(D["Kd_ss"])
    return D


def kernel_sa(D, m):
    """Kernel subspace alignment (Sec. 3.2.2): centred KPCA of each domain, T = A_s' K_st A_t."""
    ns = D["ns"]
    Ks, Kt, Kst = D["Kv_ss"], D["Kv_tt"], D["Kv_ts"].T
    Hs, Ht = np.eye(ns) - 1.0 / ns, np.eye(len(Kt)) - 1.0 / len(Kt)
    ls, Us = eigh_desc(Hs @ Ks @ Hs)
    lt, Ut = eigh_desc(Ht @ Kt @ Ht)
    As, At = Us[:, :m] / np.sqrt(ls[:m]), Ut[:, :m] / np.sqrt(lt[:m])
    T = As.T @ (Hs @ Kst @ Ht) @ At
    return dict(Xs=Us[:, :m] * np.sqrt(ls[:m]), Xt=Ut[:, :m] * np.sqrt(lt[:m]), T=T)


def evaluate(D, dec):
    return class_avg_acc(D["yt"], dec.argmax(1), D["K"])


def grid_run(name, fn, grid, D, log, preds):
    res, pr = [], []
    for params in grid:
        t0 = time.time()
        dec = fn(D, **params)
        acc = evaluate(D, dec)
        res.append(dict(params=params, acc=acc, time=time.time() - t0))
        pr.append(dec.argmax(1))
        print(f"  {name:9s} {params} -> {100 * acc:.2f}%  ({time.time() - t0:.1f}s)", flush=True)
    ib = int(np.argmax([r["acc"] for r in res]))
    log[name] = dict(best=res[ib], all=res)
    preds[name] = pr[ib]
    print(f"==> {name}: {100 * res[ib]['acc']:.2f}%  best {res[ib]['params']}", flush=True)


def run_dam2s(D, variant, lam, m, mu, kappa, sub_cache, monitor=False):
    model = DAM2S(variant, m=m, mu=mu, lam=lam, kappa=kappa)
    if monitor:
        model._monitor = lambda dec: evaluate(D, dec)
    sub = None
    if variant == "C":
        if m not in sub_cache:
            sub_cache[m] = kernel_sa(D, m)
        sub = sub_cache[m]
    model.fit(D["Es"], D["Et"], D["Ed"], D["lam_v"], D["lam_d"], D["Y"], sub)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("decaf")
    ap.add_argument("kdes")
    ap.add_argument("out_dir")
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--m", type=int, nargs="*", default=[50])
    ap.add_argument("--kappa", type=float, default=1e-3)
    ap.add_argument("--lams", type=float, nargs="*", default=LAMBDAS)
    ap.add_argument("--tag", default="", help="suffix of the result key, e.g. _wo_depth")
    ap.add_argument("--visual-kdes", action="store_true", help="visual features are RGB KDES")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    D = load(a.decaf, a.kdes, a.visual_kdes)
    print(f"source {D['ns']}  target {len(D['Xt'])}  visual rank {D['Es'].shape[1]}  depth rank {D['Ed'].shape[1]}")
    log_path, pred_path = out / "or_results.json", out / "or_preds.npz"
    log = json.load(open(log_path)) if log_path.exists() else {}
    preds = dict(np.load(pred_path)) if pred_path.exists() else {}
    want = lambda n: a.methods is None or n in a.methods

    def dump():
        json.dump(log, open(log_path, "w"), indent=1)
        np.savez(pred_path, **preds)

    grids = {
        "SVM_A": (B.svm_a, [dict()]),
        "SVM+": (B.svm_plus, [dict(gamma=g) for g in [1e-2, 1e-1, 1, 10, 100]]),
        "KCCA": (B.kcca, [dict(m=m, kappa=k) for m in [10, 20, 50, 100] for k in [1e-3, 1e-2, 1e-1]]),
        "SVM2K": (B.svm2k, [dict(Dc=d) for d in [1e-3, 1e-2, 1e-1, 1, 10]]),
        "KMM": (B.kmm, [dict(B=b) for b in [2, 5, 10, 100, 1000]]),
        "DAM": (B.dam, [dict(lam_d=l, eps=e) for l in [0.1, 1, 10] for e in [0.01, 0.1, 0.3]]),
        "SGF": (B.sgf, [dict(d=d) for d in [10, 20, 50, 100]]),
        "LMK": (B.lmk, [dict(d=d, train_on=t) for d in [10, 20, 50] for t in ["all", "landmarks"]]),
        "TCA": (B.tca, [dict(m=m, mu=mu) for mu in [0.1, 1, 10] for m in [10, 20, 50, 100]]),
        "SA": (B.sa, [dict(d=d) for d in [10, 20, 50, 100, 200]]),
        "DIP": (B.dip, [dict(d=d) for d in [10, 20, 50, 100]]),
    }
    for name, (fn, grid) in grids.items():
        if want(name):
            grid_run(name, fn, grid, D, log, preds)
            dump()

    sub_cache = {}
    for variant, mu in (("A", 1e5), ("B", 1e5), ("C", 1e4)):
        name = f"DAM2S_{variant}"
        if not want(name):
            continue
        name += a.tag
        res, best_model = [], None
        for m, lam in itertools.product(a.m, a.lams):
            model = run_dam2s(D, variant, lam, m, mu, a.kappa, sub_cache, monitor=True)
            acc = evaluate(D, model.decision_target())
            res.append(dict(params=dict(lam=lam, m=m, mu=mu, kappa=a.kappa), acc=acc,
                            time=model.train_time, obj=model.history, acc_init=model.acc_init,
                            acc_iter=model.acc_history))
            print(f"  {name} m={m} lam={lam:g} -> {100 * acc:.2f}%  (init {100 * model.acc_init:.2f}%) "
                  f"iters={len(model.history) - 1} ({model.train_time:.1f}s)", flush=True)
            if best_model is None or acc > best_model[0]:
                best_model = (acc, model)
        best = max(res, key=lambda r: r["acc"])
        log[name] = dict(best=best, all=res)
        model = best_model[1]
        preds[name] = model.decision_target().argmax(1)
        Xs_p, Xt_p = model.projected()
        np.savez(out / f"or_proj_{name}.npz", Xs=Xs_p, Xt=Xt_p)
        print(f"==> {name}: {100 * best['acc']:.2f}%  best {best['params']}", flush=True)
        dump()


if __name__ == "__main__":
    main()
