"""Baselines of Table 2.  Each function returns target decision values (n_t x K).

All kernel methods use the Gaussian kernel whose parameter is 1 / mean squared distance between
samples (Sec. 4.2).  Classifiers are one-vs-all SVMs (C = 1) solved in the empirical kernel map.
"""
import numpy as np
import torch
from cvxopt import matrix, solvers as cvx
from sklearn.svm import SVR

from common import (DEV, bandwidth, eigh_desc, empirical_map, gfk_embed, grassmann_geodesic,
                    pca_basis, rbf, sqdist)
from solvers import add_bias, svm2k_dcd, svm_dcd, svmplus_dcd

cvx.options["show_progress"] = False


def ksvm(Fs, Ft, Y, C=1.0, gamma=None):
    """Gaussian-kernel one-vs-all SVM trained on Fs, evaluated on Ft."""
    g = bandwidth(Fs, Ft) if gamma is None else gamma
    Es, Et, _ = empirical_map(rbf(Fs, Fs, g), rbf(Ft, Fs, g))
    W, _ = svm_dcd(Es, Y, C)
    return add_bias(Et) @ W


# --------------------------------------------------------------------- naive / multi-view / LUPI
def svm_a(D, C=1.0):
    W, _ = svm_dcd(D["Es"], D["Y"], C)
    return add_bias(D["Et"]) @ W


def svm_plus(D, gamma, C=1.0):
    A = svmplus_dcd(D["Kv_ss"], D["Kd_ss"], D["Y"], C, gamma)
    return (D["Kv_ts"] + 1.0) @ A


def _inv_sqrt(M, ridge):
    lam, U = eigh_desc(M)
    lam = lam + ridge * lam[0]
    return (U / np.sqrt(lam)) @ U.T


def kcca(D, m, kappa):
    mv, md = D["Es"].mean(0), D["Ed"].mean(0)
    Xs, Xt, Zs = D["Es"] - mv, D["Et"] - mv, D["Ed"] - md
    Rv, Rd = _inv_sqrt(Xs.T @ Xs, kappa), _inv_sqrt(Zs.T @ Zs, kappa)
    U, _, Vt = np.linalg.svd(Rv @ (Xs.T @ Zs) @ Rd)
    Wv, Wd = Rv @ U[:, :m], Rd @ Vt[:m].T
    Xp, Tp, Zp = Xs @ Wv, Xt @ Wv, Zs @ Wd
    g = bandwidth(Xp, Zp, Tp)
    # two SVMs in the common space; their decisions on the projected target data are fused
    return 0.5 * (ksvm(Xp, Tp, D["Y"], gamma=g) + ksvm(Zp, Tp, D["Y"], gamma=g))


def svm2k(D, Dc, C=1.0, eps=1e-3):
    WA, _, _ = svm2k_dcd(D["Es"], D["Ed"], D["Y"], C, Dc, eps)
    return add_bias(D["Et"]) @ WA


# --------------------------------------------------------------------- domain adaptation
def kmm_weights(Kss, Kts, B):
    ns, nt = Kss.shape[0], Kts.shape[0]
    kappa = (ns / nt) * Kts.sum(0)
    eps = B / np.sqrt(ns)
    G = np.vstack([np.eye(ns), -np.eye(ns), np.ones((1, ns)), -np.ones((1, ns))])
    h = np.hstack([np.full(ns, B), np.zeros(ns), ns * (1 + eps), ns * (eps - 1)])
    sol = cvx.qp(matrix(Kss + 1e-8 * np.eye(ns)), matrix(-kappa), matrix(G), matrix(h))
    return np.clip(np.array(sol["x"]).ravel(), 0, B)


def kmm(D, B, C=1.0):
    beta = kmm_weights(D["Kv_ss"], D["Kv_ts"], B)
    W, _ = svm_dcd(D["Es"], D["Y"], C * beta)
    return add_bias(D["Et"]) @ W


def dam(D, lam_d, C=1.0, eps=0.1):
    """Single-source DAM: target SVR on the virtual labels given by the source classifier
    (domain-dependent regulariser -> kernel K + I/lam_d, Duan et al. TNNLS'12)."""
    fs = svm_a(D)
    Ktt = D["Kv_tt"]
    out = np.zeros_like(fs)
    for k in range(fs.shape[1]):
        svr = SVR(kernel="precomputed", C=C, epsilon=eps).fit(Ktt + np.eye(len(Ktt)) / lam_d, fs[:, k])
        out[:, k] = svr.predict(Ktt)
    return out


def sgf(D, d, N=10):
    Xs, Xt = D["Xs"], D["Xt"]
    mu = np.vstack([Xs, Xt]).mean(0)
    path = grassmann_geodesic(pca_basis(Xs, d), pca_basis(Xt, d))[0]
    Ps = [path(t) for t in np.linspace(0, 1, N)]
    Fs = np.hstack([(Xs - mu) @ P for P in Ps])
    Ft = np.hstack([(Xt - mu) @ P for P in Ps])
    return ksvm(Fs, Ft, D["Y"])


def tca_embedding(D, mu):
    K = D["Kv_all"]
    ns, n = D["ns"], len(D["Kv_all"])
    s = np.hstack([np.full(ns, 1.0 / ns), np.full(n - ns, -1.0 / (n - ns))])
    H = np.eye(n) - 1.0 / n
    A = K @ H @ K
    Bm = K @ np.outer(s, s) @ K + mu * np.eye(n)
    L = np.linalg.cholesky(Bm)
    Li = np.linalg.inv(L)
    lam, V = eigh_desc(Li @ A @ Li.T)
    return K @ (Li.T @ V)  # columns sorted by decreasing eigenvalue


def tca(D, m, mu, cache={}):
    key = (id(D), mu)
    if key not in cache:
        cache[key] = tca_embedding(D, mu)
    Z = cache[key][:, :m]
    return ksvm(Z[:D["ns"]], Z[D["ns"]:], D["Y"])


def _zscore(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-8)


def sa(D, d):
    Xs, Xt = _zscore(D["Xs"]), _zscore(D["Xt"])
    Ps, Pt = pca_basis(Xs, d), pca_basis(Xt, d)
    return ksvm(Xs @ Ps @ (Ps.T @ Pt), Xt @ Pt, D["Y"])


def dip(D, d, iters=150, seed=0):
    """Domain invariant projection: min_W MMD^2(W'x_s, W'x_t), W'W = I (Grassmann descent)."""
    torch.manual_seed(seed)
    Xs = torch.from_numpy(D["Xs"]).to(DEV, torch.float64)
    Xt = torch.from_numpy(D["Xt"]).to(DEV, torch.float64)
    mu = torch.cat([Xs, Xt]).mean(0)
    Xs, Xt = Xs - mu, Xt - mu
    W = torch.from_numpy(pca_basis(np.vstack([D["Xs"], D["Xt"]]), d)).to(DEV, torch.float64)
    X = torch.cat([Xs, Xt])
    s2 = torch.cdist(X @ W, X @ W).pow(2).mean().item()

    def mmd(W):
        a, b = Xs @ W, Xt @ W
        k = lambda p, q: torch.exp(-torch.cdist(p, q).pow(2) / s2)
        return k(a, a).mean() + k(b, b).mean() - 2 * k(a, b).mean()

    step = 1.0
    for _ in range(iters):
        W.requires_grad_(True)
        f = mmd(W)
        g, = torch.autograd.grad(f, W)
        W = W.detach()
        rg = g - W @ (W.T @ g)  # Riemannian gradient on the Grassmann manifold
        gn = (rg ** 2).sum()
        if gn < 1e-14:
            break
        while step > 1e-8:  # Armijo backtracking with QR retraction
            Wn, _ = torch.linalg.qr(W - step * rg)
            if mmd(Wn) <= f.item() - 1e-4 * step * gn:
                break
            step *= 0.5
        W, step = Wn, step * 2.0
    W = W.detach()
    return ksvm((Xs @ W).cpu().numpy(), (Xt @ W).cpu().numpy(), D["Y"])


def _simplex_proj(v, z):
    """Euclidean projection of v onto {x >= 0, sum x = z}."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - z
    rho = np.nonzero(u - css / np.arange(1, len(u) + 1) > 0)[0][-1]
    return np.maximum(v - css[rho] / (rho + 1.0), 0.0)


def _landmarks(Kss, Kst, Yb, iters=500):
    """LMK landmark selection QP (Gong et al. ICML'13), relaxed: beta >= 0, class-balanced
    (sum_{i in c} beta_i = prior_c), minimising the MMD between the weighted source and the
    target.  The feasible set is a product of scaled simplices -> accelerated projected gradient."""
    ns, nt = Kst.shape
    cls = Yb.argmax(1)
    prior = Yb.mean(0)
    q = -2.0 / nt * Kst.sum(1)
    L = 2.0 * np.linalg.eigvalsh(Kss)[-1]
    beta = np.zeros(ns)
    for c in range(Yb.shape[1]):
        beta[cls == c] = prior[c] / max((cls == c).sum(), 1)
    y, t = beta.copy(), 1.0
    for _ in range(iters):
        g = 2.0 * Kss @ y + q
        nb = y - g / L
        for c in range(Yb.shape[1]):
            nb[cls == c] = _simplex_proj(nb[cls == c], prior[c])
        tn = (1 + np.sqrt(1 + 4 * t * t)) / 2
        y = nb + (t - 1) / tn * (nb - beta)
        beta, t = nb, tn
    return beta > 1e-2 / ns


def lmk(D, d=20, qs=(-2, -1, 0, 1, 2), train_on="all"):
    Xs, Xt = D["Xs"], D["Xt"]
    mu = np.vstack([Xs, Xt]).mean(0)
    s0 = 1.0 / bandwidth(Xs, Xt)
    Yb = (D["Y"] > 0).astype(float)
    dss, dst = sqdist(Xs, Xs), sqdist(Xs, Xt)
    Ks_comb = Kt_comb = 0.0
    used = np.zeros(len(Xs), bool)
    n_used = 0
    for q in qs:
        s2 = s0 * 4.0 ** q
        lm = _landmarks(np.exp(-dss / s2), np.exp(-dst / s2), Yb)
        if lm.sum() < d or (~lm).sum() < 2 * d:
            continue  # degenerate auxiliary task (almost all / no source samples are landmarks)
        used |= lm
        L = gfk_embed(pca_basis(Xs[~lm], d), pca_basis(np.vstack([Xt, Xs[lm]]), d))
        Fs, Ft = (Xs - mu) @ L, (Xt - mu) @ L
        g = bandwidth(Fs, Ft)
        Ks_comb = Ks_comb + rbf(Fs, Fs, g)
        Kt_comb = Kt_comb + rbf(Ft, Fs, g)
        n_used += 1
    if n_used == 0:  # no valid auxiliary task: plain GFK between the two domains
        L = gfk_embed(pca_basis(Xs, d), pca_basis(Xt, d))
        Fs, Ft = (Xs - mu) @ L, (Xt - mu) @ L
        g = bandwidth(Fs, Ft)
        Ks_comb, Kt_comb, n_used, used = rbf(Fs, Fs, g), rbf(Ft, Fs, g), 1, np.ones(len(Xs), bool)
    Ks_comb, Kt_comb = Ks_comb / n_used, Kt_comb / n_used
    idx = np.where(used)[0] if train_on == "landmarks" else np.arange(len(Xs))
    Es, Et, _ = empirical_map(Ks_comb[np.ix_(idx, idx)], Kt_comb[:, idx])
    W, _ = svm_dcd(Es, D["Y"][idx], 1.0)
    return add_bias(Et) @ W
