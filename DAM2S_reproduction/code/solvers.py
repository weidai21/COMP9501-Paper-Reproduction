"""Dual coordinate-descent solvers (numba) used by DAM2S and the baselines.

All solvers work on explicit (empirical-kernel-map) features, are one-vs-all over the K
columns of the label matrix Y in {+1,-1}^{n x K}, run the K problems in parallel, and treat
the bias as an extra constant feature (as LIBLINEAR does).

  svm_dcd      : L1-loss SVM            (Hsieh et al., ICML'08)
  svm2k_dcd    : SVM-2K two-view SVM    (Farquhar et al., NIPS'05) with eps-insensitive consistency
  svmplus_dcd  : SVM+ (LUPI)            (Vapnik & Vashist, NN'09), kernel form
"""
import numpy as np
from numba import njit, prange

BIAS = 1.0


def add_bias(X):
    return np.hstack([X, np.full((X.shape[0], 1), BIAS)])


@njit(cache=True)
def _svm_one(X, y, U, alpha, tol, max_epoch, seed):
    n, d = X.shape
    w = np.zeros(d)
    for i in range(n):
        if alpha[i] != 0.0:
            w += alpha[i] * y[i] * X[i]
    qd = np.empty(n)
    for i in range(n):
        qd[i] = X[i] @ X[i]
    np.random.seed(seed)
    idx = np.arange(n)
    for ep in range(max_epoch):
        np.random.shuffle(idx)
        pgmax, pgmin = -np.inf, np.inf
        for t in range(n):
            i = idx[t]
            G = y[i] * (w @ X[i]) - 1.0
            a = alpha[i]
            if a <= 0.0:
                pg = min(G, 0.0)
            elif a >= U[i]:
                pg = max(G, 0.0)
            else:
                pg = G
            pgmax = max(pgmax, pg)
            pgmin = min(pgmin, pg)
            if abs(pg) > 1e-12:
                an = min(max(a - G / qd[i], 0.0), U[i])
                w += (an - a) * y[i] * X[i]
                alpha[i] = an
        if pgmax - pgmin < tol:
            break
    return w


@njit(parallel=True, cache=True)
def _svm_multi(X, Y, U, A, tol, max_epoch):
    K = Y.shape[1]
    W = np.zeros((X.shape[1], K))
    for k in prange(K):
        a = A[:, k].copy()
        W[:, k] = _svm_one(X, Y[:, k].copy(), U, a, tol, max_epoch, 17 + k)
        A[:, k] = a
    return W


def svm_dcd(X, Y, C=1.0, A0=None, tol=1e-3, max_epoch=500):
    """Returns (W [(d+1) x K], alpha [n x K]); decision values = add_bias(X) @ W."""
    Xb = np.ascontiguousarray(add_bias(X), dtype=np.float64)
    n = Xb.shape[0]
    U = np.broadcast_to(np.asarray(C, dtype=np.float64), (n,)).copy()
    A = np.zeros(Y.shape) if A0 is None else np.minimum(A0.copy(), U[:, None])
    W = _svm_multi(Xb, np.ascontiguousarray(Y, dtype=np.float64), U, A, tol, max_epoch)
    return W, A


def svm_primal(X, Y, W, C=1.0):
    F = add_bias(X) @ W
    return 0.5 * (W ** 2).sum() + (C * np.maximum(0.0, 1.0 - Y * F)).sum()


@njit(cache=True)
def _svm2k_one(XA, XB, y, C, D, eps, aA, aB, b, tol, max_epoch, seed):
    n = XA.shape[0]
    wA = np.zeros(XA.shape[1])
    wB = np.zeros(XB.shape[1])
    for i in range(n):
        wA += (aA[i] * y[i] - b[i]) * XA[i]
        wB += (aB[i] * y[i] + b[i]) * XB[i]
    qA = np.empty(n)
    qB = np.empty(n)
    for i in range(n):
        qA[i] = XA[i] @ XA[i]
        qB[i] = XB[i] @ XB[i]
    np.random.seed(seed)
    idx = np.arange(n)
    for ep in range(max_epoch):
        np.random.shuffle(idx)
        viol = 0.0
        for t in range(n):
            i = idx[t]
            fA = wA @ XA[i]
            fB = wB @ XB[i]
            # view A margin variable
            G = y[i] * fA - 1.0
            a = aA[i]
            pg = min(G, 0.0) if a <= 0.0 else (max(G, 0.0) if a >= C else G)
            viol = max(viol, abs(pg))
            if abs(pg) > 1e-12:
                an = min(max(a - G / qA[i], 0.0), C)
                wA += (an - a) * y[i] * XA[i]
                fA += (an - a) * y[i] * qA[i]
                aA[i] = an
            # view B margin variable
            G = y[i] * fB - 1.0
            a = aB[i]
            pg = min(G, 0.0) if a <= 0.0 else (max(G, 0.0) if a >= C else G)
            viol = max(viol, abs(pg))
            if abs(pg) > 1e-12:
                an = min(max(a - G / qB[i], 0.0), C)
                wB += (an - a) * y[i] * XB[i]
                fB += (an - a) * y[i] * qB[i]
                aB[i] = an
            # consistency variable: min g*d + H/2 d^2 + eps*|b+d|, |b+d| <= D
            if D > 0.0:
                H = qA[i] + qB[i]
                g = fB - fA
                bu = b[i] - g / H
                bn = np.sign(bu) * max(abs(bu) - eps / H, 0.0)
                bn = min(max(bn, -D), D)
                dlt = bn - b[i]
                if dlt != 0.0:
                    viol = max(viol, abs(dlt) * H)
                    wA -= dlt * XA[i]
                    wB += dlt * XB[i]
                    b[i] = bn
        if viol < tol:
            break
    return wA, wB


@njit(parallel=True, cache=True)
def _svm2k_multi(XA, XB, Y, C, D, eps, AA, AB, Bt, tol, max_epoch):
    K = Y.shape[1]
    WA = np.zeros((XA.shape[1], K))
    WB = np.zeros((XB.shape[1], K))
    for k in prange(K):
        aA, aB, b = AA[:, k].copy(), AB[:, k].copy(), Bt[:, k].copy()
        wA, wB = _svm2k_one(XA, XB, Y[:, k].copy(), C, D, eps, aA, aB, b, tol, max_epoch, 31 + k)
        WA[:, k], WB[:, k] = wA, wB
        AA[:, k], AB[:, k], Bt[:, k] = aA, aB, b
    return WA, WB


def svm2k_dcd(XA, XB, Y, C=1.0, D=1.0, eps=1e-3, init=None, tol=1e-2, max_epoch=300):
    """SVM-2K.  Returns WA, WB and the duals (alphaA, alphaB, beta); the view-A weight vector
    is sum_i (alphaA_i y_i - beta_i) x_i."""
    XAb = np.ascontiguousarray(add_bias(XA), dtype=np.float64)
    XBb = np.ascontiguousarray(add_bias(XB), dtype=np.float64)
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    if init is None:
        AA, AB, Bt = np.zeros(Y.shape), np.zeros(Y.shape), np.zeros(Y.shape)
    else:
        AA, AB, Bt = (np.array(v, dtype=np.float64) for v in init)
        Bt = np.clip(Bt, -D, D)
    WA, WB = _svm2k_multi(XAb, XBb, Y, float(C), float(D), float(eps), AA, AB, Bt, tol, max_epoch)
    return WA, WB, (AA, AB, Bt)


def svm2k_primal(XA, XB, Y, WA, WB, C=1.0, D=1.0, eps=1e-3):
    FA, FB = add_bias(XA) @ WA, add_bias(XB) @ WB
    return (0.5 * (WA ** 2).sum() + 0.5 * (WB ** 2).sum()
            + C * np.maximum(0, 1 - Y * FA).sum() + C * np.maximum(0, 1 - Y * FB).sum()
            + D * np.maximum(0, np.abs(FA - FB) - eps).sum())


@njit(cache=True)
def _svmplus_one(K, Ks, y, C, gam, a, b, tol, max_epoch, seed):
    n = K.shape[0]
    u = np.zeros(n)  # K (a*y)
    v = np.zeros(n)  # Ks (a + b - C)
    for j in range(n):
        if a[j] != 0.0:
            u += a[j] * y[j] * K[:, j]
        c = a[j] + b[j] - C
        if c != 0.0:
            v += c * Ks[:, j]
    np.random.seed(seed)
    idx = np.arange(n)
    for ep in range(max_epoch):
        np.random.shuffle(idx)
        viol = 0.0
        for t in range(n):
            i = idx[t]
            # alpha_i
            G = -1.0 + y[i] * u[i] + v[i] / gam
            H = K[i, i] + Ks[i, i] / gam
            an = max(a[i] - G / H, 0.0)
            d = an - a[i]
            if d != 0.0:
                viol = max(viol, abs(d) * H)
                u += d * y[i] * K[:, i]
                v += d * Ks[:, i]
                a[i] = an
            # beta_i
            G = v[i] / gam
            H = Ks[i, i] / gam
            bn = max(b[i] - G / H, 0.0)
            d = bn - b[i]
            if d != 0.0:
                viol = max(viol, abs(d) * H)
                v += d * Ks[:, i]
                b[i] = bn
        if viol < tol:
            break


@njit(parallel=True, cache=True)
def _svmplus_multi(K, Ks, Y, C, gam, A, B, tol, max_epoch):
    for k in prange(Y.shape[1]):
        a, b = A[:, k].copy(), B[:, k].copy()
        _svmplus_one(K, Ks, Y[:, k].copy(), C, gam, a, b, tol, max_epoch, 7 + k)
        A[:, k], B[:, k] = a, b


def svmplus_dcd(K, Ks, Y, C=1.0, gamma=1.0, tol=1e-2, max_epoch=300):
    """SVM+ with decision kernel K and correcting kernel Ks (biases as constant features).
    Returns the coefficient matrix A*Y so that decision values on test = (K_test + 1) @ (A*Y)."""
    Kb = np.ascontiguousarray(K + BIAS ** 2)
    Ksb = np.ascontiguousarray(Ks + BIAS ** 2)
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    A = np.zeros(Y.shape)
    B = np.full(Y.shape, float(C))
    _svmplus_multi(Kb, Ksb, Y, float(C), float(gamma), A, B, tol, max_epoch)
    return A * Y
