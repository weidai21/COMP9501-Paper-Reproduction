"""Shared helpers: kernels, empirical kernel maps, Grassmann geodesics, evaluation."""
import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def sqdist(A, B):
    d = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * A @ B.T
    return np.maximum(d, 0.0)


def mean_sqdist(X):
    """Mean squared distance over all pairs of rows of X."""
    return 2.0 * (X ** 2).sum(1).mean() - 2.0 * (X.mean(0) ** 2).sum()


def rbf(A, B, gamma):
    return np.exp(-gamma * sqdist(A, B))


def bandwidth(*Xs):
    """Gaussian kernel parameter: 1 / mean squared distance between any two samples."""
    return 1.0 / mean_sqdist(np.vstack(Xs))


def eigh_desc(M):
    """Eigen-decomposition with eigenvalues in descending order (GPU, float64)."""
    t = torch.from_numpy(np.ascontiguousarray(M)).to(DEV, torch.float64)
    lam, U = torch.linalg.eigh((t + t.T) / 2)
    return lam.flip(0).cpu().numpy(), U.flip(1).cpu().numpy()


def empirical_map(Kss, Kts=None, rtol=1e-10):
    """K_ss = E_s E_s' with E_s' E_s diagonal (the eigenvalues).  Returns (E_s, E_t, eigenvalues),
    where E_t holds the target samples projected onto span{phi(x_s)}; inner products with any
    function in that span are therefore exact."""
    lam, U = eigh_desc(Kss)
    keep = lam > rtol * lam[0]
    lam, U = lam[keep], U[:, keep]
    Es = U * np.sqrt(lam)
    Et = None if Kts is None else (Kts @ U) / np.sqrt(lam)
    return Es, Et, lam


def onehot_pm(y, K):
    Y = -np.ones((len(y), K))
    Y[np.arange(len(y)), y] = 1.0
    return Y


def class_avg_acc(y, pred, K):
    """Multiclass accuracy = average of the per-class accuracies (the paper's criterion)."""
    return float(np.mean([np.mean(pred[y == k] == k) for k in range(K) if np.any(y == k)]))


def top_eigvecs(M, m):
    lam, U = eigh_desc(M)
    return U[:, :m], lam


def grassmann_geodesic(Y0, Y1):
    """Geodesic between span(Y0) and span(Y1) (orthonormal bases), Edelman et al. / Gopalan et al.
    Returns a function tau -> orthonormal basis, with tau=0 -> span(Y0), tau=1 -> span(Y1)."""
    Q1, c, Q2t = np.linalg.svd(Y0.T @ Y1)
    c = np.clip(c, -1.0, 1.0)
    theta = np.arccos(c)
    A = Y0 @ Q1
    R = Y1 @ Q2t.T - A * c
    s = np.sin(theta)
    Z = np.where(s > 1e-10, R / np.where(s > 1e-10, s, 1.0), 0.0)

    def path(tau):
        return A * np.cos(theta * tau) + Z * np.sin(theta * tau)

    return path, A, Z, theta


def gfk_embed(Ps, Pt):
    """Geodesic flow kernel (Gong et al. CVPR'12) between PCA subspaces Ps, Pt (D x d).
    Returns a linear map L (D x 2d) such that x' G y = (L'x)'(L'y)."""
    _, A, Z, th = grassmann_geodesic(Ps, Pt)
    with np.errstate(divide="ignore", invalid="ignore"):
        cc = np.where(th > 1e-10, 0.5 + np.sin(2 * th) / (4 * th), 1.0)
        ss = np.where(th > 1e-10, 0.5 - np.sin(2 * th) / (4 * th), 0.0)
        cs = np.where(th > 1e-10, (1 - np.cos(2 * th)) / (4 * th), 0.0)
    M = np.block([[np.diag(cc), np.diag(cs)], [np.diag(cs), np.diag(ss)]])
    lam, V = np.linalg.eigh(M)
    Mh = (V * np.sqrt(np.maximum(lam, 0))) @ V.T
    return np.hstack([A, Z]) @ Mh


def pca_basis(X, d):
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:d].T
