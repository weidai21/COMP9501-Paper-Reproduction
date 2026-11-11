"""DAM2S: Domain Adaptation from Multi-view to Single-view (Li, Chen, Xu, Van Gool, TPAMI'18).

Notation (paper -> code).  phi / psi are the Gaussian-kernel feature maps of the visual / depth
features.  Following footnote 1 and the preliminary CVPR'14 work, the projection matrices live in
the span of the source samples, P = Phi_s A_v, Q = Psi A_d.  We use the empirical kernel map
E (K = E E', E'E = diag(eigenvalues)) so that P'phi(x) = W' e(x) with W = E'A, and every
constraint / regulariser becomes a finite-dimensional matrix expression:

  P'Phi_s Phi_s'P = I_m        <->  W' diag(lam_v) W = I_m          (+ kappa I, KCCA-style ridge)
  Omega^mmd(P)                 <->  mu/2 ||W'(mean e(x_s) - mean e(x_t))||^2
  Delta^mfc(P,Q)               <->  -lambda tr(Q'E_d'E_s W)
  Omega^ksa(P)                 <->  mu/2 ||P - T||_F^2,  T = kernel subspace alignment

Optimisation (Sec. 3.4 and Alg. 1 of CVPR'14): alternate between
  (i)  the SVM duals for a fixed projection (one-vs-all SVMs for DAM2S_A, SVM-2K for B and C),
  (ii) the projection for fixed duals -- a generalised eigen-problem for A/B (closed form for C),
       followed by a line search on the Grassmann geodesic (linear path for C), so that the
       true objective decreases.
"""
import time

import numpy as np

from common import grassmann_geodesic, top_eigvecs
from solvers import add_bias, svm2k_dcd, svm2k_primal, svm_dcd, svm_primal

TAUS = (0.2, 0.4, 0.6, 0.8, 1.0)


class DAM2S:
    """variant in {'A', 'B', 'C'}; fit() takes the empirical-kernel-map features."""

    def __init__(self, variant, m=50, mu=1e5, lam=1e-2, C=1.0, eps=1e-3, kappa=1e-3,
                 max_iter=10, tol=1e-4, rho=1e-3, verbose=False):
        self.variant, self.m, self.mu, self.lam, self.C = variant, m, mu, lam, C
        self.eps, self.kappa, self.max_iter, self.tol, self.rho = eps, kappa, max_iter, tol, rho
        self.verbose = verbose

    # ------------------------------------------------------------------ objectives
    def _solve_A(self, U, warm):
        """DAM2S_A: K one-vs-all SVMs on the projected visual and on the projected depth data."""
        rv = self.Esw.shape[1]
        Xp, Zp = self.Esw @ U[:rv], self.Edw @ U[rv:]
        Wv, Av = svm_dcd(Xp, self.Y, self.C, None if warm is None else warm[0], tol=1e-2)
        Wd, Ad = svm_dcd(Zp, self.Y, self.C, None if warm is None else warm[1], tol=1e-2)
        mmd = 0.5 * self.mu * ((U[:rv].T @ self.mew) ** 2).sum()
        corr = -self.lam * np.trace(Zp.T @ Xp)
        obj = mmd + corr + svm_primal(Xp, self.Y, Wv, self.C) + svm_primal(Zp, self.Y, Wd, self.C)
        return obj, dict(Wv=Wv, Wd=Wd, duals=(Av, Ad))

    def _solve_B(self, U, warm):
        """DAM2S_B: SVM-2K between the projected visual view and the depth view."""
        Xp = self.Esw @ U
        WA, WB, duals = svm2k_dcd(Xp, self.Ed, self.Y, self.C, self.lam, self.eps, init=warm)
        obj = (0.5 * self.mu * ((U.T @ self.mew) ** 2).sum()
               + svm2k_primal(Xp, self.Ed, self.Y, WA, WB, self.C, self.lam, self.eps))
        return obj, dict(WA=WA, WB=WB, duals=duals)

    def _solve_C(self, P, warm):
        """DAM2S_C: SVM-2K on the kernel-SA transformed visual view and the depth view."""
        Xp = self.Xs_sub @ P
        WA, WB, duals = svm2k_dcd(Xp, self.Ed, self.Y, self.C, self.lam, self.eps, init=warm)
        obj = (0.5 * self.mu * ((P - self.T) ** 2).sum()
               + svm2k_primal(Xp, self.Ed, self.Y, WA, WB, self.C, self.lam, self.eps))
        return obj, dict(WA=WA, WB=WB, duals=duals)

    # ------------------------------------------------------------------ projection updates
    def _eig_step(self, U, sol):
        """Fix the duals and maximise tr(U' B~ U) over orthonormal U (whitened coordinates)."""
        if self.variant == "A":
            Av, Ad = sol["duals"]
            Gv = self.Esw.T @ (Av * self.Y)
            Gd = self.Edw.T @ (Ad * self.Y)
            M = self.M0.copy()
            rv = Gv.shape[0]
            M[:rv, :rv] += Gv @ Gv.T
            M[rv:, rv:] += Gd @ Gd.T
        else:
            aA, _, b = sol["duals"]
            G = self.Esw.T @ (aA * self.Y - b)
            M = self.M0 + G @ G.T
        Ustar, lam = top_eigvecs(M, self.m)
        if self.rho > 0:
            # tie-break for (near) degenerate eigenvalues: prefer the current subspace
            Ustar, _ = top_eigvecs(M + self.rho * max(lam[0], 1e-12) * (U @ U.T), self.m)
        return Ustar

    def _closed_form_C(self, P, sol):
        aA, _, b = sol["duals"]
        G = self.Xs_sub.T @ (aA * self.Y - b)
        GG = G @ G.T
        top = np.linalg.eigvalsh(GG)[-1]
        if top < 0.9 * self.mu:  # argmin_P mu/2||P-T||^2 - 1/2 tr(P'GG'P)
            return self.mu * np.linalg.solve(self.mu * np.eye(len(GG)) - GG, self.T)
        grad = self.mu * (P - self.T) - GG @ P
        return P - grad / (self.mu + top)

    # ------------------------------------------------------------------ main loop
    def fit(self, Es, Et, Ed, lam_v, lam_d, Y, sub=None):
        """Es/Et: visual empirical maps (source/target), Ed: depth empirical map (source),
        lam_v/lam_d: their eigenvalues, Y: n_s x K labels in {+1,-1}.
        sub (DAM2S_C only): dict(Xs, Xt, T) kernel-PCA coordinates and the kernel-SA matrix."""
        t0 = time.time()
        self.Y = Y
        self.Ed = Ed
        dv = lam_v + self.kappa * lam_v[0]
        self.Esw, self.Etw = Es / np.sqrt(dv), Et / np.sqrt(dv)
        self.mew = self.Esw.mean(0) - self.Etw.mean(0)
        rv = Es.shape[1]
        if self.variant == "A":
            dd = lam_d + self.kappa * lam_d[0]
            self.Edw = Ed / np.sqrt(dd)
            self.M0 = np.zeros((rv + len(dd), rv + len(dd)))
            self.M0[:rv, :rv] = -self.mu * np.outer(self.mew, self.mew)
            cross = self.lam * self.Esw.T @ self.Edw
            self.M0[:rv, rv:], self.M0[rv:, :rv] = cross, cross.T
            # initialisation: solve with the regularisers only (KCCA + MMD)
            U, _ = top_eigvecs(self.M0, self.m)
            solve = self._solve_A
        elif self.variant == "B":
            self.M0 = -self.mu * np.outer(self.mew, self.mew)
            # initialisation: MMD-only problem is degenerate -> ties broken by the leading
            # kernel-PCA directions of the source data (first m axes of the empirical map)
            U0 = np.eye(rv)[:, :self.m]
            U, _ = top_eigvecs(self.M0 + 1e-3 * np.abs(self.M0).max() * (U0 @ U0.T), self.m)
            solve = self._solve_B
        else:
            self.Xs_sub, self.Xt_sub, self.T = sub["Xs"], sub["Xt"], sub["T"]
            U = self.T.copy()  # P_0 = T (kernel subspace alignment)
            solve = self._solve_C
        obj, sol = solve(U, None)
        self.history = [obj]
        self.acc_history = []
        self.acc_init = self._monitor(self._decision(U, sol)) if hasattr(self, "_monitor") else None
        for it in range(self.max_iter):
            if self.variant == "C":
                Pstar = self._closed_form_C(U, sol)
                cand = [(1 - t) * U + t * Pstar for t in TAUS]
            else:
                path = grassmann_geodesic(U, self._eig_step(U, sol))[0]
                cand = [path(t) for t in TAUS]
            best = (obj, U, sol)
            for Uc in cand:
                o, s = solve(Uc, sol["duals"])
                if o < best[0]:
                    best = (o, Uc, s)
            improved = obj - best[0]
            obj, U, sol = best
            self.history.append(obj)
            if self.verbose:
                print(f"    iter {it + 1}: obj={obj:.4f}")
            if hasattr(self, "_monitor"):
                self.acc_history.append(self._monitor(self._decision(U, sol)))
            if improved <= self.tol * abs(obj):
                break
        self.U, self.sol = U, sol
        self.train_time = time.time() - t0
        return self

    def _decision(self, U, sol):
        if self.variant == "A":
            rv = self.Esw.shape[1]
            Tp = add_bias(self.Etw @ U[:rv])
            return 0.5 * (Tp @ sol["Wv"] + Tp @ sol["Wd"])  # fuse visual and depth classifiers
        if self.variant == "B":
            return add_bias(self.Etw @ U) @ sol["WA"]
        return add_bias(self.Xt_sub) @ sol["WA"]

    def decision_target(self):
        return self._decision(self.U, self.sol)

    def projected(self):
        """Source / target visual features in the learned common space."""
        if self.variant == "A":
            rv = self.Esw.shape[1]
            return self.Esw @ self.U[:rv], self.Etw @ self.U[:rv]
        if self.variant == "B":
            return self.Esw @ self.U, self.Etw @ self.U
        return self.Xs_sub @ self.U, self.Xt_sub
