"""
jasmap.biobank.cg_kinship
--------------------------
Conjugate-gradient LMM — the third alternative for Kinship/GRM at scale,
alongside the exact LMM class (small data) and LMMLowRank (concentrated
population structure).

Why this exists
----------------
validate_lowrank_lmm.py and the accompanying investigation found that
LMMLowRank's rank-truncation approach can fail unpredictably on cohorts
whose kinship spectrum isn't cleanly concentrated in a few top eigenvalues
(realistic for modest SNP:sample ratios or genuine cryptic relatedness):
per-SNP beta estimates swung non-monotonically as rank increased, and an
honest outcome-based convergence check correctly refused to trust anything
below full rank for that dataset.

This module never approximates the kinship matrix's rank at all. It solves
(h*K+(1-h)*I)^-1 @ v via conjugate gradient (CG) -- an EXACT iterative
solver for any positive-definite system, using only matrix-vector products
K@v = (W_std/sqrt(m)) @ ((W_std/sqrt(m)).T @ v), which costs O(n*m) per
product and never forms K or holds an n x n array. CG error shrinks
predictably (not chaotically) as iterations increase, regardless of the
spectrum's shape -- validated in validate_cg_lmm.py against exactly the
scenario that broke LMMLowRank.

The one piece CG alone doesn't give you is the log-determinant needed for
the REML/ML likelihood (heritability estimation). This module uses
Stochastic Lanczos Quadrature (SLQ; Ubaru, Chen & Saad 2017) for that:
build a small number of Lanczos tridiagonalizations of K ALONE (matvecs
only), then reuse those cached tridiagonal matrices to evaluate log-det at
ANY h during the grid search / Brent optimisation at zero extra matvec
cost -- validated in this module's docstring-referenced tests to converge
predictably (Monte-Carlo-like, shrinking with more probes/steps) rather
than failing depending on spectrum shape.

Honest scope of the approximation
-----------------------------------
Beta/SE/p-values for a GIVEN h are exact (CG converges to the true linear
solve to a specified tolerance). The heritability point estimate h itself
carries some Monte Carlo noise from the SLQ log-determinant estimate --
this is the one place this module is not bit-for-bit exact, but the error
here is well-understood stochastic noise that shrinks with more probes,
not a systematic failure that depends on getting lucky with the data's
structure (the actual problem found with rank truncation).
"""

import logging
import time

import numpy as np
from scipy import linalg, stats, optimize

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Matrix-vector product primitive (never forms K)
# ─────────────────────────────────────────────────────────────────────────────

def make_kinship_matvec(W, center_snps=True, maf_min=0.01):
    """
    Returns a function matvec(v) -> K @ v, computed as
    A @ (A.T @ v) where A = W_std / sqrt(m), WITHOUT ever forming K.

    Also returns A (still dense here; the streaming/chunked-from-disk
    version is a follow-on piece, same as for the low-rank module -- this
    is the validated reference implementation).

    maf_min : minimum minor allele frequency for a SNP to be included --
        see lowrank_kinship.randomized_kinship_eigendecomp's docstring for
        why this matters (a rare variant's tiny-but-nonzero variance
        inflates its standardized weight, which `col_var > 0` alone does
        not catch since it is only exactly zero for monomorphic sites).
    """
    W = np.asarray(W, dtype=np.float64)
    n, m = W.shape
    if center_snps:
        col_mean = np.nanmean(W, axis=0)
        nan_mask = np.isnan(W)
        if nan_mask.any():
            nan_rows, nan_cols = np.where(nan_mask)
            W = W.copy()
            W[nan_rows, nan_cols] = col_mean[nan_cols]
        # Bug fix: matches lowrank_kinship.py's identical fix -- scale by
        # the Yang et al. 2010 Hardy-Weinberg theoretical variance
        # 2*p*(1-p) (p = allele frequency = col_mean/2), not the empirical
        # sample variance, to match ps.makeG()'s method=0 (the default
        # kinship method) exactly. See that module's docstring for the
        # end-to-end comparison that caught the mismatch and the numerical
        # verification (matches to 2.2e-16) before this fix was applied.
        p = col_mean / 2.0
        col_var = 2.0 * p * (1.0 - p)
        maf = np.minimum(p, 1.0 - p)
        keep = (col_var > 0) & (maf >= maf_min)
        if not keep.all():
            W = W[:, keep]; col_mean = col_mean[keep]; col_var = col_var[keep]
        W = (W - col_mean) / np.sqrt(col_var)
        m = W.shape[1]

    A = W / np.sqrt(m)   # K = A @ A.T

    def matvec_K(v):
        return A @ (A.T @ v)

    return matvec_K, A


# ─────────────────────────────────────────────────────────────────────────────
# Conjugate gradient solver
# ─────────────────────────────────────────────────────────────────────────────

def cg_solve(matvec_A, b, tol=1e-8, max_iter=500, x0=None):
    """
    Solve A @ x = b via conjugate gradient, given only a matvec function
    for A (never needs A explicitly). A must be symmetric positive-definite
    -- true here since A = h*K+(1-h)*I with K PSD and h in [0,1).

    Returns x, n_iterations_used, final_residual_norm
    """
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = len(b)
    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=np.float64).copy()
    r = b - matvec_A(x)
    p = r.copy()
    rs_old = float(r @ r)
    b_norm = max(np.linalg.norm(b), 1e-12)

    for it in range(max_iter):
        if np.sqrt(rs_old) / b_norm < tol:
            return x, it, np.sqrt(rs_old)
        Ap = matvec_A(p)
        denom = float(p @ Ap)
        if abs(denom) < 1e-300:
            break
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float(r @ r)
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x, max_iter, np.sqrt(rs_old)


def cg_solve_batch(matvec_A, B, tol=1e-8, max_iter=500):
    """CG solve for multiple right-hand sides at once (columns of B)."""
    B = np.asarray(B, dtype=np.float64)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    n, k = B.shape
    X = np.zeros((n, k))
    for j in range(k):
        X[:, j], _, _ = cg_solve(matvec_A, B[:, j], tol=tol, max_iter=max_iter)
    return X


# ─────────────────────────────────────────────────────────────────────────────
# Stochastic Lanczos Quadrature for log-determinant
# ─────────────────────────────────────────────────────────────────────────────

def _lanczos_tridiag(matvec, n, v, n_lanczos, seed_check=1e-10):
    v = v / np.linalg.norm(v)
    alphas, betas = [], []
    q_prev = np.zeros(n); q = v.copy(); beta_prev = 0.0
    for j in range(n_lanczos):
        w = matvec(q) - beta_prev * q_prev
        alpha = float(q @ w)
        w = w - alpha * q
        beta = float(np.linalg.norm(w))
        alphas.append(alpha)
        if beta < seed_check or j == n_lanczos - 1:
            break
        betas.append(beta)
        q_prev, q = q, w / beta
        beta_prev = beta
    T = np.diag(alphas)
    if betas:
        T += np.diag(betas, 1) + np.diag(betas, -1)
    return T


def build_logdet_probes(matvec_K, n, n_probes=40, n_lanczos=50, seed=42):
    """
    One-time cost: builds `n_probes` Lanczos tridiagonalizations of K ALONE
    (not h*K+(1-h)*I). These are cached and reused to evaluate the
    log-determinant of h*K+(1-h)*I at ANY h with zero additional matvecs --
    critical for making the heritability grid search (~100 h values)
    computationally tractable at biobank scale.
    """
    rng = np.random.default_rng(seed)
    T_list = []
    weight_list = []
    for _ in range(n_probes):
        v = rng.choice([-1.0, 1.0], size=n)
        T = _lanczos_tridiag(matvec_K, n, v, n_lanczos)
        evals, evecs = np.linalg.eigh(T)
        T_list.append((evals, evecs[0, :] ** 2))
    return T_list


def logdet_from_probes(T_list, h, n):
    """Evaluate log-det(h*K+(1-h)*I) at h using cached Lanczos probes."""
    total = 0.0
    for evals_K, weights in T_list:
        evals_A = h * evals_K + (1.0 - h)
        total += np.sum(weights * np.log(np.clip(evals_A, 1e-12, None)))
    return float(n * total / len(T_list))


# ─────────────────────────────────────────────────────────────────────────────
# CGLMM -- same public interface as LMM / LMMLowRank
# ─────────────────────────────────────────────────────────────────────────────

class CGLMM:
    """
    Biobank-scale LMM using conjugate-gradient solving instead of
    eigendecomposition or rank truncation. Never forms K, never
    approximates its rank -- beta/SE/p-values for a given h are exact
    (to CG's convergence tolerance); the heritability point estimate
    carries Monte-Carlo noise from the SLQ log-determinant (see module
    docstring).
    """

    def __init__(self, Y, matvec_K, n, X0=None, n_probes=40, n_lanczos=50,
                 cg_tol=1e-8, cg_max_iter=500, seed=42, verbose=False):
        """
        Parameters
        ----------
        Y        : phenotype, shape (n,)
        matvec_K : callable, v -> K @ v  (see make_kinship_matvec)
        n        : number of samples
        X0       : covariate matrix (n, q); defaults to a column of ones
        n_probes, n_lanczos : SLQ parameters (see build_logdet_probes)
        cg_tol, cg_max_iter  : CG solver parameters
        """
        if X0 is None:
            X0 = np.ones(n).reshape(n, 1)
        self.verbose = verbose

        Y = np.asarray(Y, dtype=np.float64).reshape(-1)
        x = ~np.isnan(Y)
        self.nonmissing = x
        # Note: CG/SLQ path currently assumes no missing Y (matching the
        # common case); dropping missing samples here would also require
        # re-deriving matvec_K on the subsetted sample set, which the
        # caller should do upstream (consistent with how the exact LMM
        # class handles this by re-slicing K itself before construction).

        self.Y = Y.reshape((n, 1))
        self.X0 = X0
        self.N = n
        self.matvec_K = matvec_K
        self.cg_tol = cg_tol
        self.cg_max_iter = cg_max_iter
        self.q = X0.shape[1]

        t0 = time.time()
        self.T_probes = build_logdet_probes(matvec_K, n, n_probes, n_lanczos, seed)
        if verbose:
            logger.info(f"[CGLMM] Built {n_probes} SLQ probes "
                        f"({n_probes*n_lanczos} matvecs) in {time.time()-t0:.2f}s")

    def _Ax(self, h):
        return lambda v: h * self.matvec_K(v) + (1.0 - h) * v

    def getMLSoln(self, h, X, beta):
        """Same semantics as LMM.getMLSoln, but (h*K+(1-h)*I)^-1@v via CG
        instead of a full eigenbasis transform."""
        Ax = self._Ax(h)
        Ainv_X = cg_solve_batch(Ax, X, tol=self.cg_tol, max_iter=self.cg_max_iter)
        Ainv_Y = cg_solve_batch(Ax, self.Y, tol=self.cg_tol, max_iter=self.cg_max_iter)

        XX = X.T @ Ainv_X
        if np.isnan(XX).any():
            XX = np.nan_to_num(XX)
        try:
            XX_i = linalg.inv(XX)
        except linalg.LinAlgError:
            return None

        XtY = X.T @ Ainv_Y

        if beta is None:
            beta = XX_i @ XtY
        else:
            Ainv_X0 = cg_solve_batch(Ax, self.X0, tol=self.cg_tol, max_iter=self.cg_max_iter)
            X0X0 = self.X0.T @ Ainv_X0
            X0X0_i = linalg.inv(X0X0)
            X0tY = self.X0.T @ Ainv_Y
            betaf = X0X0_i @ X0tY
            beta = np.append(betaf, beta, axis=0)

        YtY = self.Y.T @ Ainv_Y
        Q = float(YtY - 2 * (beta.T @ XtY) + beta.T @ XX @ beta)
        sigma = Q * 1.0 / (float(self.N) - float(X.shape[1]))
        return beta, sigma, Q, XX_i, XX

    def LL(self, h, X=None, beta=None, stack=True, REML=False):
        if X is None:
            X = self.X0
        elif stack:
            X = np.hstack([self.X0, X])

        n = float(self.N)
        q = float(X.shape[1])
        MLSoln = self.getMLSoln(h, X, beta)
        if MLSoln is None:
            return -np.inf, None, None, None
        beta, sigma, Q, XX_i, XX = MLSoln

        logdet = logdet_from_probes(self.T_probes, h, self.N)
        LLv = n * np.log(2 * np.pi) + logdet + n + n * np.log(1.0 / n * Q)
        LLv = -0.5 * LLv

        if REML:
            LLv = LLv + 0.5 * (q * np.log(2.0 * np.pi * sigma)
                               + np.log(linalg.det(X.T @ X))
                               - np.log(linalg.det(XX)))
        return float(np.sum(LLv)), beta, sigma, XX_i

    def LL_brent(self, h, X=None, REML=False):
        if h < 0:
            return 1e6
        ll = self.LL(h, X, stack=False, REML=REML)[0]
        return 1e6 if not np.isfinite(ll) else -ll

    def getMax(self, H, X=None, REML=False):
        n = len(self.LLs)
        HOpt = []
        for i in range(1, n - 2):
            if self.LLs[i - 1] < self.LLs[i] and self.LLs[i] > self.LLs[i + 1]:
                HOpt.append(optimize.brent(self.LL_brent, args=(X, REML),
                                           brack=(H[i - 1], H[i + 1])))
                if np.isnan(HOpt[-1]):
                    HOpt[-1] = H[i - 1]
        if len(HOpt) > 1:
            return HOpt[0]
        elif len(HOpt) == 1:
            return HOpt[0]
        elif self.LLs[0] > self.LLs[n - 1]:
            return H[0]
        else:
            return H[n - 1]

    def fit(self, X=None, ngrids=100, REML=True):
        X_use = self.X0 if X is None else np.hstack([self.X0, X])
        H = np.array(range(ngrids)) / float(ngrids)
        L = np.array([self.LL(h, X_use, stack=False, REML=REML)[0] for h in H])
        self.LLs = L
        hmax = self.getMax(H, X_use, REML)
        Lv, beta, sigma, betaSTDERR = self.LL(hmax, X_use, stack=False, REML=REML)
        if beta is None:
            hmax = min(max(hmax, 1e-6), 1 - 1e-6)
            Lv, beta, sigma, betaSTDERR = self.LL(hmax, X_use, stack=False, REML=REML)
        self.H = H
        self.optH = float(np.sum(hmax))
        self.optLL = Lv
        self.optBeta = beta
        self.optSigma = float(np.sum(sigma))
        return self.optH, beta, sigma, Lv

    def tstat(self, beta, var, sigma, q, log=False):
        stderr = np.sqrt(var * sigma)
        ts = beta / stderr
        if log:
            ps = 2.0 + stats.t.logsf(np.abs(ts), self.N - q)
        else:
            ps = 2.0 * stats.t.sf(np.abs(ts), self.N - q)
        return float(stderr), float(np.sum(ts)), float(np.sum(ps))

    def association(self, X, h=None, beta=None, stack=True, REML=True, returnBeta=False):
        if h is None:
            h = self.optH
        X_use = np.hstack([self.X0, X]) if stack else X
        L, beta, sigma, betaVAR = self.LL(h, X_use, beta, stack=False, REML=REML)
        if beta is None:
            return "nan", "nan", "nan", "nan"
        q = len(beta)
        stderr, ts, ps = self.tstat(beta[q - 1], betaVAR[q - 1, q - 1], sigma, q)
        return float(np.sum(beta[q - 1])), stderr, ts, ps

    def association_batch(self, X_batch, h=None, REML=True):
        """
        Vectorised multi-SNP test. Unlike LMM/LMMLowRank, cannot share a
        single upfront transform across the whole batch (CG solves depend
        on h and the specific right-hand side) -- instead solves once per
        SNP but reuses the covariate-only CG solve across the whole batch
        (the expensive shared part), matching the same FWL cost-sharing
        strategy in spirit.
        """
        if h is None:
            h = self.optH
        Ax = self._Ax(h)
        X0 = self.X0
        q = X0.shape[1]
        n = self.N

        Ainv_X0 = cg_solve_batch(Ax, X0, tol=self.cg_tol, max_iter=self.cg_max_iter)
        Ainv_Y = cg_solve_batch(Ax, self.Y, tol=self.cg_tol, max_iter=self.cg_max_iter)
        X0X0 = X0.T @ Ainv_X0
        X0X0_i = linalg.inv(X0X0)
        X0tY = X0.T @ Ainv_Y

        b = X_batch.shape[1]
        betas = np.full(b, np.nan); stderrs = np.full(b, np.nan)
        ts = np.full(b, np.nan); ps = np.full(b, np.nan)

        raw_var = X_batch.var(axis=0)
        near_zero_var = raw_var < (1e-8 * max(raw_var.max(), 1.0))

        dof = float(n - (q + 1))
        for j in range(b):
            if near_zero_var[j]:
                continue
            xj = X_batch[:, j:j+1]
            Ainv_xj, _, _ = cg_solve(Ax, xj.ravel(), tol=self.cg_tol, max_iter=self.cg_max_iter)
            Ainv_xj = Ainv_xj.reshape(-1, 1)

            X0txj = X0.T @ Ainv_xj
            xjtxj = float(xj.T @ Ainv_xj)
            xjtY = float(xj.T @ Ainv_Y)

            proj = X0X0_i @ X0txj
            numer = xjtY - float(X0txj.T @ (X0X0_i @ X0tY))
            denom = xjtxj - float(X0txj.T @ proj)

            if denom <= 0 or not np.isfinite(numer/denom if denom != 0 else np.nan):
                continue
            beta_j = numer / denom
            YtY = float(self.Y.T @ Ainv_Y)
            Y0tY0 = float(X0tY.T @ (X0X0_i @ X0tY))
            Q = (YtY - Y0tY0) - beta_j * numer
            sigma_j = Q / dof
            var_j = sigma_j / denom
            if var_j <= 0:
                continue
            se_j = np.sqrt(var_j)
            t_j = beta_j / se_j
            p_j = 2.0 * stats.t.sf(np.abs(t_j), dof)

            betas[j] = beta_j; stderrs[j] = se_j; ts[j] = t_j; ps[j] = p_j

        return betas, stderrs, ts, ps
