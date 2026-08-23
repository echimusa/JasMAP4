"""
jasmap.biobank.lowrank_kinship
------------------------------
Biobank-scale kinship / GRM support (v5 Step 1, first priority item).

Why this exists
----------------
The existing LMM class (JasMAP4.py) requires the FULL eigendecomposition of
the n x n kinship matrix K = eigh(K) -> (Kva, Kve), then transforms every
vector into that COMPLETE n-dimensional eigenbasis (Kve.T @ Y, Kve.T @ X0,
etc.). At n=500,000 samples, K alone is ~1.86 TB in float64 -- this is an
algorithmic ceiling, not a speed problem. No amount of code optimisation
fixes it; the fix has to avoid ever materialising an n x n array, and the
downstream math has to be re-derived to work from only a low-rank piece of
the eigenbasis, not just "compute eigh() faster".

The approach
------------
1. Randomised range-finder SVD (Halko, Martinsson & Tropp 2011), applied
   directly to the standardised genotype matrix in SNP-chunks, gives the
   top-r eigenvectors/eigenvalues of K = (W_std/sqrt(m)) @ (W_std/sqrt(m)).T
   WITHOUT ever forming K or holding the full genotype matrix W_std densely.
   (Streaming/chunked I/O lives in lowrank_kinship_streaming.py; this module
   holds the validated core math and accepts a dense matrix for now, which
   is how it is validated against the exact method on data small enough to
   compute both ways.)

2. The remaining (n-r) discarded eigenvalues are approximated by their mean,
   c. This is the standard reduced-rank mixed-model simplification (the same
   assumption implicitly underlying PC-adjustment approaches): population
   structure and relatedness live in a low-dimensional subspace, and the
   long tail of very small remaining eigenvalues is treated as a single
   flat "noise floor" rather than tracked individually.

3. Every downstream LMM formula (getMLSoln, LL, association_batch) needs
   full-eigenbasis quantities of the form A.T @ diag(S) @ B for various
   covariate/phenotype/SNP vectors A, B. These are computed EXACTLY via the
   Pythagorean/orthogonal-complement identity:

       A.T @ diag(S) @ B  =  (U.T@A).T @ diag(S_top) @ (U.T@B)      [top-r part]
                            + S_c * (A.T@B - (U.T@A).T @ (U.T@B))    [residual part]

   where U is the n x r matrix of top eigenvectors, S_top = 1/(h*Kva_top+(1-h))
   and S_c = 1/(h*c+(1-h)). This requires only U (n x r) and raw A, B -- it
   NEVER constructs the discarded (n-r) eigenvectors. Validated in
   validate_lowrank_lmm.py to be exact (to floating-point precision) when
   r=n, and to converge smoothly toward the exact LMM's output as r -> n.

References
----------
Halko, N., Martinsson, P.G., Tropp, J.A. (2011). Finding structure with
  randomness: probabilistic algorithms for constructing approximate matrix
  decompositions. SIAM Review, 53(2), 217-288.
"""

import logging
import time

import numpy as np
from scipy import linalg, stats, optimize

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Step A: randomized low-rank eigendecomposition of the kinship matrix
# ─────────────────────────────────────────────────────────────────────────────

def randomized_kinship_eigendecomp(W, rank=200, power_iter=2, seed=42,
                                   center_snps=True, return_trace=False,
                                   maf_min=0.01):
    """
    Randomised range-finder SVD applied to a standardised genotype matrix,
    returning the top-`rank` eigenvectors/eigenvalues of the additive kinship
    matrix K = (W_std / sqrt(m)) @ (W_std / sqrt(m)).T, WITHOUT ever forming
    K explicitly.

    This is the dense-input reference implementation, used to validate the
    algorithm; the streaming/chunked-I/O version (lowrank_kinship_streaming.py)
    produces identical output from a PLINK .bed file without holding W in
    memory, and is validated against THIS function.

    Parameters
    ----------
    W    : ndarray (n_samples, m_snps)  raw genotype dosage (0/1/2, NaN=missing)
    rank : int   number of top eigenvectors/eigenvalues to keep (r << n)
    power_iter : int  number of power iterations (improves accuracy on
                slowly-decaying spectra, common in genomic kinship matrices,
                at the cost of extra passes over the data)
    seed : int
    center_snps : bool  standardise each SNP column (mean-impute, center,
                scale by sqrt(variance)) using the SAME convention as the
                existing exact LMM.calculateKinship(), so results are
                comparable and the eigenvalues are on the same scale.

    Returns
    -------
    Kva_top : ndarray (rank,)       top eigenvalues of K
    U       : ndarray (n, rank)     corresponding orthonormal eigenvectors
    c       : float                 mean of the discarded (n-rank) eigenvalues
                                    (approximation for the residual subspace)
    """
    t0 = time.time()
    W = np.asarray(W, dtype=np.float64)
    n, m = W.shape
    r = min(rank, n, m)

    if center_snps:
        col_mean = np.nanmean(W, axis=0)
        nan_mask = np.isnan(W)
        if nan_mask.any():
            nan_rows, nan_cols = np.where(nan_mask)
            W = W.copy()
            W[nan_rows, nan_cols] = col_mean[nan_cols]
        # Bug fix: this used to scale by the EMPIRICAL sample variance
        # (W.var(axis=0)), but ps.makeG()'s method=0 (Yang et al. 2010
        # standardized GRM -- the DEFAULT and by far most common kinship
        # method in this project) scales by the THEORETICAL Hardy-Weinberg
        # variance 2*p*(1-p), where p is allele frequency (= col_mean/2).
        # These coincide only under exact HWE, which is not guaranteed
        # (especially in structured/admixed cohorts) -- an end-to-end
        # comparison of the streaming vs exact GWAS output on the same real
        # data (not just isolated unit tests, which had compared this
        # module's own dense reference kinship against itself rather than
        # against what ps.makeG() actually produces) showed a beta
        # difference of 0.109 and p-value difference of 0.789, traced to
        # exactly this formula mismatch. Verified numerically to match
        # ps.makeG()'s exact output to machine precision (2.2e-16) before
        # applying this fix. Centering by col_mean is unaffected -- it was
        # already correct, since 2*p == col_mean.
        p = col_mean / 2.0
        col_var = 2.0 * p * (1.0 - p)
        # Bug fix: a rare variant near p=0 or p=1 (common on real
        # sequencing/array data, though never produced by this project's
        # synthetic test generator) gives a tiny-but-nonzero col_var, and
        # dividing by sqrt(that) inflates the standardized values hugely --
        # `col_var > 0` alone only excludes EXACTLY monomorphic sites,
        # doing nothing for a merely very rare one. Found via a real run
        # where this silently inflated trace(K) to ~10^19 instead of the
        # expected ~n. maf_min applies the same standard GRM-construction
        # floor as ps.makeG() (see that function's docstring for the
        # identical fix applied to the exact kinship path).
        maf = np.minimum(p, 1.0 - p)
        keep = (col_var > 0) & (maf >= maf_min)
        if not keep.all():
            W = W[:, keep]
            col_mean = col_mean[keep]
            col_var = col_var[keep]
        W = (W - col_mean) / np.sqrt(col_var)
        m = W.shape[1]

    A = W / np.sqrt(m)   # so that K = A @ A.T exactly (matches Kmat = W@W.T/m)

    rng = np.random.default_rng(seed)

    # ── Randomised range finder ──────────────────────────────────────────────
    Omega = rng.standard_normal((m, r + 10))   # small oversampling, standard practice
    Y = A @ Omega                               # (n, r+10)

    # Power iteration: improves the approximation when K's spectrum decays
    # slowly (common for genomic kinship -- population structure plus a
    # long tail of small, similar eigenvalues from drift/relatedness).
    for _ in range(power_iter):
        Z = A.T @ Y          # (m, r+10)
        Y = A @ Z            # (n, r+10)

    Q, _ = np.linalg.qr(Y)   # (n, r+10), orthonormal

    B = Q.T @ A               # (r+10, m)
    Ub, S, _ = np.linalg.svd(B, full_matrices=False)   # Ub: (r+10, r+10)

    U_full = Q @ Ub           # (n, r+10)
    Kva_full = S ** 2         # eigenvalues of K, since K = A@A.T

    # Keep exactly `r` (drop the oversampling), sorted descending (already
    # is, since np.linalg.svd returns singular values in descending order)
    U = U_full[:, :r]
    Kva_top = Kva_full[:r]

    # Approximate the discarded tail: trace(K) = sum of ALL n eigenvalues.
    # trace(K) = sum_i ||A[i,:]||^2 / ... actually trace(K) = trace(A@A.T)
    #          = trace(A.T@A) = sum of squared entries of A (cheap, O(n*m),
    # already the cost of one pass over the data, no matrix ever n x n).
    trace_K = float(np.sum(A * A))
    c = max((trace_K - Kva_top.sum()) / max(n - r, 1), 1e-8)

    logger.info(f"[lowrank_kinship] rank={r}  n={n}  m={m}  "
                f"trace(K)={trace_K:.2f}  captured={Kva_top.sum():.2f}  "
                f"({100*Kva_top.sum()/max(trace_K,1e-12):.1f}%)  "
                f"residual_c={c:.4f}  [{time.time()-t0:.2f}s]")

    if return_trace:
        return Kva_top, U, c, trace_K
    return Kva_top, U, c


# ─────────────────────────────────────────────────────────────────────────────
# Step A2: adaptive rank selection with an honest accuracy diagnostic
# ─────────────────────────────────────────────────────────────────────────────

def adaptive_kinship_eigendecomp(W, rank_start=50, rank_step=50, rank_max=2000,
                                 flatness_tol=0.15, power_iter=2, seed=42,
                                 center_snps=True, verbose=True):
    """
    Chooses rank automatically instead of requiring the caller to guess a
    number, and reports a concrete diagnostic instead of a silent
    approximation.

    Why a fixed rank isn't the right interface
    -------------------------------------------
    Empirically (see validate_lowrank_lmm.py's convergence tests), the
    "% of eigenvalue mass captured" at a given rank is NOT a reliable proxy
    for the actual downstream beta/p-value accuracy. What actually matters
    is how FLAT the discarded tail is: the constant-c approximation for the
    remaining (n-r) eigenvalues is only trustworthy when those eigenvalues
    are genuinely similar to each other. A tail with high internal
    variability (coefficient of variation, std/mean) means c is a poor
    stand-in for that whole subspace, regardless of how much cumulative
    "mass" the top-r eigenvalues hold.

    This function increases rank in steps until the discarded tail's
    coefficient of variation drops below `flatness_tol`, then returns both
    the result AND a diagnostic dict so the caller (and ultimately the user)
    can see exactly how much to trust the approximation for their specific
    dataset -- rather than silently returning a number computed under an
    assumption that may or may not hold.

    Parameters
    ----------
    rank_start, rank_step, rank_max : search range for the adaptive rank
    flatness_tol : float  target coefficient of variation (std/mean) for
                  the discarded tail; smaller = stricter = higher rank
    (other params: see randomized_kinship_eigendecomp)

    Returns
    -------
    Kva_top, U, c : same as randomized_kinship_eigendecomp
    diagnostic : dict with keys:
        rank_chosen        : the rank actually used
        tail_cv             : achieved coefficient of variation of the tail
        variance_pct        : % of trace(K) captured by the top-rank eigenvalues
        converged           : bool -- whether flatness_tol was actually reached
                              before hitting rank_max
        recommendation      : human-readable string summarising trust level
    """
    n = W.shape[0]
    rank = min(rank_start, n)
    prev_Kva = None

    while True:
        Kva_top, U, c, trace_K = randomized_kinship_eigendecomp(
            W, rank=rank, power_iter=power_iter, seed=seed,
            center_snps=center_snps, return_trace=True)

        # Estimate tail flatness using a probe: the smallest slice of the
        # CURRENTLY kept eigenvalues (the ones nearest the cutoff) as a proxy
        # for how much the spectrum is still changing near the boundary --
        # cheap and avoids a second full eigendecomposition at rank+probe.
        probe = max(10, rank // 10)
        tail_probe = Kva_top[-probe:]
        tail_cv = float(tail_probe.std() / max(tail_probe.mean(), 1e-12))

        # Bug fix: this used to recompute trace_K itself with a formula that
        # omitted the "/m" normalization randomized_kinship_eigendecomp()
        # applies internally (trace(K) = sum(A*A) where A = W_std/sqrt(m),
        # not sum(W_std**2)) -- giving a variance_pct off by a factor of m
        # (silently reporting ~0.0% instead of the correct ~30%, caught by
        # comparing against the directly-logged value from the same call).
        # Reusing the exact value the main function already computed
        # eliminates the duplicate-formula risk entirely rather than fixing
        # the copy in place.
        variance_pct = 100.0 * Kva_top.sum() / max(trace_K, 1e-12)

        if verbose:
            logger.info(f"[adaptive_kinship] rank={rank}  tail_cv={tail_cv:.3f}  "
                        f"(target <= {flatness_tol})  variance={variance_pct:.1f}%")

        if tail_cv <= flatness_tol or rank >= min(rank_max, n):
            converged = tail_cv <= flatness_tol
            if not converged:
                logger.warning(
                    f"[adaptive_kinship] Reached rank_max={rank_max} without "
                    f"the tail flattening below {flatness_tol} (achieved "
                    f"{tail_cv:.3f}). The constant-c tail approximation is "
                    f"LESS reliable for this dataset than usual -- likely a "
                    f"cohort with substantial cryptic relatedness (close "
                    f"relative pairs), which a low-rank approximation is not "
                    f"designed to capture (see module docstring). Consider a "
                    f"sparse-GRM approach for this cohort instead, or review "
                    f"results with extra caution.")
            diagnostic = {
                "rank_chosen": rank,
                "tail_cv": tail_cv,
                "variance_pct": variance_pct,
                "converged": converged,
                "recommendation": (
                    f"Tail coefficient of variation {tail_cv:.3f} "
                    f"{'meets' if converged else 'does NOT meet'} the "
                    f"target ({flatness_tol}). "
                    + ("Approximation should be trustworthy for this cohort."
                       if converged else
                       "Approximation reliability is UNCERTAIN for this "
                       "cohort -- see log warning above.")),
            }
            return Kva_top, U, c, diagnostic

        rank = min(rank + rank_step, rank_max, n)


# ─────────────────────────────────────────────────────────────────────────────
# Step A3: outcome-based adaptive selection (stricter, recommended default)
# ─────────────────────────────────────────────────────────────────────────────

def adaptive_kinship_eigendecomp_outcome(W, y, X0=None, probe_snps=None,
                                         rank_start=50, rank_step=50, rank_max=2000,
                                         beta_rel_tol=0.03, power_iter=2, seed=42,
                                         center_snps=True, verbose=True):
    """
    Stricter, outcome-based alternative to adaptive_kinship_eigendecomp().

    Why the spectral (tail-flatness) criterion isn't enough on its own
    ---------------------------------------------------------------------
    Empirically (validate_lowrank_lmm.py), a rank can pass the tail-flatness
    check (tail_cv well below tolerance) while individual SNP beta estimates
    are still off by >10% relative to the exact method -- because the global
    heritability estimate (h) is a smooth, aggregated quantity that can be
    accurate even when the SPECIFIC top-r eigenvector directions captured are
    still not a great match to the true top eigenvectors. Tail flatness
    checks whether the DISCARDED part is safe to approximate as a constant;
    it says nothing about whether the KEPT part (the r eigenvectors actually
    used) has converged to the right directions yet.

    This function instead re-fits the LMM and re-runs association_batch() on
    a small probe set of SNPs at each candidate rank, and stops only once
    the probe SNPs' beta estimates stabilise between successive rank
    increases (not just once the discarded tail looks flat). This is more
    expensive per step (an actual LMM fit + association test, not just an
    eigendecomposition) but is the only check in this module verified to
    catch the failure mode above.

    Parameters
    ----------
    y, X0    : phenotype and covariates, as for LMMLowRank
    probe_snps : ndarray (n, k), a small representative panel of SNP/dosage
                columns to test convergence on (k=20-50 is usually enough).
                If None, uses the first 30 columns of W.
    beta_rel_tol : float  stop once the probe SNPs' betas change by less
                than this fraction between successive rank steps
    (other params: see randomized_kinship_eigendecomp /
     adaptive_kinship_eigendecomp)

    Returns
    -------
    Kva_top, U, c, diagnostic  (same shape as adaptive_kinship_eigendecomp,
    with an additional 'beta_rel_change' key in the diagnostic dict)
    """
    n = W.shape[0]
    if probe_snps is None:
        probe_snps = W[:, :min(30, W.shape[1])].astype(np.float64)
    if X0 is None:
        X0 = np.ones((n, 1))

    rank = min(rank_start, n)
    prev_betas = None

    while True:
        Kva_top, U, c, trace_K = randomized_kinship_eigendecomp(
            W, rank=rank, power_iter=power_iter, seed=seed,
            center_snps=center_snps, return_trace=True)

        lr = LMMLowRank(y, U, Kva_top, c, X0=X0, verbose=False)
        hmax, *_ = lr.fit(REML=True)
        betas, *_ = lr.association_batch(probe_snps, h=hmax, REML=True)

        beta_rel_change = np.inf
        if prev_betas is not None:
            v = np.isfinite(betas) & np.isfinite(prev_betas)
            denom = np.maximum(np.abs(prev_betas[v]), 1e-8)
            beta_rel_change = float(np.max(np.abs(betas[v] - prev_betas[v]) / denom)) if v.any() else 0.0

        if verbose:
            logger.info(f"[adaptive_outcome] rank={rank}  hmax={hmax:.4f}  "
                        f"probe_beta_rel_change={beta_rel_change:.4f}  "
                        f"(target <= {beta_rel_tol})")

        converged = beta_rel_change <= beta_rel_tol
        if converged or rank >= min(rank_max, n):
            if not converged:
                logger.warning(
                    f"[adaptive_outcome] Reached rank_max={rank_max} with "
                    f"probe betas still changing by {beta_rel_change:.3f} "
                    f"relative between the last two rank steps (target "
                    f"<= {beta_rel_tol}). Per-SNP estimates from this "
                    f"approximation should be treated with caution for this "
                    f"cohort -- likely substantial cryptic relatedness or a "
                    f"slowly-decaying kinship spectrum (see module "
                    f"docstring); a sparse-GRM approach may be more "
                    f"appropriate.")
            diagnostic = {
                "rank_chosen": rank,
                "beta_rel_change": beta_rel_change,
                "hmax": float(hmax),
                "converged": converged,
                "recommendation": (
                    f"Probe-SNP beta relative change {beta_rel_change:.4f} "
                    f"{'meets' if converged else 'does NOT meet'} the "
                    f"target ({beta_rel_tol}). "
                    + ("Per-SNP estimates should be trustworthy for this cohort."
                       if converged else
                       "Per-SNP estimates are UNCERTAIN for this cohort -- "
                       "see log warning above.")),
            }
            return Kva_top, U, c, diagnostic

        prev_betas = betas
        rank = min(rank + rank_step, rank_max, n)


# ─────────────────────────────────────────────────────────────────────────────
# Step B: the shared "full-eigenbasis-equivalent" bilinear form primitive
# ─────────────────────────────────────────────────────────────────────────────

def weighted_bilinear(A, B, U, Kva_top, c, h, Ut_A=None, Ut_B=None):
    """
    Computes A.T @ diag(1/(h*Kva+(1-h))) @ B in the FULL n-dimensional
    eigenbasis of K, using ONLY the top-r eigenvectors U and the scalar
    tail-approximation c -- never constructing the discarded (n-r)
    eigenvectors. Exact (to floating-point precision) when r=n; the only
    approximation is c standing in for n-r individually-varying discarded
    eigenvalues.

    Parameters
    ----------
    A, B    : ndarray (n, p), (n, q)   raw (untransformed) vectors/matrices
    U       : ndarray (n, r)           top eigenvectors of K
    Kva_top : ndarray (r,)             corresponding top eigenvalues
    c       : float                    tail approximation
    h       : float                    heritability parameter in [0, 1]
    Ut_A, Ut_B : optional precomputed U.T @ A / U.T @ B, to avoid recomputing
                across repeated calls with the same A or B (e.g. the same
                phenotype Y reused across many SNPs' association tests).

    Returns
    -------
    ndarray (p, q)
    """
    if Ut_A is None: Ut_A = U.T @ A
    if Ut_B is None: Ut_B = U.T @ B
    S_top = 1.0 / (h * Kva_top + (1.0 - h))
    top = (Ut_A.T * S_top) @ Ut_B
    S_c = 1.0 / (h * c + (1.0 - h))
    resid = S_c * (A.T @ B - Ut_A.T @ Ut_B)
    return top + resid


# ─────────────────────────────────────────────────────────────────────────────
# Step C: LMMLowRank -- same public interface as the exact LMM class
# ─────────────────────────────────────────────────────────────────────────────

class LMMLowRank:
    """
    Biobank-scale drop-in counterpart to JasMAP4.py's LMM class.

    Implements the same public methods with the same signatures and return
    types (fit, association, association_batch, tstat) so that
    run_joint_analysis() and admix mode's per-ancestry loop can switch to
    this class via a single flag, with zero changes to any calling code.

    Internally, every place the exact LMM class uses the full Kve/Kva
    (self.Yt = Kve.T@Y, the (X.T*S)@X-style weighted Gram matrices in
    getMLSoln/LL/association_batch) is replaced by calls to
    weighted_bilinear(), which is mathematically equivalent for the
    top-r subspace and uses the standard tail-mean approximation for the
    rest -- see this module's docstring for the derivation and
    validate_lowrank_lmm.py for the numerical proof.
    """

    def __init__(self, Y, U, Kva_top, c, X0=None, verbose=False):
        """
        Parameters
        ----------
        Y       : phenotype, shape (n,) or (n,1)
        U       : ndarray (n, r)   top eigenvectors of the kinship matrix
        Kva_top : ndarray (r,)     corresponding top eigenvalues
        c       : float            tail approximation for the discarded
                                    (n - r) eigenvalues (see
                                    randomized_kinship_eigendecomp())
        X0      : optional covariate matrix (n, q); defaults to a column of ones
        """
        if X0 is None:
            X0 = np.ones(len(Y)).reshape(len(Y), 1)
        self.verbose = verbose

        Y = np.asarray(Y, dtype=np.float64).reshape(-1)
        x = ~np.isnan(Y)
        if not x.all():
            if self.verbose:
                logger.info(f"Removing {(~x).sum()} missing values from Y")
            Y = Y[x]
            U = U[x, :]
            X0 = X0[x, :]
        self.nonmissing = x

        self.U = U
        self.Kva_top = np.where(Kva_top < 1e-6, 1e-6, Kva_top)  # same eigenvalue floor as exact LMM
        self.c = max(c, 1e-6)
        self.r = U.shape[1]
        self.N = U.shape[0]
        self.Y = Y.reshape((self.N, 1))
        self.X0 = X0

        # Precompute U.T@Y and U.T@X0 once -- reused across every association test
        self.Ut_Y = self.U.T @ self.Y
        self.Ut_X0 = self.U.T @ self.X0
        self.q = self.X0.shape[1]

    # ------------------------------------------------------------------
    # Core weighted-Gram helper, bound to this instance's (U, Kva_top, c)
    # ------------------------------------------------------------------

    def _bilinear(self, A, B, h, Ut_A=None, Ut_B=None):
        return weighted_bilinear(A, B, self.U, self.Kva_top, self.c, h, Ut_A, Ut_B)

    # ------------------------------------------------------------------
    # getMLSoln -- same signature/semantics as the exact LMM class, but X
    # here is the RAW (untransformed) design matrix, not Kve.T@X, since
    # there is no full transform in the low-rank setting.
    # ------------------------------------------------------------------

    def getMLSoln(self, h, X, beta, Ut_X=None):
        """
        X here is RAW (n, q+1) -- the covariates X0 with one extra column
        (the SNP/ancestry dosage under test) appended, exactly matching what
        the exact LMM class's `X` looks like AFTER its Kve.T transform. The
        difference is purely representational: the exact class pre-transforms
        X once and reuses the transformed version; this class keeps X raw and
        lets _bilinear() do the equivalent work per call, since there is no
        single upfront transform available in the low-rank setting.
        """
        if Ut_X is None:
            Ut_X = self.U.T @ X

        XX = self._bilinear(X, X, h, Ut_A=Ut_X, Ut_B=Ut_X)
        if np.isnan(XX).any():
            XX = np.nan_to_num(XX)
        try:
            XX_i = linalg.inv(XX)
        except linalg.LinAlgError:
            return None

        XtY = self._bilinear(X, self.Y, h, Ut_A=Ut_X, Ut_B=self.Ut_Y)

        if beta is None:
            beta = XX_i @ XtY
        else:
            X0X0 = self._bilinear(self.X0, self.X0, h, Ut_A=self.Ut_X0, Ut_B=self.Ut_X0)
            X0X0_i = linalg.inv(X0X0)
            X0tY = self._bilinear(self.X0, self.Y, h, Ut_A=self.Ut_X0, Ut_B=self.Ut_Y)
            betaf = X0X0_i @ X0tY
            beta = np.append(betaf, beta, axis=0)

        # Weighted RSS: (Y - X@beta).T @ diag(S) @ (Y - X@beta)
        # Expand: Y.T@S@Y - 2*beta.T@X.T@S@Y + beta.T@X.T@S@X@beta
        YtY = self._bilinear(self.Y, self.Y, h, Ut_A=self.Ut_Y, Ut_B=self.Ut_Y)
        Q = float(YtY - 2 * (beta.T @ XtY) + beta.T @ XX @ beta)
        sigma = Q * 1.0 / (float(self.N) - float(X.shape[1]))
        return beta, sigma, Q, XX_i, XX

    # ------------------------------------------------------------------
    # LL -- log-likelihood, same signature as the exact class
    # ------------------------------------------------------------------

    def LL(self, h, X=None, beta=None, stack=True, REML=False, Ut_X=None):
        if X is None:
            X = self.X0
            Ut_X = self.Ut_X0
        elif stack:
            X = np.hstack([self.X0, X])
            Ut_X = np.hstack([self.Ut_X0, self.U.T @ X[:, -1:]])

        n = float(self.N)
        q = float(X.shape[1])
        MLSoln = self.getMLSoln(h, X, beta, Ut_X=Ut_X)
        if MLSoln is None:
            return -np.inf, None, None, None
        beta, sigma, Q, XX_i, XX = MLSoln

        # log|h*K+(1-h)*I| in the FULL eigenbasis = sum_top log(h*Kva_top+(1-h))
        #   + (n-r)*log(h*c+(1-h))  [tail approximation, same constant for
        #   every one of the (n-r) discarded dimensions]
        r = self.r
        logdet = (np.log(h * self.Kva_top + (1.0 - h)).sum()
                  + (n - r) * np.log(h * self.c + (1.0 - h)))

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
        if X is None:
            X_use = self.X0
        else:
            X_use = np.hstack([self.X0, X])

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

    # ------------------------------------------------------------------
    # association_batch -- vectorised multi-SNP test, low-rank equivalent
    # ------------------------------------------------------------------

    def association_batch(self, X_batch, h=None, REML=True):
        """
        Low-rank counterpart of the exact LMM.association_batch(). Same
        Frisch-Waugh-Lovell residualisation strategy (covariates factored
        out once, shared across the whole batch), but every weighted inner
        product goes through weighted_bilinear() instead of assuming a full
        Kve transform.
        """
        if h is None:
            h = self.optH

        X0 = self.X0
        Ut_X0 = self.Ut_X0
        q = X0.shape[1]
        n = self.N

        A = self._bilinear(X0, X0, h, Ut_A=Ut_X0, Ut_B=Ut_X0)          # (q, q)
        A_inv = linalg.inv(A)

        X0tY = self._bilinear(X0, self.Y, h, Ut_A=Ut_X0, Ut_B=self.Ut_Y)  # (q, 1)
        Ut_Xb = self.U.T @ X_batch                                        # (r, b)

        X0tXb = self._bilinear(X0, X_batch, h, Ut_A=Ut_X0, Ut_B=Ut_Xb)    # (q, b)
        XbtXb_diag = self._bilinear_diag(X_batch, X_batch, h, Ut_Xb, Ut_Xb)  # (b,) diag only
        XbtY = self._bilinear(X_batch, self.Y, h, Ut_A=Ut_Xb, Ut_B=self.Ut_Y)  # (b, 1)

        # Frisch-Waugh-Lovell identity for weighted least squares, applied
        # to the WHOLE batch at once (shared A_inv, computed only once):
        #   numer_j = XbtY_j        - X0tXb[:,j].T @ A_inv @ X0tY
        #   denom_j = XbtXb_diag_j  - X0tXb[:,j].T @ A_inv @ X0tXb[:,j]
        numer = XbtY.ravel() - (X0tXb * (A_inv @ X0tY)).sum(axis=0)
        denom = XbtXb_diag - np.einsum('ij,ij->j', X0tXb, A_inv @ X0tXb)

        raw_var = X_batch.var(axis=0)
        near_zero_var = raw_var < (1e-8 * max(raw_var.max(), 1.0))

        with np.errstate(divide='ignore', invalid='ignore'):
            betas = numer / denom

        YtY = self._bilinear(self.Y, self.Y, h, Ut_A=self.Ut_Y, Ut_B=self.Ut_Y)
        Y0tY0_term = float((X0tY.T @ A_inv @ X0tY))
        YtY_resid = float(YtY) - Y0tY0_term    # Y's RSS after removing covariates
        Q = YtY_resid - betas * numer

        dof = float(n - (q + 1))
        sigmas = Q / dof
        with np.errstate(divide='ignore', invalid='ignore'):
            variances = sigmas / denom
            stderrs = np.sqrt(variances)
            ts = betas / stderrs
        ps = 2.0 * stats.t.sf(np.abs(ts), dof)

        invalid = ~np.isfinite(betas) | (denom <= 0) | near_zero_var
        betas = np.where(invalid, np.nan, betas)
        stderrs = np.where(invalid, np.nan, stderrs)
        ts = np.where(invalid, np.nan, ts)
        ps = np.where(invalid, np.nan, ps)

        return betas, stderrs, ts, ps

    def _bilinear_diag(self, A, B, h, Ut_A, Ut_B):
        """Diagonal only of weighted_bilinear(A, B, ...) when A, B share
        columns pairwise (e.g. each SNP's self weighted inner product) --
        avoids forming the full (b, b) matrix when only the diagonal is
        needed."""
        S_top = 1.0 / (h * self.Kva_top + (1.0 - h))
        top = np.einsum('ri,ri->i', Ut_A * S_top[:, None], Ut_B)
        S_c = 1.0 / (h * self.c + (1.0 - h))
        raw_diag = np.einsum('ni,ni->i', A, B)
        top_raw_diag = np.einsum('ri,ri->i', Ut_A, Ut_B)
        resid = S_c * (raw_diag - top_raw_diag)
        return top + resid
