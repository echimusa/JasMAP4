"""
jasmap.biobank.auto_lmm
------------------------
Auto-selects between three Kinship/GRM strategies based on data size and,
where relevant, a cheap structure check -- so calling code (run_joint_analysis,
admix mode's per-ancestry loop) doesn't need to know which one is in play.

Selection logic
---------------
1. SMALL data (dense n x n kinship comfortably fits in memory, default
   threshold ~2 GB -> n <~ 16,000): use the exact LMM class, unchanged.
   No approximation, no reason to use one when the data doesn't demand it.

2. LARGER data: attempt LMMLowRank first, using the outcome-based adaptive
   rank selection (adaptive_kinship_eigendecomp_outcome), bounded by a rank
   budget proportional to how much time/memory is reasonable to spend
   probing. If it converges within that budget, the cohort's kinship
   spectrum is genuinely concentrated (low-rank is a good fit) -- use
   LMMLowRank, since it is faster once built.

3. If the low-rank probe does NOT converge within its rank budget (as
   found for cohorts without cleanly-concentrated structure --
   validate_lowrank_lmm.py's population-structure test case), fall back to
   CGLMM, which handles any spectrum shape correctly without rank
   truncation, at a higher but predictable computational cost.

This means the "concentrated structure" check IS the deciding factor
between tiers 2 and 3, exactly as intended -- it is just performed
automatically via the already-validated outcome-based diagnostic rather
than left for the user to judge by eye.
"""

import logging
import time

import numpy as np

from jasmap.biobank.lowrank_kinship import (
    adaptive_kinship_eigendecomp_outcome, LMMLowRank, randomized_kinship_eigendecomp)
from jasmap.biobank.cg_kinship import CGLMM, make_kinship_matvec

logger = logging.getLogger(__name__)

# Dense n x n float64 kinship matrix bytes-per-sample
_BYTES_PER_N2_ENTRY = 8

# Default memory ceiling for "exact" mode. 2 GB -> n <~ 16,000. Deliberately
# conservative: the exact LMM class ALSO needs O(n^2) scratch space during
# eigh() beyond just K itself, so staying well under available RAM matters.
DEFAULT_EXACT_MEMORY_LIMIT_BYTES = 2 * 1024**3


def recommend_lmm_strategy(n, memory_limit_bytes=None):
    """
    Pure sizing check -- no data needed, just n. Returns 'exact' or
    'approximate' (the latter meaning "try low-rank, fall back to CG").
    """
    memory_limit_bytes = memory_limit_bytes or DEFAULT_EXACT_MEMORY_LIMIT_BYTES
    dense_bytes = n * n * _BYTES_PER_N2_ENTRY
    return "exact" if dense_bytes <= memory_limit_bytes else "approximate"


def build_lmm(G, y, X0=None, memory_limit_bytes=None,
             lowrank_rank_start=50, lowrank_rank_step=50, lowrank_rank_max=500,
             lowrank_beta_rel_tol=0.03, lowrank_probe_snps=None,
             cg_n_probes=40, cg_n_lanczos=50,
             seed=42, verbose=True):
    """
    Size- and structure-aware LMM construction. Picks exact LMM, LMMLowRank,
    or CGLMM automatically and returns a ready-to-fit object with the same
    public interface across all three (fit, association, association_batch,
    tstat) -- calling code does not need to branch on which was chosen.

    Parameters
    ----------
    G   : ndarray (n_samples, n_snps)  raw genotype/ancestry dosage
    y   : ndarray (n_samples,)          phenotype
    X0  : optional covariate matrix (n_samples, q)
    memory_limit_bytes : override the default 2 GB "exact mode" ceiling
    lowrank_* : passed through to adaptive_kinship_eigendecomp_outcome
                when the approximate path is tried
    cg_n_probes, cg_n_lanczos : passed through to CGLMM when the CG
                fallback is used

    Returns
    -------
    model    : LMM, LMMLowRank, or CGLMM instance, already usable via
              .fit() / .association_batch() etc.
    strategy : dict describing what was chosen and why -- always returned
              so the caller (and ultimately the log / user) can see the
              decision, never a silent choice.
    """
    n = G.shape[0]
    t0 = time.time()
    size_recommendation = recommend_lmm_strategy(n, memory_limit_bytes)

    if size_recommendation == "exact":
        # Import here (not at module top) to avoid a circular import: this
        # module lives in jasmap.biobank, but the exact LMM class is
        # defined directly in JasMAP4.py (the main entry-point script, not
        # a package module) -- executed on demand via exec(), the same
        # pattern already used by the validate_*.py scripts in this
        # project to reuse the real class rather than reimplementing it.
        namespace = {}
        exec(open(_find_jasmap4_path()).read().split("if __name__")[0], namespace)
        LMM = namespace["LMM"]

        col_mean = G.mean(0); col_var = G.var(0)
        col_var = np.where(col_var > 0, col_var, 1.0)
        Gstd = (G - col_mean) / np.sqrt(col_var)
        K = (Gstd @ Gstd.T) / G.shape[1]

        model = LMM(y, K, X0=X0, verbose=False)
        strategy = {
            "chosen": "exact",
            "reason": f"n={n} -> dense kinship {n*n*_BYTES_PER_N2_ENTRY/1e9:.2f} GB "
                      f"fits within the {(memory_limit_bytes or DEFAULT_EXACT_MEMORY_LIMIT_BYTES)/1e9:.1f} GB limit",
            "n": n,
            "setup_time_s": time.time() - t0,
        }
        if verbose:
            logger.info(f"[auto_lmm] n={n}: using EXACT LMM ({strategy['reason']})")
        return model, strategy

    # ── "approximate" tier: try low-rank first, with the outcome-based check ──
    if verbose:
        logger.info(f"[auto_lmm] n={n}: dense kinship would need "
                    f"{n*n*_BYTES_PER_N2_ENTRY/1e9:.1f} GB -- attempting "
                    f"low-rank (concentrated-structure) path first")

    probe_snps = lowrank_probe_snps
    if probe_snps is None:
        probe_snps = G[:, :min(30, G.shape[1])].astype(np.float64)

    Kva, U, c, diag = adaptive_kinship_eigendecomp_outcome(
        G, y, X0=X0, probe_snps=probe_snps,
        rank_start=lowrank_rank_start, rank_step=lowrank_rank_step,
        rank_max=min(lowrank_rank_max, n), beta_rel_tol=lowrank_beta_rel_tol,
        seed=seed, verbose=verbose)

    if diag["converged"]:
        model = LMMLowRank(y, U, Kva, c, X0=X0, verbose=False)
        strategy = {
            "chosen": "lowrank",
            "reason": (f"kinship spectrum concentrated enough: probe-SNP beta "
                      f"stability {diag['beta_rel_change']:.4f} met the "
                      f"{lowrank_beta_rel_tol} target at rank={diag['rank_chosen']}"),
            "n": n,
            "rank_chosen": diag["rank_chosen"],
            "diagnostic": diag,
            "setup_time_s": time.time() - t0,
        }
        if verbose:
            logger.info(f"[auto_lmm] n={n}: using LOW-RANK LMM ({strategy['reason']})")
        return model, strategy

    # ── Fall back to CG: low-rank did not converge -> spectrum isn't ─────────
    # cleanly concentrated for this cohort (e.g. cryptic relatedness, or a
    # modest SNP:sample ratio) -- CG handles this correctly regardless.
    if verbose:
        logger.info(
            f"[auto_lmm] n={n}: low-rank did NOT converge within rank_max="
            f"{lowrank_rank_max} (probe-SNP beta still changing by "
            f"{diag['beta_rel_change']:.3f}, target {lowrank_beta_rel_tol}) "
            f"-- falling back to conjugate-gradient LMM (exact regardless "
            f"of spectrum shape).")

    matvec_K, _ = make_kinship_matvec(G)
    model = CGLMM(y, matvec_K, n, X0=X0, n_probes=cg_n_probes,
                  n_lanczos=cg_n_lanczos, seed=seed, verbose=verbose)
    strategy = {
        "chosen": "cg",
        "reason": (f"low-rank did not converge (probe-SNP beta change "
                  f"{diag['beta_rel_change']:.3f} > target "
                  f"{lowrank_beta_rel_tol} even at rank={diag['rank_chosen']}); "
                  f"using conjugate-gradient solving, which is correct "
                  f"regardless of the kinship spectrum's shape"),
        "n": n,
        "lowrank_diagnostic_that_triggered_fallback": diag,
        "setup_time_s": time.time() - t0,
    }
    if verbose:
        logger.info(f"[auto_lmm] n={n}: using CONJUGATE-GRADIENT LMM ({strategy['reason']})")
    return model, strategy


def _find_jasmap4_path():
    """Locate JasMAP4.py relative to the current working directory or the
    package installation, so build_lmm() works whether called from the
    project root or as an installed package."""
    import os
    candidates = [
        "JasMAP4.py",
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "JasMAP4.py"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        "Could not locate JasMAP4.py to import the exact LMM class. "
        "Run from the project root, or pass an explicit path.")


# ─────────────────────────────────────────────────────────────────────────────
# Streaming-aware entry point: build_lmm_from_bed()
# ─────────────────────────────────────────────────────────────────────────────

def _adaptive_outcome_streaming(source, y, X0, probe_snps,
                                rank_start, rank_step, rank_max,
                                beta_rel_tol, seed, verbose):
    """
    Streaming counterpart to
    lowrank_kinship.adaptive_kinship_eigendecomp_outcome(): same outcome-
    based stopping rule (re-fit LMMLowRank and check probe-SNP beta
    stability between successive rank increases), but the eigendecomposition
    itself comes from randomized_kinship_eigendecomp_streaming() instead of
    requiring a dense in-memory genotype array.
    """
    from jasmap.biobank.streaming_io import randomized_kinship_eigendecomp_streaming
    from jasmap.biobank.lowrank_kinship import LMMLowRank

    n = source.n
    rank = min(rank_start, n)
    prev_betas = None

    while True:
        Kva_top, U, c, trace_K = randomized_kinship_eigendecomp_streaming(
            source, rank=rank, power_iter=2, seed=seed, verbose=verbose)

        lr = LMMLowRank(y, U, Kva_top, c, X0=X0, verbose=False)
        hmax, *_ = lr.fit(REML=True)
        betas, *_ = lr.association_batch(probe_snps, h=hmax, REML=True)

        beta_rel_change = np.inf
        if prev_betas is not None:
            v = np.isfinite(betas) & np.isfinite(prev_betas)
            denom = np.maximum(np.abs(prev_betas[v]), 1e-8)
            beta_rel_change = (float(np.max(np.abs(betas[v] - prev_betas[v]) / denom))
                               if v.any() else 0.0)

        if verbose:
            logger.info(f"[auto_lmm/streaming] rank={rank}  hmax={hmax:.4f}  "
                        f"probe_beta_rel_change={beta_rel_change:.4f}  "
                        f"(target <= {beta_rel_tol})")

        converged = beta_rel_change <= beta_rel_tol
        if converged or rank >= min(rank_max, n):
            diagnostic = {
                "rank_chosen": rank,
                "beta_rel_change": beta_rel_change,
                "hmax": float(hmax),
                "converged": converged,
                "recommendation": (
                    f"Probe-SNP beta relative change {beta_rel_change:.4f} "
                    f"{'meets' if converged else 'does NOT meet'} the "
                    f"target ({beta_rel_tol})."),
            }
            return Kva_top, U, c, diagnostic

        prev_betas = betas
        rank = min(rank + rank_step, rank_max, n)


def build_lmm_from_bed(bed_path, y, X0=None, sample_idx=None,
                       memory_limit_bytes=None, cache_budget_gb=4.0,
                       lowrank_rank_start=50, lowrank_rank_step=50,
                       lowrank_rank_max=500, lowrank_beta_rel_tol=0.03,
                       n_probe_snps=30, cg_n_probes=40, cg_n_lanczos=50,
                       seed=42, verbose=True):
    """
    Streaming-aware counterpart to build_lmm(): reads directly from a PLINK
    .bed file rather than requiring a dense genotype array already in
    memory, so the "approximate" tier (low-rank / CG) genuinely never
    materialises the full genotype matrix -- the actual point of this
    module, completed by jasmap.biobank.streaming_io.

    Same three-tier logic as build_lmm():
      - exact:   reads the (necessarily small, by definition of this tier)
                genotype data into a dense array and uses the exact LMM
                class, same as before.
      - lowrank: uses randomized_kinship_eigendecomp_streaming() and the
                same outcome-based convergence check as build_lmm(),
                just fed by SNP chunks instead of a dense array.
      - cg:      uses the streaming matvec (make_streaming_kinship_matvec)
                instead of the dense one.

    Parameters
    ----------
    bed_path   : str  path to the .bed file (with or without extension)
    y, X0      : phenotype and covariates, as for build_lmm()
    sample_idx : optional sample index subset (e.g. after missing-phenotype
                filtering -- pass the SAME nonmissing mask used elsewhere)
    cache_budget_gb : passed to StreamingGenotypeSource -- data within this
                budget is cached in memory after one read (fast); larger
                data streams fresh from disk on every matvec call

    Returns
    -------
    model, strategy : same shape as build_lmm()
    """
    from jasmap.biobank.streaming_io import (
        StreamingGenotypeSource, make_streaming_kinship_matvec)

    t0 = time.time()
    from bed_reader import open_bed
    _bed_probe = open_bed(bed_path if bed_path.endswith(".bed") else bed_path + ".bed",
                          count_A1=False)
    n = _bed_probe.iid_count if sample_idx is None else len(sample_idx)

    size_recommendation = recommend_lmm_strategy(n, memory_limit_bytes)

    if size_recommendation == "exact":
        idx = (sample_idx, slice(None)) if sample_idx is not None else (None, slice(None))
        G = _bed_probe.read(index=idx).astype(np.float64)
        namespace = {}
        exec(open(_find_jasmap4_path()).read().split("if __name__")[0], namespace)
        LMM = namespace["LMM"]

        col_mean = G.mean(0); col_var = G.var(0)
        col_var = np.where(col_var > 0, col_var, 1.0)
        Gstd = (G - col_mean) / np.sqrt(col_var)
        K = (Gstd @ Gstd.T) / G.shape[1]

        model = LMM(y, K, X0=X0, verbose=False)
        strategy = {
            "chosen": "exact",
            "reason": f"n={n} -> dense kinship fits within the memory limit; "
                      f"read directly (no streaming needed at this size)",
            "n": n,
            "setup_time_s": time.time() - t0,
        }
        if verbose:
            logger.info(f"[auto_lmm] n={n}: using EXACT LMM ({strategy['reason']})")
        return model, strategy

    # ── "approximate" tier: streaming source, low-rank first, CG fallback ────
    source = StreamingGenotypeSource(bed_path, sample_idx=sample_idx,
                                     cache_budget_gb=cache_budget_gb,
                                     verbose=verbose)

    # Small probe SNP panel for the outcome-based diagnostic -- read
    # directly (cheap: n_probe_snps columns, not the whole file)
    probe_idx = (sample_idx, slice(0, min(n_probe_snps, source.m_total))) \
        if sample_idx is not None else (None, slice(0, min(n_probe_snps, source.m_total)))
    probe_snps = _bed_probe.read(index=probe_idx).astype(np.float64)

    if verbose:
        logger.info(f"[auto_lmm] n={n}: attempting streaming low-rank "
                    f"(concentrated-structure) path first")

    Kva, U, c, diag = _adaptive_outcome_streaming(
        source, y, X0, probe_snps,
        lowrank_rank_start, lowrank_rank_step, min(lowrank_rank_max, n),
        lowrank_beta_rel_tol, seed, verbose)

    if diag["converged"]:
        from jasmap.biobank.lowrank_kinship import LMMLowRank
        model = LMMLowRank(y, U, Kva, c, X0=X0, verbose=False)
        strategy = {
            "chosen": "lowrank",
            "reason": (f"kinship spectrum concentrated enough: probe-SNP beta "
                      f"stability {diag['beta_rel_change']:.4f} met the "
                      f"{lowrank_beta_rel_tol} target at rank={diag['rank_chosen']}"),
            "n": n,
            "rank_chosen": diag["rank_chosen"],
            "diagnostic": diag,
            "setup_time_s": time.time() - t0,
        }
        if verbose:
            logger.info(f"[auto_lmm] n={n}: using STREAMING LOW-RANK LMM ({strategy['reason']})")
        return model, strategy

    if verbose:
        logger.info(
            f"[auto_lmm] n={n}: streaming low-rank did NOT converge within "
            f"rank_max={lowrank_rank_max} (probe-SNP beta still changing by "
            f"{diag['beta_rel_change']:.3f}, target {lowrank_beta_rel_tol}) "
            f"-- falling back to streaming conjugate-gradient LMM.")

    matvec_K = make_streaming_kinship_matvec(source)
    model = CGLMM(y, matvec_K, n, X0=X0, n_probes=cg_n_probes,
                  n_lanczos=cg_n_lanczos, seed=seed, verbose=verbose)
    strategy = {
        "chosen": "cg",
        "reason": (f"low-rank did not converge (probe-SNP beta change "
                  f"{diag['beta_rel_change']:.3f} > target "
                  f"{lowrank_beta_rel_tol} even at rank={diag['rank_chosen']}); "
                  f"using streaming conjugate-gradient solving"),
        "n": n,
        "lowrank_diagnostic_that_triggered_fallback": diag,
        "setup_time_s": time.time() - t0,
    }
    if verbose:
        logger.info(f"[auto_lmm] n={n}: using STREAMING CONJUGATE-GRADIENT LMM ({strategy['reason']})")
    return model, strategy
