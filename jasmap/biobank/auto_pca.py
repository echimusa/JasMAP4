"""
jasmap.biobank.auto_pca
-------------------------
Automatically computes per-sample PCA covariates for population
stratification correction when the user does not provide --covfile.

Why this exists
----------------
Per-individual PCA eigenvectors are the standard mechanism for correcting
population stratification in association testing -- and JasMAP4's entire
purpose is analysing ADMIXED populations, where stratification is the
dominant confounder. Before this module, --covfile omitted meant
X0 = np.ones((n,1)) (intercept-only): zero stratification correction,
silently. That is exactly the same risk category as the positional-argument
bug found and fixed earlier in JasMAP4.py's LMM(...) call sites (see that
fix's comments) -- just for the "no covariates given at all" case rather
than "covariates given but silently dropped".

This module closes that gap by default: when --covfile is not given,
compute the top-K genotype-based principal components and use them (plus
an explicit intercept column) as X0, with a clear log message -- never a
silent choice, matching every other default/fallback decision in this
project.

Implementation
---------------
PCA on the standardised genotype matrix is mathematically the same
eigendecomposition already used for the kinship/GRM matrix (top
eigenvectors of K = A@A.T are exactly the top genotype PCs, by
construction). Rather than writing separate PCA code, this module reuses
the already-validated randomized_kinship_eigendecomp() /
randomized_kinship_eigendecomp_streaming() functions directly -- both
already produce exactly the (n, r) matrix of top eigenvectors needed here,
and both are independently validated (validate_lowrank_lmm.py,
validate_streaming_io.py) to match an exact dense eigendecomposition.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_AUTO_PCA_K = 10


def compute_auto_pca_dense(G, n_pcs=DEFAULT_AUTO_PCA_K, seed=42, verbose=True,
                           maf_min=0.01):
    """
    Compute top-n_pcs genotype PCA covariates from a dense genotype matrix,
    for the exact-tier case where G is already in memory.

    Parameters
    ----------
    G     : ndarray (n_samples, n_snps)  raw genotype dosage
    n_pcs : int  number of principal components to compute
    maf_min : minimum minor allele frequency for a SNP to contribute to the
        PCA -- see randomized_kinship_eigendecomp()'s docstring for why a
        rare-variant numerical-stability floor matters here too (PCA is
        computed via the same eigendecomposition as the kinship matrix).

    Returns
    -------
    X0 : ndarray (n_samples, n_pcs + 1)  intercept column followed by the
        top n_pcs PCs, ready to pass directly as X0 to LMM/LMMLowRank/CGLMM
    """
    from jasmap.biobank.lowrank_kinship import randomized_kinship_eigendecomp

    n = G.shape[0]
    r = min(n_pcs, n - 1, G.shape[1])
    _, U, _ = randomized_kinship_eigendecomp(G, rank=r, power_iter=3, seed=seed,
                                             maf_min=maf_min)

    X0 = np.hstack([np.ones((n, 1)), U[:, :r]])
    if verbose:
        logger.info(f"[auto_pca] Computed {r} genotype PCA covariate(s) "
                    f"(n={n} samples) for population stratification "
                    f"correction (no --covfile provided).")
    return X0


def compute_auto_pca_streaming(bed_path, n_pcs=DEFAULT_AUTO_PCA_K,
                               sample_idx=None, snp_idx=None, cache_budget_gb=4.0,
                               seed=42, verbose=True, maf_min=0.01):
    """
    Streaming counterpart to compute_auto_pca_dense(): computes the same
    top-n_pcs genotype PCA covariates directly from a .bed file via
    SNP-chunked streaming, for the approximate/large-n tier where the
    genotype matrix is never materialised in memory.

    snp_idx : optional array of SNP indices to restrict to (e.g. --chr
        filtering) -- passed straight through to StreamingGenotypeSource.

    Returns
    -------
    X0 : ndarray (n_samples, n_pcs + 1)  same shape/meaning as the dense version
    """
    from jasmap.biobank.streaming_io import (
        StreamingGenotypeSource, randomized_kinship_eigendecomp_streaming)

    source = StreamingGenotypeSource(bed_path, sample_idx=sample_idx,
                                     snp_idx=snp_idx, cache_budget_gb=cache_budget_gb,
                                     verbose=verbose, maf_min=maf_min)
    n = source.n
    r = min(n_pcs, n - 1, source.m_eff)
    _, U, _, _ = randomized_kinship_eigendecomp_streaming(
        source, rank=r, power_iter=2, seed=seed, verbose=verbose)

    X0 = np.hstack([np.ones((n, 1)), U[:, :r]])
    if verbose:
        logger.info(f"[auto_pca] Computed {r} genotype PCA covariate(s) "
                    f"(n={n} samples, streaming) for population "
                    f"stratification correction (no --covfile provided).")
    return X0
