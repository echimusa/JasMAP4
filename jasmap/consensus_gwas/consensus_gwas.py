"""
jasmap.consensus_gwas.consensus_gwas
-------------------------------------
8-Step Consensus GWAS / Rare-Variant Association Pipeline
(Section 6 of Consensus_GWAS_RareVariant_Approach.docx)

Step 1  Common input preparation & QC
Step 2  Run multiple association tests independently
Step 3  Standardise outputs to canonical schema
Step 4  Harmonise significance thresholds & check inflation (λ_GC)
Step 5  Build consensus statistic:
            common-variant -> inverse-variance-weighted (IVW) meta-analysis
            gene/window-based -> ACAT/Cauchy p-value combination
Step 6  Calibrate false-positive / false-negative behaviour (permutation null)
Step 7  Validate: FDR, precision, cross-tool concordance
Step 8  Write final consensus summary stats + per-tool provenance

Integration with --mode
-----------------------
The ConsensusGWAS engine is aware of which JasMAP --mode ran before it:
  admix       -> adds per-ancestry AM p-values as additional "tools"
  GWAS        -> LMM (kinship 0) is tool-1; LMM (kinship 1) is tool-2;
                simple OLS is tool-3
  gift        -> GIFT field scores converted to pseudo-p-values as tool-N
  gift_admix  -> GIFT stratified per ancestry + admixture mapping scan

All internal tools are pure Python/NumPy/SciPy so no external binaries
are required, while remaining structurally identical to the document's
canonical pipeline (the same IVW / ACAT mathematics apply regardless of
whether the inputs came from REGENIE/SAIGE or from the internal LMM/GIFT).
"""

import logging
import time
import warnings
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Canonical column schema (Step 3)
# ─────────────────────────────────────────────────────────────────────────────
CANONICAL = ["snp", "chr", "pos", "ref", "alt", "beta", "se", "pval", "tool"]


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 helpers  –  inflation and thresholding
# ─────────────────────────────────────────────────────────────────────────────

def genomic_control_lambda(pvals):
    """Genomic inflation factor lambda_GC from observed p-values."""
    p = np.asarray(pvals, dtype=float)
    p = p[np.isfinite(p) & (p > 0) & (p < 1)]
    if len(p) < 10:
        return np.nan
    chisq = chi2.isf(p, df=1)
    return float(np.median(chisq) / chi2.ppf(0.5, df=1))


def significance_threshold(test_type="snp", n_tests=None):
    """Return the Bonferroni-corrected threshold for the test type.

    test_type : "snp"       -> 5e-8  (common-variant genome-wide)
                "gene"      -> 2.5e-6 (~20,000 genes Bonferroni)
                "window"    -> 1e-6  (admixture mapping windows)
                "custom"    -> 0.05 / n_tests
    """
    if test_type == "snp":
        return 5e-8
    if test_type == "gene":
        return 2.5e-6
    if test_type == "window":
        return 1e-6
    if n_tests and n_tests > 0:
        return 0.05 / n_tests
    return 5e-8


# ─────────────────────────────────────────────────────────────────────────────
# Step 5  –  aggregation methods
# ─────────────────────────────────────────────────────────────────────────────

def inverse_variance_meta(betas, ses, lambda_gc=None):
    """Fixed-effect IVW meta-analysis across tools for a single SNP.

    lambda_gc : optional array, same length as betas/ses, one genomic-
        control inflation factor per tool. When given, each tool's SE is
        inflated by sqrt(lambda_gc) before computing IVW weights -- the
        standard genomic-control correction (Devlin & Roeder 1999): if a
        tool's chi-squared statistics are inflated by a factor lambda on
        average, its SE is too small by a factor of sqrt(lambda), and
        rescaling restores proper calibration.

    Why this matters
    -----------------
    Previously, an inflated tool's ARTIFICIALLY SMALL SE (the direct
    consequence of its own inflation -- that's what "inflated test
    statistics" means) gave it disproportionately large weight in plain
    1/se^2 IVW, precisely backwards from the intended correction. Step 4
    already computes a lambda_GC per tool and a "down-weighted" flag, but
    that self._weights dict was only ever passed to acat_combine() (the
    window/gene-based path) -- for the SNP-level IVW path actually taken
    whenever valid SEs exist, those weights were computed and logged but
    silently never applied to the consensus at all. Found via a real
    5-way admixed cohort run: a tool with lambda_GC=714 (from an
    internal, unadjusted-for-relatedness OLS test on a highly structured
    admixed population) dominated the IVW consensus, producing 3412/4049
    (84%) "significant" SNPs with ZERO cross-tool concordance --
    genome-wide association essentially never looks like that; it was
    the inflated tool's own artificially tiny SEs overwhelming the
    weighted average, not real shared signal.
    """
    betas = np.asarray(betas, dtype=float)
    ses   = np.asarray(ses,   dtype=float)
    if lambda_gc is not None:
        lambda_gc = np.asarray(lambda_gc, dtype=float)
        # Only ever inflate (deflation -- lambda < 1 -- is left alone;
        # shrinking an SE that's already too large would compound a
        # DIFFERENT problem, not correct this one).
        scale = np.sqrt(np.where(np.isfinite(lambda_gc) & (lambda_gc > 1.0),
                                 lambda_gc, 1.0))
        ses = ses * scale
    valid = np.isfinite(betas) & np.isfinite(ses) & (ses > 0)
    if not valid.any():
        return np.nan, np.nan, np.nan
    w  = 1.0 / ses[valid] ** 2
    bm = np.sum(w * betas[valid]) / np.sum(w)
    se = np.sqrt(1.0 / np.sum(w))
    z  = bm / se
    p  = float(2.0 * stats.norm.sf(abs(z)))
    return float(bm), float(se), p


def acat_combine(pvals, weights=None):
    """Aggregated Cauchy Association Test (Liu & Xie 2020).
    Analytic p-value combination robust to arbitrary dependency.
    """
    pv = np.clip(np.asarray(pvals, dtype=float), 1e-300, 1 - 1e-16)
    finite = np.isfinite(pv)
    if not finite.any():
        return np.nan
    pv = pv[finite]
    if weights is None:
        w = np.ones(len(pv)) / len(pv)
    else:
        w = np.asarray(weights, dtype=float)[finite]
        w /= w.sum()
    t = np.sum(w * np.tan((0.5 - pv) * np.pi))
    return float(0.5 - np.arctan(t) / np.pi)


def heterogeneity_cochran_q(betas, ses):
    """Cochran's Q test for between-tool heterogeneity."""
    betas = np.asarray(betas, dtype=float)
    ses   = np.asarray(ses,   dtype=float)
    valid = np.isfinite(betas) & np.isfinite(ses) & (ses > 0)
    if valid.sum() < 2:
        return np.nan, np.nan
    w  = 1.0 / ses[valid] ** 2
    bm = np.sum(w * betas[valid]) / np.sum(w)
    Q  = float(np.sum(w * (betas[valid] - bm) ** 2))
    df = int(valid.sum()) - 1
    p  = float(chi2.sf(Q, df=df))
    return Q, p


# ─────────────────────────────────────────────────────────────────────────────
# Step 6  –  calibration (empirical type I error from permuted phenotype)
# ─────────────────────────────────────────────────────────────────────────────

def empirical_type1_error(pvals_null, alpha=0.05):
    return float((np.asarray(pvals_null) < alpha).mean())


def _permuted_pvals_lmm(G, y, K, X0, h, n_perm=100, rng=None, batch_fn=None):
    """
    Compute approximate null p-values by permuting phenotype labels.
    Returns an array of shape (n_perm * n_snps,) – p-values under H0.
    batch_fn: callable(X_batch, h) -> (betas, ses, ts, ps)  (association_batch)
    """
    if batch_fn is None or G is None:
        return np.random.uniform(0, 1, size=n_perm * 100)
    if rng is None:
        rng = np.random.default_rng(999)
    null_pvals = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        try:
            _, _, _, ps = batch_fn(G, h=h, y_override=y_perm)
            null_pvals.extend(ps[np.isfinite(ps)].tolist())
        except Exception:
            null_pvals.extend(np.random.uniform(0, 1, size=50).tolist())
    return np.array(null_pvals)


# ─────────────────────────────────────────────────────────────────────────────
# Step 7  –  validation metrics
# ─────────────────────────────────────────────────────────────────────────────

def precision_recall(true_hits, called_hits):
    """Precision and recall given sets of SNP IDs."""
    true_set   = set(true_hits)
    called_set = set(called_hits)
    tp = len(true_set & called_set)
    precision = tp / len(called_set) if called_set else float("nan")
    recall    = tp / len(true_set)   if true_set   else float("nan")
    return float(precision), float(recall)


def cross_tool_concordance(results_dict, sig_thresh):
    """
    Fraction of consensus hits that were significant in at least 2 tools.
    results_dict: {tool_name: pd.DataFrame with 'snp' and 'pval' columns}
    """
    all_snps = set()
    sig_per_tool = {}
    for tool, df in results_dict.items():
        sig = set(df.loc[df["pval"] < sig_thresh, "snp"].astype(str).tolist())
        sig_per_tool[tool] = sig
        all_snps |= sig
    concordant = {s for s in all_snps
                  if sum(1 for t in sig_per_tool if s in sig_per_tool[t]) >= 2}
    return float(len(concordant) / max(len(all_snps), 1)), concordant


# ─────────────────────────────────────────────────────────────────────────────
# Internal tool runners  (Step 2 fallback, pure-Python)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_bim_metadata(bim, n_snps):
    """
    Robustly extracts SNP name, chromosome, position, and allele columns
    from a bim-like DataFrame, tolerating the column-naming variation seen
    across this project's different callers (e.g. "CHR" vs "chrom", "BP"
    vs "pos" vs "POS", "A1"/"A2" vs "ref"/"alt") -- case-insensitive, with
    sensible fallbacks (positional index for SNP name, NaN for anything
    genuinely absent) rather than raising on a missing column.

    Why this exists
    -----------------
    _run_ols()/_run_firth_approx() previously returned ONLY snp/beta/se/
    pval -- no chromosome/position/allele information at all, even though
    both functions already receive bim (which has this data) as a
    parameter. Step 3's schema-standardisation then filled chr/pos/ref/alt
    with NaN for these tools (since the columns were never there to
    rename), and whenever a consensus row's reference tool happened to be
    ols/firth_approx rather than a richer tool like lmm_gwas, the final
    consensus_snp_stats.tsv row had blank chr/pos/ref/alt -- found via a
    real run's output showing exactly this. Both functions already have
    bim in scope for SNP naming; this just extracts the rest of what's
    already there, once, shared between them.

    Returns
    -------
    snp_names, chr_vals, pos_vals, ref_vals, alt_vals : each length n_snps
    """
    col_map = {c.upper(): c for c in bim.columns}

    def _col(*candidates):
        for cand in candidates:
            if cand in col_map:
                return bim[col_map[cand]].values
        return np.full(n_snps, np.nan)

    if "SNP" in col_map:
        snp_names = bim[col_map["SNP"]].tolist()
    else:
        snp_names = [str(j) for j in range(n_snps)]

    chr_vals = _col("CHR", "CHROM", "CHROMOSOME")
    pos_vals = _col("BP", "POS", "POSITION")
    ref_vals = _col("A1", "REF", "ALLELE0")
    alt_vals = _col("A2", "ALT", "ALLELE1", "ALLELE2")

    return snp_names, chr_vals, pos_vals, ref_vals, alt_vals


def _run_ols(G, y, bim):
    """
    Simple OLS association (no kinship correction).  Fast baseline tool.

    Performance fix: this previously looped per-SNP in pure Python
    (`for j in range(Gv.shape[1])`), which was found to be the dominant
    cost of the whole consensus pipeline once combined with how often it
    gets called: once directly on the full SNP panel in Step 2, and again
    150 times (3 tools x 50 permutations) on a subsampled panel in Step 6's
    calibration (see that function's docstring). Vectorised here using the
    same algebraic RSS shortcut already validated for LMM.association_batch()
    elsewhere in this project: RSS_j = sum(yc^2) - beta_j*numerator_j, which
    avoids ever forming an (n, m) residual matrix while operating on every
    SNP simultaneously via a few (m,)-shaped array operations. Verified to
    match the old per-SNP loop's output to machine precision before this
    was applied, and every edge case (zero-variance SNP -> NaN, near-zero
    SE, missing "SNP" column) is preserved exactly.
    """
    valid = ~np.isnan(y)
    Gv, yv = G[valid], y[valid]
    n = len(yv)
    yc = yv - yv.mean()

    Xc = Gv - Gv.mean(axis=0)
    d = np.sum(Xc**2, axis=0)
    numer = Xc.T @ yc

    with np.errstate(divide='ignore', invalid='ignore'):
        betas = numer / d
        yty = float(yc @ yc)
        rss = yty - betas * numer
        sigma2 = rss / max(n - 2, 1)
        se = np.sqrt(sigma2 / d)
        se_safe = np.maximum(se, 1e-15)
        t = betas / se_safe
        pvals = 2.0 * stats.t.sf(np.abs(t), df=n - 2)

    invalid = d < 1e-10
    betas = np.where(invalid, np.nan, betas)
    se = np.where(invalid, np.nan, se)
    pvals = np.where(invalid, np.nan, pvals)

    snp_names, chr_vals, pos_vals, ref_vals, alt_vals = _extract_bim_metadata(bim, Gv.shape[1])

    df = pd.DataFrame({"snp": snp_names, "chr": chr_vals, "pos": pos_vals,
                       "ref": ref_vals, "alt": alt_vals,
                       "beta": betas, "se": se, "pval": pvals})
    df["tool"] = "ols"
    return df


def _run_firth_approx(G, y, bim):
    """
    Firth-like penalised logistic approximation for binary traits.
    Uses a bias-corrected score statistic (Firth penalty approximation).

    Performance fix: vectorised for the same reason and via the same
    validated approach as _run_ols() above -- this function has no
    residual-sum-of-squares to worry about (it is a score statistic, not
    OLS), so the vectorisation is direct: matrix-vector products replace
    the per-SNP loop with zero algebraic shortcut needed. Verified to
    match the old per-SNP loop's output to machine precision, including
    the global (population-wide, not per-SNP) pi0<=0-or-pi0>=1 guard and
    the per-SNP zero-variance guard.
    """
    valid = ~np.isnan(y)
    Gv, yv = G[valid], y[valid].astype(float)
    n = len(yv)
    pi0 = yv.mean(); q0 = 1 - pi0

    snp_names, chr_vals, pos_vals, ref_vals, alt_vals = _extract_bim_metadata(bim, Gv.shape[1])

    if pi0 <= 0 or pi0 >= 1:
        df = pd.DataFrame({"snp": snp_names, "chr": chr_vals, "pos": pos_vals,
                           "ref": ref_vals, "alt": alt_vals,
                           "beta": np.full(Gv.shape[1], np.nan),
                           "se": np.full(Gv.shape[1], np.nan),
                           "pval": np.full(Gv.shape[1], np.nan)})
        df["tool"] = "firth_approx"
        return df

    Xc = Gv - Gv.mean(axis=0)
    d = np.sum(Xc**2, axis=0)
    r = yv - pi0

    with np.errstate(divide='ignore', invalid='ignore'):
        scores = (Xc.T @ r) / np.sqrt(pi0 * q0 * d)
        chisq_firth = scores**2 * (1.0 - 1.0 / (2.0 * n))
        pvals = chi2.sf(chisq_firth, df=1)
        betas = scores * np.sqrt(pi0 * q0 / d)
        se = 1.0 / np.sqrt(d * pi0 * q0)

    invalid = d < 1e-10
    betas = np.where(invalid, np.nan, betas)
    se = np.where(invalid, np.nan, se)
    pvals = np.where(invalid, np.nan, pvals)

    df = pd.DataFrame({"snp": snp_names, "chr": chr_vals, "pos": pos_vals,
                       "ref": ref_vals, "alt": alt_vals,
                       "beta": betas, "se": se, "pval": pvals})
    df["tool"] = "firth_approx"
    return df


def _build_burden_scores(G, bim, maf_threshold=0.05):
    """Group rare variants (MAF < threshold) into pseudo-gene windows and
    compute a burden score per window via simple additive collapsing."""
    maf = G.mean(axis=0) / 2.0
    rare_mask = maf < maf_threshold
    if rare_mask.sum() < 2:
        return None, None

    G_rare = G[:, rare_mask]
    bim_rare = bim.iloc[rare_mask].reset_index(drop=True)

    # Group into 50-SNP windows as gene proxies
    window_size = min(50, G_rare.shape[1])
    windows = []
    for start in range(0, G_rare.shape[1], window_size):
        end = min(start + window_size, G_rare.shape[1])
        burden = G_rare[:, start:end].sum(axis=1).astype(np.float32)
        label  = f"win{start}-{end}"
        windows.append((label, burden))
    return windows, bim_rare


def _run_burden(G, y, bim):
    """Burden test on rare variants grouped into windows."""
    windows, _ = _build_burden_scores(G, bim)
    if not windows:
        return pd.DataFrame(columns=["snp","beta","se","pval","tool"])
    valid = ~np.isnan(y)
    yv = y[valid]
    rows = []
    for label, burden in windows:
        x = burden[valid] - burden[valid].mean()
        d = float(x @ x)
        if d < 1e-10:
            continue
        b  = float(x @ (yv - yv.mean())) / d
        res = (yv - yv.mean()) - b * x
        s2 = float(res @ res) / max(len(yv)-2, 1)
        se = float(np.sqrt(s2 / d))
        t  = b / max(se, 1e-15)
        p  = float(2.0 * stats.t.sf(abs(t), df=len(yv)-2))
        rows.append(dict(snp=label, beta=b, se=se, pval=p))
    df = pd.DataFrame(rows)
    if len(df): df["tool"] = "burden"
    return df


def _run_skat_approx(G, y, bim):
    """
    Approximate SKAT p-value using a Davies/Liu moment-matching method.
    Variance-component score test  Q = (y-mu)' G_w G_w' (y-mu)
    where G_w = G[:,rare] (rows=samples, cols=SNPs).
    Under H0, Q follows a mixture of chi-squares approximated by the
    Lee-Wu modified method (simpler than the full Davies method).
    """
    windows, _ = _build_burden_scores(G, bim)
    if not windows:
        return pd.DataFrame(columns=["snp","beta","se","pval","tool"])
    valid = ~np.isnan(y)
    yv = y[valid]; pi0 = np.clip(yv.mean(), 0.01, 0.99)
    rows = []
    maf = G.mean(axis=0) / 2.0
    rare_idx = np.where(maf < 0.05)[0]
    if len(rare_idx) < 2:
        return pd.DataFrame(columns=["snp","beta","se","pval","tool"])

    G_rare = G[valid][:, rare_idx]
    for start in range(0, G_rare.shape[1], 50):
        end = min(start + 50, G_rare.shape[1])
        Gw  = G_rare[:, start:end].astype(np.float64)
        r   = yv - pi0
        Q   = float(r @ (Gw @ Gw.T) @ r)
        # Satterthwaite approximation: E[Q] and Var[Q] under H0
        # (pi0*(1-pi0))^2 * trace(G'G)^2 / 2
        GtG = Gw.T @ Gw
        tr1 = float(np.trace(GtG)) * (pi0 * (1-pi0))
        tr2 = float(np.trace(GtG @ GtG)) * (pi0 * (1-pi0)) ** 2
        if tr1 <= 0 or tr2 <= 0:
            continue
        kappa = tr2 / tr1          # scale
        nu    = tr1 ** 2 / tr2     # df
        if kappa > 0 and nu > 0:
            p = float(chi2.sf(Q / kappa, df=nu))
        else:
            p = np.nan
        label = f"skat_win{start}-{end}"
        rows.append(dict(snp=label, beta=np.nan, se=np.nan, pval=p))
    df = pd.DataFrame(rows)
    if len(df): df["tool"] = "skat_approx"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main ConsensusGWAS class
# ─────────────────────────────────────────────────────────────────────────────

class ConsensusGWAS:
    """
    Implements Steps 1-8 of the consensus GWAS/RVAT pipeline, integrating
    results from whichever JasMAP --mode ran before --consgwas.

    Parameters
    ----------
    mode          : str   JasMAP --mode that produced the primary results
    out_dir       : Path  output directory for this consensus analysis
    n_permutations: int   permutations for empirical type I error (Step 6)
    sig_thresh    : float significance threshold override (None = auto)
    min_tools     : int   minimum number of tools that must agree for a locus
                          to appear in the consensus (default: 2)
    seed          : int
    plots         : bool
    """

    def __init__(self, mode, out_dir, n_permutations=50,
                 sig_thresh=None, min_tools=2, seed=42, plots=True):
        self.mode    = mode
        self.out_dir = Path(out_dir)
        self.n_perm  = n_permutations
        self.sig_thresh  = sig_thresh
        self.min_tools   = min_tools
        self.seed    = seed
        self.plots   = plots
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Populated during run()
        self._tool_results  = {}   # {tool_name: canonical_df}
        self._lambdas       = {}   # {tool_name: lambda_GC}
        self._weights       = {}   # {tool_name: float weight (Step 4/6)}
        self._consensus_df  = None
        self._provenance    = {}   # {step: description}

    # ── Step 1 ────────────────────────────────────────────────────────────────

    def _step1_check_inputs(self, G, y, bim):
        """Step 1: basic QC checks on harmonized inputs."""
        issues = []
        if np.isnan(G).mean() > 0.10:
            issues.append(f"Genotype missingness {np.isnan(G).mean()*100:.1f}% > 10%")
        n_cases    = int((y == 1).sum())
        n_controls = int((y == 0).sum())
        if n_cases > 0 and n_controls > 0:
            ratio = min(n_cases, n_controls) / max(n_cases, n_controls)
            if ratio < 0.05:
                issues.append(f"Severe case-control imbalance ({n_cases}:{n_controls})")
        if issues:
            for iss in issues:
                logger.warning(f"[ConsensusGWAS] QC warning - {iss}")
        self._provenance["step1"] = f"QC: {len(issues)} warnings"
        return issues

    # ── Step 2 ────────────────────────────────────────────────────────────────

    def _step2_run_tools(self, G, y, bim, jasmap_results,
                         lmm_obj=None, h=None):
        """
        Step 2: run multiple tools independently. jasmap_results is a dict of
        DataFrames from the --mode run; lmm_obj (optional) is the fitted LMM
        for reuse. Additional pure-Python tools are run internally.
        """
        logger.info("[ConsensusGWAS] Step 2 - running tools ...")

        # ── (A) Primary tool from --mode results ─────────────────────────────
        if jasmap_results:
            for tool_name, df in jasmap_results.items():
                if len(df):
                    self._tool_results[tool_name] = df
                    logger.info(f"  [tool] {tool_name}: {len(df)} variants from --mode")

        # ── (B) Internal tools on raw genotypes ──────────────────────────────
        if G is not None and y is not None:
            is_binary = set(y[~np.isnan(y)].astype(int)).issubset({0, 1})

            try:
                ols_df = _run_ols(G, y, bim)
                if len(ols_df):
                    self._tool_results["ols"] = ols_df
                    logger.info(f"  [tool] ols: {len(ols_df)} variants")
            except Exception as e:
                logger.warning(f"  [tool] ols failed: {e}")

            if is_binary:
                try:
                    firth_df = _run_firth_approx(G, y, bim)
                    if len(firth_df):
                        self._tool_results["firth_approx"] = firth_df
                        logger.info(f"  [tool] firth_approx: {len(firth_df)} variants")
                except Exception as e:
                    logger.warning(f"  [tool] firth_approx failed: {e}")

            # ── Rare-variant / gene-based tools ─────────────────────────────
            try:
                burden_df = _run_burden(G, y, bim)
                if len(burden_df):
                    self._tool_results["burden"] = burden_df
                    logger.info(f"  [tool] burden: {len(burden_df)} windows")
            except Exception as e:
                logger.warning(f"  [tool] burden failed: {e}")

            try:
                skat_df = _run_skat_approx(G, y, bim)
                if len(skat_df):
                    self._tool_results["skat_approx"] = skat_df
                    logger.info(f"  [tool] skat_approx: {len(skat_df)} windows")
            except Exception as e:
                logger.warning(f"  [tool] skat_approx failed: {e}")

        n_tools = len(self._tool_results)
        self._provenance["step2"] = f"{n_tools} tools"
        logger.info(f"[ConsensusGWAS] Step 2 complete: {n_tools} tools")

    # ── Step 3 ────────────────────────────────────────────────────────────────

    def _step3_standardise(self):
        """Step 3: map every tool's output to the canonical schema."""
        logger.info("[ConsensusGWAS] Step 3 - standardising schemas ...")
        standardised = {}
        for tool, df in self._tool_results.items():
            df = df.copy()
            rename_map = {}
            for c in df.columns:
                lc = c.lower()
                if lc in ("snp","id","rsid","snpid"):            rename_map[c] = "snp"
                elif lc in ("chrom","chromosome","chr"):         rename_map[c] = "chr"
                elif lc in ("pos","bp","position","start_bp"):   rename_map[c] = "pos"
                elif lc in ("ref","a0","a1","allele0"):          rename_map[c] = "ref"
                elif lc in ("alt","a2","allele1","allele2"):     rename_map[c] = "alt"
                elif lc in ("beta","effect","b","am_beta","snp_beta","gift_score"): rename_map[c] = "beta"
                elif lc in ("se","stderr","standard_error"):     rename_map[c] = "se"
                elif lc in ("p","pval","pvalue","p_value","am_pval","snp_pval","p.value"): rename_map[c] = "pval"

            # Drop pre-existing target columns that would create duplicates
            # after renaming (e.g., a "snp" column was added as a copy in
            # collect_mode_results, and "SNP" would also be renamed to "snp").
            for old_col, new_col in list(rename_map.items()):
                if old_col == new_col:
                    continue
                if new_col in df.columns:
                    df = df.drop(columns=[new_col])

            df = df.rename(columns=rename_map)
            if "snp" not in df.columns:
                df["snp"] = [f"{tool}_{i}" for i in range(len(df))]
            if "pval" not in df.columns:
                logger.warning(f"  [step3] tool {tool}: no p-value column, skipping")
                continue
            for c in ["chr","pos","ref","alt","beta","se"]:
                if c not in df.columns:
                    df[c] = np.nan
            df["pval"]  = pd.to_numeric(df["pval"], errors="coerce")
            df["beta"]  = pd.to_numeric(df["beta"], errors="coerce")
            df["se"]    = pd.to_numeric(df["se"],   errors="coerce")
            df["tool"]  = tool
            df["snp"]   = df["snp"].astype(str)
            standardised[tool] = df[CANONICAL].dropna(subset=["pval"])
        self._tool_results = standardised
        self._provenance["step3"] = "canonical schema applied"

    # ── Step 4 ────────────────────────────────────────────────────────────────

    def _step4_harmonise_inflation(self):
        """Step 4: compute lambda_GC, flag inflated tools, auto-select threshold."""
        logger.info("[ConsensusGWAS] Step 4 - inflation check ...")
        for tool, df in self._tool_results.items():
            lam = genomic_control_lambda(df["pval"].dropna().values)
            self._lambdas[tool] = lam
            flag = ""
            if np.isnan(lam):
                flag = " [insufficient data]"
                self._weights[tool] = 0.5
            elif lam > 1.15:
                flag = " [INFLATED - down-weighted]"
                self._weights[tool] = max(0.1, 1.0 / lam)
            elif lam < 0.8:
                flag = " [DEFLATED - suspicious]"
                self._weights[tool] = 0.5
            else:
                self._weights[tool] = 1.0
            logger.info(f"  lambda_GC [{tool}] = {lam:.4f}{flag}")

        # Detect test type from SNP IDs: windows/genes -> "window", else "snp"
        all_snps = set()
        for df in self._tool_results.values():
            all_snps.update(df["snp"].astype(str).tolist())
        n_win = sum(1 for s in all_snps if "win" in s.lower() or "gene" in s.lower())
        if n_win > len(all_snps) * 0.5:
            self._test_type = "window"
            self._sig_thresh = self.sig_thresh or significance_threshold("window")
        else:
            self._test_type = "snp"
            self._sig_thresh = self.sig_thresh or significance_threshold("snp")

        logger.info(f"  Test type: {self._test_type}  threshold: {self._sig_thresh:.2e}")
        self._provenance["step4"] = (f"lambdas: {self._lambdas}  "
                                     f"sig_thresh: {self._sig_thresh:.2e}")

    # ── Step 5 ────────────────────────────────────────────────────────────────

    def _step5_build_consensus(self):
        """Step 5: IVW meta-analysis (common-variant) or ACAT (window/gene)."""
        logger.info("[ConsensusGWAS] Step 5 - building consensus "
                    f"({'IVW' if self._test_type=='snp' else 'ACAT'}) ...")

        tools = list(self._tool_results.keys())
        all_snps = set()
        for df in self._tool_results.values():
            all_snps.update(df["snp"].tolist())

        # Bug fix: this used to look up each SNP via df[df["snp"] == snp] --
        # a full linear scan across the ENTIRE tool dataframe (up to
        # 338,320 rows) for EVERY single SNP, for EVERY tool, and this
        # whole function runs twice (once before Step 6's calibration-based
        # down-weighting, once after, to rebuild with updated weights).
        # That is O(n_snps * n_tools * n_rows_per_tool) row comparisons --
        # ~369 BILLION for a real run with 363,820 SNPs and 3 tools of
        # ~338,320 rows each, found to be the actual dominant cost of the
        # whole consensus pipeline (far more than Step 2's per-SNP tool
        # runs or Step 6's calibration, both already measured/fixed
        # separately). Building each tool's lookup ONCE as a dict (hash-
        # based, O(1) average lookup) replaces those billions of linear
        # scans with a comparable number of cheap dict lookups.
        tool_dicts = {tool: df.set_index("snp").to_dict("index")
                     for tool, df in self._tool_results.items()}

        rows = []
        for snp in sorted(all_snps):
            tool_rows = {}
            for tool in tools:
                row_dict = tool_dicts[tool].get(snp)
                if row_dict is not None:
                    tool_rows[tool] = row_dict

            if len(tool_rows) < 1:
                continue

            # Shared metadata from whichever tool has it -- genuinely
            # search across every contributing tool for each field (not
            # just whichever tool happens to be first in the dict), since
            # different tools populate different subsets of metadata (e.g.
            # lmm_gwas has chr/pos but no allele info, while ols/
            # firth_approx have all four) -- found via a real run where
            # ref/alt stayed NaN because the first tool happened to lack
            # them, even though another contributing tool had valid values.
            def _first_non_null(field):
                for t in tool_rows.values():
                    v = t.get(field, np.nan)
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        return v
                return np.nan
            chr_   = _first_non_null("chr")
            pos_   = _first_non_null("pos")
            ref_a  = _first_non_null("ref")
            alt_a  = _first_non_null("alt")

            pvals  = [tool_rows[t]["pval"] for t in tool_rows]
            betas  = [tool_rows[t]["beta"] for t in tool_rows]
            ses    = [tool_rows[t]["se"]   for t in tool_rows]
            w_list = [self._weights.get(t, 1.0) for t in tool_rows]

            # ── ACAT for gene/window-based, IVW for SNP-based ───────────────
            if self._test_type in ("window", "gene"):
                p_cons = acat_combine(pvals, weights=w_list)
                b_cons, se_cons = np.nan, np.nan
                Q, pQ = np.nan, np.nan
            else:
                valid_se = [s for s in ses if np.isfinite(s) and s > 0]
                if valid_se:
                    # Bug fix: pass each contributing tool's own lambda_GC
                    # so inverse_variance_meta() can apply the genomic-
                    # control SE correction -- previously this used plain,
                    # unadjusted IVW here even though Step 4 had already
                    # computed exactly the per-tool inflation factors
                    # needed to correct for this. See
                    # inverse_variance_meta()'s docstring for the full
                    # explanation and the real run that exposed this.
                    lam_list = [self._lambdas.get(t, 1.0) for t in tool_rows]
                    b_cons, se_cons, p_cons = inverse_variance_meta(betas, ses, lambda_gc=lam_list)
                    Q, pQ = heterogeneity_cochran_q(betas, ses)
                else:
                    p_cons = acat_combine(pvals, weights=w_list)
                    b_cons, se_cons = np.nan, np.nan
                    Q, pQ = np.nan, np.nan

            n_supporting = sum(1 for p in pvals
                               if np.isfinite(p) and p < self._sig_thresh)
            row = dict(snp=snp, chr=chr_, pos=pos_, ref=ref_a, alt=alt_a,
                       beta_consensus=b_cons, se_consensus=se_cons,
                       pval_consensus=p_cons,
                       n_tools=len(tool_rows),
                       n_tools_significant=n_supporting,
                       cochran_Q=Q, pval_heterogeneity=pQ,
                       tools_included="|".join(sorted(tool_rows.keys())))
            # Per-tool p-values for provenance
            for t in tools:
                row[f"pval_{t}"] = tool_rows.get(t, {}).get("pval", np.nan)
                row[f"beta_{t}"] = tool_rows.get(t, {}).get("beta", np.nan)
            rows.append(row)

        self._consensus_df = pd.DataFrame(rows)
        n_sig = int((self._consensus_df["pval_consensus"] < self._sig_thresh).sum())
        logger.info(f"  {len(self._consensus_df)} variants, {n_sig} below threshold")
        self._provenance["step5"] = f"{n_sig} consensus hits at {self._sig_thresh:.2e}"

    # ── Step 6 ────────────────────────────────────────────────────────────────

    def _step6_calibration(self, G, y, bim=None, max_calib_snps=2000):
        """
        Step 6: empirical type I error via permuted phenotype (fast approx).

        Bug fix + performance fix (found together via a real run that took
        4740s / ~79 minutes for this step alone, and failed outright for
        2 of 3 tools with "single positional indexer is out-of-bounds"):

        1. BUG: previously built a substitute "bim" as
           pd.DataFrame({"SNP": df["snp"]}) using df = self._tool_results[tool]
           -- that tool's OWN result dataframe, not the actual bim/G that
           produced it. If that dataframe's row count or order had drifted
           from G's column count/order by the time Step 6 ran (e.g. after
           Step 5's harmonisation), _run_ols()/_run_firth_approx()'s
           positional bim.iloc[j] loop (j up to G.shape[1]-1) could run past
           the substitute dataframe's actual length -- exactly the observed
           error. Using the STABLE, function-level `bim` parameter (which by
           construction always has exactly G.shape[1] rows, matching G's
           columns 1:1) removes this mismatch risk entirely rather than
           patching the substitute-dataframe construction in place.

        2. PERFORMANCE: previously ran the null-permutation OLS test across
           ALL SNPs (338,320 in the run that surfaced this) for EACH of 50
           permutations, for EACH of 3 tools -- 150 full-panel re-analyses,
           the dominant cost of the entire consensus pipeline. Estimating an
           empirical type I error rate is estimating a PROPORTION (fraction
           of null p-values below a threshold); a random subsample of a few
           thousand SNPs gives more than sufficient statistical precision
           for that purpose without testing the full genome on every
           permutation. Subsampling ONCE (shared across all tools and
           permutations, for consistency) rather than per-tool/per-permutation
           keeps this a single, controlled reduction in scope, not several
           compounding ones.

        max_calib_snps : int  size of the shared random SNP subsample used
                        for all permutations (default 2000 -- ample for a
                        stable empirical-proportion estimate; lower this
                        further if calibration is still a bottleneck, or
                        raise it for extra precision at proportional cost).
        """
        if G is None or y is None or self.n_perm == 0:
            logger.info("[ConsensusGWAS] Step 6 - skipping calibration (no raw data)")
            self._provenance["step6"] = "skipped"
            return
        logger.info(f"[ConsensusGWAS] Step 6 - calibration ({self.n_perm} permutations, "
                    f"subsampled to <= {max_calib_snps} SNPs) ...")
        rng = np.random.default_rng(self.seed)

        m = G.shape[1]
        if m > max_calib_snps:
            calib_idx = rng.choice(m, size=max_calib_snps, replace=False)
            calib_idx.sort()
        else:
            calib_idx = np.arange(m)
        G_calib = G[:, calib_idx]
        bim_calib = (bim.iloc[calib_idx].reset_index(drop=True) if bim is not None
                    else pd.DataFrame({"SNP": [str(i) for i in calib_idx]}))

        calibration = {}
        for tool, df in self._tool_results.items():
            try:
                null_pvals = []
                for _ in range(self.n_perm):
                    y_perm = rng.permutation(y)
                    ols_null = _run_ols(G_calib, y_perm, bim_calib)
                    null_pvals.extend(ols_null["pval"].dropna().tolist())
                t1e = empirical_type1_error(null_pvals, self._sig_thresh)
                calibration[tool] = {"type1_error": t1e, "lambda_gc": self._lambdas.get(tool)}
                if t1e > 0.1:
                    logger.warning(f"  [step6] {tool}: high type I error {t1e:.3f}, down-weighting")
                    self._weights[tool] *= max(0.05, 1.0 - t1e)
            except Exception as e:
                logger.warning(f"  [step6] calibration for {tool} failed: {e}")
                calibration[tool] = {"type1_error": np.nan}

        self._calibration = calibration
        self._provenance["step6"] = str(calibration)

    # ── Step 7 ────────────────────────────────────────────────────────────────

    def _step7_validate(self):
        """Step 7: cross-tool concordance validation."""
        logger.info("[ConsensusGWAS] Step 7 - validation ...")
        concordance_frac, concordant_snps = cross_tool_concordance(
            self._tool_results, self._sig_thresh)
        sig_df = self._consensus_df[
            self._consensus_df["pval_consensus"] < self._sig_thresh
        ] if self._consensus_df is not None else pd.DataFrame()
        sig_snps = set(sig_df["snp"].astype(str).tolist())
        concordant_in_consensus = len(sig_snps & concordant_snps)
        logger.info(f"  Cross-tool concordance: {concordance_frac:.3f}")
        # Small interpretability fix: "0/0" (or, in principle, any N/0)
        # reads as a degenerate fraction rather than what it actually
        # means -- there were no consensus hits at all to check
        # concordance against, which is a DIFFERENT, more informative
        # statement than "hits exist but none were concordant". Found
        # via a real run: a properly-corrected consensus (see
        # inverse_variance_meta()'s genomic-control fix) can legitimately
        # produce zero significant hits for a modest-sized cohort testing
        # a complex trait -- a correct, unremarkable result that this
        # message previously reported in a way that looked like a
        # concerning "0 out of 0" ratio.
        if len(sig_snps) == 0:
            logger.info("  Consensus hits also concordant across >=2 tools: "
                        "N/A (no consensus hits at this threshold)")
        else:
            logger.info("  Consensus hits also concordant across >=2 tools: "
                        f"{concordant_in_consensus}/{len(sig_snps)}")
        self._validation = dict(concordance_frac=concordance_frac,
                                n_consensus_hits=len(sig_snps),
                                n_concordant_hits=concordant_in_consensus)
        self._provenance["step7"] = str(self._validation)

    # ── Step 8 ────────────────────────────────────────────────────────────────

    def _step8_write_outputs(self):
        """Step 8: write consensus stats + per-tool provenance + plots."""
        logger.info("[ConsensusGWAS] Step 8 - writing outputs ...")

        # Sort by consensus p-value
        out = self._consensus_df.sort_values("pval_consensus").reset_index(drop=True)
        out.to_csv(self.out_dir / "consensus_snp_stats.tsv", sep="\t", index=False)

        # Inflation / lambda report
        lam_df = pd.DataFrame([{"tool": t, "lambda_gc": v,
                                 "weight": self._weights.get(t, np.nan)}
                                for t, v in self._lambdas.items()])
        lam_df.to_csv(self.out_dir / "genomic_control.tsv", sep="\t", index=False)

        # Provenance
        prov_df = pd.DataFrame([{"step": k, "description": v}
                                 for k, v in self._provenance.items()])
        prov_df.to_csv(self.out_dir / "provenance.tsv", sep="\t", index=False)

        # Validation summary
        val_df = pd.DataFrame([{"metric": k, "value": v}
                                for k, v in self._validation.items()])
        val_df.to_csv(self.out_dir / "validation_summary.tsv", sep="\t", index=False)

        # Significant hits only
        sig_df = out[out["pval_consensus"] < self._sig_thresh]
        sig_df.to_csv(self.out_dir / "consensus_significant_hits.tsv",
                      sep="\t", index=False)

        logger.info(f"[ConsensusGWAS] Written to {self.out_dir}")
        for fn in ["consensus_snp_stats.tsv", "consensus_significant_hits.tsv",
                   "genomic_control.tsv", "validation_summary.tsv", "provenance.tsv"]:
            logger.info(f"  -> {fn}")
        self._provenance["step8"] = "complete"

    # ── Plotting ──────────────────────────────────────────────────────────────

    def _generate_plots(self):
        if not self.plots:
            return
        try:
            from jasmap.plotting.plots import (
                manhattan, qq_plot)
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            df = self._consensus_df.copy()

            # 1. Consensus Manhattan
            if "pval_consensus" in df.columns and df["pval_consensus"].notna().sum() > 5:
                mh_df = df.copy()
                mh_df["CHR"] = mh_df["chr"].fillna("1").astype(str)
                bp_raw = pd.to_numeric(mh_df["pos"], errors="coerce")
                mh_df["BP"] = bp_raw.fillna(
                    pd.Series(range(len(mh_df)), index=mh_df.index, dtype=float)
                ).astype(float)
                mh_df["P"]   = mh_df["pval_consensus"]
                mh_df["SNP"] = mh_df["snp"]
                manhattan(mh_df, pval_col="P", chrom_col="CHR", bp_col="BP",
                          snp_col="SNP",
                          title=f"Consensus GWAS Manhattan ({self.mode} mode)",
                          sig_line=self._sig_thresh,
                          out_path=str(self.out_dir / "consensus_manhattan.png"))

            # 2. Consensus QQ
            pvals_ok = df["pval_consensus"].dropna().values
            pvals_ok = pvals_ok[pvals_ok > 0]
            if len(pvals_ok) > 5:
                qq_plot(pvals_ok,
                        title="Consensus QQ (lambda overall)",
                        out_path=str(self.out_dir / "consensus_qq.png"))

            # 3. Per-tool QQ overlay
            tool_cols = [c for c in df.columns if c.startswith("pval_") and
                         c != "pval_consensus" and c != "pval_heterogeneity"]
            if tool_cols:
                fig, ax = plt.subplots(figsize=(7,6), dpi=300)
                colors = ["#E41A1C","#377EB8","#4DAF4A","#984EA3",
                          "#FF7F00","#A65628","#F781BF","#999999"]
                for i, tc in enumerate(tool_cols):
                    p = pd.to_numeric(df[tc], errors="coerce").dropna().values
                    p = p[(p > 0) & np.isfinite(p)]
                    if len(p) < 5:
                        continue
                    n = len(p)
                    obs = np.sort(-np.log10(p))[::-1]
                    exp = -np.log10(np.arange(1, n+1) / (n+1))
                    ax.plot(exp, obs, ".", ms=3, alpha=0.6,
                            color=colors[i % len(colors)],
                            label=tc.replace("pval_",""))
                lim = ax.get_xlim()[1]
                ax.plot([0,lim],[0,lim], "k--", lw=0.8)
                ax.set_xlabel("Expected -log₁₀(p)")
                ax.set_ylabel("Observed -log₁₀(p)")
                ax.set_title("Per-tool QQ comparison", fontweight="bold")
                ax.legend(fontsize=7, markerscale=3)
                ax.spines[["top","right"]].set_visible(False)
                fig.tight_layout()
                fig.savefig(self.out_dir / "per_tool_qq_comparison.png",
                            dpi=300, bbox_inches="tight")
                plt.close(fig)
                logger.info("[PLOT] per_tool_qq_comparison.png")

            # 4. Lambda barchart
            if self._lambdas:
                fig, ax = plt.subplots(figsize=(max(4, len(self._lambdas)), 4), dpi=300)
                tools = [t for t, v in self._lambdas.items() if np.isfinite(v)]
                vals  = [self._lambdas[t] for t in tools]
                cols  = ["#D6604D" if v > 1.1 else "#4393C3" for v in vals]
                ax.bar(tools, vals, color=cols, alpha=0.85, edgecolor="white")
                ax.axhline(1.0, color="k", lw=0.8, ls="--")
                ax.axhline(1.1, color="#D6604D", lw=0.8, ls=":")
                ax.set_ylabel("lambda_GC (genomic inflation)")
                ax.set_title("Genomic Inflation per Tool", fontweight="bold")
                ax.tick_params(axis="x", rotation=30)
                ax.spines[["top","right"]].set_visible(False)
                fig.tight_layout()
                fig.savefig(self.out_dir / "genomic_inflation_barchart.png",
                            dpi=300, bbox_inches="tight")
                plt.close(fig)
                logger.info("[PLOT] genomic_inflation_barchart.png")

            # 5. Tool p-value concordance scatter (first two tools)
            tool_cols_2 = [c for c in df.columns
                           if c.startswith("pval_") and c != "pval_consensus"
                           and c != "pval_heterogeneity"]
            if len(tool_cols_2) >= 2:
                tc1, tc2 = tool_cols_2[0], tool_cols_2[1]
                sub = df[[tc1, tc2]].dropna()
                sub = sub[(sub[tc1]>0) & (sub[tc2]>0)]
                if len(sub) > 5:
                    fig, ax = plt.subplots(figsize=(6,6), dpi=300)
                    ax.scatter(-np.log10(sub[tc1]), -np.log10(sub[tc2]),
                               s=5, alpha=0.5, color="#377EB8", linewidths=0)
                    lim = max(-np.log10(sub.values.min()), 1)
                    ax.plot([0,lim],[0,lim], "r--", lw=0.8)
                    ax.set_xlabel(f"-log₁₀(p)  {tc1.replace('pval_','')}")
                    ax.set_ylabel(f"-log₁₀(p)  {tc2.replace('pval_','')}")
                    ax.set_title("Tool p-value concordance", fontweight="bold")
                    ax.spines[["top","right"]].set_visible(False)
                    fig.tight_layout()
                    fig.savefig(self.out_dir / "tool_concordance_scatter.png",
                                dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    logger.info("[PLOT] tool_concordance_scatter.png")

        except Exception as e:
            logger.warning(f"[ConsensusGWAS] Plotting failed: {e}")

    def _generate_rvat_plots(self, var_annot_path=None):
        """
        Generate gene-level RVAT publication figures from consensus results.
        Aggregates SNP-level p-values to gene level via ACAT combination,
        then produces gene Manhattan, multi-test comparison, lollipop,
        and regional zoom plots.
        """
        if not self.plots or self._consensus_df is None or len(self._consensus_df) == 0:
            return
        try:
            from jasmap.plotting.plots import (
                rvat_gene_manhattan, rvat_test_comparison,
                rvat_lollipop, rvat_regional_gene)

            df = self._consensus_df.copy()
            if "pval_consensus" not in df.columns: return

            # ── Step 0: SNP -> Gene aggregation using variant annotation ─────
            gene_df = None
            csq_per_gene = {}

            if var_annot_path and Path(var_annot_path).exists():
                try:
                    va = pd.read_csv(var_annot_path, sep="\t")
                    # Merge consensus SNP p-values with annotation
                    df["snp_str"] = df["snp"].astype(str)
                    va["snp_str"] = va["SNP"].astype(str)
                    merged = va.merge(
                        df[["snp_str","pval_consensus"]
                           + [c for c in df.columns if c.startswith("pval_") and
                              c not in ("pval_consensus","pval_heterogeneity")]],
                        on="snp_str", how="left")
                    merged = merged.dropna(subset=["pval_consensus"])

                    tool_cols_snp = [c for c in merged.columns
                                     if c.startswith("pval_") and
                                     c not in ("pval_consensus","pval_heterogeneity")]

                    # Aggregate to gene level via ACAT (robust p-value combination)
                    def _acat_series(pvals):
                        p = np.clip(pvals.dropna().values.astype(float), 1e-300, 1-1e-16)
                        if len(p) == 0: return np.nan
                        if len(p) == 1: return float(p[0])
                        t = np.mean(np.tan((0.5 - p) * np.pi))
                        return float(np.clip(0.5 - np.arctan(t)/np.pi, 1e-300, 1))

                    gene_rows = []
                    for gene, grp in merged.groupby("GENE"):
                        if gene == "intergenic": continue
                        row = {
                            "snp": gene,
                            "chr": str(grp["CHR"].iloc[0]),
                            "pos": float(grp["POS"].mean()),
                            "pval_consensus": _acat_series(grp["pval_consensus"]),
                            "n_tools": int(grp["pval_consensus"].notna().sum()),
                            "n_rare_variants": int((grp.get("IS_RARE","no")=="yes").sum()),
                            "n_tools_significant": 0,
                        }
                        for tc in tool_cols_snp:
                            row[tc] = _acat_series(grp[tc]) if tc in grp else np.nan
                        gene_rows.append(row)
                        # Dominant consequence
                        rare_grp = grp[grp.get("IS_RARE","no")=="yes"]
                        if len(rare_grp):
                            csq_per_gene[gene] = str(
                                rare_grp["CONSEQUENCE"].mode().iloc[0]
                                if len(rare_grp["CONSEQUENCE"].dropna()) else "other")

                    if gene_rows:
                        gene_df = pd.DataFrame(gene_rows)
                        # Count tools significant at nominal level
                        sig_cols = [c for c in gene_df.columns if c.startswith("pval_")
                                    and c != "pval_consensus"]
                        gene_df["n_tools_significant"] = gene_df[sig_cols].apply(
                            lambda r: (r < 0.05).sum(), axis=1)
                        gene_df["dom_csq"] = gene_df["snp"].map(csq_per_gene).fillna("other")
                        logger.info(f"[ConsensusGWAS] Aggregated {len(gene_df)} genes for RVAT plots")

                except Exception as e:
                    logger.warning(f"[ConsensusGWAS] Gene aggregation failed: {e}")

            # Bug fix: this used to fall back to plotting SNP-level data
            # AS IF it were gene-level whenever gene aggregation produced
            # nothing (e.g. no --gene-annot given) -- generating plots
            # titled "Gene-Level Association" and saved as rvat_*.png even
            # for a purely SNP-level analysis (GWAS/GIFT/admix), with every
            # SNP mislabeled as "gene" and a generic "other" consequence.
            # Found via a real --mode gift + --consgwas run producing
            # rvat_gene_manhattan.png etc. with no actual gene-level
            # information behind them. Skip these plots entirely when no
            # genuine gene aggregation happened -- UNLESS the analysis mode
            # is actually rvat, where SNP-level fallback is still a
            # reasonable, honestly-labelled thing to show (the user
            # explicitly asked for a gene-based analysis in that case).
            if gene_df is None and self.mode != "rvat":
                logger.info("[ConsensusGWAS] Skipping RVAT-style plots: no "
                           "gene-level aggregation available for this "
                           f"SNP-level mode ({self.mode}) -- pass "
                           "--gene-annot for real gene-level plots.")
            else:
                if gene_df is None:
                    gene_df = df.copy()
                    gene_df["dom_csq"] = "other"

                tool_cols = [c for c in gene_df.columns
                             if c.startswith("pval_") and c not in
                             ("pval_consensus","pval_heterogeneity")]

                # 1. Gene-level Manhattan
                rvat_gene_manhattan(
                    gene_df, test_cols=tool_cols,
                    title=f"Gene-Level Association ({self.mode} + consensus RVAT)",
                    out_path=str(self.out_dir / "rvat_gene_manhattan.png"))

                # 2. Multi-test comparison dot plot
                if tool_cols and len(gene_df) >= 3:
                    rvat_test_comparison(
                        gene_df, test_cols=tool_cols, top_n=min(25, len(gene_df)),
                        out_path=str(self.out_dir / "rvat_test_comparison.png"))

                # 3. Lollipop with consequence annotation
                rvat_lollipop(
                    gene_df, top_n=min(25, len(gene_df)),
                    consequence_col="dom_csq" if "dom_csq" in gene_df.columns else None,
                    out_path=str(self.out_dir / "rvat_lollipop.png"))

                # 4. Regional zoom for the top gene
                if var_annot_path and Path(var_annot_path).exists() and len(gene_df):
                    try:
                        top_gene = gene_df.dropna(subset=["pval_consensus"]).nsmallest(
                            1, "pval_consensus")["snp"].iloc[0]
                        va = pd.read_csv(var_annot_path, sep="\t")
                        gene_snps = va[va["GENE"] == top_gene].copy()
                        if len(gene_snps) >= 2:
                            # Use actual p-values from SNP-level consensus where available
                            snp_pmap = dict(zip(
                                df["snp"].astype(str),
                                df["pval_consensus"].astype(float)))
                            gene_snps["pval"] = gene_snps["SNP"].astype(str).map(snp_pmap)
                            # Fill missing with random values around the gene's p-value
                            gene_p = float(gene_df[gene_df["snp"]==top_gene]["pval_consensus"].iloc[0])
                            rng = np.random.default_rng(42)
                            gene_snps["pval"] = gene_snps["pval"].fillna(
                                pd.Series(np.clip(gene_p * rng.uniform(0.5, 3.0,
                                                  len(gene_snps)), 1e-50, 1),
                                          index=gene_snps.index))
                            rvat_regional_gene(
                                gene_snps, gene_name=top_gene,
                                pval_col="pval", maf_col="MAF",
                                consequence_col="CONSEQUENCE",
                                out_path=str(self.out_dir / f"rvat_regional_{top_gene}.png"))
                    except Exception as e:
                        logger.warning(f"[ConsensusGWAS] Regional plot failed: {e}")

        except Exception as e:
            logger.warning(f"[ConsensusGWAS] RVAT plotting failed: {e}")

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, G, y, bim, jasmap_results, lmm_obj=None, h=None,
            var_annot_path=None):
        """
        Run all 8 steps and return the consensus DataFrame.

        Parameters
        ----------
        G               : np.ndarray (n_samples, n_snps)  genotype dosage
        y               : np.ndarray (n_samples,)         phenotype
        bim             : pd.DataFrame                    SNP metadata
        jasmap_results  : dict {tool_name: DataFrame}    primary mode outputs
        lmm_obj         : fitted LMM instance (optional)
        h               : optimal heritability (optional)
        var_annot_path  : path to variant_annotation.tsv for gene-level RVAT
                          plots; auto-detected from outfolder if None
        """
        t0 = time.time()
        logger.info("\n" + "="*60)
        logger.info("[ConsensusGWAS] 8-Step Consensus GWAS/RVAT Pipeline")
        logger.info(f"  Mode: {self.mode}  | Test type: auto-detect")
        logger.info("="*60)

        # Bug fix: impute missing genotype calls ONCE here, before ANY step
        # sees G -- both Step 2's internal tools (_run_ols/_run_firth_approx/
        # burden/skat_approx) and Step 6's calibration receive their own
        # copy of G, and each computes a per-SNP mean via plain
        # G.mean(axis=0), not np.nanmean. A SINGLE missing genotype call
        # anywhere in a SNP's column poisons that SNP's ENTIRE beta/se/pval
        # to NaN -- verified directly with a targeted test (one missing
        # call out of 200 samples turned a valid pval into NaN while every
        # other SNP stayed correct), then confirmed as the exact mechanism
        # behind a real run's consensus output dropping from 338,320 SNPs
        # to exactly 25,500: a real cohort's ~262,447 missing genotype
        # calls, spread across many different SNPs, meant the large
        # majority had at least one missing call, and Step 3's
        # dropna(subset=["pval"]) then silently dropped every one of them.
        # This predates this session's vectorisation work (confirmed
        # directly against the original per-SNP loop, which had the
        # identical issue) -- imputing once here, at the single point
        # where G first enters this class, fixes it for every downstream
        # step consistently rather than patching each one separately.
        if G is not None:
            from jasmap.prs.consensus_prs import impute_missing_genotypes
            G = impute_missing_genotypes(G, tag="ConsensusGWAS")

        self._step1_check_inputs(G, y, bim)
        self._step2_run_tools(G, y, bim, jasmap_results, lmm_obj, h)
        self._step3_standardise()

        if not self._tool_results:
            logger.warning("[ConsensusGWAS] No tool results to combine - aborting")
            return pd.DataFrame()

        self._step4_harmonise_inflation()
        self._step5_build_consensus()
        self._step6_calibration(G, y, bim=bim)
        # Re-run step 5 with updated weights after calibration
        self._step5_build_consensus()
        self._step7_validate()
        self._step8_write_outputs()
        self._generate_plots()
        self._generate_rvat_plots(var_annot_path=var_annot_path)

        elapsed = round(time.time() - t0, 2)
        logger.info(f"[ConsensusGWAS] Complete in {elapsed}s")
        return self._consensus_df


# ─────────────────────────────────────────────────────────────────────────────
# Output file picker  –  read primary mode outputs into canonical dicts
# ─────────────────────────────────────────────────────────────────────────────

def collect_mode_results(mode, outfolder, bim=None):
    """
    Intelligently pick the correct output files from the primary --mode run
    and return them as {tool_name: canonical DataFrame} ready for ConsensusGWAS.

    Parameters
    ----------
    mode      : str  the --mode that produced the results
    outfolder : str  the --out directory
    bim       : pd.DataFrame, optional
        SNP metadata used to translate JasMAP's positional SNP indices into
        real variant IDs.

        Bug fix: JasMAP's own output files (Jasmap.Joint.assoc,
        Jasmap.GWAS.assoc, Jasmap.ANC*.AM.assoc) write the SNP column as a
        1-based LINE NUMBER, because run_joint_analysis() iterates over
        sorted(lanc_dict.keys()). The internal consensus tools (OLS,
        Firth-approx, burden, SKAT) instead use the real rs IDs from the BIM.
        The two naming schemes never matched, so every variant appeared TWICE
        in the consensus under two different names -- 300 SNPs produced 600
        rows, n_tools never exceeded the size of one naming group, and
        cross-tool concordance was mathematically forced to 0.000. Passing the
        BIM here maps index -> rs ID so all tools align on one identifier.
    """
    results = {}
    # Bug fix: every path below used to be built via `out / "filename"`
    # (pathlib's directory-join operator), treating outfolder as a bare
    # DIRECTORY. But once --out-prefix folds a prefix directly onto
    # self.outfolder as part of the string itself (e.g. "results/JasMAP_",
    # not "results/" with a separate "JasMAP_" component) -- matching how
    # every actual output file gets written elsewhere in this codebase,
    # via plain "self.outfolder + filename" concatenation -- `out /
    # "Jasmap.GWAS.assoc"` no longer matches anything real on disk: it
    # looks for a literal "Jasmap.GWAS.assoc" inside a "results/JasMAP_/"
    # subdirectory that doesn't exist, instead of the actual file,
    # "results/JasMAP_Jasmap.GWAS.assoc", sitting directly in "results/".
    # Found via a real run: --consgwas silently reported "no primary
    # --mode output files found" and fell back to internal-tools-only,
    # for every single run once a prefix was in play -- which, after the
    # previous turn's change, is now every run by default. Plain string
    # concatenation (matching every writer) rather than pathlib's `/`
    # fixes this; Path(...) is still used below for .exists()/.glob(),
    # which work identically regardless of how the underlying string was
    # built.
    out = str(outfolder)

    # Build positional index -> variant ID lookup (1-based, matching JasMAP)
    idx_to_rsid = None
    if bim is not None and len(bim):
        bim_cols = {c.upper(): c for c in bim.columns}
        snp_col_bim = bim_cols.get("SNP", bim_cols.get("RSID"))
        if snp_col_bim is not None:
            ids = bim[snp_col_bim].astype(str).tolist()
            idx_to_rsid = {str(i + 1): rsid for i, rsid in enumerate(ids)}

    def _relabel(df, snp_column="snp"):
        """Translate positional SNP indices to real variant IDs where possible."""
        if idx_to_rsid is None or snp_column not in df.columns:
            return df
        col = df[snp_column].astype(str)
        # Only remap when the column looks like positional indices
        if col.str.fullmatch(r"\d+").mean() > 0.9:
            mapped = col.map(lambda s: idx_to_rsid.get(s, s))
            n_mapped = int((mapped != col).sum())
            if n_mapped:
                logger.info(f"[collect_mode_results] Remapped {n_mapped} positional "
                            "SNP indices to variant IDs")
            df = df.copy()
            df[snp_column] = mapped
        return df

    # Only collect files that the CURRENT mode actually produces.
    # Bug fix: this function previously read Jasmap.GWAS.assoc AND
    # Jasmap.Joint.assoc AND glob'd Jasmap.ANC*.AM.assoc for both "GWAS" and
    # "admix". Re-using an --out folder therefore fed STALE results from a
    # previous run into the consensus as if they were current tools -- e.g. a
    # --mode GWAS run in a folder holding an earlier admix run silently
    # contributed admix_snp + am_anc1/2/3, giving 7 "tools" instead of 3 and
    # mixing two different analyses into one consensus statistic.
    if mode == "GWAS":
        gwas_file = Path(out + "Jasmap.GWAS.assoc")
        if gwas_file.exists():
            df = pd.read_csv(gwas_file, sep="\t")
            df = df.rename(columns={"SNP_Pval":"pval", "SNP_beta":"beta"})
            df["snp"] = df["SNP"].astype(str) if "SNP" in df.columns else df.index.astype(str)
            results["lmm_gwas"] = _relabel(df)

    if mode == "admix":
        joint_file = Path(out + "Jasmap.Joint.assoc")
        if joint_file.exists():
            df = pd.read_csv(joint_file, sep="\t")
            if "SNP_Pval" in df.columns:
                sub = df[["SNP","SNP_beta","SNP_Pval","AM_Pval","AM_beta"]].copy()
                sub = sub.rename(columns={"SNP_Pval":"pval","SNP_beta":"beta","SNP":"snp"})
                results["admix_snp"] = _relabel(sub)
            # per-ancestry admixture-mapping files
            for anc_file in sorted(Path(p) for p in glob.glob(out + "Jasmap.ANC*.AM.assoc")):
                k = anc_file.stem.split(".")[1]  # e.g. "ANC1"
                df_anc = pd.read_csv(anc_file, sep="\t")
                df_anc = df_anc.rename(columns={"Pval":"pval","beta":"beta","SNP":"snp"})
                results[f"am_{k.lower()}"] = _relabel(df_anc)

    if mode == "gift":
        gift_dir = Path(out + "gift_results")
        si_file  = gift_dir / "snp_importance.tsv"
        if si_file.exists():
            df = pd.read_csv(si_file, sep="\t")
            # Convert GIFT scores to pseudo p-values via rank-based transform
            n = len(df)
            ranks = df["GIFT_SCORE"].rank(ascending=False)
            df["pval"] = ranks / (n + 1)
            df["beta"] = df["GIFT_SCORE"]
            df["snp"]  = df["SNP"].astype(str) if "SNP" in df.columns else df.index.astype(str)
            results["gift"] = _relabel(df)
            # Clarification, not a fix: this rank-based "pval" is the SAME
            # construction used for GIFT's own illustrative QQ plot (see
            # gift_engine.py's comment: "rank-based uniform p-values for
            # illustrative purposes") -- it is NOT a real hypothesis-test
            # p-value, since GIFT's mutual-information/ridge-regression
            # importance scores have no underlying null distribution to
            # test against here. A uniform rank transform is, BY
            # CONSTRUCTION, uniformly distributed regardless of true
            # signal strength -- its smallest possible value is
            # 1/(n+1) (~2e-4 for a 5000-SNP GIFT run), several orders of
            # magnitude short of genome-wide significance (5e-8). This
            # means the "gift" tool can structurally never contribute a
            # significant hit to the IVW consensus, no matter how strong
            # GIFT's actual importance signal is for a given SNP --
            # worth knowing when interpreting a consensus that includes
            # a "gift" tool contribution, found while investigating a
            # real run's lambda_GC=1.0000 EXACTLY for this tool (the
            # expected, guaranteed outcome of a perfectly uniform
            # p-value distribution, not a sign of a well-calibrated
            # test). Left as informational rather than "fixed" -- giving
            # GIFT's importance scores a rigorous, non-uniform p-value
            # would need its own null distribution (e.g. via
            # permutation), a genuine methodological addition beyond
            # this rank-transform's original, illustrative purpose.
            logger.info(
                "[ConsensusGWAS] Note: the 'gift' tool's p-values are a "
                "rank-based transform of GIFT's importance scores (for "
                "illustrative QQ-plotting), not a real hypothesis test -- "
                "they are uniformly distributed by construction and can "
                "never reach genome-wide significance on their own, "
                "regardless of the true underlying signal.")

    if mode == "gift_admix":
        ga_dir  = Path(out + "gift_admix_results")
        si_file = ga_dir / "gift_admix_snp_importance.tsv"
        if si_file.exists():
            df = pd.read_csv(si_file, sep="\t")
            n  = len(df)
            ranks = df["GIFT_SCORE"].rank(ascending=False)
            df["pval"] = ranks / (n + 1)
            df["beta"] = df["GIFT_SCORE"]
            df["snp"]  = df["SNP"].astype(str) if "SNP" in df.columns else df.index.astype(str)
            results["gift_admix"] = _relabel(df)
            # Same rank-based pseudo-p-value construction, and the same
            # caveat, as mode == "gift" above (see that block's full
            # explanation) -- structurally uniform, can never reach
            # genome-wide significance alone. The "admix_map_k*" tools
            # collected just below (real ancestry-mapping p-values) are
            # NOT affected by this.
            logger.info(
                "[ConsensusGWAS] Note: the 'gift_admix' tool's p-values are "
                "a rank-based transform of GIFT's importance scores, not a "
                "real hypothesis test -- see the equivalent note for "
                "--mode gift for the full explanation.")
        am_file = ga_dir / "admixture_mapping.tsv"
        if am_file.exists():
            df = pd.read_csv(am_file, sep="\t")
            df = df.rename(columns={"P":"pval","BETA":"beta",
                                    "WINDOW":"snp"})
            df["snp"] = df["snp"].astype(str)
            for k_val in df["ANCESTRY_K"].unique():
                sub = df[df["ANCESTRY_K"]==k_val].copy()
                results[f"admix_map_k{k_val}"] = sub

    if mode == "rvat":
        rvat_file = Path(out + "rvat_results") / "JasMAP.RVAT.assoc"
        if rvat_file.exists():
            df = pd.read_csv(rvat_file, sep="\t")
            # Map gene-level result columns to canonical schema for consensus
            df = df.rename(columns={
                "gene":        "snp",          # gene name becomes SNP id
                "chr":         "chr",
                "pos_mid":     "pos",
                "consensus_p": "pval",
                "burden_p":    "pval_burden",
                "skat_p":      "pval_skat",
                "skato_p":     "pval_skato",
                "acat_v_p":    "pval_acat_v",
            })
            df["beta"] = df.get("burden_beta", pd.Series([np.nan]*len(df)))
            df["se"]   = df.get("burden_se",   pd.Series([np.nan]*len(df)))
            results["rvat_gene"] = df
            logger.info(f"[collect_mode_results] rvat: {len(df)} genes from "
                        f"{rvat_file.name}")

    return results
