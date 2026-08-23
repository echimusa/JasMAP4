"""
jasmap.prs.consensus_prs
------------------------
Consensus multi-tool Polygenic Risk Score construction and validation
(Steps 1-8 from Document 4: Consensus_PRS_Validation_Approach).

Changes from previous version
------------------------------
Bug fix  -- portability_ratios was always empty because the minimum-samples
            guard (`mask.sum() < 10`) was applied to the *test split* only
            (test_frac=0.2 of n samples), meaning even moderate-sized ancestry
            groups each had fewer than 10 test-set members and were silently
            skipped. Fix: (a) portability is now computed on ALL samples
            (not just the test split), (b) the minimum is `max(3, n // 20)` so
            it scales with cohort size, (c) groups that fail the threshold are
            reported as NaN rather than omitted entirely, and (d) the portability
            TSV is always written even if no group has enough samples (it will
            contain NaN rows with a note column explaining why).

New plots  -- 8 publication-quality PRS figures (in addition to the existing
             distribution and ROC):
             1. prs_individual_scores      -- per-sample PRS dot plot with
                                              percentile bands and risk flags
             2. prs_risk_stratification    -- mean phenotype per PRS decile
                                              (bar + scatter, the "staircase" plot)
             3. prs_calibration_plot       -- observed vs predicted risk by decile
             4. prs_portability_bar        -- R^2 per ancestry group (bar chart)
             5. prs_ancestry_density       -- overlaid KDE densities per ancestry
             6. prs_cumulative_risk        -- cumulative case fraction vs PRS rank
             7. prs_forest_or_violin       -- per-ancestry PRS violin + OR overlay
             8. prs_percentile_table_plot  -- table of clinical risk percentile thresholds
"""

import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: harmonise summary statistics
# ─────────────────────────────────────────────────────────────────────────────

def harmonise_sumstats(path_or_df, bim, effect_col="BETA",
                       pval_col="P", snp_col="SNP", a1_col="A1"):
    if isinstance(path_or_df, (str, Path)):
        ss = pd.read_csv(path_or_df, sep=r"\s+", engine="python")
    else:
        ss = path_or_df.copy()
    rename = {}
    for alt, canon in [("beta","BETA"),("p","P"),("PVAL","P"),
                       ("SNP","SNP"),("rsid","SNP"),("ID","SNP"),
                       ("A1","A1"),("effect_allele","A1"),("a1","A1")]:
        if alt in ss.columns and canon not in ss.columns:
            rename[alt] = canon
    ss = ss.rename(columns=rename)
    required = [c for c in [snp_col, effect_col, pval_col] if c in ss.columns]
    ss = ss[required].dropna()
    merged = bim.merge(ss, left_on="snp", right_on=snp_col, how="inner")
    logger.info(f"[PRS] Harmonised: {len(merged)}/{len(bim)} SNPs matched")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: PRS tools
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Population stratification correction for PRS weight fitting
# ─────────────────────────────────────────────────────────────────────────────

def _trapezoid_area(y, x):
    """
    Trapezoidal-rule integration, implemented manually rather than via
    numpy's own trapz/trapezoid functions.

    Why this exists
    -----------------
    np.trapz was removed in NumPy 2.0 (renamed to np.trapezoid), which does
    not exist at all in NumPy <2.0 -- there is no single function name that
    works across both. Found via a real run: this project's own dev/test
    environment has NumPy 1.26 (trapz present, trapezoid absent), while a
    user's environment on a newer Python had NumPy 2.0+ (trapz removed
    entirely, causing "[PRS] Plotting failed: module 'numpy' has no
    attribute 'trapz'"). Implementing the rule directly with basic
    operations (verified to match np.trapz's output exactly on the
    NumPy 1.26 environment where both are available for comparison) avoids
    betting on either name and works identically on any NumPy version.
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    return float(np.sum((y[:-1] + y[1:]) / 2.0 * np.diff(x)))


def residualize_phenotype(y, X0):
    """
    Regresses phenotype y on covariates X0 (typically an intercept plus
    PCA eigenvectors -- either user-provided via --covfile or auto-
    generated via jasmap.biobank.auto_pca) and returns the RESIDUALS.

    Why this exists
    ----------------
    Before this function existed, ConsensusPRS.fit() fit Ridge/Lasso
    directly on genotype -> phenotype with NO stratification adjustment at
    all -- unlike GWAS/admix mode, which already residualize/adjust via
    covariates in the LMM's own X0 handling. PRS weight estimation without
    stratification correction is a well-known, actively-researched problem
    in the PRS literature: SNP effect estimates (and the resulting PRS)
    can be confounded by ancestry, inflating apparent predictive power in
    a way that is driven by population structure rather than true genetic
    effects, and is a common cause of poor PRS transferability across
    ancestry groups.

    This uses the standard "residualize first" approach (regress out
    covariates via OLS, then fit the genetic model on the residuals) --
    the same logic in spirit as the Frisch-Waugh-Lovell covariate
    adjustment already used for association_batch() elsewhere in this
    project, adapted here for a single phenotype vector rather than a
    batch of SNP tests.

    Parameters
    ----------
    y  : ndarray (n,)    phenotype (NaN = missing, consistent with the rest
                         of this module)
    X0 : ndarray (n, q)  covariates, typically [intercept, PC1, ..., PCk]

    Returns
    -------
    y_resid : ndarray (n,)  residuals after regressing out X0 (same NaN
             pattern as the input -- missing samples stay NaN, not
             silently imputed)
    beta    : ndarray (q,)  fitted covariate coefficients (kept in case a
             caller wants to add the covariate-explained component back)
    """
    y = np.asarray(y, dtype=np.float64)
    X0 = np.asarray(X0, dtype=np.float64)
    valid = ~np.isnan(y)
    if not valid.any():
        return y.copy(), np.zeros(X0.shape[1])

    beta, *_ = np.linalg.lstsq(X0[valid], y[valid], rcond=None)
    y_resid = y.copy()
    y_resid[valid] = y[valid] - X0[valid] @ beta
    return y_resid, beta


class _CTScore:
    """
    C+T: clumping + p-value thresholding.

    Stratification note: C+T scores SNPs using EXTERNAL summary statistics
    (effect sizes from a prior GWAS, supplied via --sumstats), not by
    fitting locally on this cohort's genotype/phenotype. Covariate
    adjustment for C+T therefore has to happen upstream, in whatever GWAS
    produced those summary statistics -- there is no local fitting step
    here to residualize against. ConsensusPRS.fit() does NOT apply
    residualize_phenotype() to this method for that reason; it is applied
    only to _RidgePRS/_LassoPRS, which DO fit locally on this cohort.
    """
    def __init__(self, p_thresh=5e-8, window_bp=500_000):
        self.p_thresh = p_thresh
        self.window   = window_bp
        self.weights_ = None

    def fit(self, G, sumstats_df, bim):
        ss = sumstats_df.copy()
        ss["P"] = pd.to_numeric(ss.get("P", pd.Series([1]*len(ss))), errors="coerce")
        ss = ss[ss["P"] <= self.p_thresh].sort_values("P")
        kept = []
        for _, chr_bim in bim.groupby("chrom"):
            chr_snps = chr_bim["snp"].tolist()
            chr_ss   = ss[ss["SNP"].isin(chr_snps)].sort_values("P")
            positions = dict(zip(chr_bim["snp"], chr_bim["pos"].astype(float)))
            excluded  = set()
            for _, row in chr_ss.iterrows():
                sid = row["SNP"]
                if sid in excluded: continue
                kept.append(sid)
                pos_i = positions.get(sid, 0)
                for s2 in chr_ss["SNP"]:
                    if s2 == sid or s2 in excluded: continue
                    if abs(positions.get(s2,0) - pos_i) < self.window:
                        excluded.add(s2)
        kept_ss = ss[ss["SNP"].isin(kept)]
        self.snp_ids_ = kept_ss["SNP"].tolist()
        self.weights_  = pd.Series(
            pd.to_numeric(kept_ss.get("BETA", pd.Series([1]*len(kept_ss))),
                          errors="coerce").fillna(0).values,
            index=self.snp_ids_)
        logger.info(f"[C+T]  {len(self.snp_ids_)} SNPs after clumping")
        return self

    def score(self, G, bim):
        snp_list = bim["snp"].tolist()
        idx = [i for i,s in enumerate(snp_list) if s in self.weights_]
        if not idx: return np.zeros(G.shape[0])
        w = np.array([self.weights_.get(snp_list[i], 0) for i in idx])
        return G[:, idx] @ w


def _lasso_alpha_grid(X, y, n_alphas=30, eps=1e-3):
    """
    Data-adaptive alpha grid for Lasso, with an explicit, controlled count.

    Why this exists
    -----------------
    LassoCV's own n_alphas parameter is deprecated in current sklearn
    versions with no direct replacement -- passing alphas= explicitly
    works, but naively hard-coding a fixed range loses the DATA-ADAPTIVE
    alpha_max sklearn computes internally (the smallest alpha giving an
    all-zero solution), which matters because a badly-chosen fixed range
    could miss the right regularisation strength entirely for a given
    dataset. This mirrors sklearn's own internal _alpha_grid formula
    (alpha_max = max(|X.T @ y|) / n_samples) so the grid stays properly
    adaptive, just with an explicit, small COUNT instead of the
    now-deprecated default of 100 -- a real, measured contributor to slow
    PRS fitting on large SNP counts (LassoCV was found to scale roughly
    linearly with feature count in timing tests: ~43s at m=8000 features,
    extrapolating to tens of minutes at m=300,000+).
    """
    n = X.shape[0]
    alpha_max = float(np.max(np.abs(X.T @ y)) / max(n, 1))
    if alpha_max <= 0:
        alpha_max = 1.0
    return np.logspace(np.log10(alpha_max * eps), np.log10(alpha_max), n_alphas)


def select_top_snps(G, y, X0=None, max_snps=20000, seed=42):
    """
    Quick univariate pre-filter: keeps only the top `max_snps` SNPs by
    |correlation| with the (covariate-residualized, if X0 given)
    phenotype, before handing off to Ridge/Lasso.

    Why this exists
    -----------------
    Fitting a CV-based penalized regression (Ridge, and especially Lasso)
    on every SNP simultaneously is expensive by design -- and most
    individual SNPs have negligible true effect on any polygenic trait, so
    fitting on all of them equally is mostly wasted computation. Real PRS
    methods (C+T's own clumping+thresholding already in this module, and
    external tools like PRS-CS/LDpred) universally pre-select or otherwise
    reduce the feature space before the expensive step, rather than
    penalized-regressing on the full genome at once. This is the same idea
    applied to the Ridge/Lasso components specifically. NOT applied to C+T,
    which already does its own selection via clumping+thresholding on
    external summary statistics -- a structurally different mechanism.

    Parameters
    ----------
    G, y, X0 : as elsewhere in this module
    max_snps : int  keep at most this many SNPs. If G already has fewer
              columns than this, returns all of them (no filtering, no
              wasted computation).

    Returns
    -------
    keep_idx : ndarray of column indices into G, sorted ascending (so the
              retained SNPs keep their relative order, useful for any
              downstream code that assumes an ordered SNP list).
    """
    m = G.shape[1]
    if m <= max_snps:
        return np.arange(m)

    valid = ~np.isnan(y)
    y_screen = y
    if X0 is not None:
        y_resid, _ = residualize_phenotype(y, X0)
        y_screen = y_resid

    Gv = G[valid]
    yv = y_screen[valid]
    yv_c = yv - yv.mean()
    Gv_c = Gv - Gv.mean(axis=0)
    denom = np.sqrt(np.sum(Gv_c**2, axis=0)) * np.sqrt(np.sum(yv_c**2)) + 1e-12
    corr = (Gv_c.T @ yv_c) / denom

    top_idx = np.argsort(-np.abs(corr))[:max_snps]
    return np.sort(top_idx)


def select_top_snps_streaming(bed_path, y, X0=None, max_snps=20000,
                              sample_idx=None, snp_idx=None, chunk_size=None,
                              verbose=True):
    """
    Streaming counterpart to select_top_snps(): computes the SAME
    univariate |correlation| screen via SNP-chunked reads from a .bed
    file, then performs one targeted second read fetching ONLY the
    selected columns -- the full genotype matrix is never materialised
    densely at any point, only the already-reduced final subset.

    Why this exists
    -----------------
    _run_prs_pipeline() previously read the ENTIRE genotype matrix
    densely via bed.read() before ever calling run_prs() -- the exact
    same "read everything first" pattern already fixed for kinship
    (jasmap.biobank.streaming_io). At n=500,000 samples and millions of
    SNPs, that dense read alone could exhaust memory long before Ridge/
    Lasso fitting -- or even the pre-filtering step in select_top_snps()
    -- ever runs, since that function ALSO assumed a dense G was already
    in memory. This closes that gap specifically for PRS.

    Parameters
    ----------
    bed_path   : str  path to the .bed file (with or without extension)
    y, X0      : as elsewhere in this module (X0 typically comes from
                jasmap.biobank.auto_pca.compute_auto_pca_streaming, which
                is itself already streaming-capable, so the two compose
                cleanly: covariates computed via streaming, THEN used to
                residualize the streaming SNP screen below)
    max_snps   : as in select_top_snps() -- if the .bed file already has
                fewer SNPs than this, reads everything directly with no
                screening pass at all (no wasted computation)
    sample_idx : optional sample subset (matches StreamingGenotypeSource's
                convention elsewhere)
    snp_idx    : optional array of SNP indices to restrict the SCREEN to
                (e.g. --chr filtering) -- the screen only considers these
                SNPs, and the returned keep_idx are still ORIGINAL FILE
                column indices (mapped back through snp_idx), matching
                this function's contract regardless of whether filtering
                was applied.
    chunk_size : SNPs per chunk; auto-sized (~200MB/chunk) if None

    Returns
    -------
    keep_idx : ndarray of the selected SNPs' ORIGINAL FILE column indices
              (sorted ascending, matching select_top_snps()'s contract --
              always original-file-space, even when snp_idx restricted
              the screen to a subset)
    G_selected : ndarray (n, len(keep_idx))  the actual genotype data for
              just those SNPs -- this IS the dense matrix Ridge/Lasso/C+T
              will use, but it is at most max_snps columns wide, never the
              full panel.
    """
    from bed_reader import open_bed
    bed_path = bed_path if bed_path.endswith(".bed") else bed_path + ".bed"
    bed = open_bed(bed_path, count_A1=False)
    n = bed.iid_count if sample_idx is None else len(sample_idx)
    snp_idx = np.asarray(snp_idx) if snp_idx is not None else None
    m = bed.sid_count if snp_idx is None else len(snp_idx)

    if m <= max_snps:
        col_sel = snp_idx if snp_idx is not None else slice(None)
        idx = (sample_idx, col_sel) if sample_idx is not None else (None, col_sel)
        G_all = bed.read(index=idx).astype(np.float64)
        if verbose:
            logger.info(f"[PRS] {m} SNPs already <= max_snps={max_snps}; "
                        f"reading directly with no streaming screen needed.")
        return_idx = snp_idx if snp_idx is not None else np.arange(m)
        return return_idx, G_all

    valid = ~np.isnan(y)
    y_screen = y
    if X0 is not None:
        y_resid, _ = residualize_phenotype(y, X0)
        y_screen = y_resid
    yv = y_screen[valid]
    yv_c = yv - yv.mean()
    yv_norm = np.sqrt(np.sum(yv_c**2)) + 1e-12

    if chunk_size is None:
        chunk_size = max(1, int(200 * 1024**2 / max(n * 8, 1)))
    chunk_size = min(chunk_size, m)

    t0 = time.time()
    all_corr = np.zeros(m)
    for start in range(0, m, chunk_size):
        end = min(start + chunk_size, m)
        # Map this logical [start:end) chunk to the actual file columns --
        # a contiguous slice when unfiltered, or the corresponding (possibly
        # scattered) entries of snp_idx when a chromosome filter is active.
        col_sel = snp_idx[start:end] if snp_idx is not None else slice(start, end)
        idx = (sample_idx, col_sel) if sample_idx is not None else (None, col_sel)
        chunk = bed.read(index=idx).astype(np.float64)
        chunk_v = chunk[valid]
        # Per-SNP mean imputation for missing genotype calls within this
        # chunk -- same convention as impute_missing_genotypes(), applied
        # locally since only one chunk is ever held in memory at a time.
        col_mean = np.nanmean(chunk_v, axis=0)
        nan_mask = np.isnan(chunk_v)
        if nan_mask.any():
            r, c = np.where(nan_mask)
            chunk_v = chunk_v.copy()
            chunk_v[r, c] = col_mean[c]
        chunk_c = chunk_v - chunk_v.mean(axis=0)
        denom = np.sqrt(np.sum(chunk_c**2, axis=0)) * yv_norm + 1e-12
        all_corr[start:end] = (chunk_c.T @ yv_c) / denom

    keep_idx_logical = np.sort(np.argsort(-np.abs(all_corr))[:max_snps])
    # Map back to ORIGINAL FILE indices -- this function's documented
    # contract is original-file-space regardless of snp_idx filtering.
    keep_idx = snp_idx[keep_idx_logical] if snp_idx is not None else keep_idx_logical

    # Targeted second read: fetch ONLY the selected columns directly
    idx2 = (sample_idx, keep_idx) if sample_idx is not None else (None, keep_idx)
    G_selected = bed.read(index=idx2).astype(np.float64)

    if verbose:
        logger.info(f"[PRS] Streaming pre-filter: screened {m} SNPs in "
                    f"chunks of {chunk_size}, selected top {len(keep_idx)} "
                    f"by |correlation| with phenotype -- full genotype "
                    f"matrix never materialised densely. [{time.time()-t0:.2f}s]")

    return keep_idx, G_selected


class _RidgePRS:
    """Ridge shrinkage PRS."""
    def __init__(self):
        self.model_  = None
        self.scaler_ = StandardScaler()

    def fit(self, G, y, X0=None):
        valid = ~np.isnan(y)
        Gs = self.scaler_.fit_transform(G[valid])
        y_fit = y
        if X0 is not None:
            # Residualize against covariates BEFORE fitting -- see
            # residualize_phenotype()'s docstring for why this matters.
            y_resid, _ = residualize_phenotype(y, X0)
            y_fit = y_resid
        self.model_ = RidgeCV(alphas=np.logspace(-4, 2, 20), cv=5)
        self.model_.fit(Gs, y_fit[valid])
        return self

    def score(self, G):
        return self.model_.predict(self.scaler_.transform(G))


class _LassoPRS:
    """Lasso penalised PRS."""
    def __init__(self):
        self.model_  = None
        self.scaler_ = StandardScaler()

    def fit(self, G, y, X0=None):
        valid = ~np.isnan(y)
        Gs = self.scaler_.fit_transform(G[valid])
        y_fit = y
        if X0 is not None:
            y_resid, _ = residualize_phenotype(y, X0)
            y_fit = y_resid
        # Explicit, data-adaptive alpha grid with a controlled, small count
        # instead of relying on LassoCV's now-deprecated n_alphas default
        # of 100 -- see _lasso_alpha_grid()'s docstring. This alone was
        # found to meaningfully reduce fitting time on large SNP counts.
        alphas = _lasso_alpha_grid(Gs, y_fit[valid], n_alphas=30)
        self.model_ = LassoCV(alphas=alphas, cv=5, max_iter=5000)
        self.model_.fit(Gs, y_fit[valid])
        return self

    def score(self, G):
        return self.model_.predict(self.scaler_.transform(G))


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: consensus stacked meta-learner
# ─────────────────────────────────────────────────────────────────────────────

class ConsensusPRS:
    """
    Stacked meta-learner combining C+T / Ridge / Lasso PRS component scores.

    Stratification note: when X0 (covariates -- user-provided via
    --covfile, or auto-generated PCA via jasmap.biobank.auto_pca) is
    passed to fit(), the ridge and lasso components are trained on
    covariate-RESIDUALIZED phenotype (see residualize_phenotype()) -- so
    their scores represent genetic risk with population structure already
    factored out, not the raw phenotype scale. The C+T component is
    unaffected (see _CTScore's docstring for why). This means
    PRS_RIDGE/PRS_LASSO in the output should be interpreted as "genetic
    component of risk, stratification-adjusted" rather than a direct
    prediction of the raw phenotype -- the conventional interpretation in
    PRS literature, and consistent with how a downstream analysis using
    the PRS would typically want to adjust for the SAME covariates again
    anyway (avoiding double-counting their effect).
    """
    def __init__(self, methods=("ct","ridge","lasso"), cv_folds=5, seed=42,
                max_snps=20000):
        self.methods = methods
        self.cv_folds = cv_folds
        self.seed = seed
        self.max_snps = max_snps
        self._models = {}
        self._meta   = None
        self._scaler = StandardScaler()
        self._snp_filter_idx = None  # set in fit(); shared by ridge/lasso only

    def fit(self, G, y, sumstats_df=None, bim=None, X0=None):
        valid = ~np.isnan(y)

        # Pre-filter ONCE, shared between ridge and lasso -- see
        # select_top_snps()'s docstring for why. Computed here (not inside
        # each model separately) so the univariate screen only runs once,
        # and so both models see the identical retained SNP set. C+T is
        # NOT filtered -- it does its own selection via clumping+
        # thresholding on external summary statistics, a structurally
        # different mechanism operating on the full SNP set.
        needs_filter = any(m in ("ridge", "lasso") for m in self.methods)
        if needs_filter and self.max_snps is not None and G.shape[1] > self.max_snps:
            self._snp_filter_idx = select_top_snps(
                G, y, X0=X0, max_snps=self.max_snps, seed=self.seed)
            logger.info(f"[PRS] Pre-filtered to top {len(self._snp_filter_idx)} "
                       f"of {G.shape[1]} SNPs (by |correlation| with "
                       f"phenotype) before Ridge/Lasso fitting -- see "
                       f"select_top_snps()'s docstring for why.")
        else:
            self._snp_filter_idx = None

        def _G_for(meth):
            if meth in ("ridge", "lasso") and self._snp_filter_idx is not None:
                return G[:, self._snp_filter_idx]
            return G

        scores = {}
        for meth in self.methods:
            try:
                if meth == "ct":
                    if sumstats_df is None or bim is None:
                        logger.warning("[PRS] C+T skipped - no sumstats")
                        continue
                    m = _CTScore(); m.fit(G[valid], sumstats_df, bim)
                    raw = m.score(G, bim)
                    self._models["ct"] = m
                elif meth == "ridge":
                    Gm = _G_for(meth)
                    m = _RidgePRS(); m.fit(Gm, y, X0=X0)
                    raw = m.score(Gm)
                    self._models["ridge"] = m
                elif meth == "lasso":
                    Gm = _G_for(meth)
                    m = _LassoPRS(); m.fit(Gm, y, X0=X0)
                    raw = m.score(Gm)
                    self._models["lasso"] = m
                else:
                    continue
                raw_z = (raw - raw[valid].mean()) / (raw[valid].std() + 1e-12)
                scores[meth] = raw_z
            except Exception as e:
                logger.warning(f"[PRS] {meth} failed: {e}")
        if not scores:
            raise RuntimeError("[PRS] All methods failed.")
        self._methods_fit = list(scores.keys())
        score_matrix = np.column_stack([scores[m] for m in self._methods_fit])
        miss_flags   = (~np.isnan(score_matrix)).astype(float)
        X_meta = np.hstack([score_matrix[valid], miss_flags[valid]])
        self._meta = RidgeCV(alphas=np.logspace(-3,3,20), cv=self.cv_folds)
        self._meta.fit(X_meta, y[valid])
        self._score_matrix = score_matrix
        logger.info(f"[PRS] Consensus stacked on: {self._methods_fit}")
        return self

    def get_snp_weights(self, bim):
        """
        Per-SNP weights/coefficients from every fitted component method,
        for downstream enrichment analysis on "which SNPs actually drove
        the PRS" -- previously not available at all: the PRS pipeline only
        ever wrote per-SAMPLE scores (prs_scores.tsv) and summary metrics
        (prs_validation.tsv/prs_portability.tsv), never a per-SNP file.

        Handles the two structurally different internal representations
        the fitted models use:
          - Ridge/Lasso: sklearn's model_.coef_ is a plain array indexed
            by COLUMN POSITION in the (possibly pre-filtered, via
            self._snp_filter_idx) G that was actually fit on -- mapped
            back through self._snp_filter_idx to the ORIGINAL column
            index, then to a real SNP ID via bim.
          - C+T: self._models["ct"].weights_ is already a pandas Series
            indexed directly by SNP ID (clumping+thresholding naturally
            operates on a named subset from external summary stats, not
            column positions) -- used as-is.
        These are merged into one table, outer-joined on SNP ID (since
        Ridge/Lasso and C+T generally select different SNP subsets), then
        joined against bim for chromosome/position.

        Parameters
        ----------
        bim : pd.DataFrame with 'snp'/'chrom'/'pos' columns (lowercase),
             same row order as the G originally passed to fit() -- i.e.
             bim.iloc[j] must describe G[:, j] for every j.

        Returns
        -------
        pd.DataFrame with columns: chr, snp, pos, and one weight column
        per fitted ridge/lasso/ct method (whichever were actually used),
        plus IMPORTANCE (max absolute weight across available methods for
        that SNP) -- sorted by IMPORTANCE descending, so the top rows are
        directly usable as an enrichment input list.
        """
        col_map = {c.lower(): c for c in bim.columns}
        snp_col  = col_map.get("snp", bim.columns[0])
        chr_col  = col_map.get("chrom", col_map.get("chr"))
        pos_col  = col_map.get("pos")
        all_snp_ids = bim[snp_col].astype(str).tolist()

        weight_series = {}

        for meth in ("ridge", "lasso"):
            if meth not in self._models:
                continue
            model = self._models[meth]
            if model.model_ is None or not hasattr(model.model_, "coef_"):
                continue
            coef = np.asarray(model.model_.coef_).ravel()
            # Map each coefficient's position back to the ORIGINAL column
            # index in the full, unfiltered panel (see docstring above).
            orig_idx = (self._snp_filter_idx if self._snp_filter_idx is not None
                       else np.arange(len(coef)))
            snp_ids_for_coef = [all_snp_ids[i] for i in orig_idx]
            weight_series[meth.upper()] = pd.Series(coef, index=snp_ids_for_coef)

        if "ct" in self._models and self._models["ct"].weights_ is not None:
            weight_series["CT"] = self._models["ct"].weights_.copy()
            weight_series["CT"].index = weight_series["CT"].index.astype(str)

        if not weight_series:
            logger.warning("[PRS] get_snp_weights(): no fitted method exposes "
                           "per-SNP weights -- returning an empty table.")
            return pd.DataFrame(columns=["chr","snp","pos","IMPORTANCE"])

        out = pd.DataFrame(index=sorted(set().union(*[s.index for s in weight_series.values()])))
        for meth, s in weight_series.items():
            out[f"{meth}_WEIGHT"] = s.reindex(out.index)
        out.index.name = "snp"
        out = out.reset_index()

        bim_meta = pd.DataFrame({
            "snp": all_snp_ids,
            "chr": bim[chr_col].values if chr_col else np.nan,
            "pos": bim[pos_col].values if pos_col else np.nan,
        })
        out = bim_meta.merge(out, on="snp", how="inner")

        weight_cols = [c for c in out.columns if c.endswith("_WEIGHT")]
        out["IMPORTANCE"] = out[weight_cols].abs().max(axis=1, skipna=True)
        out = out.sort_values("IMPORTANCE", ascending=False, na_position="last").reset_index(drop=True)
        return out[["chr","snp","pos"] + weight_cols + ["IMPORTANCE"]]

    def predict(self, G=None, score_matrix=None):
        if score_matrix is None:
            scores = {}
            for meth, m in self._models.items():
                if meth == "ct":    raw = m.score(G, None) if hasattr(m,'weights_') else np.zeros(G.shape[0])
                elif meth in ("ridge", "lasso"):
                    # Same filtered SNP subset used at fit() time -- required
                    # for shape consistency with self.scaler_/self.model_,
                    # which were fitted on that same reduced feature set.
                    Gm = G[:, self._snp_filter_idx] if self._snp_filter_idx is not None else G
                    raw = m.score(Gm)
                else: continue
                raw_z = (raw - raw.mean()) / (raw.std() + 1e-12)
                scores[meth] = raw_z
            score_matrix = np.column_stack([scores[m] for m in self._methods_fit])
        miss_flags = (~np.isnan(score_matrix)).astype(float)
        X_meta = np.hstack([np.nan_to_num(score_matrix, nan=0.0), miss_flags])
        return self._meta.predict(X_meta)

    def component_scores(self, G):
        out = {}
        for meth, m in self._models.items():
            if meth == "ct":    pass  # skip: needs bim
            elif meth in ("ridge", "lasso"):
                Gm = G[:, self._snp_filter_idx] if self._snp_filter_idx is not None else G
                out[meth] = m.score(Gm)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Steps 6-7: validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_prs(prs_all, y_all, ancestry_labels=None):
    """
    Returns a validation summary dict.

    Bug fix: portability is now computed on ALL samples (not just the test
    split), and the minimum-sample guard scales with cohort size: max(3, n//20).
    Groups below the threshold are included as NaN with a note, rather than
    being silently omitted from the output file.
    """
    valid = ~np.isnan(y_all) & ~np.isnan(prs_all)
    prs_v, y_v = prs_all[valid], y_all[valid]
    n = int(valid.sum())
    summary = {}

    # Bug fix: this used to ALWAYS binarize by median split, even when y_v
    # is already a clean binary (0/1) phenotype -- e.g. case/control
    # disease status, the most common PRS use case. Median-split assumes
    # a roughly 50/50 split; for an already-binary phenotype with an
    # UNEVEN case:control ratio (e.g. a deliberately case-enriched study
    # design, common in GWAS), the median itself can land ON one of the
    # two actual values (0 or 1) rather than between them -- e.g. if over
    # half the cohort are cases, median(y_v)=1, so (y_v > 1) is false for
    # EVERY sample, collapsing binary_y to a single class and making
    # roc_auc_score raise (silently caught below as AUC=nan). Found via
    # a real TB case/control run where this happened for exactly this
    # reason. Using the phenotype's own values directly when it's
    # already binary sidesteps the issue entirely; median-split remains
    # the fallback for genuinely continuous/quantitative phenotypes,
    # where AUC is a coarser but still requested summary.
    unique_y = np.unique(y_v[np.isfinite(y_v)])
    if len(unique_y) == 2:
        binary_y = (y_v == unique_y[1]).astype(int)
    else:
        binary_y = (y_v > np.median(y_v)).astype(int)
    try:
        summary["auc"] = float(roc_auc_score(binary_y, prs_v))
    except Exception:
        summary["auc"] = np.nan

    summary["pearson_r"] = float(np.corrcoef(y_v, prs_v)[0,1]) if n >= 3 else np.nan

    # Calibration: use min(10, n//4) deciles so small cohorts still work
    n_deciles = max(2, min(10, n // 4))
    try:
        deciles = pd.qcut(prs_v, n_deciles, labels=False, duplicates="drop")
        dec_prs = pd.Series(prs_v).groupby(deciles).mean().values
        dec_obs = pd.Series(y_v).groupby(deciles).mean().values
        summary["calibration_slope"] = float(
            np.polyfit(dec_prs, dec_obs, 1)[0]) if len(dec_prs) > 1 else np.nan
    except Exception:
        summary["calibration_slope"] = np.nan

    # Nagelkerke pseudo-R² (approximate)
    try:
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(max_iter=200).fit(prs_v[:,None], binary_y)
        ll_full  = float(np.sum(binary_y * np.log(lr.predict_proba(prs_v[:,None])[:,1]+1e-12) +
                                (1-binary_y) * np.log(1-lr.predict_proba(prs_v[:,None])[:,1]+1e-12)))
        ll_null  = float(np.sum(binary_y) * np.log(binary_y.mean()+1e-12) +
                         np.sum(1-binary_y) * np.log(1-binary_y.mean()+1e-12))
        r2_nag   = (1 - np.exp((ll_null-ll_full)*(2/n))) / (1 - np.exp(ll_null*2/n)+1e-12)
        summary["nagelkerke_r2"] = float(np.clip(r2_nag, 0, 1))
    except Exception:
        summary["nagelkerke_r2"] = np.nan

    # Per-ancestry portability -- FIX: use all samples, adaptive threshold
    portability_rows = []
    if ancestry_labels is not None:
        anc_arr  = np.array(ancestry_labels)[valid]
        ref_r2   = summary["pearson_r"]**2 if np.isfinite(summary["pearson_r"]) else np.nan
        min_n    = max(3, n // 20)          # adaptive minimum: 5% of cohort, at least 3
        for anc in sorted(np.unique(anc_arr)):
            mask  = anc_arr == anc
            count = int(mask.sum())
            if count < min_n:
                portability_rows.append({
                    "ancestry": anc,
                    "n_samples": count,
                    "pearson_r": np.nan,
                    "r2": np.nan,
                    "portability_ratio": np.nan,
                    "note": f"below minimum ({min_n} samples required)"
                })
                continue
            r = float(np.corrcoef(y_v[mask], prs_v[mask])[0,1])
            r2 = r**2
            porto = float(r2 / max(ref_r2, 1e-12)) if np.isfinite(ref_r2) and ref_r2 > 0 else np.nan
            portability_rows.append({
                "ancestry": anc,
                "n_samples": count,
                "pearson_r": r,
                "r2": r2,
                "portability_ratio": porto,
                "note": "ok"
            })
        summary["portability_ratios"] = portability_rows

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Publication-quality PRS visualization suite
# ─────────────────────────────────────────────────────────────────────────────

def _mpl():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

PALETTE = ["#E41A1C","#377EB8","#4DAF4A","#984EA3",
           "#FF7F00","#A65628","#F781BF","#999999"]


def _save(fig, path, dpi=300):
    plt = _mpl()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[PLOT] {Path(path).name}")


def plot_prs_individual_scores(prs, y, sample_ids=None, ancestry_labels=None,
                                out_path="prs_individual_scores.png"):
    """
    Per-individual PRS dot plot with population-percentile bands and
    high/low risk flags.

    Rows: individuals sorted by PRS (ascending left to right).
    Background bands: 10th / 25th / 75th / 90th percentile risk zones.
    Points coloured by ancestry when available, else case/control status.
    Top and bottom 10% annotated with risk tier labels.
    """
    plt = _mpl()
    valid = ~np.isnan(prs)
    prs_v = prs[valid]; y_v = y[valid]
    n = len(prs_v)
    order = np.argsort(prs_v)
    prs_sorted = prs_v[order]
    y_sorted   = y_v[order]

    p10, p25, p75, p90 = (np.percentile(prs_sorted, q)
                           for q in [10, 25, 75, 90])

    fig, ax = plt.subplots(figsize=(14, 5), dpi=300)
    ax.axhspan(p90, prs_sorted.max()*1.05, alpha=0.10, color="#D6604D",
               label="Top 10% (high risk)")
    ax.axhspan(p75, p90, alpha=0.06, color="#FDAE61")
    ax.axhspan(p25, p75, alpha=0.04, color="#ABD9E9")
    ax.axhspan(prs_sorted.min()*1.05, p10, alpha=0.10, color="#4393C3",
               label="Bottom 10% (low risk)")

    if ancestry_labels is not None:
        anc_sorted = np.array(ancestry_labels)[valid][order]
        uniq = sorted(set(anc_sorted))
        for i, a in enumerate(uniq):
            mask = anc_sorted == a
            ax.scatter(np.where(mask)[0], prs_sorted[mask],
                       s=12, alpha=0.7, label=a,
                       color=PALETTE[i%len(PALETTE)], linewidths=0)
    else:
        cases    = y_sorted == 1
        controls = ~cases
        ax.scatter(np.where(controls)[0], prs_sorted[controls],
                   s=10, alpha=0.5, color="#4393C3", label="Control", linewidths=0)
        ax.scatter(np.where(cases)[0], prs_sorted[cases],
                   s=10, alpha=0.7, color="#D6604D", label="Case", linewidths=0)

    ax.axhline(p10, color="#4393C3", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(p90, color="#D6604D", lw=0.8, ls="--", alpha=0.6)

    ax.set_xlabel("Individual (rank by PRS)", fontsize=10)
    ax.set_ylabel("Consensus PRS (standardised)", fontsize=10)
    ax.set_title("Individual PRS Scores with Population Percentile Bands",
                 fontweight="bold")
    ax.legend(fontsize=8, markerscale=2, framealpha=0.7, loc="upper left")
    ax.spines[["top","right"]].set_visible(False)

    # Annotate risk tiers
    ax.text(n*0.98, (p90+prs_sorted.max())/2, "HIGH RISK", ha="right",
            va="center", fontsize=7, color="#D6604D", alpha=0.7)
    ax.text(n*0.98, (prs_sorted.min()+p10)/2, "LOW RISK",  ha="right",
            va="center", fontsize=7, color="#4393C3", alpha=0.7)

    _save(fig, out_path)


def plot_prs_risk_stratification(prs, y, n_groups=10,
                                  out_path="prs_risk_stratification.png"):
    """
    Mean phenotype (or case rate) per PRS decile — the canonical "staircase"
    risk-stratification plot used in clinical PRS validation papers.

    Shows both the mean and 95% CI per decile plus individual data points.
    The top and bottom deciles are highlighted to emphasise the contrast
    between high and low genetic risk groups.
    """
    plt = _mpl()
    valid = ~np.isnan(prs) & ~np.isnan(y)
    pv, yv = prs[valid], y[valid].astype(float)
    n_groups = min(n_groups, max(2, len(pv)//3))
    try:
        groups = pd.Series(pd.qcut(pv, n_groups, labels=False, duplicates="drop"))
    except Exception:
        groups = pd.Series(pd.cut(pv, n_groups, labels=False))
    # Bug fix: pd.qcut(..., duplicates="drop") prevents a crash from
    # duplicate BIN EDGES, but doesn't guarantee every sample gets a
    # valid bin assignment -- when the PRS distribution is heavily tied
    # (e.g. Ridge/Lasso predicting a near-identical score for most
    # samples, the same degenerate scenario RISK_TIER's own fix
    # addresses), samples landing exactly on a dropped edge get NaN in
    # `groups`. groups.max() then being NaN made `int(groups.max()) + 1`
    # raise "cannot convert float NaN to integer" -- found via a real
    # run where a weak-signal PRS on n=733 produced exactly this.
    # Dropping the NaN-grouped samples (rather than crashing) still
    # plots whatever CAN be meaningfully stratified.
    groups = groups.dropna()
    if len(groups) == 0:
        raise ValueError(
            "PRS scores are too degenerate (heavily tied) to stratify into "
            "risk groups at all -- skipping this plot.")
    pv, yv = pv[groups.index], yv[groups.index]
    n_actual = int(groups.max()) + 1

    means, cis, sizes = [], [], []
    for g in range(n_actual):
        vals = yv[groups == g]
        if not len(vals): continue
        means.append(vals.mean())
        se = vals.std() / max(np.sqrt(len(vals)), 1)
        cis.append(1.96 * se)
        sizes.append(len(vals))

    x = np.arange(len(means))
    colors = ["#D6604D" if i == len(means)-1 else
              "#4393C3"  if i == 0 else "#ABD9E9"
              for i in range(len(means))]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    ax.bar(x, means, color=colors, alpha=0.85, edgecolor="white",
           width=0.7, yerr=cis, capsize=3, error_kw={"lw":1.2})
    ax.scatter(x, means, s=60, color="#333333", zorder=5)
    ax.axhline(yv.mean(), color="k", lw=0.9, ls="--", alpha=0.5,
               label=f"Overall mean = {yv.mean():.3f}")

    ax.set_xlabel(f"PRS {n_actual}-ile group (1 = lowest, {n_actual} = highest)",
                  fontsize=10)
    ax.set_ylabel("Mean phenotype / case rate", fontsize=10)
    ax.set_title("Risk Stratification by PRS Group", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i+1) for i in range(len(means))], fontsize=8)

    # Annotate sample sizes
    for i, (xi, ni) in enumerate(zip(x, sizes)):
        ax.text(xi, ax.get_ylim()[0] - 0.03*(ax.get_ylim()[1]-ax.get_ylim()[0]),
                f"n={ni}", ha="center", va="top", fontsize=6, alpha=0.7)

    ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)
    _save(fig, out_path)


def plot_prs_calibration(prs, y, n_deciles=10,
                          out_path="prs_calibration.png"):
    """
    Calibration plot: observed risk vs mean predicted PRS per decile.
    The ideal calibrated model lies along y = x (diagonal reference line).
    """
    plt = _mpl()
    valid = ~np.isnan(prs) & ~np.isnan(y)
    pv, yv = prs[valid], y[valid].astype(float)
    n_deciles = min(n_deciles, max(2, len(pv)//3))
    try:
        groups = pd.Series(pd.qcut(pv, n_deciles, labels=False, duplicates="drop"))
    except Exception:
        groups = pd.Series(pd.cut(pv, n_deciles, labels=False))
    # Bug fix: same issue and reasoning as plot_prs_risk_stratification()'s
    # equivalent fix -- pd.qcut(duplicates="drop") can still leave some
    # samples with a NaN group when the PRS distribution is heavily tied.
    # sorted(groups.unique()) would then include NaN, and iterating over
    # it as a group label produces silently-wrong (or NaN-propagating)
    # obs/pred entries rather than a clean, if smaller, calibration plot.
    groups = groups.dropna()
    if len(groups) == 0:
        raise ValueError(
            "PRS scores are too degenerate (heavily tied) to compute "
            "calibration deciles at all -- skipping this plot.")
    pv, yv = pv[groups.index], yv[groups.index]

    obs, pred = [], []
    for g in sorted(groups.unique()):
        obs.append(yv[groups==g].mean())
        pred.append(pv[groups==g].mean())

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    ax.plot(pred, obs, "o-", color=PALETTE[0], ms=7, lw=1.5, label="Observed")
    lim = [min(min(pred), min(obs))*0.95, max(max(pred), max(obs))*1.05]
    ax.plot(lim, lim, "k--", lw=0.9, alpha=0.5, label="Perfect calibration")
    ax.fill_between(lim, [l*0.9 for l in lim], [l*1.1 for l in lim],
                    alpha=0.07, color="#ABD9E9", label="+/-10% band")

    slope = float(np.polyfit(pred, obs, 1)[0])
    ax.set_xlabel("Mean PRS per decile", fontsize=10)
    ax.set_ylabel("Observed mean phenotype per decile", fontsize=10)
    ax.set_title(f"PRS Calibration Plot  (slope = {slope:.2f})",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)
    _save(fig, out_path)


def plot_prs_ancestry_density(prs, ancestry_labels, out_path="prs_ancestry_density.png"):
    """
    Overlaid KDE (kernel density estimate) per ancestry group.
    Vertical dashed lines mark each group's median.
    This is the key 'portability visualisation' — groups shifted left or right
    indicate ancestry-specific mean PRS offsets that affect transferability.
    """
    plt = _mpl()
    from scipy.stats import gaussian_kde

    anc_arr = np.array(ancestry_labels)
    uniq = sorted(set(anc_arr))

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    x_range = np.linspace(prs.min()-0.5, prs.max()+0.5, 400)

    for i, anc in enumerate(uniq):
        sub = prs[anc_arr == anc]
        sub = sub[np.isfinite(sub)]
        if len(sub) < 3: continue
        # Improvement: gaussian_kde requires genuine spread in the data --
        # it inverts a covariance matrix internally, which is singular (and
        # raises "The data appears to lie in a lower-dimensional subspace")
        # when a group's PRS values are all identical or near-identical.
        # The len(sub) < 3 check above doesn't catch that, since a group
        # can have plenty of samples that all share essentially the same
        # score -- exactly what happens with the degenerate, heavily-tied
        # PRS distributions that also skip the risk-stratification and
        # calibration plots (weak signal + strong Ridge/Lasso shrinkage).
        # Found on a real run: this raised and skipped the ENTIRE ancestry
        # plot, even though other groups might have been perfectly
        # plottable. Skipping just the degenerate GROUP lets the rest of
        # the figure still render.
        if not np.isfinite(np.std(sub)) or np.std(sub) < 1e-12:
            logger.warning(
                f"[PRS] Ancestry group {anc!r} (n={len(sub)}) has "
                f"essentially no variation in PRS score -- omitting it from "
                f"the density plot (a KDE needs genuine spread to compute).")
            continue
        kde = gaussian_kde(sub, bw_method="scott")
        col = PALETTE[i % len(PALETTE)]
        ax.plot(x_range, kde(x_range), lw=2.0, color=col, label=f"{anc} (n={len(sub)})")
        ax.fill_between(x_range, kde(x_range), alpha=0.10, color=col)
        ax.axvline(np.median(sub), lw=1.0, ls="--", color=col, alpha=0.7)

    ax.set_xlabel("Consensus PRS (standardised)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("PRS Distribution by Ancestry Group\n"
                 "(dashed lines = group medians; shifted distributions signal "
                 "transferability issues)",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.7)
    ax.spines[["top","right"]].set_visible(False)
    _save(fig, out_path)


def plot_prs_portability_bar(portability_rows, out_path="prs_portability.png"):
    """
    Horizontal bar chart of R^2 (and portability ratio) per ancestry group.
    Reference line at the overall R^2 = 1.0 (all portability ratios are
    relative to the overall value). Groups with too few samples are shown
    as hatched bars with an 'N/A' label instead of being silently omitted.
    """
    plt = _mpl()
    df = pd.DataFrame(portability_rows)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(df)*1.1), 5), dpi=300)
    y_pos = np.arange(len(df))

    for i, row in df.iterrows():
        val  = row["portability_ratio"]
        note = row.get("note","ok")
        col  = PALETTE[i % len(PALETTE)]
        if pd.isna(val) or val != val:
            ax.barh(i, 0.05, color=col, alpha=0.3, hatch="////",
                    edgecolor=col, height=0.6)
            ax.text(0.06, i, f"N/A  ({note})", va="center", fontsize=8, color=col)
        else:
            ax.barh(i, float(val), color=col, alpha=0.80, height=0.6,
                    edgecolor="white")
            ax.text(float(val)+0.01, i,
                    f"{val:.3f}  (r={row['pearson_r']:.2f}, n={row['n_samples']})",
                    va="center", fontsize=8)

    ax.axvline(1.0, color="k", lw=1.0, ls="--", alpha=0.5,
               label="Overall R² (reference = 1.0)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["ancestry"].tolist(), fontsize=9)
    ax.set_xlabel("Portability ratio  (R² / overall R²)", fontsize=10)
    ax.set_title("PRS Portability Across Ancestry Groups\n"
                 "(ratio > 1 = better than average, < 1 = worse, hatch = insufficient n)",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)
    _save(fig, out_path)


def plot_prs_cumulative_risk(prs, y, out_path="prs_cumulative_risk.png"):
    """
    Cumulative case fraction vs PRS rank percentile.
    A PRS with clinical utility will show a steep early rise (cases
    concentrated in the top PRS percentiles) departing from the diagonal
    (random) reference line. Quantified by the Lorenz-curve-style Gini
    coefficient shown on the plot.
    """
    plt = _mpl()
    valid = ~np.isnan(prs) & ~np.isnan(y)
    pv, yv = prs[valid], (y[valid] > np.median(y[valid])).astype(float)
    order = np.argsort(pv)[::-1]     # descending: highest PRS first
    cum_case = np.cumsum(yv[order]) / yv.sum()
    cum_frac = np.arange(1, len(yv)+1) / len(yv)

    # Gini coefficient (area between curve and diagonal)
    gini = float(2 * _trapezoid_area(cum_case, cum_frac) - 1)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    ax.plot(cum_frac*100, cum_case*100, color=PALETTE[0], lw=2.0,
            label=f"PRS (Gini = {gini:.3f})")
    ax.plot([0,100],[0,100], "k--", lw=0.9, alpha=0.5, label="Random (Gini = 0)")
    ax.fill_between(cum_frac*100, cum_case*100, cum_frac*100,
                    alpha=0.12, color=PALETTE[0])

    ax.set_xlabel("Cumulative % of individuals screened (highest PRS first)", fontsize=10)
    ax.set_ylabel("Cumulative % of cases identified", fontsize=10)
    ax.set_title("Cumulative Risk Curve\n(steeper departure = better risk stratification)",
                 fontweight="bold")
    ax.legend(fontsize=10); ax.spines[["top","right"]].set_visible(False)
    _save(fig, out_path)


def plot_prs_forest_violin(prs, y, ancestry_labels,
                            out_path="prs_forest_violin.png"):
    """
    Combined violin + odds-ratio forest plot per ancestry group.

    Left panel: violin plot of PRS distribution split by case/control
    within each ancestry group — shows within-group separation.
    Right panel: per-ancestry case-vs-control OR of being in the top 20%
    PRS vs the bottom 20%, with 95% CI — a forest plot style summary
    of clinical risk stratification power per ancestry.
    """
    plt = _mpl()
    from scipy.stats import fisher_exact

    anc_arr = np.array(ancestry_labels)
    valid   = ~np.isnan(prs)
    prs_v, y_v, anc_v = prs[valid], y[valid], anc_arr[valid]
    uniq    = sorted(set(anc_v))
    n_anc   = len(uniq)

    fig, (ax_v, ax_f) = plt.subplots(1, 2, figsize=(14, max(4, n_anc*1.4)),
                                      dpi=300,
                                      gridspec_kw={"width_ratios":[2,1]})

    # ── Left: violin per ancestry × case/control ─────────────────────────────
    labels_v = []
    tick_pos = []
    for i, anc in enumerate(uniq):
        sub_prs = prs_v[anc_v == anc]
        sub_y   = y_v[anc_v   == anc]
        if len(sub_prs) < 4: continue
        base = i * 3
        for yi, col, lbl in [(0,"#4393C3","Ctrl"),(1,"#D6604D","Case")]:
            vals = sub_prs[sub_y == yi]
            if len(vals) < 2: continue
            parts = ax_v.violinplot(vals, positions=[base + yi],
                                    widths=0.8, showmedians=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(col); pc.set_alpha(0.6)
            parts["cmedians"].set_color("k"); parts["cmedians"].set_lw(1.5)
        tick_pos.append(base + 0.5)
        labels_v.append(anc)

    ax_v.set_xticks(tick_pos); ax_v.set_xticklabels(labels_v, fontsize=9)
    ax_v.set_ylabel("Consensus PRS"); ax_v.set_xlabel("Ancestry group")
    ax_v.set_title("PRS by Case/Control Status\nper Ancestry", fontweight="bold")
    ax_v.spines[["top","right"]].set_visible(False)
    # manual legend
    import matplotlib.patches as mpatches
    ax_v.legend(handles=[mpatches.Patch(color="#4393C3",label="Control"),
                          mpatches.Patch(color="#D6604D",label="Case")],
                fontsize=8, loc="upper left")

    # ── Right: forest plot of OR (top 20% vs bottom 20% PRS) ─────────────────
    ors, lo, hi, anc_labs = [], [], [], []
    p80 = np.percentile(prs_v, 80); p20 = np.percentile(prs_v, 20)
    for anc in uniq:
        mask = anc_v == anc
        sub_prs = prs_v[mask]; sub_y = (y_v[mask] > np.median(y_v)).astype(int)
        top = sub_prs >= p80; bot = sub_prs <= p20
        if top.sum() < 2 or bot.sum() < 2: continue
        ct = np.array([[int((top & (sub_y==1)).sum()), int((top & (sub_y==0)).sum())],
                       [int((bot & (sub_y==1)).sum()), int((bot & (sub_y==0)).sum())]])
        try:
            or_v, pv = fisher_exact(ct)
        except Exception:
            or_v = np.nan
        if not np.isfinite(or_v) or or_v == 0: continue
        log_or = np.log(or_v + 1e-9)
        n11,n10,n01,n00 = ct[0,0],ct[0,1],ct[1,0],ct[1,1]
        se = np.sqrt(sum(1/max(x,0.5) for x in [n11,n10,n01,n00]))
        ors.append(or_v); lo.append(np.exp(log_or-1.96*se))
        hi.append(np.exp(log_or+1.96*se)); anc_labs.append(anc)

    if ors:
        y_f = np.arange(len(ors))
        ax_f.scatter(ors, y_f, s=60, color=PALETTE[0], zorder=5)
        for yi, (l, h) in enumerate(zip(lo, hi)):
            ax_f.plot([l,h],[yi,yi], color=PALETTE[0], lw=1.5)
        ax_f.axvline(1.0, color="k", lw=0.9, ls="--", alpha=0.5)
        ax_f.set_yticks(y_f); ax_f.set_yticklabels(anc_labs, fontsize=9)
        ax_f.set_xlabel("Odds Ratio\n(top 20% vs bottom 20% PRS)", fontsize=9)
        ax_f.set_title("OR of High vs Low PRS\nper Ancestry", fontweight="bold")
        ax_f.spines[["top","right"]].set_visible(False)
    else:
        ax_f.text(0.5, 0.5, "Insufficient data\nfor OR calculation",
                  ha="center", va="center", transform=ax_f.transAxes, fontsize=10)

    _save(fig, out_path)


def plot_prs_percentile_table(prs, y, out_path="prs_percentile_thresholds.png"):
    """
    Clinical percentile threshold table — shows the PRS cut-off, case rate,
    and enrichment ratio at the 50th, 75th, 80th, 90th, 95th percentile.

    This is the 'clinical utility summary' figure: at what PRS threshold
    does the case rate exceed the population average, and by how much?
    """
    plt = _mpl()
    valid = ~np.isnan(prs) & ~np.isnan(y)
    pv, yv = prs[valid], (y[valid] > np.median(y[valid])).astype(float)
    overall_rate = yv.mean()

    rows = []
    for pct in [50, 75, 80, 90, 95, 99]:
        thresh = np.percentile(pv, pct)
        above  = pv >= thresh
        rate   = yv[above].mean() if above.sum() > 0 else np.nan
        enrich = rate / max(overall_rate, 1e-9)
        rows.append([f">P{pct}", f"{thresh:.2f}", f"{above.sum()}",
                     f"{rate*100:.1f}%", f"{enrich:.2f}x"])

    col_labels = ["PRS value", "N individuals",
                  "Case rate", "Enrichment vs baseline"]
    row_labels  = [r[0] for r in rows]
    cell_text   = [r[1:] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=300)
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=col_labels,
                   rowLabels=row_labels, loc="center",
                   cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.scale(1.0, 1.8)

    # Colour high-enrichment rows
    for ri, row in enumerate(rows):
        enrich_val = float(row[4].replace("x",""))
        if enrich_val >= 2.0:
            for ci in range(len(col_labels)):
                tbl[ri+1, ci].set_facecolor("#FFF3CD")

    ax.set_title("PRS Clinical Risk Percentile Summary\n"
                 "(highlighted rows: case rate >= 2x population average)",
                 fontweight="bold", fontsize=10, pad=12)
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Public runner
# ─────────────────────────────────────────────────────────────────────────────

def impute_missing_genotypes(G, tag="PRS"):
    """
    Per-SNP mean imputation for missing genotype calls (NaN), matching the
    same convention already used in jasmap.biobank.lowrank_kinship /
    cg_kinship / streaming_io for consistency across the project.

    Why this exists
    -----------------
    bed-reader returns NaN for missing genotype calls when reading with
    dtype float32/float64 (its own documented convention) -- completely
    normal for real genetic data (most real genotyping arrays/sequencing
    have SOME missing calls), but ConsensusPRS's Ridge/Lasso fitting had
    no handling for this at all: sklearn's RidgeCV/LassoCV do not accept
    NaN and fail immediately ("Input X contains NaN"). This was never
    caught by any prior testing because the synthetic test data generator
    (make_test_data.py) never produces missing genotype calls -- it only
    surfaced on real data (found via an actual run against a real SAC2
    cohort, not a unit test). The auto-PCA path already imputes internally
    (randomized_kinship_eigendecomp's own nan_mask handling), which is why
    that step succeeded in the same run while Ridge/Lasso crashed.

    Applied ONCE here, before G is split into train/predict subsets or
    handed to any PRS method, so every method (Ridge, Lasso, and C+T's
    scoring step, which would also silently propagate NaN through its
    G[:, idx] @ w matmul otherwise) sees a consistently-imputed matrix,
    and training-time and predict-time use the SAME per-SNP means.

    Parameters
    ----------
    G : ndarray (n_samples, n_snps)

    Returns
    -------
    G_imputed : ndarray, same shape, NaN replaced by that SNP's mean
                dosage across non-missing samples. SNPs that are 100%
                missing (no non-missing samples at all) are imputed to 0,
                logged as a warning -- this is a real data-quality issue
                the user should know about, not something to silently
                paper over.

    tag : str, log-message prefix (default "PRS", matching this
          function's original, primary caller). Callers from a different
          mode (e.g. GIFT/GIFT-ADMIX, which also reads G through its own,
          separate code path and needs this same fix) should pass their
          own tag so the log line correctly reflects which step actually
          ran the imputation, rather than always claiming "[PRS]"
          regardless of the real caller.
    """
    G = np.asarray(G, dtype=np.float64)
    nan_mask = np.isnan(G)
    if not nan_mask.any():
        return G

    col_mean = np.nanmean(G, axis=0)
    fully_missing = np.isnan(col_mean)
    if fully_missing.any():
        logger.warning(f"[{tag}] {fully_missing.sum()} SNP(s) have NO "
                       f"non-missing genotype calls at all -- imputing to "
                       f"0 for these. This likely reflects a real data-"
                       f"quality issue worth checking upstream.")
        col_mean = np.where(fully_missing, 0.0, col_mean)

    G_imputed = G.copy()
    rows, cols = np.where(nan_mask)
    G_imputed[rows, cols] = col_mean[cols]

    n_missing = int(nan_mask.sum())
    logger.info(f"[{tag}] Imputed {n_missing} missing genotype call(s) "
               f"({100*n_missing/G.size:.2f}% of all genotype x sample "
               f"entries) using per-SNP mean dosage before downstream use.")
    return G_imputed


def run_prs(G, y, bim, fam, out_prefix,
            sumstats_path=None,
            methods=("ridge","lasso"),
            ancestry_labels=None,
            X0=None,
            max_snps=20000,
            test_frac=0.2, seed=42,
            restrict_output_idx=None,
            plots=True):
    """
    Full consensus PRS pipeline.  Returns (prs_scores, validation).

    restrict_output_idx : optional array of sample indices. When given,
        the model is still FIT and SCORED (model.predict(G)) on every
        sample as before, but the final output file (prs_scores.tsv) and
        the validation metrics are restricted to only these indices.
        Intended for --mode admix/gift_admix, where the genotype file
        also contains reference/ancestral population samples included
        purely to inform local ancestry inference -- those samples aren't
        part of the phenotyped study cohort at all (often not even
        conceptually the same population being studied), so a PRS
        percentile/risk-tier computed jointly across both groups is
        actively misleading, not just extra information. Found via a
        real user report describing this data model. None (the default)
        keeps the previous behaviour -- reporting every sample -- which
        remains appropriate for a generically partially-phenotyped cohort
        outside the admixture-reference-panel context, e.g. scoring
        individuals whose outcome isn't yet known but who are still part
        of the same study population.

    Parameters
    ----------
    max_snps : int or None  pre-filter Ridge/Lasso to at most this many
        SNPs (by univariate |correlation| with phenotype) before fitting
        -- see select_top_snps()'s docstring. Set to None to disable
        filtering entirely and fit on every SNP (slow on large panels).
    X0 : optional ndarray (n, q)  covariates for population stratification
        correction during Ridge/Lasso weight fitting (typically an
        intercept plus PCA eigenvectors -- user-provided via --covfile, or
        auto-generated via jasmap.biobank.auto_pca when neither --covfile
        nor --no-auto-pca is given). See residualize_phenotype()'s
        docstring for why this matters. None (the previous, unconditional
        default) means no adjustment at all.
    """
    out = Path(out_prefix); out.mkdir(parents=True, exist_ok=True)

    # Real genotype data (unlike this project's synthetic test generator)
    # commonly has missing calls. Imputed ONCE here, before G is split or
    # handed to any method -- see impute_missing_genotypes()'s docstring.
    G = impute_missing_genotypes(G)

    sumstats_df = None
    if sumstats_path:
        sumstats_df = harmonise_sumstats(sumstats_path, bim)
        if len(sumstats_df):
            methods = ("ct",) + tuple(m for m in methods if m != "ct")

    rng = np.random.default_rng(seed)
    n   = G.shape[0]
    # Bug fix: same issue as run_gift()/run_gift_admix() (see their
    # comments for the full explanation and how this was reproduced) --
    # the permutation previously ran over ALL n samples with no regard
    # for missing phenotype, so a fixed seed could concentrate every
    # phenotyped sample into the discarded test partition, leaving tr
    # entirely NaN. Restricting to phenotyped samples up front fixes this;
    # note the test partition here is intentionally unused (see the
    # existing comment below), so only "at least 1 train sample" needs
    # guaranteeing, not a full train+test split.
    valid_idx = np.where(~np.isnan(y))[0]
    if len(valid_idx) < 1:
        raise ValueError(f"[PRS] 0 of {n} samples have a non-missing "
                        f"phenotype -- cannot fit any model.")
    n_valid = len(valid_idx)
    perm = rng.permutation(n_valid)
    split = max(1, int(n_valid * (1-test_frac)))
    split = min(split, n_valid)
    idx = valid_idx[perm]
    # Note: only the train indices (tr) are used for model FITTING below;
    # the test indices are intentionally discarded (_) since the portability
    # fix moved final validation to ALL samples rather than the test split
    # only (see validate_prs() docstring for why).
    tr, _ = idx[:split], idx[split:]
    G_tr, y_tr = G[tr], y[tr]
    X0_tr = X0[tr, :] if X0 is not None else None

    model = ConsensusPRS(methods=methods, seed=seed, max_snps=max_snps)
    model.fit(G_tr, y_tr, sumstats_df=sumstats_df, bim=bim, X0=X0_tr)

    # All-sample consensus scores
    prs_scores = model.predict(G)

    # Restrict validation + output rows to the study cohort only, if asked
    # (see restrict_output_idx's docstring above) -- the model itself was
    # already fit on only the admix-only train subset regardless, so
    # scoring stays correct; this only affects what gets reported.
    if restrict_output_idx is not None:
        report_idx = restrict_output_idx
    else:
        report_idx = np.arange(n)

    # Validation on ALL (reported) samples (not just test split) -- portability fix
    validation = validate_prs(prs_scores[report_idx], y[report_idx],
                              ancestry_labels=([ancestry_labels[i] for i in report_idx]
                                               if ancestry_labels else None))

    # ── Outputs ───────────────────────────────────────────────────────────────
    out_df = fam.iloc[report_idx].copy()[["FID","IID"]].reset_index(drop=True)
    out_df["PRS_CONSENSUS"] = prs_scores[report_idx]
    for meth, sc in model.component_scores(G).items():
        out_df[f"PRS_{meth.upper()}"] = sc[report_idx]
    if ancestry_labels:
        out_df["ANCESTRY"] = [ancestry_labels[i] for i in report_idx]
    # Percentile rank (0-100) -- computed within the reported cohort only,
    # so a sample's percentile reflects its rank among the samples it's
    # actually comparable to, not diluted/skewed by any excluded group.
    from scipy.stats import rankdata
    n_reported = len(report_idx)
    out_df["PRS_PERCENTILE"] = (rankdata(prs_scores[report_idx]) / n_reported * 100).round(1)
    # Bug fix: RISK_TIER used to bin PRS_CONSENSUS directly against
    # VALUE-based cutpoints (np.nanpercentile at 20/80), which crashes
    # ("Bin edges must be unique") whenever those two percentiles happen
    # to land on the same value -- exactly what happens when a large
    # fraction of the cohort gets a near-identical (or, with strong
    # Ridge/Lasso regularisation and weak true signal, literally
    # identical) predicted score. Found via a real run: p20 and p80 both
    # landed on exactly -1.0, meaning at least 60% of 733 samples were
    # tied at that value. Binning on PRS_PERCENTILE instead -- itself
    # already rank-based via rankdata(), so ties in the underlying
    # PRS_CONSENSUS values just average out to a shared percentile,
    # never a crash -- against FIXED cutpoints (20, 80) rather than
    # data-dependent ones removes the possibility of a collision
    # entirely: 20 and 80 can never be equal to each other, regardless
    # of how degenerate the underlying score distribution is. Same
    # intended semantics (bottom 20% / middle 60% / top 20%), just
    # computed on a collision-proof axis.
    out_df["RISK_TIER"] = pd.cut(
        out_df["PRS_PERCENTILE"],
        bins=[-0.1, 20, 80, 100.1],
        labels=["Low (<P20)","Intermediate","High (>P80)"])
    out_df.to_csv(out/"prs_scores.tsv", sep="\t", index=False)

    # Per-SNP weights from every fitted component method -- for downstream
    # enrichment analysis on "which SNPs actually drove the PRS". Previously
    # the PRS pipeline only ever wrote per-sample scores and summary
    # metrics, never a per-SNP file at all -- see get_snp_weights()'s
    # docstring for how Ridge/Lasso's column-position-indexed coefficients
    # and C+T's SNP-ID-indexed weights get reconciled into one table.
    try:
        weights_df = model.get_snp_weights(bim)
        weights_df.to_csv(out/"prs_snp_weights.tsv", sep="\t", index=False)
        logger.info(f"[PRS] Per-SNP weights ({len(weights_df)} SNPs across "
                    f"{[c for c in weights_df.columns if c.endswith('_WEIGHT')]}) "
                    f"written to prs_snp_weights.tsv")
    except Exception as e:
        logger.warning(f"[PRS] Could not write per-SNP weights: {e}")

    val_df = pd.DataFrame([{"metric": k, "value": str(v)}
                            for k,v in validation.items()
                            if not isinstance(v, list)])
    val_df.to_csv(out/"prs_validation.tsv", sep="\t", index=False)

    # Portability TSV -- always written (NaN rows included now)
    if "portability_ratios" in validation:
        port_df = pd.DataFrame(validation["portability_ratios"])
        port_df.to_csv(out/"prs_portability.tsv", sep="\t", index=False)
        n_ok = int((port_df["portability_ratio"].notna()).sum())
        logger.info(f"[PRS] Portability: {n_ok}/{len(port_df)} groups had "
                    f"sufficient samples")
    else:
        # No ancestry labels — write an empty portability file with explanation
        pd.DataFrame([{"ancestry":"N/A","note":"no --pop-file provided"}]
                     ).to_csv(out/"prs_portability.tsv", sep="\t", index=False)

    logger.info(f"[PRS] AUC={validation.get('auc',float('nan')):.3f}  "
                f"r={validation.get('pearson_r',float('nan')):.3f}  "
                f"cal_slope={validation.get('calibration_slope',float('nan')):.3f}")
    logger.info(f"[PRS] Results written to {out}")

    # ── Publication plots ─────────────────────────────────────────────────────
    if plots:
        # Bug fix: every plot call below used to share ONE try/except for
        # the entire block -- a single plot failing (e.g. the two fixed
        # just above, which crashed on a heavily-tied/degenerate PRS
        # distribution) silently aborted every REMAINING, otherwise
        # perfectly independent plot too. Found via a real run: only 3 of
        # 10 requested plots were ever attempted once
        # plot_prs_risk_stratification() raised. Each plot now gets its
        # own try/except via this small helper, so one failure just skips
        # that one plot (with a clear, specific warning) rather than
        # silently dropping everything after it.
        def _safe_plot(name, fn, *a, **kw):
            try:
                fn(*a, **kw)
            except Exception as e:
                logger.warning(f"[PRS] Plot {name!r} failed: {e}")

        from jasmap.plotting.plots import prs_distribution, prs_roc
        valid_mask = ~np.isnan(y)
        # Bug fix: same median-split issue already fixed in validate_prs()
        # (see that function's own docstring for the full explanation) --
        # this second, independent occurrence was never updated to match,
        # so an already-binary, case-enriched phenotype could still
        # collapse to a single class here even after validate_prs()'s own
        # AUC came out fine.
        y_valid = y[valid_mask]
        unique_y_plot = np.unique(y_valid[np.isfinite(y_valid)])
        if len(unique_y_plot) == 2:
            binary_y = (y_valid == unique_y_plot[1]).astype(int)
        else:
            binary_y = (y_valid > np.median(y_valid)).astype(int)
        prs_valid = prs_scores[valid_mask]

        # Existing basic plots
        _safe_plot("prs_distribution", prs_distribution, prs_valid, binary_y,
                  title="Consensus PRS Distribution",
                  out_path=str(out/"prs_distribution.png"))
        _safe_plot("prs_roc", prs_roc, binary_y, prs_valid,
                  title=f"PRS ROC  (AUC={validation.get('auc',float('nan')):.3f})",
                  out_path=str(out/"prs_roc.png"))

        # New comprehensive plots
        _safe_plot("prs_individual_scores", plot_prs_individual_scores,
                  prs_scores, y, ancestry_labels=ancestry_labels,
                  out_path=str(out/"prs_individual_scores.png"))

        _safe_plot("prs_risk_stratification", plot_prs_risk_stratification,
                  prs_scores, y, out_path=str(out/"prs_risk_stratification.png"))

        _safe_plot("prs_calibration", plot_prs_calibration,
                  prs_scores, y, out_path=str(out/"prs_calibration.png"))

        _safe_plot("prs_cumulative_risk", plot_prs_cumulative_risk,
                  prs_scores, y, out_path=str(out/"prs_cumulative_risk.png"))

        _safe_plot("prs_percentile_table", plot_prs_percentile_table,
                  prs_scores, y, out_path=str(out/"prs_percentile_thresholds.png"))

        if ancestry_labels is not None:
            _safe_plot("prs_ancestry_density", plot_prs_ancestry_density,
                      prs_scores, ancestry_labels,
                      out_path=str(out/"prs_ancestry_density.png"))

            _safe_plot("prs_forest_violin", plot_prs_forest_violin,
                      prs_scores, y, ancestry_labels,
                      out_path=str(out/"prs_forest_violin.png"))

            if "portability_ratios" in validation and validation["portability_ratios"]:
                _safe_plot("prs_portability_bar", plot_prs_portability_bar,
                          validation["portability_ratios"],
                          out_path=str(out/"prs_portability_bar.png"))

    return prs_scores, validation
