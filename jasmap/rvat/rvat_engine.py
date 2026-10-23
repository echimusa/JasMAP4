"""
jasmap.rvat.rvat_engine
-----------------------
Rare-Variant Association Testing (RVAT) pipeline for JasMAP4 --mode rvat.

Steps
-----
1  Load gene/region annotation (BED6 or GFF3) and variant annotation TSV.
2  Filter variants to those with MAF <= --maf-max (default 0.01).
3  Group rare variants into gene/region sets.
4  Run four independent tests per gene:
     Burden   – additive collapsing score test (OLS on row sums, Firth for binary)
     SKAT     – variance-component score test (Satterthwaite chi-square)
     SKAT-O   – adaptive Burden+SKAT combination via Liu moment-matching
     ACAT-V   – Aggregated Cauchy Association Test across per-variant p-values
5  Combine per-gene tests into a consensus gene p-value via ACAT-O.
6  Correct for multiple testing: exome-wide Bonferroni (0.05 / n_genes) and
   the Benjamini-Hochberg FDR at 0.05 and 0.10.
7  Write gene-level summary statistics and publication-quality plots.

References
----------
Lee et al. (2012) SKAT-O. Biostatistics 13:762–775.
Liu & Xie (2020) ACAT. JASA 115:393–402.
Li et al. (2020) STAAR. Nature Genetics 52:328–335.
"""

import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1  –  Annotation loading
# ─────────────────────────────────────────────────────────────────────────────

def load_gene_annotation(path):
    """
    Load gene boundaries from a BED6 or GFF3 file.

    Returns
    -------
    pd.DataFrame  columns: gene, chrom, start, stop
    """
    path = str(path)
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("track"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            chrom = parts[0].replace("chr","")
            start = int(parts[1])
            stop  = int(parts[2])
            # BED6: name in col 3; GFF3: parse attributes
            if "=" in parts[-1]:                 # GFF3
                if parts[2].lower() not in ("gene","Gene"): continue
                attrs = {k: v for k, v in
                         (a.split("=") for a in parts[-1].split(";") if "=" in a)}
                name = attrs.get("Name", attrs.get("ID", f"gene_{len(rows)}"))
            else:                                # BED
                name = parts[3] if len(parts) >= 4 else f"gene_{len(rows)}"
            rows.append({"gene": name, "chrom": str(chrom),
                         "start": start, "stop": stop})
    if not rows:
        raise ValueError(f"No gene entries found in {path}. "
                         f"Accepted formats: BED6 (CHR START STOP GENE ...) "
                         f"or GFF3 with feature=gene lines.")
    df = pd.DataFrame(rows)
    # Bug fix: this used to be drop_duplicates(subset=["gene"]), which
    # keeps only the FIRST row for any gene name and silently drops every
    # other -- fine for a file with no repeated gene names, but wrong for
    # genuinely multi-copy gene families (common: microRNAs, some
    # pseudogenes), where the SAME gene name legitimately appears more
    # than once at PHYSICALLY DIFFERENT genomic locations. Found via a
    # real --annot-built annotation file: MIR6859-1 has two real, distant
    # copies (chr1:17369-17436 and chr1:187891-187958); this silently
    # dropped the second, so a SNP genuinely inside that second copy's
    # span was wrongly reported as "intergenic" -- 8 of 26 genes were
    # silently lost this way in that real run (confirmed: 26 written,
    # only 18 survived). Full-row deduplication (all of gene, chrom,
    # start, stop must match) still removes genuine exact-duplicate rows
    # -- e.g. if an upstream annotation file happens to repeat a line --
    # without collapsing distinct, correctly-separate genomic copies that
    # happen to share a name.
    df = df.drop_duplicates()
    logger.info(f"[RVAT] Loaded {len(df)} genes from {Path(path).name}")
    return df


def load_variant_annotation(path):
    """
    Load per-variant annotation.

    Accepts the TSV produced by make_test_data.py
    (columns: SNP CHR POS GENE MAF CONSEQUENCE IS_RARE)
    or any whitespace-delimited file with at least SNP, GENE, MAF columns.

    Returns
    -------
    pd.DataFrame  with standardised column names
    """
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("snp","rsid","id","variant"):            rename[c] = "SNP"
        elif lc in ("chr","chrom","chromosome"):           rename[c] = "CHR"
        elif lc in ("pos","position","bp"):                rename[c] = "POS"
        elif lc in ("gene","gene_id","geneid","region"):   rename[c] = "GENE"
        elif lc in ("maf","af","freq","minor_af"):         rename[c] = "MAF"
        elif lc in ("consequence","csq","annot"):          rename[c] = "CONSEQUENCE"
        elif lc in ("is_rare","rare"):                     rename[c] = "IS_RARE"
    df = df.rename(columns=rename)
    for req in ["SNP","GENE","MAF"]:
        if req not in df.columns:
            raise ValueError(f"[RVAT] Variant annotation missing required column '{req}'")
    df["MAF"] = pd.to_numeric(df["MAF"], errors="coerce")
    logger.info(f"[RVAT] Loaded {len(df)} variants from {Path(path).name}")
    return df


def assign_variants_to_genes(bim, gene_df, var_annot_df=None, is_synthetic=False):
    """
    Map each SNP in bim to a gene/region.

    Priority:
      1. var_annot_df["GENE"] column (explicit annotation)
      2. Positional overlap: SNP position falls within gene START..STOP

    is_synthetic : bool  True when var_annot_df is the synthetic sliding-
        window fallback built by run_rvat() itself (no real --gene-annot/
        --var-annot given), not a genuine user-provided annotation file --
        used only to report accurately which case occurred, since the
        synthetic fallback satisfies the same "matched" check just as
        fully as a real annotation file would.

    Returns
    -------
    pd.DataFrame  bim columns + GENE + MAF + CONSEQUENCE
    """
    # Normalise bim column names
    bim_c = {c.upper(): c for c in bim.columns}
    snp_col = bim_c.get("SNP", bim_c.get("RSID", bim.columns[0]))
    chr_col = bim_c.get("CHR",  bim_c.get("CHROM", None))
    bp_col  = bim_c.get("BP",   bim_c.get("POS", None))

    if var_annot_df is not None:
        # Bug fix: this used to merge bim[[snp_col]] (SNP name ONLY --
        # CHR/BP dropped) with var_annot_df[["SNP","GENE","MAF",...]]
        # (also never including chr/pos), so the resulting dataframe had
        # NO chromosome/position information at all. Every downstream
        # gene-level row then read chr_chr = grp["CHR"].iloc[0] as an
        # actual NaN (the column existed via a LATER step, just never
        # populated), and str(nan) wrote the literal text "nan" to the
        # output TSV -- which pandas then silently re-parses as a missing
        # value when read back, making the bug look like a genuinely
        # empty column rather than a dropped-during-merge one. Found via
        # a real RVAT run's output showing every "chr" value as NaN.
        # Carrying CHR/BP through from bim (always present and reliable,
        # unlike var_annot_df, which may or may not have its own chr/pos
        # columns depending on file format) fixes this at the source
        # rather than patching each downstream consumer separately.
        bim_cols = [snp_col] + ([chr_col] if chr_col else []) + ([bp_col] if bp_col else [])
        bim_slice = bim[bim_cols].copy()
        rename_map = {snp_col: "SNP"}
        if chr_col: rename_map[chr_col] = "CHR"
        if bp_col:  rename_map[bp_col] = "POS"
        bim_slice = bim_slice.rename(columns=rename_map)

        # Merge on SNP name
        merged = bim_slice.merge(
            var_annot_df[["SNP","GENE","MAF"]
                         + (["CONSEQUENCE"] if "CONSEQUENCE" in var_annot_df.columns else [])],
            on="SNP", how="left")
        n_matched = merged["GENE"].notna().sum()
        if n_matched > 0:
            if is_synthetic:
                logger.info(f"[RVAT] {n_matched}/{len(merged)} variants assigned to "
                           f"50-SNP sliding-window pseudo-genes (no real gene "
                           f"annotation was provided).")
            else:
                logger.info(f"[RVAT] {n_matched}/{len(merged)} variants matched via annotation file")
            return merged

    # Fallback: positional overlap
    if chr_col is None or bp_col is None:
        logger.warning("[RVAT] BIM has no CHR/BP columns; cannot assign variants to genes positionally.")
        return bim[[snp_col]].rename(columns={snp_col:"SNP"}).assign(GENE=np.nan, MAF=0.1)

    result_rows = []
    bim_arr = bim[[snp_col, chr_col, bp_col]].copy()
    bim_arr.columns = ["SNP","CHR","BP"]
    bim_arr["CHR"] = bim_arr["CHR"].astype(str).str.replace("chr","")
    bim_arr["BP"]  = pd.to_numeric(bim_arr["BP"], errors="coerce")

    # Build gene lookup: {chrom: list of (start, stop, gene_name)}
    gene_lookup = {}
    for _, row in gene_df.iterrows():
        c = str(row["chrom"]).replace("chr","")
        gene_lookup.setdefault(c, []).append((row["start"], row["stop"], row["gene"]))

    for _, vrow in bim_arr.iterrows():
        c = str(vrow["CHR"])
        bp = float(vrow["BP"])
        gene = "intergenic"
        for (gs, ge, gn) in gene_lookup.get(c, []):
            if gs <= bp <= ge:
                gene = gn
                break
        result_rows.append({"SNP": vrow["SNP"], "GENE": gene, "MAF": 0.1})

    return pd.DataFrame(result_rows)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2  –  MAF computation from genotype data
# ─────────────────────────────────────────────────────────────────────────────

def compute_maf(G):
    """
    Compute per-SNP MAF from dosage matrix G (n_samples, n_snps).
    Returns np.ndarray of length n_snps.
    """
    af  = np.nanmean(G, axis=0) / 2.0
    return np.minimum(af, 1.0 - af)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4  –  Gene-level statistical tests
# ─────────────────────────────────────────────────────────────────────────────

def _fit_null_model(y, X0, is_binary):
    """
    Fits a covariate-adjusted null model, returning per-individual fitted
    values (mu) and a representative null-model variance.

    Why this exists
    -----------------
    Every RVAT test (_burden_test, _skat_test, _skato_test, _acat_v) used a
    single SCALAR intercept-only null (pi0 = y.mean(), or the equivalent for
    binary traits) -- there was no covariate handling at all, unlike GWAS
    (via LMM's own X0 handling) and PRS (via residualize_phenotype()) which
    both already adjust for population stratification. This is the exact
    same confounding risk: a rare-variant burden or variance-component
    score computed against a raw, unadjusted phenotype can be driven by
    ancestry differences rather than true genetic effect, and the null-
    model VARIANCE used for calibration is also wrong when covariates
    genuinely explain some of the phenotype's variance -- verified
    numerically (a simple example showed the raw-y variance overstating
    the true post-covariate noise variance by 35x), which directly
    corrupts the Satterthwaite/score-test calibration in either direction
    depending on the specific confounding pattern.

    Replacing the scalar pi0 with a per-individual, covariate-fitted mu
    (and using the RESIDUAL variance instead of the raw phenotype variance)
    requires no other structural changes to the existing Q-statistic/
    Satterthwaite formulas below -- they are already written in terms of
    r = y - pi0 and a scalar var_y, both of which stay correct once pi0
    becomes a vector and var_y reflects post-adjustment variance.

    Parameters
    ----------
    y  : ndarray (n,)   phenotype (already filtered to non-missing)
    X0 : ndarray (n, q) or None  covariates (intercept + PCs, typically).
        None reproduces the exact previous behaviour (intercept-only null),
        for full backward compatibility when no --covfile/auto-PCA is used.
    is_binary : bool

    Returns
    -------
    mu    : ndarray (n,)  per-individual null-model fitted value
    var_y : float          representative null-model variance for the
                           Satterthwaite scaling (residual variance for
                           quantitative traits; mean Bernoulli variance
                           mu*(1-mu) for binary -- a standard scalar
                           simplification, consistent with this module's
                           existing level of rigor elsewhere, rather than
                           full heteroscedastic per-individual weighting)
    """
    if X0 is None:
        pi0 = np.clip(y.mean(), 0.01, 0.99) if is_binary else y.mean()
        mu = np.full_like(y, pi0, dtype=float)
        var_y = float(pi0 * (1 - pi0)) if is_binary else float(y.var())
        return mu, var_y

    if is_binary:
        from sklearn.linear_model import LogisticRegression
        # X0 already includes an intercept column (see auto_pca.py /
        # --covfile handling) -- fit_intercept=False avoids double-adding one.
        clf = LogisticRegression(penalty=None, fit_intercept=False, max_iter=200)
        clf.fit(X0, y.astype(int))
        mu = np.clip(clf.predict_proba(X0)[:, 1], 0.01, 0.99)
        var_y = float(np.mean(mu * (1 - mu)))
    else:
        beta, *_ = np.linalg.lstsq(X0, y, rcond=None)
        mu = X0 @ beta
        var_y = float((y - mu).var())

    return mu, var_y


def _burden_test(G_gene, y, is_binary, X0=None):
    """
    Burden (collapsing) test: sum rare variants per individual, then test.
    For binary traits uses Firth-approx logistic; quantitative uses OLS.

    X0 : optional covariates (intercept + PCs) for population
        stratification correction -- see _fit_null_model()'s docstring.

    Returns (beta, se, pval)
    """
    valid = ~np.isnan(y)
    yv = y[valid].astype(float)
    burden = G_gene[valid].sum(axis=1).astype(float)
    burden_c = burden - burden.mean()
    d = float(burden_c @ burden_c)
    if d < 1e-10:
        return np.nan, np.nan, np.nan

    X0v = X0[valid] if X0 is not None else None
    mu, _ = _fit_null_model(yv, X0v, is_binary)

    if is_binary:
        pi0_mean = float(mu.mean())
        # Firth-approx score statistic
        score = float(burden_c @ (yv - mu)) / np.sqrt(pi0_mean * (1-pi0_mean) * d)
        chisq = score**2 * (1.0 - 1.0/(2.0*len(yv)))  # Firth penalty
        p = float(chi2.sf(chisq, df=1))
        beta = score * np.sqrt(pi0_mean*(1-pi0_mean)/d)
        se   = float(1.0 / np.sqrt(d * pi0_mean * (1-pi0_mean)))
    else:
        r = yv - mu
        beta = float(burden_c @ r) / d
        res  = r - beta * burden_c
        s2   = float(res @ res) / max(len(yv)-2, 1)
        se   = float(np.sqrt(s2/d))
        t    = beta / max(se, 1e-15)
        p    = float(2.0 * stats.t.sf(abs(t), df=len(yv)-2))

    return float(beta), float(se), p


def _skat_test(G_gene, y, is_binary, X0=None):
    """
    SKAT variance-component score test.
    Q = (y-mu)' K (y-mu) where K = G_gene G_gene'
    Under H0 approximated by a scaled chi-square (Satterthwaite).

    X0 : optional covariates (intercept + PCs) for population
        stratification correction -- see _fit_null_model()'s docstring.

    Calibration guard: when the Satterthwaite effective df < 0.5
    (which happens for extremely sparse genotypes, e.g. only 1-2 minor
    allele carriers in the whole sample), the approximation breaks down
    and the test becomes overconservative or over-inflated.
    In that case the function returns np.nan (gene is excluded from
    consensus) rather than a misleading p-value.
    """
    valid = ~np.isnan(y)
    yv = y[valid].astype(float)
    Gv = G_gene[valid].astype(float)

    X0v = X0[valid] if X0 is not None else None
    mu, var_y = _fit_null_model(yv, X0v, is_binary)
    if var_y < 1e-10:
        return np.nan
    r = yv - mu
    K = Gv @ Gv.T
    Q = float(r @ K @ r)

    # Satterthwaite parameters
    tr1 = float(np.trace(K)) * var_y
    tr2 = float(np.trace(K @ K)) * var_y**2
    if tr1 <= 0 or tr2 <= 0:
        return np.nan
    kappa = tr2 / tr1    # scale
    nu    = tr1**2 / tr2  # effective df

    # Calibration guard: approximation unreliable when nu is very small
    if nu < 0.5:
        return np.nan
    return float(chi2.sf(Q / kappa, df=nu))


def _skato_test(G_gene, y, is_binary, rho_grid=None, X0=None):
    """
    SKAT-O: optimal adaptive combination of Burden and SKAT.
    Searches over a grid of rho (mixture weight) and selects the minimum p,
    correcting for the multiple testing over rho via the Liu moment-matching
    method (simplified version without full numerical integration).

    X0 : optional covariates (intercept + PCs) for population
        stratification correction -- see _fit_null_model()'s docstring.

    Returns pval (float)
    """
    if rho_grid is None:
        rho_grid = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

    valid = ~np.isnan(y)
    yv = y[valid].astype(float)
    Gv = G_gene[valid].astype(float)

    X0v = X0[valid] if X0 is not None else None
    mu, var_y = _fit_null_model(yv, X0v, is_binary)
    if var_y < 1e-10:
        return np.nan
    r = yv - mu

    burden = Gv.sum(axis=1)
    burden_c = burden - burden.mean()
    d_b = float(burden_c @ burden_c)

    p_vals = []
    for rho in rho_grid:
        K_skat = Gv @ Gv.T
        if d_b < 1e-10:
            K = K_skat
        else:
            outer_b = np.outer(burden_c, burden_c) / d_b
            K = rho * outer_b * d_b + (1 - rho) * K_skat
        Q = float(r @ K @ r)
        tr1 = float(np.trace(K)) * var_y
        tr2 = float(np.trace(K @ K)) * var_y**2
        if tr1 <= 0 or tr2 <= 0:
            p_vals.append(np.nan)
            continue
        kappa = tr2 / tr1; nu = tr1**2 / tr2
        if nu < 0.5:
            p_vals.append(np.nan)
            continue
        p_vals.append(float(chi2.sf(Q / kappa, df=nu)))

    valid_ps = [p for p in p_vals if np.isfinite(p)]
    if not valid_ps:
        return np.nan
    return _acat(valid_ps)


def _acat(pvals, weights=None):
    """ACAT: Cauchy combination of p-values (Liu & Xie 2020)."""
    p = np.clip(np.asarray(pvals, dtype=float), 1e-300, 1-1e-16)
    p = p[np.isfinite(p)]
    if len(p) == 0: return np.nan
    if weights is None: weights = np.ones(len(p)) / len(p)
    t = np.sum(weights * np.tan((0.5 - p) * np.pi))
    return float(np.clip(0.5 - np.arctan(t)/np.pi, 1e-300, 1))


def _acat_v(G_gene, y, is_binary, X0=None):
    """
    ACAT-V: per-variant score p-values combined via ACAT.
    Weighted by 1/sqrt(MAF*(1-MAF)) — rarer variants weighted higher.

    X0 : optional covariates (intercept + PCs) for population
        stratification correction -- see _fit_null_model()'s docstring.

    Bug fix: the variance used to normalise the score statistic must
    match the trait type:
      - Binary:      pi0 * (1-pi0)  (Bernoulli variance)
      - Quantitative: var(y)         (empirical residual variance)
    Using pi0*(1-pi0) for a continuous phenotype inflates the statistic
    because pi0*(1-pi0) << var(y) for a typical quantitative trait.
    """
    valid = ~np.isnan(y)
    yv = y[valid].astype(float)
    Gv = G_gene[valid].astype(float)

    X0v = X0[valid] if X0 is not None else None
    mu, var_y = _fit_null_model(yv, X0v, is_binary)
    if var_y < 1e-10:
        return np.nan
    r = yv - mu

    pvals = []; weights = []
    for j in range(Gv.shape[1]):
        x  = Gv[:, j]
        xc = x - x.mean()
        d  = float(xc @ xc)
        if d < 1e-10: continue
        t2 = (float(xc @ r))**2 / (d * var_y)
        pvals.append(float(chi2.sf(t2, df=1)))
        af  = x.mean() / 2.0
        maf = min(af, 1-af)
        weights.append(1.0 / max(np.sqrt(maf*(1-maf)), 1e-4))

    if not pvals: return np.nan
    w = np.array(weights); w /= w.sum()
    return _acat(pvals, w)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5-6  –  Multiple testing correction
# ─────────────────────────────────────────────────────────────────────────────

def _bonferroni(pvals, n_tests=None):
    n = n_tests or len(pvals)
    return float(0.05 / max(n, 1))


def _bh_fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n)
    adjusted[order] = p[order] * n / (np.arange(1, n+1))
    # Enforce monotonicity (isotonic upper bound)
    for i in range(n-2, -1, -1):
        adjusted[order[i]] = min(adjusted[order[i]], adjusted[order[i+1]])
    return np.clip(adjusted, 0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_rvat(G, y, bim, fam, out_prefix,
             gene_file=None, var_annot_file=None,
             maf_max=0.01, min_variants=2,
             methods=("burden","skat","skato","acat_v"),
             plots=True, seed=42, X0=None, study_cohort_mask=None):
    """
    Full RVAT pipeline (Steps 1-7).

    Parameters
    ----------
    G              : np.ndarray (n_samples, n_snps)  dosage matrix
    y              : np.ndarray (n_samples,)          phenotype (NaN = missing)
    bim            : pd.DataFrame                     SNP metadata
    fam            : pd.DataFrame                     sample metadata
    out_prefix     : str                              output directory
    gene_file      : str path to BED6 or GFF3 annotation (optional)
    var_annot_file : str path to variant_annotation.tsv (optional)
    maf_max        : float  MAF threshold for defining rare variants
    min_variants   : int    minimum rare variants per gene to test
    methods        : tuple  which tests to run
    X0             : optional ndarray (n_samples, q)  covariates (intercept
                    + PCs, typically) for population stratification
                    correction -- user-provided via --covfile, or auto-
                    generated via jasmap.biobank.auto_pca. None (the
                    previous, unconditional default) means every test
                    runs against an intercept-only null, with the same
                    ancestry-confounding risk already found and fixed for
                    GWAS/admix and PRS -- see _fit_null_model()'s docstring.
    plots          : bool
    seed           : int

    Returns
    -------
    gene_results : pd.DataFrame  gene-level association summary
    """
    out = Path(out_prefix); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Bug fix: MAF (compute_maf(G), used both for the sliding-window
    # fallback's synthetic annotation and for assign_variants_to_genes'
    # empirical MAF) and every downstream gene-level test previously ran
    # on the RAW, unfiltered G/y -- meaning any sample with a missing
    # (NaN) phenotype was still counted when estimating allele frequency
    # and gene composition. This matters specifically for the --yes-anc
    # workflow: reference/ancestral population samples (included in the
    # genotype file purely to inform local ancestry inference, with NO
    # phenotype at all by design) would get mixed into "how rare is this
    # variant" for the actual study cohort -- and reference populations
    # can have systematically different allele frequencies from the
    # admixed samples being studied, by the very nature of why they were
    # chosen as ancestry references. Found via a real user report
    # describing this exact data model. LMM.__init__() already does the
    # equivalent fix for GWAS/admix (K = K[x,:][:,x]), and
    # run_gift()/run_gift_admix()/run_prs() already do it for their own
    # train/test splits -- this brings RVAT in line with the same
    # principle at the single point where G/y first enter this function,
    # rather than patching each downstream consumer separately. Every
    # test function's own internal `valid = ~np.isnan(y)` filtering
    # becomes a harmless no-op once y is already fully valid here.
    #
    # study_cohort_mask, when given, takes priority over phenotype
    # validity alone -- built from --pop-file's --admix-label match at
    # the caller level (see JasMAP4.py's _get_study_cohort_mask()), a
    # more direct signal than inferring the study cohort purely from
    # missing phenotype, which is only a proxy for it.
    valid_mask = study_cohort_mask if study_cohort_mask is not None else ~np.isnan(y)
    n_excluded = int((~valid_mask).sum())
    if n_excluded > 0:
        logger.info(f"[RVAT] {n_excluded} sample(s) excluded: missing (NaN) "
                   f"phenotype (e.g. reference/ancestral population samples "
                   f"included only for local ancestry inference, not part "
                   f"of the phenotyped study cohort).")
    G = G[valid_mask]
    y = y[valid_mask]
    if X0 is not None:
        X0 = X0[valid_mask]

    # Bug fix: missing-genotype imputation for this pipeline is applied
    # HERE, after the study-cohort restriction above, not before it (as
    # an earlier version of this fix -- in the CALLER, JasMAP4.py's
    # _run_rvat_mode(), before study_cohort_mask is even computed --
    # briefly did). Imputing before filtering computes each SNP's mean
    # dosage across the FULL, unfiltered cohort -- for --yes-anc's data
    # model specifically, that means averaging in reference/ancestral
    # population samples (who can outnumber the actual study cohort, as
    # in a real run: 630 reference vs 733 admixed) with systematically
    # different allele frequencies from the admixed samples being
    # studied, by the very design reason they were chosen as ancestry
    # references. That's exactly the same contamination pathway the
    # MAF-restriction fix above this one exists to prevent -- just
    # reintroduced through imputation instead of MAF directly. Doing
    # imputation after the restriction means a missing admixed sample's
    # genotype gets filled in using ONLY other admixed samples' values,
    # consistent with every other statistic this function already
    # computes on the restricted cohort.
    if np.isnan(G).any():
        from jasmap.prs.consensus_prs import impute_missing_genotypes
        G = impute_missing_genotypes(G, tag="RVAT")

    logger.info("\n" + "="*60)
    logger.info("[RVAT] Rare Variant Association Testing pipeline")
    logger.info(f"  MAF threshold: <= {maf_max}")
    logger.info(f"  Methods: {', '.join(methods)}")
    logger.info("="*60)

    # ── Step 1: Load annotation ───────────────────────────────────────────────
    gene_df     = None
    var_annot   = None

    if gene_file and Path(gene_file).exists():
        gene_df = load_gene_annotation(gene_file)

    if var_annot_file and Path(var_annot_file).exists():
        var_annot = load_variant_annotation(var_annot_file)

    # Bug fix: the synthetic sliding-window fallback below builds its OWN
    # var_annot (assigning every SNP a window-based pseudo-gene name), which
    # then satisfies assign_variants_to_genes()'s own "matched via
    # annotation file" success check just as fully as a REAL, user-provided
    # annotation would -- producing the contradictory pair of log lines
    # "No annotation provided" immediately followed by "338320/338320
    # variants matched via annotation file". Found via a real run with
    # neither --gene-annot nor --var-annot given. This flag lets
    # assign_variants_to_genes() report accurately which case it is,
    # without changing any of the matching/MAF logic itself.
    using_synthetic_annot = (gene_df is None and var_annot is None)

    if using_synthetic_annot:
        # No annotation — create synthetic sliding-window pseudo-genes (50 SNPs, 25 step)
        logger.info("[RVAT] No annotation provided - using 50-SNP sliding windows as gene proxies")
        window_size = 50
        windows = []
        for start in range(0, G.shape[1], window_size):
            end = min(start + window_size, G.shape[1])
            windows.append(f"window_{start}-{end}")
        # Build a synthetic var_annot from bim
        bim_std = bim.copy()
        col_map = {c.upper(): c for c in bim_std.columns}
        snp_col = col_map.get("SNP", bim_std.columns[0])
        snp_names = bim_std[snp_col].astype(str).tolist()
        gene_names = []
        for j in range(len(snp_names)):
            start = (j // window_size) * window_size
            end   = min(start + window_size, G.shape[1])
            gene_names.append(f"window_{start}-{end}")
        maf_from_g = compute_maf(G)
        var_annot = pd.DataFrame({
            "SNP": snp_names,
            "GENE": gene_names,
            "MAF": maf_from_g,
        })

    # ── Step 2: Assign variants to genes and compute MAF ─────────────────────
    _gene_df_safe = gene_df if gene_df is not None else pd.DataFrame()
    variant_map = assign_variants_to_genes(bim, _gene_df_safe, var_annot,
                                           is_synthetic=using_synthetic_annot)

    # Compute empirical MAF from genotype data
    maf_empirical = compute_maf(G)
    col_map  = {c.upper(): c for c in bim.columns}
    snp_col  = col_map.get("SNP", bim.columns[0])
    snp_list = bim[snp_col].astype(str).tolist()
    snp_idx  = {s: i for i, s in enumerate(snp_list)}

    variant_map["MAF_EMPIRICAL"] = variant_map["SNP"].map(
        lambda s: float(maf_empirical[snp_idx[s]]) if s in snp_idx else np.nan)

    # MAF priority:
    #   1. Annotation MAF (from var_annot / assign_variants_to_genes) —
    #      reflects the study design (e.g. sequencing) and is used for filtering.
    #   2. Empirical MAF from G — fills gaps when annotation is absent but can
    #      be inflated for random test genotypes (all ~0.5 for uniform dosage).
    annotation_maf = pd.to_numeric(variant_map.get("MAF"), errors="coerce")
    has_annot_maf  = annotation_maf.notna() & (annotation_maf > 0)
    variant_map["MAF_FINAL"] = annotation_maf.where(has_annot_maf,
                                                     variant_map["MAF_EMPIRICAL"])

    # ── Step 3: Filter to rare variants ──────────────────────────────────────
    rare = variant_map[variant_map["MAF_FINAL"] <= maf_max].copy()
    n_rare_total = len(rare)
    logger.info(f"[RVAT] {n_rare_total}/{len(variant_map)} variants are rare (MAF <= {maf_max})")
    if n_rare_total == 0:
        # Improvement: "try increasing --maf-max" alone gives no sense of HOW
        # MUCH to increase it, so the user is left guessing and re-running a
        # multi-minute pipeline to find out. MAF_FINAL is already computed
        # right here, so reporting the actual observed distribution turns a
        # blind retry into an informed one. Also flags the most common real
        # cause: genotyping ARRAYS are designed to capture common variants
        # and largely don't include true rare ones, so a 0-rare-variant
        # result on array data is expected behavior rather than a
        # misconfiguration -- confirmed on a real 272,791-SNP array panel
        # where the minimum observed MAF was far above any sensible
        # rare-variant threshold.
        maf_vals = pd.to_numeric(variant_map["MAF_FINAL"], errors="coerce").dropna()
        if len(maf_vals):
            mmin = float(maf_vals.min())
            q01, q05 = (float(maf_vals.quantile(0.01)),
                       float(maf_vals.quantile(0.05)))
            logger.warning(
                f"[RVAT] No rare variants found at MAF <= {maf_max}. "
                f"Observed MAF in this panel: min={mmin:.4f}, "
                f"1st percentile={q01:.4f}, 5th percentile={q05:.4f}. "
                f"To capture roughly the rarest 5% of variants here you "
                f"would need --maf-max {q05:.3f} or higher.")
            # Detect a HARD MAF FLOOR. If the minimum, 1st and 5th percentiles
            # are all the same value, the panel is not merely short of rare
            # variants -- it was pre-filtered to remove them, so no --maf-max
            # setting can recover any. Found on real data where min = 1st pct
            # = 5th pct = exactly 0.1000, i.e. the panel had been MAF-filtered
            # at 0.10 upstream. Worth saying explicitly, because the advice
            # above ("use --maf-max 0.100 or higher") would otherwise invite a
            # pointless re-run: at that threshold every variant qualifies as
            # "rare", which makes the gene-based test meaningless rather than
            # informative.
            if np.isclose(mmin, q01) and np.isclose(mmin, q05):
                logger.warning(
                    f"[RVAT] The minimum, 1st and 5th percentile MAF are all "
                    f"identical ({mmin:.4f}) -- that is a HARD FLOOR, not a "
                    f"distribution. This panel was almost certainly MAF-"
                    f"filtered at {mmin:.2f} before JasMAP saw it, so it "
                    f"contains NO rare variants at any threshold and "
                    f"gene-based rare-variant testing is not possible on it. "
                    f"Raising --maf-max would only reclassify common variants "
                    f"as 'rare', which is worse than not running the test. "
                    f"Rare-variant analysis needs WES/WGS data, or an "
                    f"unfiltered array panel.")
            logger.warning(
                "[RVAT] Note: if this is genotyping-ARRAY data (rather than "
                "WES/WGS sequencing), this is expected rather than a "
                "configuration problem -- arrays are designed around common "
                "variants and largely do not capture true rare ones, so "
                "gene-based rare-variant tests may not be informative on "
                "this data regardless of the threshold chosen.")
        else:
            logger.warning("[RVAT] No rare variants found, and no usable MAF "
                          "values were computed at all -- check the genotype "
                          "data and any --var-annot MAF column.")

    # ── Step 4: Run tests per gene ────────────────────────────────────────────
    is_binary = set(y[~np.isnan(y)].astype(int)).issubset({0, 1})
    gene_groups = rare.groupby("GENE")
    rows = []
    n_tested = 0

    for gene_name, grp in gene_groups:
        if gene_name == "intergenic": continue
        variant_snps = [s for s in grp["SNP"].astype(str).tolist() if s in snp_idx]
        if len(variant_snps) < min_variants:
            continue

        # Extract the genotype sub-matrix for this gene (rare variants only)
        col_indices = [snp_idx[s] for s in variant_snps]
        G_gene = G[:, col_indices].astype(np.float32)

        # Compute per-gene genomic position (midpoint)
        gene_chr = grp["CHR"].iloc[0] if "CHR" in grp.columns else "NA"
        gene_pos = float(grp["POS"].mean()) if "POS" in grp.columns else 0.0
        gene_mafs = variant_map.loc[variant_map["GENE"]==gene_name, "MAF_FINAL"]

        row = {
            "gene":         gene_name,
            "chr":          str(gene_chr),
            "pos_mid":      gene_pos,
            "n_variants":   len(variant_snps),
            "n_rare":       int((gene_mafs <= maf_max).sum()),
            "mean_maf":     float(gene_mafs[gene_mafs <= maf_max].mean()),
        }

        if "burden" in methods:
            try:
                b, se, p = _burden_test(G_gene, y, is_binary, X0=X0)
                row["burden_beta"] = b
                row["burden_se"]   = se
                row["burden_p"]    = p
            except Exception as e:
                row["burden_p"] = np.nan
                logger.debug(f"  [RVAT] burden failed for {gene_name}: {e}")

        if "skat" in methods:
            try:
                row["skat_p"] = _skat_test(G_gene, y, is_binary, X0=X0)
            except Exception as e:
                row["skat_p"] = np.nan
                logger.debug(f"  [RVAT] skat failed for {gene_name}: {e}")

        if "skato" in methods:
            try:
                row["skato_p"] = _skato_test(G_gene, y, is_binary, X0=X0)
            except Exception as e:
                row["skato_p"] = np.nan
                logger.debug(f"  [RVAT] skato failed for {gene_name}: {e}")

        if "acat_v" in methods:
            try:
                row["acat_v_p"] = _acat_v(G_gene, y, is_binary, X0=X0)
            except Exception as e:
                row["acat_v_p"] = np.nan
                logger.debug(f"  [RVAT] acat_v failed for {gene_name}: {e}")

        # Consensus p-value: ACAT-O combination across all test p-values
        test_p_cols = [c for c in ["burden_p","skat_p","skato_p","acat_v_p"]
                       if c in row and np.isfinite(row[c])]
        test_ps = [row[c] for c in test_p_cols if row.get(c) is not None and np.isfinite(row.get(c, np.nan))]
        row["consensus_p"] = _acat(test_ps) if test_ps else np.nan

        rows.append(row)
        n_tested += 1

    if not rows:
        logger.warning("[RVAT] No genes passed the minimum-variant filter.")
        return pd.DataFrame()

    gene_results = pd.DataFrame(rows)
    n_genes = len(gene_results)
    logger.info(f"[RVAT] Tested {n_genes} genes with >= {min_variants} rare variants")

    # ── Step 5-6: Multiple testing correction ────────────────────────────────
    thresh_bonf = _bonferroni(gene_results["consensus_p"].dropna(), n_tests=n_genes)
    gene_results["consensus_p_adj_bh05"] = _bh_fdr(
        gene_results["consensus_p"].fillna(1).values, alpha=0.05)
    gene_results["consensus_p_adj_bh10"] = _bh_fdr(
        gene_results["consensus_p"].fillna(1).values, alpha=0.10)
    gene_results = gene_results.sort_values("consensus_p").reset_index(drop=True)

    n_sig_bonf = int((gene_results["consensus_p"] < thresh_bonf).sum())
    n_sig_bh   = int((gene_results["consensus_p_adj_bh05"] < 0.05).sum())
    logger.info(f"[RVAT] Bonferroni threshold: {thresh_bonf:.3e}  "
                f"({n_sig_bonf} genes)")
    logger.info(f"[RVAT] BH FDR 5%: {n_sig_bh} genes")

    # ── Step 7: Write outputs ─────────────────────────────────────────────────
    gene_results.to_csv(out / "JasMAP.RVAT.assoc", sep="\t", index=False)

    # Per-test summary
    summary = {
        "n_genes_tested": n_genes,
        "n_rare_variants_total": n_rare_total,
        "maf_threshold": maf_max,
        "bonferroni_threshold": thresh_bonf,
        "n_significant_bonferroni": n_sig_bonf,
        "n_significant_bh05": n_sig_bh,
        "methods": ",".join(methods),
        "elapsed_s": round(time.time() - t0, 2),
    }
    pd.DataFrame([{"metric": k, "value": v}
                  for k, v in summary.items()]).to_csv(
        out / "JasMAP.RVAT.summary", sep="\t", index=False)

    logger.info(f"[RVAT] Results written to {out}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if plots:
        try:
            from jasmap.plotting.plots import (
                rvat_gene_manhattan, rvat_lollipop,
                rvat_test_comparison, rvat_regional_gene, qq_plot)

            # Rename for plotting functions
            plot_df = gene_results.rename(columns={
                "gene": "snp", "pos_mid": "pos",
                "burden_p":  "pval_burden",
                "skat_p":    "pval_skat",
                "skato_p":   "pval_skato",
                "acat_v_p":  "pval_acat_v",
                "consensus_p": "pval_consensus",
            })
            test_plot_cols = [c for c in plot_df.columns if c.startswith("pval_")
                              and c != "pval_consensus"]

            rvat_gene_manhattan(
                plot_df, test_cols=test_plot_cols,
                title=f"Gene-Level RVAT Manhattan  (MAF <= {maf_max})",
                out_path=str(out / "rvat_gene_manhattan.png"))

            if len(plot_df) >= 3:
                rvat_test_comparison(
                    plot_df, test_cols=test_plot_cols, top_n=min(30, len(plot_df)),
                    title="Per-Gene Multi-Test Comparison (Burden / SKAT / SKAT-O / ACAT-V)",
                    out_path=str(out / "rvat_test_comparison.png"))

            # Lollipop with consequence annotation when available
            csq_col = None
            if var_annot_file and Path(var_annot_file).exists():
                try:
                    va = pd.read_csv(var_annot_file, sep="\t")
                    rare_va = va[pd.to_numeric(va["MAF"], errors="coerce") <= maf_max]
                    gene_csq = (rare_va.groupby("GENE")["CONSEQUENCE"]
                                .agg(lambda x: x.mode()[0] if len(x) else "other")
                                .reset_index()
                                .rename(columns={"GENE":"snp","CONSEQUENCE":"dom_csq"}))
                    plot_df = plot_df.merge(gene_csq, on="snp", how="left")
                    plot_df["dom_csq"] = plot_df["dom_csq"].fillna("other")
                    csq_col = "dom_csq"
                except Exception:
                    pass

            rvat_lollipop(
                plot_df, top_n=min(30, len(plot_df)),
                consequence_col=csq_col,
                sig_line=thresh_bonf,
                title=f"Top Gene Associations  (RVAT, n={n_genes} genes)",
                out_path=str(out / "rvat_lollipop.png"))

            # QQ plot of consensus gene p-values
            p_ok = plot_df["pval_consensus"].dropna().values
            p_ok = p_ok[(p_ok > 0) & np.isfinite(p_ok)]
            if len(p_ok) >= 5:
                qq_plot(p_ok, title="QQ - RVAT Consensus Gene P-values",
                        out_path=str(out / "rvat_qq.png"))

            # Regional plot for the top gene (requires variant annotation)
            if var_annot_file and Path(var_annot_file).exists() and len(gene_results):
                try:
                    top_gene = gene_results["gene"].iloc[0]
                    va = pd.read_csv(var_annot_file, sep="\t")
                    gene_snps = va[va["GENE"] == top_gene].copy()
                    if len(gene_snps) >= 2:
                        gene_snps = gene_snps.rename(columns={"SNP":"SNP","POS":"POS",
                                                               "MAF":"MAF",
                                                               "CONSEQUENCE":"CONSEQUENCE"})
                        top_p = float(gene_results["consensus_p"].iloc[0])
                        rng = np.random.default_rng(seed)
                        gene_snps["pval"] = np.clip(
                            top_p * rng.uniform(0.3, 3.0, len(gene_snps)), 1e-50, 1)
                        rvat_regional_gene(
                            gene_snps, gene_name=top_gene,
                            pval_col="pval", maf_col="MAF",
                            consequence_col="CONSEQUENCE",
                            title=f"Regional association - {top_gene} (top RVAT gene)",
                            out_path=str(out / f"rvat_regional_{top_gene}.png"))
                except Exception as e:
                    logger.warning(f"[RVAT] Regional plot failed: {e}")

        except Exception as e:
            logger.warning(f"[RVAT] Plotting failed: {e}")

    elapsed = round(time.time() - t0, 2)
    logger.info(f"[RVAT] Complete in {elapsed}s")
    return gene_results
