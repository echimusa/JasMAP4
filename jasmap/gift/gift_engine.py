"""
jasmap.gift.gift_engine
-----------------------
GIFT core integrated into JasMAP.

Provides:
  GIFTEngine            – wraps GenomicInformationalFieldTheory (gift_core.py)
  GIFTAdmixtureEngine   – wraps GIFTAdmixture (gift_fullpipeline.py)
  run_gift()            – called from JasMAP2.py --mode gift
  run_gift_admix()      – called from JasMAP2.py --mode gift_admix
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── re-export the core classes ────────────────────────────────────────────────

try:
    from jasmap.gift.gift_core import GenomicInformationalFieldTheory
except ImportError:
    GenomicInformationalFieldTheory = None

try:
    from jasmap.gift.gift_fullpipeline import (
        GIFTAdmixture as _GIFTAdmixture,
        GlobalAncestryInfer as _GlobalAncestryInfer,
        LocalAncestryInfer as _LocalAncestryInfer,
        AdmixtureMapping as _AdmixtureMapping,
    )
except ImportError:
    _GIFTAdmixture = None
    _GlobalAncestryInfer = None
    _LocalAncestryInfer  = None
    _AdmixtureMapping    = None


# ─────────────────────────────────────────────────────────────────────────────
# run_gift  –  pure GIFT without ancestry
# ─────────────────────────────────────────────────────────────────────────────

def run_gift(G, y, bim, fam, out_prefix,
             n_components=20, lambda_reg=1e-3,
             field_temp=1.0, propagator_tau=0.5,
             mi_chunk=512, test_frac=0.2, seed=42,
             max_snps=5000, plots=True):
    """
    Standalone GIFT association analysis (--mode gift).

    Parameters
    ----------
    G           : np.ndarray float32  (n_samples, n_snps)
    y           : np.ndarray float64  phenotype (NaN = missing)
    bim         : pd.DataFrame        SNP metadata (CHR, SNP, BP, A1, A2)
    fam         : pd.DataFrame        sample metadata (FID, IID)
    out_prefix  : str                 output directory / prefix
    max_snps    : int or None  GIFT's mutual-information matrix and field
                 propagator are BOTH dense (n_snps, n_snps) matrices, and
                 the propagator additionally needs a matrix exponential
                 (expm) -- O(p^2) memory and O(p^3) compute in the SNP
                 count, fundamentally different scaling from GWAS/RVAT/PRS
                 (which scale with SAMPLE count, typically far smaller than
                 SNP count). Found via a real run: at p=338,320 SNPs, the
                 MI matrix alone needs 853 GiB -- an outright crash
                 (numpy._core._exceptions._ArrayMemoryError), not a slow-
                 but-completing run. Above max_snps (default 5000, chosen
                 so the MI matrix is a tractable ~200MB and the matrix
                 exponential completes in seconds to low minutes), a quick
                 univariate pre-filter (reusing PRS's already-validated
                 select_top_snps(), keeping the SNPs most correlated with
                 phenotype) reduces to a size where both steps are
                 actually feasible, with a clear log message rather than a
                 silent behaviour change. Set to None to disable this cap
                 entirely (not recommended above a few thousand SNPs).

    Returns
    -------
    snp_scores  : np.ndarray  normalised GIFT importance per SNP
    y_pred_te   : np.ndarray  test-set predictions
    model       : GenomicInformationalFieldTheory
    """
    if GenomicInformationalFieldTheory is None:
        raise ImportError("gift_core.py could not be imported - check jasmap/gift/")

    out = Path(out_prefix)
    out.mkdir(parents=True, exist_ok=True)

    gift_snp_idx = None
    if max_snps is not None and G.shape[1] > max_snps:
        from jasmap.prs.consensus_prs import select_top_snps
        gift_snp_idx = select_top_snps(G, y, X0=None, max_snps=max_snps, seed=seed)
        logger.info(
            f"[GIFT] {G.shape[1]} SNPs exceeds max_snps={max_snps} -- GIFT's "
            f"mutual-information matrix and field propagator are both dense "
            f"(n_snps x n_snps), with the propagator also needing a matrix "
            f"exponential (O(n_snps^3) compute). Pre-filtered to the top "
            f"{max_snps} SNPs by |correlation| with phenotype (same method "
            f"already validated for PRS's pre-filtering) to keep both steps "
            f"tractable. Pass --gift-max-snps to change this, or 0 to disable "
            f"(not recommended above a few thousand SNPs -- this crashed "
            f"outright, not just slowly, above roughly 5,000-10,000 SNPs).")
        G = G[:, gift_snp_idx]
        bim = bim.iloc[gift_snp_idx].reset_index(drop=True)

    rng = np.random.default_rng(seed)
    # Bug fix: the train/test split previously permuted ALL n samples,
    # with no regard for which have a missing (NaN) phenotype -- found via
    # a real run that crashed with "Found array with 0 sample(s)" deep
    # inside sklearn's StandardScaler, reproduced exactly by constructing
    # a case where a fixed seed's permutation happened to concentrate
    # every phenotyped sample into the OTHER partition, leaving this
    # function's train split entirely NaN. Real cohorts are commonly only
    # partially phenotyped (PLINK's -9 missing-phenotype convention), so
    # this was not a theoretical edge case. Restricting the permutation to
    # only the phenotyped samples up front means neither partition can end
    # up empty due to missingness -- only a genuine shortage of phenotyped
    # samples overall would, and that now raises a clear, immediate error
    # instead of an opaque crash deep inside model fitting.
    valid_mask = ~np.isnan(y)
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) < 2:
        raise ValueError(
            f"[GIFT] Only {len(valid_idx)} of {len(y)} samples have a "
            f"non-missing phenotype -- need at least 2 (one for training, "
            f"one for testing) to proceed.")
    n = len(valid_idx)
    perm = rng.permutation(n)
    split = max(1, int(n * (1-test_frac)))
    split = min(split, n - 1)  # always leave at least 1 test sample
    tr, te = valid_idx[perm[:split]], valid_idx[perm[split:]]
    G_tr, G_te = G[tr], G[te]
    y_tr, y_te = y[tr], y[te]
    fam_te = fam.iloc[te].reset_index(drop=True)

    logger.info(f"[GIFT] {len(tr)} train / {len(te)} test samples")

    t0 = time.time()
    model = GenomicInformationalFieldTheory(
        n_components=n_components, lambda_reg=lambda_reg,
        field_temp=field_temp, propagator_tau=propagator_tau,
        mi_chunk_size=mi_chunk)
    model.fit(G_tr, y_tr)
    logger.info(f"[GIFT] Fit complete in {time.time()-t0:.1f}s")

    te_valid = ~np.isnan(y_te)
    y_pred_te = model.predict(G_te[te_valid])
    snp_scores = model.snp_importance(G_te[te_valid])
    U = model.field_potential(G_te[te_valid])

    # ── Save outputs ─────────────────────────────────────────────────────────
    si = bim.copy(); si["GIFT_SCORE"] = snp_scores
    si = si.sort_values("GIFT_SCORE", ascending=False)
    si.to_csv(out/"snp_importance.tsv", sep="\t", index=False)

    pred_df = fam_te[["FID","IID"]].copy()
    pred_df["Y_TRUE"] = y_te[te_valid]
    pred_df["Y_PRED"] = y_pred_te
    pred_df.to_csv(out/"gift_predictions.tsv", sep="\t", index=False)

    fp_df = fam_te[["FID","IID"]].copy()
    fp_df["FIELD_POTENTIAL"] = U
    fp_df.to_csv(out/"field_potential.tsv", sep="\t", index=False)

    logger.info(f"[GIFT] Results written to {out}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if plots:
        try:
            from jasmap.plotting.plots import (
                gift_snp_importance, gift_field_potential,
                qq_plot, pca_scatter)

            gift_snp_importance(bim, snp_scores, top_n=min(30, len(snp_scores)),
                                title="GIFT SNP Importance",
                                out_path=str(out/"gift_snp_importance.png"))

            gift_field_potential(U, fam_te,
                                 out_path=str(out/"gift_field_potential.png"))

            # QQ plot of GIFT scores converted to approximate p-values
            # (rank-based uniform p-values for illustrative purposes)
            ranks = np.argsort(np.argsort(snp_scores)) + 1
            pvals = ranks / (len(ranks)+1)
            qq_plot(pvals, title="QQ - GIFT Scores (rank-based)",
                    out_path=str(out/"gift_qq.png"))

            # PCA of compressed field representation
            Z = model.pca.transform(
                model._propagate(model._encode_field(G_te[te_valid])))
            pca_scatter(Z[:, :2], title="GIFT Field PCA",
                        out_path=str(out/"gift_field_pca.png"))

        except Exception as e:
            logger.warning(f"[GIFT] Plotting failed: {e}")

    return snp_scores, y_pred_te, model


# ─────────────────────────────────────────────────────────────────────────────
# run_gift_admix  –  GIFT + multi-way admixture mapping
# ─────────────────────────────────────────────────────────────────────────────

def run_gift_admix(G, y, bim, fam, local_ancestry,
                   global_ancestry, out_prefix,
                   n_ancestries=3, ancestry_mode="stratified",
                   n_components=15, lambda_reg=1e-3,
                   mi_chunk=512, test_frac=0.2, seed=42,
                   pop_labels=None, max_snps=5000, plots=True):
    """
    GIFT + multi-way admixture mapping (--mode gift_admix).

    Parameters
    ----------
    G               : np.ndarray (n_samples, n_snps)
    y               : np.ndarray phenotype
    bim, fam        : DataFrames
    local_ancestry  : np.ndarray (n_samples, n_windows, K)  from ConsensusLAI
    global_ancestry : np.ndarray (n_samples, K)             from ConsensusLAI
    out_prefix      : str
    ancestry_mode   : "stratified" | "covariate" | "interaction" | "all"
    pop_labels      : list of K ancestry names
    max_snps        : int or None  GIFTAdmixture has the SAME dense
                     (n_snps, n_snps) mutual-information/field-propagator
                     scaling as plain GIFT (run_gift(), see its max_snps
                     docstring) -- confirmed by reproducing the crash
                     directly at a small, safe scale (killed by the OS,
                     out of memory, during the propagator's matrix
                     exponential step). The fix here is more delicate than
                     run_gift()'s: local_ancestry is stored per WINDOW,
                     then expanded to per-SNP resolution using the
                     ORIGINAL, full SNP count and a fixed 50-SNP/25-step
                     window scheme -- filtering SNPs BEFORE that expansion
                     would misalign it (windows are defined over the
                     original index space, not an arbitrary correlation-
                     selected subset). So the expansion always happens
                     first, unchanged, exactly as before; only AFTER that,
                     if needed, are G/SA/bim reduced together to a shared
                     subset -- using SEPARATELY-NAMED variables so the
                     admixture-mapping scan further below (a different,
                     already-tractable computation, window-count scaling
                     rather than SNP-count squared) keeps using the
                     original, unfiltered G/bim/local_ancestry, exactly as
                     before. Same default (5000) and semantics as
                     run_gift()'s cap; 0 or None disables it.
    """
    if _GIFTAdmixture is None:
        raise ImportError("gift_fullpipeline.py could not be imported - check jasmap/gift/")

    pop_labels = pop_labels or [f"Anc{k+1}" for k in range(n_ancestries)]
    out = Path(out_prefix)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    # Bug fix: same issue and fix as run_gift() above (see its comment for
    # the full explanation and how this was reproduced) -- restricting the
    # permutation to only phenotyped samples up front means neither
    # partition can end up empty due to missingness alone.
    valid_mask = ~np.isnan(y)
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) < 2:
        raise ValueError(
            f"[GIFT-ADMIX] Only {len(valid_idx)} of {len(y)} samples have "
            f"a non-missing phenotype -- need at least 2 (one for "
            f"training, one for testing) to proceed.")
    n = len(valid_idx)
    perm = rng.permutation(n)
    split = max(1, int(n * (1-test_frac)))
    split = min(split, n - 1)
    tr, te = valid_idx[perm[:split]], valid_idx[perm[split:]]
    G_tr, G_te = G[tr], G[te]
    y_tr, y_te = y[tr], y[te]
    Q_tr, Q_te = global_ancestry[tr], global_ancestry[te]
    LA_tr = local_ancestry[tr] if local_ancestry is not None else None
    fam_te = fam.iloc[te].reset_index(drop=True)

    logger.info(f"[GIFT-ADMIX] {len(tr)} train / {len(te)} test | mode={ancestry_mode}")

    # Build per-SNP ancestry tensor from window-level local ancestry --
    # ALWAYS using the full, original SNP count first (see max_snps'
    # docstring above for why this ordering matters).
    if LA_tr is not None:
        n_snps = G.shape[1]
        def _expand(LA):
            SA = np.zeros((LA.shape[0], n_snps, n_ancestries), np.float32)
            wins = [np.arange(s, min(s+50, n_snps))
                    for s in np.arange(0, n_snps, 25)]
            for wi, idx2 in enumerate(wins[:LA.shape[1]]):
                SA[:, idx2, :] = LA[:, wi:wi+1, :]
            return SA
        SA_tr = _expand(LA_tr)
        SA_te = _expand(local_ancestry[te])
    else:
        SA_tr = SA_te = None

    # NOW, only for the GIFT model fit/scoring below (never for the
    # admixture-mapping scan further down, which keeps using the
    # original, unfiltered G/bim/local_ancestry): reduce G/SA/bim
    # together to the same SNP subset if the panel exceeds max_snps.
    # Separately-named variables throughout so nothing downstream can
    # accidentally pick up the filtered versions where the original was
    # intended.
    G_tr_fit, G_te_fit, SA_tr_fit, SA_te_fit, bim_fit = G_tr, G_te, SA_tr, SA_te, bim
    if max_snps is not None and G.shape[1] > max_snps:
        from jasmap.prs.consensus_prs import select_top_snps
        gift_snp_idx = select_top_snps(G_tr, y_tr, X0=None, max_snps=max_snps, seed=seed)
        logger.info(
            f"[GIFT-ADMIX] {G.shape[1]} SNPs exceeds max_snps={max_snps} -- "
            f"GIFTAdmixture's mutual-information matrix and field "
            f"propagator are both dense (n_snps x n_snps) per ancestry, "
            f"with the propagator also needing a matrix exponential -- "
            f"confirmed to crash outright (killed, out of memory) above "
            f"roughly this scale, same as plain GIFT. Pre-filtered to the "
            f"top {max_snps} SNPs by |correlation| with phenotype for the "
            f"GIFT model fit only -- the admixture-mapping scan below is "
            f"unaffected (it scales with window count, not SNP count "
            f"squared, and still uses the full panel). Note: unlike plain "
            f"GIFT, this propagator step runs once per ancestry "
            f"({n_ancestries} here for 'stratified' mode), so the SAME "
            f"max_snps value will run noticeably slower here than for "
            f"plain --mode gift (measured roughly cubic scaling with SNP "
            f"count, consistent with the underlying matrix exponential) -- "
            f"lower --gift-max-snps further if this is still too slow for "
            f"your ancestry count. Pass --gift-max-snps to change this, or "
            f"0 to disable.")
        G_tr_fit = G_tr[:, gift_snp_idx]
        G_te_fit = G_te[:, gift_snp_idx]
        bim_fit = bim.iloc[gift_snp_idx].reset_index(drop=True)
        if SA_tr is not None:
            SA_tr_fit = SA_tr[:, gift_snp_idx, :]
            SA_te_fit = SA_te[:, gift_snp_idx, :]

    t0 = time.time()
    model = _GIFTAdmixture(
        k=n_ancestries, n_components=n_components,
        lambda_reg=lambda_reg, mi_chunk=mi_chunk,
        ancestry_mode=ancestry_mode)
    model.fit(G_tr_fit, y_tr, Q_tr, SA_tr_fit)
    logger.info(f"[GIFT-ADMIX] Fit complete in {time.time()-t0:.1f}s")

    # Scores and predictions
    snp_scores = model.snp_importance(G_te_fit, Q_te, SA_te_fit)
    te_valid = ~np.isnan(y_te)
    y_pred_te = model.gift_scores(G_te_fit[te_valid], Q_te[te_valid],
                                   SA_te_fit[te_valid] if SA_te_fit is not None else None)

    # ── Admixture mapping scan ────────────────────────────────────────────────
    # Deliberately uses the ORIGINAL, unfiltered G/bim/local_ancestry --
    # this scan scales with window count, not SNP count squared, so it
    # was never at risk from the same crash and does not need filtering.
    admix_df = pd.DataFrame()
    if local_ancestry is not None and _AdmixtureMapping is not None:
        # local_ancestry from ConsensusLAI is (n_samples, n_windows, K)
        # but runs with windows that may not align to the 50/25 defaults used here.
        # Use only as many windows as actually exist.
        n_wins_avail = local_ancestry.shape[1]
        wins_for_scan = [np.arange(s, min(s+50, G.shape[1]))
                         for s in np.arange(0, G.shape[1], 25)][:n_wins_avail]
        if wins_for_scan:
            admix_df = _AdmixtureMapping.scan(local_ancestry, y, bim, wins_for_scan)
            admix_df.to_csv(out/"admixture_mapping.tsv", sep="\t", index=False)

    # ── Save outputs ──────────────────────────────────────────────────────────
    si = bim_fit.copy(); si["GIFT_SCORE"] = snp_scores
    si.sort_values("GIFT_SCORE", ascending=False).to_csv(
        out/"gift_admix_snp_importance.tsv", sep="\t", index=False)

    pred_df = fam_te[["FID","IID"]].copy()
    pred_df["Y_TRUE"] = y_te[te_valid]
    pred_df["Y_PRED"] = y_pred_te
    pred_df.to_csv(out/"gift_admix_predictions.tsv", sep="\t", index=False)

    pd.DataFrame(global_ancestry[te],
                 columns=pop_labels).to_csv(
        out/"global_ancestry.tsv", sep="\t", index=False)

    logger.info(f"[GIFT-ADMIX] Results written to {out}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if plots:
        try:
            from jasmap.plotting.plots import (
                gift_snp_importance, global_ancestry_bar,
                admixture_mapping_plot, lai_heatmap, pca_scatter)

            gift_snp_importance(bim_fit, snp_scores,
                                title="GIFT-Admix SNP Importance",
                                out_path=str(out/"gift_admix_snp_importance.png"))

            global_ancestry_bar(global_ancestry[te],
                                pop_labels=pop_labels,
                                title="Global Ancestry (test set)",
                                out_path=str(out/"global_ancestry.png"))

            if len(admix_df):
                admixture_mapping_plot(admix_df, pop_labels=pop_labels,
                                       out_path=str(out/"admixture_mapping.png"))

            if local_ancestry is not None:
                lai_heatmap(local_ancestry[te], ancestry_idx=0,
                            title=f"Local Ancestry - {pop_labels[0]}",
                            out_path=str(out/"lai_heatmap.png"))

            # PCA of global ancestry proportions
            from sklearn.decomposition import PCA as SKPCA
            pca = SKPCA(n_components=min(3, n_ancestries))
            Z = pca.fit_transform(global_ancestry)
            dom = [pop_labels[np.argmax(global_ancestry[i])] for i in range(len(global_ancestry))]
            pca_scatter(Z, labels=dom,
                        var_explained=pca.explained_variance_ratio_*100,
                        title="PCA of Global Ancestry",
                        out_path=str(out/"pca_global_ancestry.png"))

        except Exception as e:
            logger.warning(f"[GIFT-ADMIX] Plotting failed: {e}")

    return snp_scores, y_pred_te, model, admix_df
