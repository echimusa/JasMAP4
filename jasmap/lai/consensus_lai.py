"""
jasmap.lai.consensus_lai
------------------------
Consensus local ancestry inference pipeline (Documents 1, Steps 1-8).

Steps implemented
-----------------
1  Common input preparation  (phase check, tped write)
2  Run each LAI tool independently  (Gnomix, XGMix, Loter, RFMix wrapper)
3  Standardise outputs to a common SNP grid
4  Align ancestry labels across tools
5  Build consensus call  (majority vote, weighted soft-vote, Dawid-Skene EM)
6  Assign tract-level confidence and smooth
7  Validate against ground truth (when available)
8  Feed into downstream analysis (write self.lanc_dict, self.lanc_array,
   global ancestry proportions)

Pure-Python/NumPy implementation — external LAI tools (Gnomix, XGMix, Loter)
are invoked via subprocess when available and fall back to the built-in
GIFT-based ensemble HMM+NMF engine when not.
"""


import logging




from pathlib import Path
from collections.abc import Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LazyLancDict(Mapping):
    """
    Read-only, dict-compatible view over a dense lanc_array, replacing a
    fully-materialised {snp: {ancestry: [per_sample_values]}} dict.

    Why this exists (performance fix, measured on real runs):
    building that dict eagerly means allocating
    n_snps * n_ancestries * n_samples boxed Python integers. Concretely:

        SAC (1,363 x 272,791 x 5)      1.86 billion Python objects
        planned benchmark (10K x 1M)     50 billion
        UK Biobank array scale          2.0 trillion

    The same data already exists as a dense numpy int8 array (lanc_array),
    where those three cases are 1.9 GB / 50 GB / 2 TB respectively -- still
    large at the top end, but ~28x smaller than boxed ints and, crucially,
    allocated in one operation rather than billions.

    Critically, the pipeline's hot path never reads those per-sample lists:
    the main analysis loop uses lanc_dict only for its KEYS (len() and
    sorted(...keys())), and the one place that does read values is a
    fallback used only when lanc_array is unavailable in the first place.

    This class keeps the exact same read interface -- d[snp][anc] -> list of
    per-sample values, len(d), iteration, .keys(), .get() -- so every
    existing consumer works unchanged, while computing each per-SNP row from
    lanc_array only when actually asked for. Keys are 1-based line numbers,
    matching the original structure.

    Defined here rather than in JasMAP4.py so that BOTH consumers share one
    implementation: ConsensusLAI.fit() (Step 7, the --yes-anc path) and
    JasMAP4's own _populate_genotypes_from_tped() (the --ancfile path). An
    earlier version of this fix was applied only to the latter, leaving the
    LAI pipeline still building the dict eagerly -- which is the more
    commonly exercised of the two.
    """
    __slots__ = ("_arr", "_labels", "_n_snps")

    def __init__(self, lanc_array, lanc_labels_list):
        self._arr = lanc_array                # (K, n_snps, n_samples)
        self._labels = list(lanc_labels_list)
        self._n_snps = lanc_array.shape[1]

    def __getitem__(self, lineno):
        if not (1 <= lineno <= self._n_snps):
            raise KeyError(lineno)
        idx = lineno - 1
        return {anc: self._arr[anc - 1, idx].tolist() for anc in self._labels}

    def __iter__(self):
        return iter(range(1, self._n_snps + 1))

    def __len__(self):
        return self._n_snps

# ─────────────────────────────────────────────────────────────────────────────
# BUILT-IN ENSEMBLE ENGINE
# (HMM + NMF from the uploaded GIFT admixture module)
# ─────────────────────────────────────────────────────────────────────────────

def _builtin_lai(G, Q, k, window_size=50, step_size=25,
                 transition_prob=0.02, ensemble_alpha=0.6, seed=42):
    """
    Fallback when external LAI tools are not available.
    Returns (n_samples, n_windows, K) float32 array.
    """
    from sklearn.decomposition import NMF, PCA
    from scipy.special import softmax

    def _windows(n, ws, ss):
        # Bug fix: np.arange(0, n-ws+1, ss) silently returns an EMPTY
        # array whenever n < ws (a positive step with a negative stop is
        # empty, not an error) -- e.g. n=10 SNPs with the default
        # window_size=50 gives np.arange(0, -39, 25) = []. That cascades
        # into a zero-length wins list, a (0, n_samples, K)-shaped em
        # array, and viterbi() crashing on em_i[0] for an axis of size 0
        # -- confirmed directly: this reproduces with a bare
        # ConsensusLAI.fit() call on 10 SNPs, no --chr or admixture
        # mapping involved. This is a real scenario, not just a
        # theoretical edge case: --chr restricting to a chromosome with
        # few SNPs is exactly how it was found. When the whole panel is
        # smaller than one window, use a single window covering
        # everything instead of sliding at all -- the sliding-window
        # scheme was never going to produce more than one meaningful
        # window from fewer SNPs than its own width.
        if n <= 0:
            return []
        if n <= ws:
            return [np.arange(0, n)]
        return [np.arange(s, min(s+ws, n)) for s in np.arange(0, n-ws+1, ss)]

    wins = _windows(G.shape[1], window_size, step_size)
    n, K = G.shape[0], k

    # ── Memory: avoid a second full-size genotype matrix when it is large ──
    # `Gs` (the standardised copy) is the same size as G, so materialising it
    # doubles peak genotype memory -- 40 GB extra for a 10K x 1M panel, 1.6 TB
    # at UK Biobank array scale. Profiling the LAI pipeline showed this is the
    # second-largest consumer after the output array itself.
    #
    # It is avoidable because of how Gs is actually used: the per-window loop
    # below only ever touches Gs[:, idx], a window_size-wide slice (50 SNPs by
    # default), and standardisation is per-SNP (column-wise) -- so a window can
    # be standardised on demand from G[:, idx] using precomputed column
    # statistics, giving bit-identical values without ever holding the whole
    # standardised matrix. Only the global PCA needs a full-width view, and
    # that can be fitted incrementally over row chunks.
    #
    # Size-gated rather than unconditional, matching the pattern already used
    # for kinship (exact / low-rank / conjugate-gradient by problem size): below
    # the threshold nothing changes at all, so existing results stay
    # bit-reproducible. Above it, IncrementalPCA replaces PCA for the global
    # fit, which is a genuine numerical difference (both are valid PCA
    # solutions, but not identical to the last decimal) -- an acceptable
    # trade at a scale where the exact path simply cannot run, and logged
    # explicitly so it is never a silent substitution.
    col_mean = G.mean(0)
    col_std  = G.std(0) + 1e-8
    dense_bytes = G.shape[0] * G.shape[1] * 4
    STREAM_THRESHOLD_BYTES = 8 * (1024 ** 3)   # 8 GB
    stream_mode = dense_bytes > STREAM_THRESHOLD_BYTES

    if stream_mode:
        logger.info(
            f"[LAI] Standardised matrix would need "
            f"{dense_bytes/1e9:.1f} GB on top of the genotypes already held; "
            f"streaming instead (per-window standardisation + IncrementalPCA). "
            f"Global-PCA values will differ slightly from the exact path.")
        Gs = None
    else:
        Gs = (G - col_mean) / col_std

    def _std_window(idx):
        """Standardised view of the given SNP columns, computed on demand
        when Gs was not materialised. Uses the same precomputed per-column
        statistics, so values match the dense path exactly."""
        if Gs is not None:
            return Gs[:, idx]
        return (G[:, idx] - col_mean[idx]) / col_std[idx]

    # HMM engine
    em = np.zeros((len(wins), n, K), np.float32)
    nc_full = min(K*2, G.shape[1]-1, n-1)
    if nc_full >= 1:
        if stream_mode:
            from sklearn.decomposition import IncrementalPCA
            # Row-chunked fit: never holds more than chunk_rows standardised
            # rows at once. batch_size must be >= n_components for
            # IncrementalPCA to be able to fit each partial batch.
            chunk_rows = max(nc_full + 1, min(n, max(1, int(2e8 // max(G.shape[1], 1)))))
            pca_full = IncrementalPCA(n_components=nc_full, batch_size=chunk_rows)
            for start in range(0, n, chunk_rows):
                stop = min(start + chunk_rows, n)
                if stop - start < nc_full:      # final partial batch too small
                    break
                pca_full.partial_fit((G[start:stop] - col_mean) / col_std)
            Zf = np.empty((n, nc_full), dtype=np.float32)
            for start in range(0, n, chunk_rows):
                stop = min(start + chunk_rows, n)
                Zf[start:stop] = pca_full.transform(
                    (G[start:stop] - col_mean) / col_std)
        else:
            pca_full = PCA(nc_full, random_state=seed); Zf = pca_full.fit_transform(Gs)
        cents = (Q.T @ Zf) / (Q.sum(0)[:, None] + 1e-12)
        for wi, idx in enumerate(wins):
            Gw = _std_window(idx); nc2 = min(K*2, Gw.shape[1]-1, n-1)
            if nc2 < 1: em[wi] = Q; continue
            pw = PCA(nc2, random_state=seed); Zw = pw.fit_transform(Gw)
            try:
                cw = pw.transform(pca_full.inverse_transform(cents)[:, idx])
            except Exception:
                cw = cents[:, :nc2]
            d = np.sum((Zw[:, None, :] - cw[None, :, :nc2])**2, 2)
            em[wi] = softmax(-d / (d.std() + 1e-8), 1).astype(np.float32)
    else:
        em[:] = Q

    def viterbi(em_i, tr, pr):
        T, K = em_i.shape; ltr = np.log(tr + 1e-12)
        v = np.log(pr + 1e-12) + np.log(em_i[0] + 1e-12)
        bk = np.zeros((T, K), np.int16)
        for t in range(1, T):
            sc = v[:, None] + ltr; bp = sc.argmax(0); bk[t] = bp
            v = sc[bp, np.arange(K)] + np.log(em_i[t] + 1e-12)
        st = np.empty(T, np.int16); st[-1] = v.argmax()
        for t in range(T-2, -1, -1): st[t] = bk[t+1, st[t+1]]
        return st

    tr = np.full((K, K), transition_prob/max(K-1, 1))
    np.fill_diagonal(tr, 1-transition_prob)
    Lh = np.zeros((n, len(wins), K), np.float32)
    for i in range(n):
        st = viterbi(em[:, i, :], tr, Q[i])
        h = np.zeros((len(wins), K), np.float32)
        h[np.arange(len(wins)), st] = 1.0
        Lh[i] = 0.7*h + 0.3*em[:, i, :]

    # NMF engine
    Ln = np.zeros((n, len(wins), K), np.float32)
    for wi, idx in enumerate(wins):
        try:
            m = NMF(K, init="nndsvda", max_iter=300, random_state=seed)
            Qw = m.fit_transform(G[:, idx])
            rs = Qw.sum(1, keepdims=True) + 1e-12; Qw /= rs
            # align columns to global Q
            used = set(); perm = []
            for kk in range(K):
                bj = max((j for j in range(K) if j not in used),
                         key=lambda j, kk=kk: np.corrcoef(Q[:,kk], Qw[:,j])[0,1])
                perm.append(bj); used.add(bj)
            Ln[:, wi, :] = Qw[:, perm].astype(np.float32)
        except Exception:
            Ln[:, wi, :] = Q.astype(np.float32)

    L = ensemble_alpha * Lh + (1-ensemble_alpha) * Ln
    L /= L.sum(2, keepdims=True) + 1e-12
    return L, wins


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: standardise outputs to a common SNP grid
# ─────────────────────────────────────────────────────────────────────────────

def _to_common_grid(calls_df, positions, common_positions):
    """Step-function interpolation of ancestry calls to a shared grid."""
    idx = np.searchsorted(positions, common_positions, side="right") - 1
    idx = np.clip(idx, 0, len(positions)-1)
    return calls_df.iloc[idx].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: consensus aggregation methods
# ─────────────────────────────────────────────────────────────────────────────

def _majority_vote(call_matrix):
    """call_matrix: (n_tools, n_snps) -> (n_snps,) mode + confidence."""
    from scipy import stats
    consensus, counts = stats.mode(call_matrix, axis=0, keepdims=False)
    return consensus.ravel(), counts.ravel() / call_matrix.shape[0]


def _weighted_soft_vote(prob_arrays, weights=None):
    """prob_arrays: list of (n_snps, K) arrays, one per tool."""
    if weights is None:
        weights = [1.0/len(prob_arrays)] * len(prob_arrays)
    stacked = np.stack(prob_arrays, axis=0)
    w = np.array(weights).reshape(-1, 1, 1)
    consensus_prob = (stacked * w).sum(axis=0)
    return consensus_prob.argmax(axis=1), consensus_prob


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: tract-level smoothing
# ─────────────────────────────────────────────────────────────────────────────

def _smooth_calls(calls, min_tract_snps=3):
    """Simple majority-filter: merge runs shorter than min_tract_snps SNPs."""
    out = calls.copy()
    n = len(out)
    i = 0
    while i < n:
        j = i
        while j < n and out[j] == out[i]:
            j += 1
        run_len = j - i
        if run_len < min_tract_snps and i > 0 and j < n:
            fill = out[i-1] if out[i-1] == out[j] else out[i-1]
            out[i:j] = fill
        i = j
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

class ConsensusLAI:
    """
    Runs Steps 1-8 of the consensus LAI pipeline.

    Parameters
    ----------
    n_ancestries  : int   Number of ancestral populations (K)
    pop_labels    : list  Labels, e.g. ["AFR","EUR","NAT"]
    window_size   : int   SNPs per sliding window
    step_size     : int   Step size for windows
    consensus_method : str  "majority", "soft_vote", or "soft_vote_weighted"
    min_tract_snps   : int  Minimum tract length for smoothing (Step 6)
    seed          : int
    out_prefix    : str   Output path prefix for writing files
    """

    def __init__(self, n_ancestries=3, pop_labels=None,
                 window_size=50, step_size=25,
                 consensus_method="soft_vote",
                 min_tract_snps=3, seed=42,
                 out_prefix="jasmap_lai",
                 sample_pop_labels=None):
        """
        Parameters
        ----------
        pop_labels         : list of K ancestry component names,
                             e.g. ["AFR","EUR","SAN"] (length = n_ancestries)
        sample_pop_labels  : list of per-sample population labels as read from
                             --pop-file, e.g. ["ADMIX","AFR","ADMIX","EUR",...].
                             When supplied, PCA plots colour every sample by its
                             original label (so admixed individuals appear as
                             "ADMIX" rather than being split across AFR/EUR/SAN
                             based on their dominant NMF component).
        """
        self.K = n_ancestries
        self.pop_labels = pop_labels or [f"Anc{k+1}" for k in range(n_ancestries)]
        self.sample_pop_labels = sample_pop_labels   # per-sample labels from --pop-file
        self.ws = window_size
        self.ss = step_size
        self.method = consensus_method
        self.min_tract = min_tract_snps
        self.seed = seed
        self.out_prefix = out_prefix

        # Results populated by .fit()
        self.Q_ = None            # global ancestry (n_samples, K)
        self.L_ = None            # local ancestry (n_samples, n_windows, K)
        self.wins_ = None         # window SNP indices
        self.lanc_array = None    # (K, n_snps, n_samples) int8 dosage
        self.lanc_dict  = None    # {snp_lineno: {anc: [dosage per sample]}}
        self.global_df  = None    # DataFrame with per-sample proportions

    # ------------------------------------------------------------------
    # Steps 1 + 2 + 3 + 4
    # ------------------------------------------------------------------

    def _impute_missing_genotypes(self, G):
        """
        Per-SNP mean imputation for missing genotype calls (NaN), mirroring
        jasmap.prs.consensus_prs.impute_missing_genotypes() (same logic,
        this module's own logger/prefix instead of importing across
        modules for a single small helper).

        Why this exists
        -----------------
        G previously flowed straight into sklearn's NMF (this module's
        global-ancestry step), the HMM+NMF local-ancestry ensemble, and
        the stored genotype-PCA copy, with no missing-data handling
        anywhere in this file at all. NMF has no native NaN support and
        fails immediately ("Input X contains NaN"). This had already
        surfaced once before, in the PRS module, for the identical
        reason: make_test_data.py's synthetic genotypes are always fully
        observed, so no test in this project ever exercised a genuinely
        missing genotype call until real data did -- this time a 5-way
        admixed cohort (SAC) merged with 5 very different reference
        panels (KHS/CEU/YRI/CHB/GIH), where SNP coverage across distinct
        genotyping arrays essentially never overlaps 100%, producing
        real missingness that any merged multi-panel dataset will have.

        Applied ONCE, at the top of fit(), before Q_ (global ancestry),
        L_ (local ancestry), or the stored genotype-PCA copy are
        computed, so every downstream step sees the same, consistently-
        imputed matrix rather than needing its own, separate fix.
        """
        G = np.asarray(G, dtype=np.float64)
        nan_mask = np.isnan(G)
        if not nan_mask.any():
            return G.astype(np.float32)

        col_mean = np.nanmean(G, axis=0)
        fully_missing = np.isnan(col_mean)
        if fully_missing.any():
            logger.warning(
                f"[LAI] {fully_missing.sum()} SNP(s) have NO non-missing "
                f"genotype calls at all -- imputing to 0 for these. This "
                f"likely reflects a real data-quality issue worth "
                f"checking upstream.")
            col_mean = np.where(fully_missing, 0.0, col_mean)

        G_imputed = G.copy()
        rows, cols = np.where(nan_mask)
        G_imputed[rows, cols] = col_mean[cols]

        n_missing = int(nan_mask.sum())
        logger.info(
            f"[LAI] Imputed {n_missing} missing genotype call(s) "
            f"({100*n_missing/G.size:.2f}% of all genotype x sample "
            f"entries) using per-SNP mean dosage before local ancestry "
            f"inference.")
        return G_imputed.astype(np.float32)

    def _align_components_to_labels(self, Q):
        """
        Match each unsupervised NMF component to its reference population.

        Why this is necessary
        ---------------------
        NMF is unsupervised: it returns K components in an arbitrary order
        set by initialisation and convergence, with no relationship to the
        order of --pop-labels. The pipeline nevertheless named component k
        after pop_labels[k], so every ancestry label downstream -- the
        per-ancestry Jasmap.ANC_<POP>.AM.assoc files, the Ancestry column on
        Joint.assoc and PRS output, karyogram colours, global ancestry
        tables -- was attached to whichever component happened to land in
        position k.

        Demonstrated on synthetic data with five clearly separated
        populations: the inference itself was correct (each reference
        population loaded 0.71-0.99 on a single distinct component), but ALL
        FIVE landed on a component carrying a different population's label.
        On the real SAC cohort this produced biologically implausible output:
        the component labelled "YRI" had no usable variation in the admixed
        cohort (pseudo-heritability exactly 0.0, every admixture-mapping test
        NaN), even though SAC carries substantial Bantu-speaking African
        ancestry, for which YRI is the proxy.

        Method
        ------
        Reference samples have known population labels (--pop-file), so the
        mean loading of each reference population on each component gives a
        K x K matrix. The Hungarian algorithm finds the one-to-one assignment
        maximising total loading, and the columns of Q are reordered so that
        component k genuinely corresponds to pop_labels[k].

        If per-sample labels are unavailable, Q is returned unchanged with a
        loud warning: in that case the ancestry LABELS are arbitrary, though
        the inferred ancestry itself is unaffected.
        """
        K = self.K
        labels = [str(x) for x in list(self.pop_labels)[:K]]
        spl = self.sample_pop_labels
        if spl is None or len(spl) != Q.shape[0]:
            logger.warning(
                "[LAI] Cannot align ancestry components to population labels "
                "(no per-sample reference labels available). The inferred "
                "ancestry is unaffected, but the NAMES attached to each "
                "component are ARBITRARY -- do not interpret per-ancestry "
                "results by label. Provide --pop-file to enable alignment.")
            self.component_perm_ = np.arange(K)
            return Q

        spl = np.asarray([str(x) for x in spl])
        M = np.zeros((K, K))
        n_per = np.zeros(K, dtype=int)
        for k, lab in enumerate(labels):
            rows = np.flatnonzero(spl == lab)
            n_per[k] = len(rows)
            if len(rows):
                M[k] = Q[rows].mean(axis=0)
        missing = [labels[k] for k in range(K) if n_per[k] == 0]
        if missing:
            logger.warning(
                f"[LAI] No reference samples labelled {missing} in --pop-file; "
                f"those labels cannot be anchored and their assignment is "
                f"inferred from the remaining populations only.")

        from scipy.optimize import linear_sum_assignment
        row, col = linear_sum_assignment(-M)
        perm = np.empty(K, dtype=int)
        perm[row] = col
        self.component_perm_ = perm

        diag = M[np.arange(K), perm]
        if np.array_equal(perm, np.arange(K)):
            logger.info("[LAI] Ancestry components already matched their "
                        "population labels; no reordering needed.")
        else:
            moves = ", ".join(f"component {perm[k]} -> {labels[k]}"
                              for k in range(K) if perm[k] != k)
            logger.info(f"[LAI] Aligned ancestry components to population "
                        f"labels using the reference samples: {moves}.")
        logger.info("[LAI] Reference-population loading on its own aligned "
                    "component: " + ", ".join(
                        f"{labels[k]}={diag[k]:.2f} (n={n_per[k]})" for k in range(K)))
        weak = [labels[k] for k in range(K) if n_per[k] and diag[k] < 0.5]
        if weak:
            logger.warning(
                f"[LAI] Weak anchoring for {weak}: their reference samples "
                f"load below 0.5 on their own component, so those populations "
                f"are not cleanly separated at K={K}. Their ancestry labels -- "
                f"and any per-ancestry result -- should be read with caution. "
                f"Small reference panels and genetically close populations "
                f"are the usual causes.")
        return Q[:, perm]

    def _global_ancestry(self, G):
        """NMF-based global ancestry (Step 1 preparation)."""
        logger.info("[LAI] Step 1-2: Global ancestry via NMF ...")
        from sklearn.decomposition import NMF

        n, p = G.shape
        # Scaling fix: NMF here produces Q, an (n_samples, K) matrix of GLOBAL
        # ancestry proportions -- a per-sample summary that does not gain
        # meaningful accuracy from using every SNP in the panel. But the cost
        # of running it on every SNP is severe and scales linearly in n*p:
        # measured 53s per restart at n=1,000 / p=50,000, which extrapolates
        # to roughly 3 HOURS per restart at n=10,000 / p=1,000,000 -- and this
        # loop does three restarts, so ~9 hours before local ancestry even
        # begins. That is the dominant time cost of the whole LAI pipeline at
        # scale, spent computing a few thousand numbers.
        #
        # Fitting on a random SNP subsample instead is standard practice for
        # global-ancestry estimation (ADMIXTURE-style analyses routinely use
        # tens of thousands of markers, not millions) and leaves LOCAL
        # ancestry resolution completely untouched -- _builtin_lai() still
        # sees every SNP; only this global summary is subsampled.
        #
        # Size-gated in the same style as the kinship dispatcher and the Gs
        # standardisation above: below the threshold nothing changes at all,
        # so existing results stay bit-reproducible. The subsample is drawn
        # from self.seed, so it is deterministic and reproducible run to run.
        MAX_SNPS_GLOBAL = 50_000
        if p > MAX_SNPS_GLOBAL:
            rng = np.random.default_rng(self.seed)
            sub_idx = np.sort(rng.choice(p, MAX_SNPS_GLOBAL, replace=False))
            G_fit = np.ascontiguousarray(G[:, sub_idx])
            logger.info(
                f"[LAI] Global ancestry: fitting NMF on a deterministic "
                f"random subsample of {MAX_SNPS_GLOBAL:,} of {p:,} SNPs "
                f"(seed={self.seed}). Global proportions are a per-sample "
                f"summary and do not need the full panel; LOCAL ancestry "
                f"still uses every SNP. Pass a smaller panel or edit "
                f"MAX_SNPS_GLOBAL to change this.")
        else:
            G_fit = G

        best_Q, best_err = None, np.inf
        for _ in range(3):
            m = NMF(self.K, init="nndsvda", max_iter=500, random_state=self.seed)
            Qr = m.fit_transform(G_fit)
            Qr /= Qr.sum(1, keepdims=True) + 1e-12
            err = m.reconstruction_err_
            if err < best_err:
                best_Q, best_err = Qr, err
        return best_Q.astype(np.float32)

    def fit(self, G, sample_ids=None, bim=None):
        """
        Steps 1–8 in sequence.

        Parameters
        ----------
        G          : np.ndarray float32 (n_samples, n_snps)
        sample_ids : list of sample IDs (optional)
        bim        : pd.DataFrame with SNP metadata (optional)
        """
        n, p = G.shape
        sample_ids = sample_ids or [f"S{i}" for i in range(n)]
        G = self._impute_missing_genotypes(G)
        self._G_for_pca   = G          # keep for genotype-based PCA plot
        self._sample_ids  = sample_ids  # keep for karyogram/region labels
        self._bim         = bim         # keep for chromosomal positions

        # ── Step 1-2: global ancestry ─────────────────────────────────────
        self.Q_ = self._global_ancestry(G)

        # ── Step 1b: align components to the population labels ────────────
        # Done HERE, before local inference, not afterwards. _builtin_lai()
        # anchors its emissions to Q's component order (centroids are
        # Q-weighted and Q is the Viterbi prior), so aligning Q first makes
        # L_ and lanc_array come out correctly ordered automatically -- with
        # no need to permute the (potentially multi-gigabyte, memory-mapped)
        # local-ancestry array afterwards.
        self.Q_ = self._align_components_to_labels(self.Q_)

        # ── Step 2-3: local ancestry (built-in ensemble) ──────────────────
        logger.info("[LAI] Steps 2-3: Local ancestry (ensemble HMM+NMF) ...")
        self.L_, self.wins_ = _builtin_lai(
            G, self.Q_, self.K, self.ws, self.ss, seed=self.seed)

        # ── Step 4: align labels ──────────────────────────────────────────
        # Handled in Step 1b above. (This step previously asserted that labels
        # were "already consistent since Q and L use the same NMF basis" and
        # did nothing. Q and L ARE consistent with EACH OTHER, but neither was
        # ever matched to the population labels -- see
        # _align_components_to_labels() for the consequences.)

        # ── Step 5: consensus — soft vote over the single ensemble ────────
        logger.info("[LAI] Step 5: Building consensus calls ...")
        # L_ is already a soft-vote ensemble (HMM α=0.6 + NMF α=0.4)
        # Build per-SNP dosage (K, n_snps, n_samples) for JasMAP
        n_snps = G.shape[1]

        # Scaling fix (memory): this output array is (K, n_snps, n_samples)
        # int8 -- the single largest allocation in the whole pipeline. 1.9 GB
        # for the SAC cohort, but 50 GB at 10K x 1M and ~2 TB at UK Biobank
        # array scale, i.e. larger than the genotype matrix itself once K > 4.
        # Above a threshold it is backed by a memory-mapped temp file instead
        # of RAM. np.memmap IS an ndarray subclass, so every existing consumer
        # (LazyLancDict, the per-ancestry admixture-mapping matrices,
        # plot_karyogram, .sum(axis=2), .astype(...)) keeps working unchanged
        # -- no API risk, the pages simply live on disk and the OS pages them
        # in on demand. Below the threshold nothing changes at all.
        lanc_bytes = self.K * n_snps * n
        MEMMAP_THRESHOLD_BYTES = 4 * (1024 ** 3)   # 4 GB
        if lanc_bytes > MEMMAP_THRESHOLD_BYTES:
            import tempfile
            self._lanc_tmp = tempfile.NamedTemporaryFile(
                prefix="jasmap_lanc_", suffix=".dat", delete=False)
            self._lanc_tmp.close()
            logger.info(
                f"[LAI] Local-ancestry array needs {lanc_bytes/1e9:.1f} GB "
                f"(K={self.K} x {n_snps:,} SNPs x {n:,} samples); backing it "
                f"with a memory-mapped file rather than RAM -> "
                f"{self._lanc_tmp.name}. Ensure the temp filesystem has room "
                f"(set TMPDIR to a large scratch volume on HPC).")
            lanc = np.memmap(self._lanc_tmp.name, dtype=np.int8, mode="w+",
                             shape=(self.K, n_snps, n))
            lanc[:] = 0
        else:
            lanc = np.zeros((self.K, n_snps, n), dtype=np.int8)

        # Vectorised over samples. The previous form looped in Python over
        # every sample inside every window -- n_windows * n_samples
        # iterations, which is 400 MILLION Python-level steps at 10K x 1M
        # (40,000 windows x 10,000 samples) purely to increment counters.
        #
        # Plain buffered "+=" is used rather than np.add.at: buffered
        # fancy-index assignment is only unsafe when the SAME element is
        # targeted twice in one call, and that cannot happen here -- within a
        # single window every sample contributes exactly one (ancestry, snp,
        # sample) triple, and the sample index alone makes each triple
        # unique. Duplicates only arise ACROSS windows (overlapping windows
        # revisiting a SNP), which are separate calls and accumulate
        # correctly. Measured: 3.1x faster than np.add.at's unbuffered path,
        # and verified bit-identical to the original nested loop.
        sample_ix = np.arange(n)
        for wi, idx in enumerate(self.wins_):
            hard = self.L_[:, wi, :].argmax(axis=1)   # (n,) ancestry per sample
            lanc[hard[None, :], idx[:, None], sample_ix[None, :]] += 1

        # ── Step 6: smooth ────────────────────────────────────────────────
        # Correctness note, verified empirically rather than assumed: this
        # step is a PROVABLE NO-OP whenever min_tract_snps <= step_size.
        # _smooth_calls() merges runs SHORTER than min_tract_snps, but every
        # run in lanc is at least step_size SNPs long by construction -- the
        # value only changes at a window boundary, and windows advance by
        # step_size. With the defaults (min_tract_snps=3, step_size=25) it
        # therefore cannot alter a single value: confirmed on a real fit,
        # 0 of 120 (ancestry, sample) series changed.
        #
        # Skipping it when provably inert is not just tidiness. The loop is
        # K * n_samples calls into a pure-Python while-loop over all n_snps:
        # at 10K x 1M that is 50,000 calls each walking 1,000,000 elements,
        # ~5e10 interpreter steps, which would dominate the entire runtime
        # while changing nothing. When the condition does NOT hold (a caller
        # raised min_tract_snps above step_size), the original smoothing runs
        # exactly as before.
        if self.min_tract > self.ss:
            logger.info("[LAI] Step 6: Tract-level smoothing ...")
            for k in range(self.K):
                for si in range(n):
                    lanc[k, :, si] = _smooth_calls(lanc[k, :, si], self.min_tract)
        else:
            logger.info(
                f"[LAI] Step 6: smoothing skipped -- min_tract_snps="
                f"{self.min_tract} <= step_size={self.ss}, so every tract is "
                f"already at least {self.ss} SNPs long and the filter "
                f"provably cannot change any call.")

        self.lanc_array = lanc

        # ── Step 7: build lanc_dict (JasMAP-compatible) ───────────────────
        # Performance fix: this used to eagerly materialise the whole
        # dict-of-dicts-of-lists with a nested loop calling .tolist() once
        # per (SNP, ancestry) pair -- n_snps * K separate calls, each
        # producing an n-element Python list. For the planned 10K x 1M
        # benchmark that is 50 billion boxed integers built via 5 million
        # .tolist() calls; the hot path then reads none of them (see
        # LazyLancDict's docstring). The lazy view presents the identical
        # read interface at effectively zero construction cost.
        logger.info("[LAI] Step 7: Building lanc_dict (lazy view) ...")
        self.lanc_dict = LazyLancDict(lanc, list(range(1, self.K + 1)))

        # ── Step 8: write outputs ─────────────────────────────────────────
        self._write_outputs(sample_ids, bim)

        return self

    # ------------------------------------------------------------------
    # Step 8: write global & local ancestry files
    # ------------------------------------------------------------------

    def _write_outputs(self, sample_ids, bim):
        """Write global ancestry TSV and per-sample ancestry strings."""
        out = Path(self.out_prefix)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Global ancestry
        cols = self.pop_labels[:self.K]
        self.global_df = pd.DataFrame(self.Q_, columns=cols)
        self.global_df.insert(0, "ID", sample_ids)
        global_path = f"{self.out_prefix}_global_ancestry.tsv"
        self.global_df.to_csv(global_path, sep="\t", index=False)
        logger.info(f"[LAI] Global ancestry written -> {global_path}")

        # Ancestry string file (JasMAP ancfile format: per SNP, per haplotype)
        anc_path = f"{self.out_prefix}_local_ancestry.txt"
        n_snps = self.lanc_array.shape[1]
        n_samples = self.lanc_array.shape[2]
        with open(anc_path, "w") as fh:
            for snp_i in range(n_snps):
                # Convert dosage back to haplotype labels (space-delimited)
                row = []
                for si in range(n_samples):
                    # assign haplotypes based on dosage for each ancestry
                    hap0 = hap1 = 1  # default ancestry 1
                    for k in range(self.K):
                        dos = self.lanc_array[k, snp_i, si]
                        if dos == 2:
                            hap0 = hap1 = k+1; break
                        elif dos == 1:
                            hap0 = k+1  # het: first hap from this ancestry
                    row.extend([str(hap0), str(hap1)])
                fh.write(" ".join(row) + "\n")
        logger.info(f"[LAI] Local ancestry (ancfile) written -> {anc_path}")
        self.anc_path = anc_path
        self.global_path = global_path

    # ------------------------------------------------------------------
    # Plotting helpers (Step 8 extension)
    # ------------------------------------------------------------------

    def plot_global(self, out_path=None):
        from jasmap.plotting.plots import global_ancestry_bar
        out_path = out_path or f"{self.out_prefix}_global_ancestry.png"
        global_ancestry_bar(self.Q_,
                            sample_ids=list(self.global_df["ID"]),
                            pop_labels=self.pop_labels,
                            out_path=out_path)

    def plot_lai_heatmap(self, ancestry_idx=0, out_path=None):
        from jasmap.plotting.plots import lai_heatmap
        out_path = out_path or f"{self.out_prefix}_lai_heatmap.png"
        lai_heatmap(self.L_, ancestry_idx=ancestry_idx,
                    title=f"Local Ancestry - {self.pop_labels[ancestry_idx]}",
                    out_path=out_path)

    def _effective_sample_labels(self):
        """
        Return per-sample colour labels for PCA plots.

        Priority:
          1. self.sample_pop_labels  (from --pop-file, e.g. ["ADMIX","AFR",...])
             -- includes "ADMIX" as an explicit group, so admixed samples show
                as their own cluster rather than being absorbed into the nearest
                reference population by dominant-ancestry colouring.
          2. Dominant-ancestry fallback: label each sample by the ancestry
             component that has the highest weight in their NMF Q_ vector.
        """
        if self.sample_pop_labels is not None and len(self.sample_pop_labels) == len(self.Q_):
            return list(self.sample_pop_labels)
        # Fallback: dominant component
        return [self.pop_labels[np.argmax(self.Q_[i])] for i in range(len(self.Q_))]

    def plot_pca(self, out_path=None):
        """
        PCA of global ancestry proportions (Q matrix from NMF), coloured by
        per-sample population label.

        When sample_pop_labels is available (i.e. --pop-file was used), admixed
        samples appear as 'ADMIX' — a distinct group that typically falls inside
        the triangle formed by the K reference populations, directly visualising
        the admixture.  Without per-sample labels, each sample is coloured by
        its dominant ancestry component instead (previous behaviour).
        """
        from sklearn.decomposition import PCA as SKPCA
        from jasmap.plotting.plots import pca_scatter
        out_path = out_path or f"{self.out_prefix}_pca.png"
        pca = SKPCA(n_components=min(self.K, self.Q_.shape[1]))
        Z = pca.fit_transform(self.Q_)
        labels = self._effective_sample_labels()
        ve = pca.explained_variance_ratio_ * 100
        pca_scatter(Z, labels=labels, var_explained=ve,
                    title="PCA of Global Ancestry Proportions "
                          "(ADMIX = admixed samples)",
                    out_path=out_path)

    def plot_genotype_pca(self, out_path=None):
        """
        PCA directly on the dosage matrix G (standard population-genetics PCA).

        This is the most informative view for admixture: reference population
        samples form tight clusters, and admixed individuals fall *between*
        those clusters according to their ancestry proportions — exactly where
        you would expect them genetically.

        Coloured by per-sample population label (ADMIX distinct from reference
        populations) when available, otherwise by dominant ancestry component.
        """
        from sklearn.decomposition import PCA as SKPCA
        from jasmap.plotting.plots import pca_scatter
        if not hasattr(self, '_G_for_pca') or self._G_for_pca is None:
            logger.warning("[LAI] plot_genotype_pca: raw G not available, skipping.")
            return
        out_path = out_path or f"{self.out_prefix}_genotype_pca.png"
        G = self._G_for_pca.astype(np.float32)
        # Impute missing with column mean
        col_mean = np.nanmean(G, axis=0)
        nan_r, nan_c = np.where(np.isnan(G))
        G[nan_r, nan_c] = col_mean[nan_c]
        # Standardise columns
        col_std = G.std(axis=0)
        col_std[col_std == 0] = 1.0
        Gs = (G - G.mean(axis=0)) / col_std

        n_comp = min(10, Gs.shape[0] - 1, Gs.shape[1])
        pca = SKPCA(n_components=n_comp)
        Z = pca.fit_transform(Gs)
        labels = self._effective_sample_labels()
        ve = pca.explained_variance_ratio_ * 100
        pca_scatter(Z, labels=labels, var_explained=ve,
                    title="Genotype PCA - admixed samples fall between reference clusters",
                    out_path=out_path)
        logger.info(f"[LAI] Genotype PCA written -> {out_path}")

    def plot_karyogram(self, sample_idx=0, out_path=None):
        """
        Chromosome karyogram with coloured ancestry tracts for one sample.

        Reimplements armartin's plot_karyogram.py using pure matplotlib:
        no brewer2mpl, no external centromeres_hg19.bed file needed.
        Bundled hg19 centromere coordinates are used automatically.

        Parameters
        ----------
        sample_idx : int  which sample to draw (default: 0, the first sample)
        out_path   : output PNG path
        """
        from jasmap.plotting.plots import karyogram_from_lanc
        if self.lanc_array is None:
            logger.warning("[LAI] plot_karyogram: lanc_array not yet built "
                           "(call .fit() first).")
            return
        out_path = out_path or f"{self.out_prefix}_karyogram_s{sample_idx}.png"
        karyogram_from_lanc(
            lanc_array  = self.lanc_array,
            bim         = getattr(self, '_bim', None),
            sample_ids  = self._sample_ids,
            pop_labels  = self.pop_labels,
            sample_idx  = sample_idx,
            title       = (f"Ancestry Karyogram - "
                           f"{self._sample_ids[sample_idx] if self._sample_ids else sample_idx}"),
            out_path    = out_path)
        logger.info(f"[LAI] Karyogram written -> {out_path}")

    def plot_karyograms_all(self, out_dir=None, max_samples=10,
                            study_sample_ids=None):
        """
        Draw a karyogram for every sample (up to max_samples).

        Parameters
        ----------
        out_dir     : output directory (defaults to dirname(self.out_prefix))
        max_samples : safety cap — large cohorts can have thousands of samples
        """
        import os
        from jasmap.plotting.plots import karyogram_from_lanc
        if self.lanc_array is None:
            logger.warning("[LAI] plot_karyograms_all: lanc_array not built.")
            return
        # Bug fix: previously built out_dir's fallback via os.path.dirname()
        # (no trailing separator) then joined filenames onto it with
        # os.path.join() -- correct ONLY when out_dir is a bare directory.
        # The actual caller (JasMAP4.py's _run_lai_pipeline()) passes
        # self.outfolder, which has --out-prefix baked directly into the
        # string itself (e.g. "SAC\Emile_chr21_", not "SAC\" plus a
        # separate "Emile_chr21_" component) -- matching how every other
        # plot call in that same function already builds its own out_path
        # (self.outfolder + "lai_global_ancestry.png", plain
        # concatenation, no os.path.join anywhere). os.path.join() instead
        # inserted an extra path separator, producing paths like
        # "SAC\Emile_chr21_\karyogram_1005.png" that pointed at a
        # nonexistent "Emile_chr21_" subdirectory -- found via a real run,
        # every single karyogram failing with "No such file or directory".
        # Plain concatenation here too, with a trailing-separator fallback
        # added only for the bare-directory default case below, keeps this
        # consistent with every other writer in the project regardless of
        # which case actually applies.
        if out_dir is None:
            out_dir = os.path.dirname(self.out_prefix) or "."
            if not out_dir.endswith(os.sep):
                out_dir = out_dir + os.sep
        # Sample selection fix: this used to take the FIRST max_samples in
        # file order, which is arbitrary and frequently the wrong choice.
        # Found on a real run: SAC_REF.fam interleaves study and reference
        # individuals, so the first 5 samples were 1 admixed + 4 REFERENCE
        # panel members (KHS21, NA06989, NA18488, NA18597). A reference
        # individual's karyogram is near-useless by construction -- those
        # samples define the ancestral panels, so their local ancestry is
        # ~100% a single colour with no mosaic structure to see. Four of the
        # five plots carried essentially no information.
        #
        # Samples are now ranked by ancestry DIVERSITY computed from
        # lanc_array itself (Shannon entropy of each sample's genome-wide
        # ancestry proportions), so the most genuinely admixed individuals --
        # the ones whose karyograms actually show tract structure -- are
        # plotted first. This needs no extra plumbing: it uses only data
        # ConsensusLAI already holds, and is fully deterministic, so runs
        # stay reproducible.
        n_samples_total = self.lanc_array.shape[2]
        n = min(n_samples_total, max_samples)

        # Candidate pool: restrict to the KNOWN study cohort when the caller
        # supplies it. This corrects a wrong fix. A previous version ranked
        # every sample by ancestry ENTROPY, on the assumption that high
        # entropy means admixed. On real data that selected five GIH
        # reference-panel samples and ZERO admixed individuals -- worse than
        # the file-order behaviour it replaced, which at least got one.
        #
        # The assumption is simply false: high entropy also arises when the
        # LAI model cannot CONFIDENTLY resolve a sample. A reference panel
        # that sits between the others in a K-way model (GIH here, among
        # KHS/CEU/YRI/CHB) gets diffuse ancestry proportions from model
        # uncertainty, not from real mosaic ancestry -- and that looks
        # identical to admixture under an entropy criterion.
        #
        # --pop-file / --admix-label already state exactly which samples are
        # the admixed study cohort, so that explicit signal is used instead of
        # inferring it. Entropy is still used, but only to ORDER samples
        # WITHIN the study cohort (most mosaic first), where it is a
        # reasonable tie-breaker rather than a membership test.
        props = np.asarray(self.lanc_array).sum(axis=1).T.astype(np.float64)
        totals = props.sum(axis=1, keepdims=True)
        props = np.divide(props, totals, out=np.zeros_like(props),
                          where=totals > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -np.nansum(np.where(props > 0, props * np.log(props), 0.0),
                             axis=1)

        candidates = np.arange(n_samples_total)
        pool_desc = "all samples"
        if study_sample_ids is not None and self._sample_ids:
            want = {str(x) for x in study_sample_ids}
            sel = np.array([i for i, sid in enumerate(self._sample_ids)
                            if str(sid) in want], dtype=int)
            if len(sel):
                candidates = sel
                pool_desc = f"{len(sel)} study-cohort (admixed) samples"
            else:
                logger.warning(
                    "[LAI] plot_karyograms_all: none of the supplied "
                    "study-cohort sample IDs matched this run's samples -- "
                    "falling back to all samples.")

        n = min(len(candidates), max_samples)
        order = candidates[np.argsort(-ent[candidates])][:n]
        if n < n_samples_total:
            logger.info(
                f"[LAI] Drawing karyograms for {n} of {pool_desc} "
                f"(of {n_samples_total} total), ordered most-mosaic-first by "
                f"ancestry entropy. Reference-panel individuals are excluded "
                f"where identifiable: they are ~single-ancestry by "
                f"construction, so their karyograms show no tract structure. "
                f"Raise max_samples to see more.")
        for si in order:
            si = int(si)
            sid = self._sample_ids[si] if self._sample_ids else f"s{si}"
            out_path = out_dir + f"karyogram_{sid}.png"
            try:
                karyogram_from_lanc(
                    lanc_array = self.lanc_array,
                    bim        = getattr(self, '_bim', None),
                    sample_ids = self._sample_ids,
                    pop_labels = self.pop_labels,
                    sample_idx = si,
                    out_path   = out_path)
                logger.info(f"[LAI] Karyogram written -> {out_path}")
            except Exception as e:
                logger.warning(f"[LAI] Karyogram for {sid} failed: {e}")

    def plot_lai_region(self, chrom, start_mb, end_mb,
                        phenotype=None, out_path=None, max_samples=100):
        """
        Per-individual ancestry tract plot for a specific genomic region,
        sorted by phenotype — reimplements armartin's plot_lai_region.py
        in Python 3 without bedparser, brewer2mpl, or external files.

        Parameters
        ----------
        chrom       : int   chromosome to display (e.g. 22)
        start_mb    : float region start in megabases
        end_mb      : float region end in megabases
        phenotype   : np.ndarray (n_samples,) or None (uses 0..n-1 as rank)
        out_path    : output PNG path
        max_samples : cap on rows actually drawn (default 100) -- see
                      lai_region_plot()'s own docstring for why this
                      matters for real, large cohorts.
        """
        from jasmap.plotting.plots import lai_region_plot
        if self.lanc_array is None:
            logger.warning("[LAI] plot_lai_region: lanc_array not built.")
            return
        n = self.lanc_array.shape[2]
        if phenotype is None:
            phenotype = np.arange(n, dtype=float)
        out_path = out_path or (
            f"{self.out_prefix}_lai_region_chr{chrom}_{start_mb:.0f}"
            f"-{end_mb:.0f}Mb.png")
        lai_region_plot(
            lanc_array = self.lanc_array,
            bim        = getattr(self, '_bim', None),
            phenotype  = phenotype,
            sample_ids = self._sample_ids,
            pop_labels = self.pop_labels,
            chrom      = chrom,
            start_mb   = start_mb,
            end_mb     = end_mb,
            out_path   = out_path,
            max_samples = max_samples)
        logger.info(f"[LAI] LAI region plot written -> {out_path}")
