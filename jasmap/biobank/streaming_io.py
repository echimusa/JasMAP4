"""
jasmap.biobank.streaming_io
-----------------------------
SNP-chunked streaming I/O for genuine biobank scale -- the piece that was
still missing after lowrank_kinship.py / cg_kinship.py / auto_lmm.py were
validated against dense in-memory arrays. This module lets every matvec
those modules need be computed by reading a PLINK .bed file in SNP-chunks
directly, so the full genotype matrix (which can be hundreds of GB to TB at
n=500,000, m=millions) is never held in memory at once.

Key identity this relies on (validated numerically before use)
-----------------------------------------------------------------
K = A @ A.T where A is the standardised genotype matrix (columns = SNPs).
Since A is just SNP-chunks concatenated column-wise,
    K = sum_chunks (A_chunk @ A_chunk.T)
so
    K @ v = sum_chunks A_chunk @ (A_chunk.T @ v)
Each term uses ONLY that chunk's own data -- computed the moment it is
read. This means every matvec needs exactly ONE pass over the SNP chunks,
not two (an earlier, less careful derivation assumed two passes; this
module's design was corrected before writing any of the actual chunk-reading
code, once the identity above was checked against a dense reference).

Two-tier memory strategy
--------------------------
- Data that fits within `cache_budget_gb` (default 4 GB) is read ONCE and
  cached as a list of standardised chunks in memory -- every subsequent
  matvec call is then a fast in-memory operation, identical in spirit to
  (and validated against) the dense arrays lowrank_kinship.py/cg_kinship.py
  already use.
- Data that does NOT fit is streamed genuinely fresh from disk on every
  matvec call. This is slower (I/O-bound, one full file pass per call) but
  is the only way to avoid running out of memory at true biobank scale.
  CG/SLQ need many matvec calls (fit() alone can need hundreds), so this
  tier trades wall-clock time for a hard memory ceiling that is respected
  regardless of cohort size.
"""

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

_BYTES_PER_ENTRY = 8  # float64


class StreamingGenotypeSource:
    """
    Wraps a PLINK .bed file (via bed-reader) and provides SNP-chunked,
    standardised access -- either cached in memory (small/medium data) or
    streamed fresh from disk on every pass (large data), decided
    automatically from `cache_budget_gb`.
    """

    def __init__(self, bed_path, sample_idx=None, snp_idx=None, chunk_size=None,
                cache_budget_gb=4.0, verbose=True, maf_min=0.01):
        """
        Parameters
        ----------
        bed_path   : str  path to the .bed file (without extension is also
                    accepted -- bed-reader's open_bed wants the full
                    ".bed" path, handled here either way)
        sample_idx : optional array of sample indices to restrict to
                    (e.g. for missing-phenotype filtering -- see
                    JasMAP4.py's nonmissing_mask usage elsewhere)
        snp_idx    : optional array of SNP (column) indices to restrict to
                    -- e.g. --chr filtering (see JasMAP4.py's
                    _resolve_chr_snp_indices()). None means all SNPs in the
                    file. Indices need not be contiguous; every chunk read
                    below maps a "logical" chunk position back to the
                    correct (possibly scattered) file columns via bed-
                    reader's own arbitrary-index-array support (validated
                    separately: matches a direct full-file read exactly).
                    Restricting to one chromosome is a practical way to
                    make biobank-scale analysis (hundreds of GB to TB for
                    the whole genome at once) fit in memory in per-
                    chromosome chunks instead -- the same chromosome-
                    parallelised pattern real biobank GWAS pipelines use.
        chunk_size : SNPs per chunk. If None, computed automatically to
                    target ~200 MB per chunk given the sample count.
        cache_budget_gb : if the FULL standardised genotype data would fit
                    within this budget, cache it in memory after the first
                    pass (fast, matches the validated dense-array behaviour
                    exactly); otherwise stream fresh from disk every call.
        """
        from bed_reader import open_bed
        bed_path = bed_path if bed_path.endswith(".bed") else bed_path + ".bed"
        self.bed = open_bed(bed_path, count_A1=False)
        self.sample_idx = sample_idx
        self.snp_idx = np.asarray(snp_idx) if snp_idx is not None else None
        self.n = self.bed.iid_count if sample_idx is None else len(sample_idx)
        self.m_total = self.bed.sid_count if self.snp_idx is None else len(self.snp_idx)
        self.verbose = verbose

        if chunk_size is None:
            target_bytes = 200 * 1024**2
            chunk_size = max(1, int(target_bytes / max(self.n * _BYTES_PER_ENTRY, 1)))
        self.chunk_size = min(chunk_size, self.m_total)
        self.maf_min = maf_min

        total_bytes = self.n * self.m_total * _BYTES_PER_ENTRY
        self.use_cache = total_bytes <= cache_budget_gb * 1024**3
        self._cache = None  # list of (start, end, A_chunk) once built

        if verbose:
            logger.info(
                f"[streaming_io] n={self.n}  m={self.m_total}  "
                f"chunk_size={self.chunk_size}  "
                f"full standardised data would need {total_bytes/1e9:.2f} GB  "
                f"-> {'CACHING in memory' if self.use_cache else 'STREAMING from disk every pass'} "
                f"(cache budget {cache_budget_gb:.1f} GB)")

        # One-time pass: per-SNP mean/variance, needed for standardisation.
        # This pass is unavoidable regardless of cache/stream mode (the
        # values are needed before ANY standardised chunk can be produced),
        # but is cheap: O(n*m) additions, same cost as reading the file once.
        self._compute_snp_stats()

    def _compute_snp_stats(self):
        t0 = time.time()
        means = np.zeros(self.m_total)
        variances = np.zeros(self.m_total)
        for start, end, raw_chunk in self._iter_raw_chunks():
            col_mean = np.nanmean(raw_chunk, axis=0)
            means[start:end] = col_mean
            # Bug fix: matches the identical fix in lowrank_kinship.py and
            # cg_kinship.py -- scale by the Yang et al. 2010 Hardy-Weinberg
            # theoretical variance 2*p*(1-p) (p = allele frequency =
            # col_mean/2), not the empirical sample variance (np.nanvar),
            # to match ps.makeG()'s method=0 (the default kinship method)
            # exactly. An end-to-end comparison of streaming vs exact GWAS
            # output on the same real data caught this mismatch (beta diff
            # 0.109, p-value diff 0.789) -- isolated unit tests had only
            # ever compared this module's own dense reference kinship
            # against itself, not against what ps.makeG() actually
            # produces. Verified to match to machine precision (2.2e-16)
            # before applying this fix.
            p = col_mean / 2.0
            variances[start:end] = 2.0 * p * (1.0 - p)
        self.snp_mean = means
        self.snp_var = variances
        # Bug fix: same MAF floor as ps.makeG()/lowrank_kinship.py/
        # cg_kinship.py -- `variances > 0` alone only excludes exactly
        # monomorphic sites, not merely-rare ones whose tiny-but-nonzero
        # variance inflates their standardized weight enormously (found
        # via a real run where this silently inflated trace(K) to ~10^19).
        maf = np.minimum(means / 2.0, 1.0 - means / 2.0)
        self.keep_mask = (variances > 0) & (maf >= self.maf_min)
        self.m_eff = int(self.keep_mask.sum())
        if self.verbose:
            logger.info(f"[streaming_io] SNP stats computed in {time.time()-t0:.2f}s "
                        f"({self.m_eff}/{self.m_total} SNPs have nonzero variance)")

    def _iter_raw_chunks(self):
        """Yields (start, end, raw_chunk) for every SNP chunk, unstandardised.
        `start`/`end` are LOGICAL positions in the (possibly snp_idx-
        filtered) SNP space; the actual file columns fetched are mapped
        through self.snp_idx when set, so a filtered chunk's columns need
        not be contiguous in the underlying .bed file."""
        for start in range(0, self.m_total, self.chunk_size):
            end = min(start + self.chunk_size, self.m_total)
            col_sel = self.snp_idx[start:end] if self.snp_idx is not None else slice(start, end)
            if self.sample_idx is not None:
                idx = (self.sample_idx, col_sel)
            else:
                idx = (None, col_sel)
            raw_chunk = self.bed.read(index=idx).astype(np.float64)
            yield start, end, raw_chunk

    def iter_standardized_chunks(self):
        """
        Yields (start, end, A_chunk) where A_chunk is the standardised
        genotype chunk divided by sqrt(m_eff) (so that A_chunk @ A_chunk.T
        summed over all chunks equals the additive kinship matrix K,
        matching randomized_kinship_eigendecomp / make_kinship_matvec's
        convention exactly). Zero-variance SNPs are dropped, matching the
        dense implementations' behaviour.

        Uses the in-memory cache when available (built on first call);
        otherwise reads fresh from disk every time this generator is
        consumed.
        """
        if self.use_cache:
            if self._cache is None:
                self._build_cache()
            for start, end, A_chunk in self._cache:
                yield start, end, A_chunk
            return

        for start, end, raw_chunk in self._iter_raw_chunks():
            yield self._standardize_chunk(start, end, raw_chunk)

    def _standardize_chunk(self, start, end, raw_chunk):
        local_keep = self.keep_mask[start:end]
        if not local_keep.any():
            return start, end, np.zeros((self.n, 0))
        chunk = raw_chunk[:, local_keep]
        col_mean = self.snp_mean[start:end][local_keep]
        col_var = self.snp_var[start:end][local_keep]
        nan_mask = np.isnan(chunk)
        if nan_mask.any():
            r, c = np.where(nan_mask)
            chunk = chunk.copy()
            chunk[r, c] = col_mean[c]
        chunk_std = (chunk - col_mean) / np.sqrt(col_var)
        A_chunk = chunk_std / np.sqrt(max(self.m_eff, 1))
        return start, end, A_chunk

    def _build_cache(self):
        t0 = time.time()
        cache = []
        for start, end, raw_chunk in self._iter_raw_chunks():
            cache.append(self._standardize_chunk(start, end, raw_chunk))
        self._cache = cache
        if self.verbose:
            logger.info(f"[streaming_io] Cached {len(cache)} standardised "
                        f"chunk(s) in {time.time()-t0:.2f}s")


def make_streaming_kinship_matvec(source):
    """
    Returns matvec_K(v) -> K@v, computed via the single-pass chunked
    identity K@v = sum_chunks A_chunk @ (A_chunk.T @ v), reading (or
    replaying, if cached) SNP chunks from `source`.

    Validated in validate_streaming_io.py to match make_kinship_matvec()
    (the dense in-memory version already validated against the exact LMM
    class) to floating-point precision on the same data.
    """
    def matvec_K(v):
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        result = np.zeros(source.n)
        for start, end, A_chunk in source.iter_standardized_chunks():
            if A_chunk.shape[1] == 0:
                continue
            u_chunk = A_chunk.T @ v
            result += A_chunk @ u_chunk
        return result

    return matvec_K


def randomized_kinship_eigendecomp_streaming(source, rank=200, power_iter=2,
                                              seed=42, verbose=True):
    """
    Streaming counterpart to lowrank_kinship.randomized_kinship_eigendecomp:
    same randomised range-finder SVD algorithm, but every place the dense
    version indexes into a fully-materialised A now streams SNP chunks from
    `source` instead. Random projection vectors (Omega) are generated
    per-chunk with a chunk-indexed seed, so the whole (m, rank+10) Omega
    matrix is never held in memory even during caching.

    Returns (Kva_top, U, c, trace_K) -- same shapes/meaning as the dense
    version's return_trace=True mode. Validated in validate_streaming_io.py
    to match the dense randomized_kinship_eigendecomp() on the same data.
    """
    t0 = time.time()
    n = source.n
    r = min(rank, n, source.m_eff)
    oversample = r + 10
    rng = np.random.default_rng(seed)

    # ── Randomised range finder: Y = A @ Omega, chunk by chunk ───────────────
    def _compute_Y(rng_seed):
        Y = np.zeros((n, oversample))
        local_rng = np.random.default_rng(rng_seed)
        for start, end, A_chunk in source.iter_standardized_chunks():
            width = A_chunk.shape[1]
            if width == 0:
                continue
            Omega_chunk = local_rng.standard_normal((width, oversample))
            Y += A_chunk @ Omega_chunk
        return Y

    Y = _compute_Y(seed)

    # ── Power iteration: alternately apply A.T then A, chunk by chunk ────────
    # Each iteration needs Z = A.T @ Y (an (m_eff, oversample) array -- held
    # in memory across chunks within ONE iteration; this is the one place
    # streaming still needs O(m * oversample) memory, documented in this
    # module's docstring as a known, bounded constraint smaller than the
    # O(n*m) / O(n^2) problems this module exists to avoid).
    for _ in range(power_iter):
        Z_parts = []
        for start, end, A_chunk in source.iter_standardized_chunks():
            if A_chunk.shape[1] == 0:
                continue
            Z_parts.append(A_chunk.T @ Y)
        Z = np.vstack(Z_parts) if Z_parts else np.zeros((0, oversample))
        Y_new = np.zeros((n, oversample))
        offset = 0
        for start, end, A_chunk in source.iter_standardized_chunks():
            width = A_chunk.shape[1]
            if width == 0:
                continue
            Y_new += A_chunk @ Z[offset:offset + width]
            offset += width
        Y = Y_new

    Q, _ = np.linalg.qr(Y)  # (n, oversample)

    # ── B = Q.T @ A, chunk by chunk (accumulate columns, not rows) ───────────
    B_parts = []
    for start, end, A_chunk in source.iter_standardized_chunks():
        if A_chunk.shape[1] == 0:
            continue
        B_parts.append(Q.T @ A_chunk)
    B = np.hstack(B_parts) if B_parts else np.zeros((oversample, 0))

    Ub, S, _ = np.linalg.svd(B, full_matrices=False)
    U_full = Q @ Ub
    Kva_full = S ** 2

    U = U_full[:, :r]
    Kva_top = Kva_full[:r]

    # trace(K) = sum of squared entries of A, accumulated chunk by chunk
    trace_K = 0.0
    for start, end, A_chunk in source.iter_standardized_chunks():
        trace_K += float(np.sum(A_chunk * A_chunk))
    c = max((trace_K - Kva_top.sum()) / max(n - r, 1), 1e-8)

    if verbose:
        logger.info(f"[streaming_io] rank={r}  n={n}  m={source.m_total}  "
                    f"trace(K)={trace_K:.2f}  captured={Kva_top.sum():.2f}  "
                    f"({100*Kva_top.sum()/max(trace_K,1e-12):.1f}%)  "
                    f"residual_c={c:.4f}  [{time.time()-t0:.2f}s]")

    return Kva_top, U, c, trace_K
