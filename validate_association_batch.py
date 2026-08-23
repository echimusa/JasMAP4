"""Validates LMM.association_batch() (vectorized, multi-SNP) against
LMM.association() (the original, scalar, per-SNP method) on synthetic data,
checking numerical agreement and reporting the speedup.
"""
import sys
import numpy as np
import time

sys.path.insert(0, '.')
exec(open('JasMAP4.py').read().split("if __name__")[0])


def run_check(n, n_snps, n_covar, seed):
    rng = np.random.default_rng(seed)

    # Build a valid PSD kinship matrix from a small random effects matrix.
    Z = rng.standard_normal((n, 8))
    K = Z @ Z.T / 8.0

    Y = rng.standard_normal((n, 1))
    X0 = np.hstack([np.ones((n, 1)), rng.standard_normal((n, n_covar))])

    lmm = LMM(Y.copy(), K.copy(), X0=X0.copy())
    hmax, beta_null, sigma, LL_null = lmm.fit()

    # A batch of SNPs, dosage-like 0/1/2 (a couple of them made monomorphic
    # to also exercise the "invalid" -> nan path).
    snps = rng.integers(0, 3, size=(n, n_snps)).astype(np.float64)
    snps[:, 0] = 1.0   # force one monomorphic ("fixed") column

    # --- scalar path (original association(), one call per SNP) ---
    t0 = time.time()
    scalar_betas, scalar_se, scalar_ts, scalar_ps = [], [], [], []
    for j in range(n_snps):
        b, se, ts, ps = lmm.association(X=snps[:, [j]], h=hmax, beta=None,
                                         stack=True, REML=True, returnBeta=False)
        scalar_betas.append(b); scalar_se.append(se); scalar_ts.append(ts); scalar_ps.append(ps)
    t_scalar = time.time() - t0
    def _to_scalar(x):
        if isinstance(x, str) and x == 'nan':
            return np.nan
        return float(np.asarray(x).ravel()[0])

    scalar_betas = np.array([_to_scalar(b) for b in scalar_betas])
    scalar_se = np.array([_to_scalar(b) for b in scalar_se])
    scalar_ts = np.array([_to_scalar(b) for b in scalar_ts])
    scalar_ps = np.array([_to_scalar(b) for b in scalar_ps])

    # --- batched path ---
    t0 = time.time()
    batch_betas, batch_se, batch_ts, batch_ps = lmm.association_batch(snps, h=hmax, REML=True)
    t_batch = time.time() - t0

    def maxdiff(a, b):
        mask = np.isfinite(a) & np.isfinite(b)
        if not mask.any():
            return 0.0
        return float(np.max(np.abs(a[mask] - b[mask])))

    # The intentionally-monomorphic column (index 0) is numerically unstable
    # in BOTH methods (dividing by a near-zero denominator along different
    # floating-point paths) and is not a meaningful correctness check either
    # way; exclude it from the strict numerical comparison, but still check
    # separately (below) that the batched method flags it as invalid (nan).
    scalar_betas_chk = scalar_betas[1:]; batch_betas_chk = batch_betas[1:]
    scalar_se_chk = scalar_se[1:]; batch_se_chk = batch_se[1:]
    scalar_ts_chk = scalar_ts[1:]; batch_ts_chk = batch_ts[1:]
    scalar_ps_chk = scalar_ps[1:]; batch_ps_chk = batch_ps[1:]

    nan_match = np.array_equal(np.isnan(scalar_betas_chk), np.isnan(batch_betas_chk))
    degenerate_col_flagged = bool(np.isnan(batch_betas[0]))

    print(f"n={n:4d} n_snps={n_snps:5d} n_covar={n_covar}  "
          f"beta_diff={maxdiff(scalar_betas_chk, batch_betas_chk):.2e}  "
          f"se_diff={maxdiff(scalar_se_chk, batch_se_chk):.2e}  "
          f"ts_diff={maxdiff(scalar_ts_chk, batch_ts_chk):.2e}  "
          f"ps_diff={maxdiff(scalar_ps_chk, batch_ps_chk):.2e}  "
          f"nan_pattern_match={nan_match}  degenerate_col_flagged={degenerate_col_flagged}  "
          f"scalar={t_scalar*1000:8.2f}ms  batch={t_batch*1000:7.2f}ms  "
          f"speedup={t_scalar/max(t_batch,1e-9):7.1f}x")

    ok = (maxdiff(scalar_betas_chk, batch_betas_chk) < 1e-6 and
          maxdiff(scalar_se_chk, batch_se_chk) < 1e-4 and
          maxdiff(scalar_ps_chk, batch_ps_chk) < 1e-6 and
          nan_match and degenerate_col_flagged)
    return ok


if __name__ == "__main__":
    all_ok = True
    for (n, n_snps, n_covar) in [(50, 20, 2), (100, 200, 3), (300, 2000, 4)]:
        all_ok &= run_check(n, n_snps, n_covar, seed=n * n_snps)

    print()
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
