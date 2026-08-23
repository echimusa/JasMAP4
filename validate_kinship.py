"""Standalone correctness check: compares the OLD (pre-optimization) and NEW
(vectorized) calculateKinship() implementations on random data with missing
values, for both center=False and center=True, and reports the max abs diff.
Run this before trusting the optimized JasMAP4.py on real data.
"""
import numpy as np
import time

rng = np.random.default_rng(42)


def old_calculateKinship(K, center=False):
    """Verbatim port of the ORIGINAL algorithm (matrixMult -> plain np.dot,
    since np.dot and linalg.fblas.dgemm are numerically equivalent -- only
    the dispatch mechanism differed, not the arithmetic)."""
    W = np.array([i for i in K])
    n = W.shape[0]
    m = W.shape[1]
    keep = []
    for i in range(m):
        mn = W[True ^ np.isnan(W[:, i]), i].mean()
        W[np.isnan(W[:, i]), i] = mn
        vr = W[:, i].var()
        if vr == 0:
            continue
        keep.append(i)
        W[:, i] = (W[:, i] - mn) / np.sqrt(vr)
    W = W[:, keep]
    Kout = np.dot(W, W.T) * 1.0 / float(m)
    if center:
        P = np.diag(np.repeat(1, n)) - 1 / float(n) * np.ones((n, n))
        S = np.trace(np.dot(np.dot(P, Kout), P))
        K_n = (n - 1) * Kout / S
        return K_n
    return Kout


def new_calculateKinship(K, center=False):
    """The optimized version (logic copied from JasMAP4.py's LMM.calculateKinship,
    with logger calls stripped for standalone testing)."""
    W = np.array(K, dtype=np.float64)
    n, m = W.shape

    col_mean = np.nanmean(W, axis=0)
    nan_mask = np.isnan(W)
    if nan_mask.any():
        nan_rows, nan_cols = np.where(nan_mask)
        W[nan_rows, nan_cols] = col_mean[nan_cols]

    col_var = W.var(axis=0)
    keep = col_var > 0
    if not keep.all():
        W = W[:, keep]
        col_mean = col_mean[keep]
        col_var = col_var[keep]

    W = (W - col_mean) / np.sqrt(col_var)
    Kmat = (W @ W.T) / float(m)

    if center:
        S = np.trace(Kmat) - Kmat.sum() / float(n)
        K_n = (n - 1) * Kmat / S
        return K_n
    return Kmat


def make_test_matrix(n, m, missing_frac=0.02, n_fixed_cols=3, seed=0):
    r = np.random.default_rng(seed)
    W = r.integers(0, 3, size=(n, m)).astype(np.float64)
    # force a few fixed (zero-variance) columns, to test the "keep" filtering path
    for c in r.choice(m, size=min(n_fixed_cols, m), replace=False):
        W[:, c] = 1.0
    # inject missing values
    n_missing = int(missing_frac * n * m)
    rows = r.integers(0, n, size=n_missing)
    cols = r.integers(0, m, size=n_missing)
    W[rows, cols] = np.nan
    return W


if __name__ == "__main__":
    all_ok = True
    for (n, m) in [(10, 20), (50, 200), (100, 2000)]:
        W = make_test_matrix(n, m, seed=n * m)

        t0 = time.time()
        old_res = old_calculateKinship(W.copy(), center=False)
        t_old = time.time() - t0

        t0 = time.time()
        new_res = new_calculateKinship(W.copy(), center=False)
        t_new = time.time() - t0

        diff = np.max(np.abs(old_res - new_res))
        ok = diff < 1e-10
        all_ok &= ok
        print(f"[center=False] n={n:4d} m={m:5d}  max_abs_diff={diff:.3e}  "
              f"OK={ok}  old={t_old*1000:7.2f}ms  new={t_new*1000:7.2f}ms  "
              f"speedup={t_old/max(t_new,1e-9):6.1f}x")

        t0 = time.time()
        old_res_c = old_calculateKinship(W.copy(), center=True)
        t_old_c = time.time() - t0

        t0 = time.time()
        new_res_c = new_calculateKinship(W.copy(), center=True)
        t_new_c = time.time() - t0

        diff_c = np.max(np.abs(old_res_c - new_res_c))
        ok_c = diff_c < 1e-8
        all_ok &= ok_c
        print(f"[center=True ] n={n:4d} m={m:5d}  max_abs_diff={diff_c:.3e}  "
              f"OK={ok_c}  old={t_old_c*1000:7.2f}ms  new={t_new_c*1000:7.2f}ms  "
              f"speedup={t_old_c/max(t_new_c,1e-9):6.1f}x")

    print()
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
