"""
validate_streaming_io.py
--------------------------
Validates jasmap.biobank.streaming_io against the dense in-memory
implementations already validated (make_kinship_matvec / cg_solve in
cg_kinship.py, randomized_kinship_eigendecomp in lowrank_kinship.py) --
so this is validated transitively against the exact LMM class too, via
the chain already proven in validate_cg_lmm.py / validate_lowrank_lmm.py.

Checks
------
1. Streaming matvec_K matches the dense matvec_K on the same .bed file,
   in BOTH cache mode (small data) and forced-streaming mode (data
   artificially forced past the cache budget).
2. Streaming randomized eigendecomposition matches the dense version's
   eigenvalues/eigenvectors (up to sign) on the same data.
3. A full CGLMM built from the streaming matvec matches one built from
   the dense matvec, on the same .bed file.
"""

import sys, os, subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open('JasMAP4.py').read().split("if __name__")[0])

from jasmap.biobank.streaming_io import (
    StreamingGenotypeSource, make_streaming_kinship_matvec,
    randomized_kinship_eigendecomp_streaming)
from jasmap.biobank.cg_kinship import make_kinship_matvec, cg_solve, CGLMM
from jasmap.biobank.lowrank_kinship import randomized_kinship_eigendecomp

PASS = 0; FAIL = 0

def check(name, cond, msg=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {name}  {msg}")


# ── Build a small synthetic .bed file to test against ────────────────────────
import tempfile
tmpdir = tempfile.mkdtemp()
sys.path.insert(0, '.')
from make_test_data import make_geno_dosage, write_plink_bed

rng = np.random.default_rng(5)
n, m = 120, 800
dosage = make_geno_dosage(m, n, seed=5)   # (n_snps, n_samples) per make_test_data's convention
iids = [f"IND{i}" for i in range(n)]
write_plink_bed(tmpdir, "test", dosage, iids)
bed_path = os.path.join(tmpdir, "test.bed")

# Dense reference genotype matrix (samples x SNPs), matching bed-reader's own
# read() orientation, for building the dense comparison arrays
from bed_reader import open_bed
bed = open_bed(bed_path, count_A1=False)
G_dense = bed.read().astype(np.float64)   # (n, m)


print("=== 1. Streaming matvec_K vs dense matvec_K (cache mode) ===")
source_cached = StreamingGenotypeSource(bed_path, cache_budget_gb=4.0, verbose=False)
matvec_stream_cached = make_streaming_kinship_matvec(source_cached)
matvec_dense, _ = make_kinship_matvec(G_dense)

rng2 = np.random.default_rng(1)
for trial in range(5):
    v = rng2.standard_normal(n)
    r_dense = matvec_dense(v)
    r_stream = matvec_stream_cached(v)
    check(f"trial {trial}: streaming (cached) matches dense",
          np.allclose(r_dense, r_stream, atol=1e-8),
          f"max diff={np.max(np.abs(r_dense-r_stream)):.2e}")


print("\n=== 2. Streaming matvec_K vs dense matvec_K (forced streaming mode) ===")
source_streamed = StreamingGenotypeSource(bed_path, cache_budget_gb=1e-6, verbose=False)
check("forced-streaming source did NOT choose to cache", not source_streamed.use_cache)
matvec_stream_forced = make_streaming_kinship_matvec(source_streamed)
for trial in range(5):
    v = rng2.standard_normal(n)
    r_dense = matvec_dense(v)
    r_stream = matvec_stream_forced(v)
    check(f"trial {trial}: streaming (forced disk reads) matches dense",
          np.allclose(r_dense, r_stream, atol=1e-8),
          f"max diff={np.max(np.abs(r_dense-r_stream)):.2e}")


print("\n=== 3. Streaming randomized eigendecomposition vs dense ===")
Kva_dense, U_dense, c_dense, trace_dense = randomized_kinship_eigendecomp(
    G_dense, rank=n, power_iter=3, seed=7, return_trace=True)
Kva_stream, U_stream, c_stream, trace_stream = randomized_kinship_eigendecomp_streaming(
    source_cached, rank=n, power_iter=3, seed=7, verbose=False)

check("trace(K) matches", abs(trace_dense - trace_stream) < 1e-6,
      f"dense={trace_dense:.4f} stream={trace_stream:.4f}")
check("top eigenvalues match (full rank)",
      np.allclose(Kva_dense[:20], Kva_stream[:20], rtol=1e-3),
      f"max diff={np.max(np.abs(Kva_dense[:20]-Kva_stream[:20])):.2e}")

# Eigenvectors can differ by sign; compare via the subspace they span
# (U_dense.T @ U_stream should be close to a signed permutation for the
# leading, well-separated eigenvalue)
overlap = abs(float(U_dense[:, 0] @ U_stream[:, 0]))
check("top eigenvector direction matches (up to sign)", overlap > 0.99,
      f"|dot product|={overlap:.4f}")


print("\n=== 4. Full CGLMM built from streaming matvec vs dense matvec ===")
y = rng.standard_normal(n) + G_dense[:, :5].sum(axis=1) * 0.1
X0 = np.ones((n, 1))

cg_dense = CGLMM(y, matvec_dense, n, X0=X0, n_probes=20, n_lanczos=30, seed=3, verbose=False)
hmax_dense, beta_dense, *_ = cg_dense.fit(REML=True)

cg_stream = CGLMM(y, matvec_stream_cached, n, X0=X0, n_probes=20, n_lanczos=30, seed=3, verbose=False)
hmax_stream, beta_stream, *_ = cg_stream.fit(REML=True)

check("CGLMM hmax matches (streaming vs dense matvec)",
      abs(hmax_dense - hmax_stream) < 1e-6,
      f"dense={hmax_dense:.6f} stream={hmax_stream:.6f}")
check("CGLMM beta matches (streaming vs dense matvec)",
      np.allclose(beta_dense.ravel(), beta_stream.ravel(), atol=1e-6),
      f"dense={beta_dense.ravel()} stream={beta_stream.ravel()}")

X_batch = G_dense[:, 10:20]
b_dense, se_dense, t_dense, p_dense = cg_dense.association_batch(X_batch, h=hmax_dense, REML=True)
b_stream, se_stream, t_stream, p_stream = cg_stream.association_batch(X_batch, h=hmax_stream, REML=True)
v = np.isfinite(b_dense) & np.isfinite(b_stream)
check("association_batch betas match (streaming vs dense matvec)",
      np.allclose(b_dense[v], b_stream[v], atol=1e-6),
      f"max diff={np.max(np.abs(b_dense[v]-b_stream[v])):.2e}")


print(f"\n{'='*50}")
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("\nALL CHECKS PASSED")
else:
    sys.exit(1)
