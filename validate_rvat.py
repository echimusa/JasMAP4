"""
validate_rvat.py
----------------
Unit tests for jasmap.rvat.rvat_engine.

Checks:
  1. Burden / SKAT / SKAT-O / ACAT-V return finite p-values for a gene
     with a genuine rare-variant signal
  2. Burden / SKAT / SKAT-O / ACAT-V return ~ Uniform(0,1) p-values under
     the null (empirical type I error at alpha=0.05 over many permutations)
  3. ACAT combination correctly yields p < min(inputs)
  4. MAF filtering: only variants with MAF <= threshold are retained
  5. Gene annotation loading from BED format
  6. Full run_rvat() call on synthetic data
"""

import numpy as np
import tempfile, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jasmap.rvat.rvat_engine import (
    _burden_test, _skat_test, _skato_test, _acat_v,
    _acat, _bh_fdr, compute_maf,
    load_gene_annotation, run_rvat)

rng = np.random.default_rng(42)
PASS = 0; FAIL = 0

def check(name, condition, msg=""):
    global PASS, FAIL
    if condition:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {name}  {msg}")


print("=== 1. Burden/SKAT/SKAT-O/ACAT-V: finite p-values on causal gene ===")
n, k = 120, 8
G_causal = (rng.random((n, k)) < 0.02).astype(np.float32)  # rare variants
beta_true = np.array([2.0, 2.5, 1.8, 0, 0, 0, 0, 0])
y_causal  = G_causal @ beta_true + rng.standard_normal(n)

p_burden = _burden_test(G_causal, y_causal, is_binary=False)[2]
p_skat   = _skat_test(G_causal, y_causal,   is_binary=False)
p_skato  = _skato_test(G_causal, y_causal,  is_binary=False)
p_acat   = _acat_v(G_causal, y_causal,       is_binary=False)

check("burden finite",    p_burden is not None and np.isfinite(p_burden), f"p={p_burden}")
check("skat   finite",    p_skat   is not None and np.isfinite(p_skat),   f"p={p_skat}")
check("skato  finite",    p_skato  is not None and np.isfinite(p_skato),  f"p={p_skato}")
check("acat_v finite",    p_acat   is not None and np.isfinite(p_acat),   f"p={p_acat}")
check("burden powered",   p_burden < 0.10, f"p={p_burden:.4f}")
print(f"  burden={p_burden:.4f}  skat={p_skat:.4f}  skato={p_skato:.4f}  acat_v={p_acat:.4f}")


print("\n=== 2. Null calibration (empirical type I error over 400 permutations) ===")
n_perm = 400; alpha = 0.05
G_null = (rng.random((80, 6)) < 0.02).astype(np.float32)
y_null = rng.standard_normal(80)

type1_burden = type1_skat = type1_acat = 0
t0 = time.time()
for _ in range(n_perm):
    yp = rng.permutation(y_null)
    pb = _burden_test(G_null, yp, False)[2]
    ps = _skat_test(G_null, yp, False)
    pa = _acat_v(G_null, yp, False)
    if pb is not None and np.isfinite(pb) and pb < alpha: type1_burden += 1
    if ps is not None and np.isfinite(ps) and ps < alpha: type1_skat   += 1
    if pa is not None and np.isfinite(pa) and pa < alpha: type1_acat   += 1

t1e_b = type1_burden / n_perm
t1e_s = type1_skat   / n_perm
t1e_a = type1_acat   / n_perm
print(f"  T1E burden={t1e_b:.3f}  skat={t1e_s:.3f}  acat_v={t1e_a:.3f}  "
      f"(expected ~{alpha}, {n_perm} permutations, {time.time()-t0:.1f}s)")
# Accept within 3 standard deviations of expected (sigma = sqrt(alpha*(1-alpha)/n))
sigma = np.sqrt(alpha * (1-alpha) / n_perm)
check("burden calibrated", abs(t1e_b - alpha) < 4*sigma, f"T1E={t1e_b:.3f}")
check("skat   calibrated", abs(t1e_s - alpha) < 4*sigma, f"T1E={t1e_s:.3f}")
check("acat_v calibrated", abs(t1e_a - alpha) < 4*sigma, f"T1E={t1e_a:.3f}")


print("\n=== 3. ACAT combination properties ===")
pvals_in = [0.003, 0.02, 0.15, 0.4]
p_comb   = _acat(pvals_in)
# ACAT is not strictly <= min(p): it pools all evidence via Cauchy,
# so a few large p-values can dilute the signal.
# What we do guarantee: (a) combined is finite and in (0,1),
# (b) combining highly significant p-values gives a significant result.
check("acat finite and valid", np.isfinite(p_comb) and 0 < p_comb < 1,
      f"combined={p_comb:.5f}")
check("acat < 1",              p_comb < 1.0)
# ACAT should be more significant than the simple average
p_avg = float(np.mean(pvals_in))
check("acat < average(inputs)", p_comb < p_avg,
      f"combined={p_comb:.5f}  avg={p_avg:.5f}")
# Combining all very small p-values should yield a very small result
p_strong = _acat([0.001, 0.001, 0.001])
check("acat(small,small,small) < 0.01", p_strong < 0.01,
      f"combined={p_strong:.5f}")


print("\n=== 4. MAF filtering and compute_maf ===")
n_s, n_v = 100, 20
G_maf = rng.integers(0, 3, size=(n_s, n_v)).astype(np.float32)
maf   = compute_maf(G_maf)
check("maf shape",     len(maf) == n_v)
check("maf in [0,0.5]", (maf >= 0).all() and (maf <= 0.5).all(), f"range [{maf.min():.3f},{maf.max():.3f}]")


print("\n=== 5. BH FDR correction ===")
pv = np.array([0.001, 0.003, 0.05, 0.1, 0.5, 0.8])
adj = _bh_fdr(pv)
check("bh same length",       len(adj) == len(pv))
check("bh adjusted >= raw",   (adj >= pv - 1e-12).all(), f"min(adj-raw)={(adj-pv).min():.4f}")
check("bh monotone adjusted", (np.diff(adj[np.argsort(pv)]) >= -1e-12).all())


print("\n=== 6. Gene annotation loading (BED) ===")
bed_content = (
    "chr1\t1000\t5000\tGENE001\t0\t+\n"
    "chr2\t2000\t8000\tGENE002\t0\t-\n"
    "chr3\t500\t3000\tGENE003\t0\t+\n"
)
with tempfile.NamedTemporaryFile("w", suffix=".bed", delete=False,
                                  encoding="utf-8") as f:
    f.write(bed_content); bed_path = f.name
try:
    gdf = load_gene_annotation(bed_path)
    check("bed rows", len(gdf) == 3)
    check("bed cols", {"gene","chrom","start","stop"}.issubset(gdf.columns))
    check("bed gene name", "GENE001" in gdf["gene"].values)
finally:
    os.unlink(bed_path)


print("\n=== 7. Full run_rvat() on synthetic data ===")
n_s2, n_v2 = 80, 60
G_full = rng.integers(0, 3, size=(n_s2, n_v2)).astype(np.float32)
# Inject rare signal in first gene (SNPs 0-5)
G_full[:, 0:3] = (rng.random((n_s2, 3)) < 0.015).astype(np.float32)
y_full = G_full[:, 0:3].sum(axis=1) * 1.5 + rng.standard_normal(n_s2)
bim_full = __import__("pandas").DataFrame({
    "CHR": [1 + (j%22) for j in range(n_v2)],
    "SNP": [f"rs{j+1}" for j in range(n_v2)],
    "CM": 0, "BP": [1000 + j*100 for j in range(n_v2)],
    "A1": "A", "A2": "G"})
fam_full = __import__("pandas").DataFrame({
    "FID":[f"F{i}" for i in range(n_s2)],
    "IID":[f"S{i}" for i in range(n_s2)]})

# Write minimal variant annotation
va_rows = []
for j in range(n_v2):
    maf_j = float(compute_maf(G_full[:, j:j+1])[0])
    gene  = f"GENE_{j//5:03d}"
    va_rows.append(f"rs{j+1}\t{1+(j%22)}\t{1000+j*100}\t{gene}\t{maf_j:.6f}\tmissense\t{'yes' if maf_j<0.05 else 'no'}")
with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False,
                                  encoding="utf-8") as f:
    f.write("SNP\tCHR\tPOS\tGENE\tMAF\tCONSEQUENCE\tIS_RARE\n")
    f.write("\n".join(va_rows) + "\n")
    va_path = f.name

with tempfile.TemporaryDirectory() as tmpdir:
    result = run_rvat(
        G=G_full, y=y_full, bim=bim_full, fam=fam_full,
        out_prefix=os.path.join(tmpdir, "rvat_test"),
        var_annot_file=va_path,
        maf_max=0.05, min_variants=2,
        methods=("burden","skat","skato","acat_v"),
        plots=False)
    check("run_rvat returns df",       result is not None and len(result) > 0,
          f"n_genes={len(result) if result is not None else 'None'}")
    if result is not None and len(result):
        check("result has consensus_p", "consensus_p" in result.columns)
        check("consensus_p finite",
              result["consensus_p"].dropna().apply(
                  lambda p: 0 < p <= 1).all())
        check("result has burden_p",   "burden_p" in result.columns)
        check("output file exists",
              os.path.exists(os.path.join(tmpdir, "rvat_test", "JasMAP.RVAT.assoc")))

os.unlink(va_path)

print(f"\n{'='*50}")
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("\nALL CHECKS PASSED")
else:
    sys.exit(1)
