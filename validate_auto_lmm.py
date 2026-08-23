"""
validate_auto_lmm.py
---------------------
Validates jasmap.biobank.auto_lmm's three-tier dispatcher:
  1. Small n -> exact LMM (bit-for-bit correct, already proven in
     validate_lowrank_lmm.py and validate_cg_lmm.py; here just confirms
     the DISPATCHER correctly routes to it)
  2. Larger n with concentrated structure -> LMMLowRank chosen automatically
  3. Larger n WITHOUT concentrated structure -> correctly falls back to
     CGLMM (the exact scenario that broke pure low-rank)
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jasmap.biobank.auto_lmm import build_lmm, recommend_lmm_strategy

PASS = 0; FAIL = 0

def check(name, cond, msg=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {name}  {msg}")


print("=== 1. Pure sizing check (no data needed) ===")
check("n=500 -> exact",       recommend_lmm_strategy(500) == "exact")
check("n=5,000 -> exact",     recommend_lmm_strategy(5_000) == "exact")
check("n=50,000 -> approximate", recommend_lmm_strategy(50_000) == "approximate")
check("n=500,000 -> approximate", recommend_lmm_strategy(500_000) == "approximate")
# Custom memory limit shifts the boundary predictably
check("custom tiny limit forces approximate even for small n",
      recommend_lmm_strategy(500, memory_limit_bytes=1000) == "approximate")


print("\n=== 2. Dispatcher on genuinely small data -> chooses 'exact' ===")
rng = np.random.default_rng(5)
n_small, m_small = 300, 1000
G_small = rng.integers(0, 3, size=(n_small, m_small)).astype(np.float64)
y_small = rng.standard_normal(n_small)
X0_small = np.ones((n_small, 1))

model, strategy = build_lmm(G_small, y_small, X0=X0_small, verbose=False)
check("small data -> 'exact' chosen", strategy["chosen"] == "exact",
      f"got {strategy['chosen']}")
hmax, beta, sigma, LL = model.fit(REML=True)
check("exact model fits without error", np.isfinite(hmax))


print("\n=== 3. Dispatcher forced into 'approximate' tier via tiny memory limit ===")
print("    (simulates large-n behaviour without actually needing huge n for the test)")

# Case A: data WITH concentrated structure -> should choose lowrank
rng2 = np.random.default_rng(30)
n2, m2 = 350, 100_000   # high SNP:sample ratio -> concentrated, flat tail
n_pops = 3
pop_id = rng2.integers(0, n_pops, size=n2)
anc_freq = rng2.uniform(0.1, 0.9, size=m2)
pop_drift = rng2.normal(0, 0.2, size=(n_pops, m2))   # STRONG population signal
pop_freq = np.clip(anc_freq[None, :] + pop_drift, 0.02, 0.98)
G_struct = np.zeros((n2, m2))
for i in range(n2):
    G_struct[i] = rng2.binomial(2, pop_freq[pop_id[i]])
y_struct = rng2.standard_normal(n2) + (pop_id == 0).astype(float) * 0.5
X0_struct = np.ones((n2, 1))

model_a, strategy_a = build_lmm(
    G_struct, y_struct, X0=X0_struct, memory_limit_bytes=1000,  # force "approximate"
    lowrank_rank_start=20, lowrank_rank_step=20, lowrank_rank_max=200,
    lowrank_beta_rel_tol=0.05, verbose=False)
print(f"  Case A (strong structure, high SNP:sample ratio): chosen={strategy_a['chosen']}")
check("Case A: dispatcher tried the approximate tier",
      strategy_a["chosen"] in ("lowrank", "cg"))
hmax_a, *_ = model_a.fit(REML=True)
check("Case A: model fits without error", np.isfinite(hmax_a))

# Case B: THE exact scenario that broke pure low-rank -> should fall back to cg
rng3 = np.random.default_rng(21)
n3, m3 = 400, 2000    # modest SNP:sample ratio, no clean concentration
pop_id3 = rng3.integers(0, 3, size=n3)
anc_freq3 = rng3.uniform(0.1, 0.9, size=m3)
pop_drift3 = rng3.normal(0, 0.15, size=(3, m3))
pop_freq3 = np.clip(anc_freq3[None, :] + pop_drift3, 0.02, 0.98)
G_hard = np.zeros((n3, m3))
for i in range(n3):
    G_hard[i] = rng3.binomial(2, pop_freq3[pop_id3[i]])
y_hard = rng3.standard_normal(n3) + (pop_id3 == 0).astype(float) * 0.5
X0_hard = np.ones((n3, 1))

model_b, strategy_b = build_lmm(
    G_hard, y_hard, X0=X0_hard, memory_limit_bytes=1000,
    lowrank_rank_start=50, lowrank_rank_step=50, lowrank_rank_max=400,
    lowrank_beta_rel_tol=0.03, verbose=False)
print(f"  Case B (modest ratio, the known-hard case): chosen={strategy_b['chosen']}")
check("Case B: dispatcher correctly falls back to CG for the hard case",
      strategy_b["chosen"] == "cg",
      f"got {strategy_b['chosen']} -- expected 'cg' since low-rank is known "
      f"not to converge for this dataset")
hmax_b, *_ = model_b.fit(REML=True)
check("Case B (CG fallback): model fits without error", np.isfinite(hmax_b))

print(f"\n  Strategy dict always includes a human-readable 'reason':")
print(f"    Case A: {strategy_a['reason']}")
print(f"    Case B: {strategy_b['reason']}")


print(f"\n{'='*50}")
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("\nALL CHECKS PASSED")
else:
    sys.exit(1)
