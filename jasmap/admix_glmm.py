"""
admix_glmm.py -- logistic mixed-model (GLMM) admixture mapping
==============================================================

An ALTERNATIVE to the default per-ancestry LMM admixture-mapping test,
selected with --am-model glmm. The default (--am-model lmm) is unchanged.

Model, fitted separately for each ancestry k
--------------------------------------------
    logit P(y_i = 1) = x_i' beta + delta_k q_ik + gamma_k a_ik(l) + u_i
    u ~ N(0, tau_k K_k)

    y_i     binary phenotype (0 = control, 1 = case)
    x_i     intercept + genotype PCs (+ any --covfile covariates)
    q_ik    global ancestry proportion of ancestry k
    a_ik(l) expected local-ancestry-k dosage at locus l (continuous, 0..2)
    K_k     ancestry-k kinship, built from local-ancestry-k dosage on the
            TESTED cohort only
    H0: gamma_k = 0, tested by a score test with saddlepoint approximation.

Why a GLMM, and why here
------------------------
JasMAP's default fits a Gaussian LMM to a binary outcome. With balanced case
and control numbers that is well calibrated. Under strong imbalance it is
not, and -- importantly -- the miscalibration runs in OPPOSITE directions
depending on how common the ancestry is. In null simulations at 87.6% cases
(the SAC cohort's ratio), with global ancestry adjusted, the linear model
gave lambda_GC 0.70 for a major ancestry and 1.17 for a minor one, while
logistic regression stayed near 1.0 for both. A bias that differs by
ancestry cannot be removed by a single genomic-control correction. This is
the problem SAIGE (Zhou et al., Nat. Genet. 2018) was built for; minor
ancestry components behave like low-frequency variants.

Covariates: PCs AND global ancestry
-----------------------------------
Both are included because they control different things:
  * q_k targets the confounding specific to ancestry k -- a person with high
    global ancestry k has high local ancestry k everywhere, so any
    association between phenotype and GLOBAL ancestry would otherwise
    surface at every locus. It is the canonical admixture-mapping adjustment.
  * Genotype PCs capture residual structure q_k does not: fine-scale
    structure within ancestries, structure within the admixed cohort, and
    non-ancestry structure such as batch.
In an admixed cohort the top PCs largely span the same space as the global
ancestry proportions, so the two are partly collinear. That does not affect
the score test -- it depends only on the projection onto the covariate
space, which is well defined however that space is parametrised -- but it
means the individual PC and q_k coefficients are not separately
interpretable. They are nuisance parameters, so that is acceptable.

Computation
-----------
Expected dosage is constant across SNPs covered by the same pair of LAI
windows, so the number of DISTINCT tests is roughly the number of windows
(~11,000 genome-wide) rather than the number of SNPs. Distinct dosage
patterns are tested once and results mapped back to every SNP sharing them.
The null GLMM is fitted once per ancestry by penalised quasi-likelihood
(PQL) with REML for tau; each locus is then a cheap score test.

References: Breslow & Clayton, JASA 1993 (PQL); Chen et al., AJHG 2016
(GMMAT score test); Dey et al., AJHG 2017 and Zhou et al., Nat. Genet. 2018
(saddlepoint approximation for unbalanced case-control data).
"""
import logging
import numpy as np
from scipy import stats, optimize
from scipy.special import expit

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# kinship
# --------------------------------------------------------------------------
def ancestry_kinship(dosage_patterns):
    """Ancestry-specific kinship from local-ancestry dosage.

    dosage_patterns : (n_samples, n_patterns) -- distinct dosage patterns for
    ONE ancestry, restricted to the tested cohort. Columns with no variation
    are dropped; the rest are standardised and K = Z Z' / m.
    """
    D = np.asarray(dosage_patterns, dtype=np.float64)
    sd = D.std(axis=0)
    keep = sd > 0
    if not keep.any():
        return None
    Z = (D[:, keep] - D[:, keep].mean(axis=0)) / sd[keep]
    return (Z @ Z.T) / keep.sum()


# --------------------------------------------------------------------------
# null model
# --------------------------------------------------------------------------
def _safe_inv(M):
    try:
        return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(M)


def fit_null_glmm(y, X, K, max_iter=50, tol=1e-5):
    """Fit logit P(y=1) = X beta + u, u ~ N(0, tau K) by PQL + REML.

    Returns dict with mu (fitted probabilities incl. the random effect),
    P (the projection matrix used by the score test), tau and convergence.
    If K is None, or tau is estimated at zero, this reduces exactly to
    ordinary logistic regression.
    """
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    n = len(y)

    # start from ordinary logistic regression (IRLS)
    beta = np.zeros(X.shape[1])
    for _ in range(50):
        mu = np.clip(expit(X @ beta), 1e-8, 1 - 1e-8)
        W = mu * (1 - mu)
        step = _safe_inv(X.T @ (W[:, None] * X)) @ (X.T @ (y - mu))
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    u = np.zeros(n)
    tau = 0.0
    converged = K is None

    if K is not None:
        for it in range(max_iter):
            eta = X @ beta + u
            mu = np.clip(expit(eta), 1e-8, 1 - 1e-8)
            W = mu * (1 - mu)
            z = eta + (y - mu) / W                     # working response

            def neg_reml(log_tau):
                t = np.exp(log_tau)
                S = np.diag(1.0 / W) + t * K
                try:
                    L = np.linalg.cholesky(S)
                except np.linalg.LinAlgError:
                    return 1e30
                Si = np.linalg.inv(L).T @ np.linalg.inv(L)
                XtSiX = X.T @ Si @ X
                sign, logdet_x = np.linalg.slogdet(XtSiX)
                if sign <= 0:
                    return 1e30
                Pm = Si - Si @ X @ _safe_inv(XtSiX) @ X.T @ Si
                return 0.5 * (2 * np.sum(np.log(np.diag(L))) + logdet_x + z @ Pm @ z)

            res = optimize.minimize_scalar(neg_reml, bounds=(-12, 4), method="bounded")
            new_tau = float(np.exp(res.x)) if res.fun < neg_reml(-12) - 1e-9 else 0.0

            S = np.diag(1.0 / W) + new_tau * K
            Si = _safe_inv(S)
            new_beta = _safe_inv(X.T @ Si @ X) @ (X.T @ Si @ z)
            new_u = new_tau * K @ Si @ (z - X @ new_beta)

            delta = max(np.max(np.abs(new_beta - beta)), abs(new_tau - tau))
            beta, u, tau = new_beta, new_u, new_tau
            if delta < tol:
                converged = True
                break

    eta = X @ beta + u
    mu = np.clip(expit(eta), 1e-8, 1 - 1e-8)
    W = mu * (1 - mu)
    S = np.diag(1.0 / W) + (tau * K if K is not None and tau > 0 else 0.0)
    Si = _safe_inv(S)
    P = Si - Si @ X @ _safe_inv(X.T @ Si @ X) @ X.T @ Si
    return {"mu": mu, "W": W, "P": P, "tau": tau, "beta": beta,
            "converged": converged, "X": X}


# --------------------------------------------------------------------------
# score test + saddlepoint approximation
# --------------------------------------------------------------------------
def _spa_pvalue(s, g, mu):
    """Two-sided saddlepoint p-value for S = g'(y - mu), y_i ~ Bernoulli(mu_i).

    Lugannani-Rice approximation to the tail of the Bernoulli score
    distribution, which -- unlike the normal approximation -- captures the
    skewness that strong case-control imbalance induces.
    """
    def K0(t):
        return np.sum(np.log1p(mu * np.expm1(g * t))) - t * np.sum(g * mu)

    def K1(t):
        e = np.exp(g * t)
        return np.sum(mu * g * e / (1 - mu + mu * e)) - np.sum(g * mu)

    def K2(t):
        e = np.exp(g * t)
        d = 1 - mu + mu * e
        return np.sum((1 - mu) * mu * g * g * e / (d * d))

    def tail(q):                       # P(S >= q) for q on either side
        lo, hi = -1.0, 1.0
        for _ in range(60):
            if K1(lo) < q:
                break
            lo *= 2
        for _ in range(60):
            if K1(hi) > q:
                break
            hi *= 2
        try:
            t = optimize.brentq(lambda t: K1(t) - q, lo, hi, xtol=1e-12)
        except ValueError:
            return np.nan
        w2 = 2 * (t * q - K0(t))
        if w2 <= 0 or t == 0:
            return np.nan
        w = np.sign(t) * np.sqrt(w2)
        v = t * np.sqrt(K2(t))
        return float(1 - stats.norm.cdf(w) + stats.norm.pdf(w) * (1 / v - 1 / w))

    a, b = tail(abs(s)), tail(-abs(s))
    if np.isnan(a) or np.isnan(b):
        return np.nan
    upper = a
    lower = 1 - b                      # P(S <= -|s|)
    return float(np.clip(upper + lower, 0, 1))


def score_test(A, null, spa=True, spa_threshold=2.0):
    """Score-test every column of A (n_samples x n_tests) against the null.

    Returns (beta, z, pvalue) arrays. beta is the one-step estimate
    T / Var(T), as reported by SAIGE. Columns with no variation return NaN.
    spa=True applies the saddlepoint approximation where |z| exceeds
    spa_threshold (as in SAIGE); below it the normal approximation is
    accurate and much cheaper. spa=False uses the normal approximation
    throughout -- adequate when cases and controls are roughly balanced, but
    its TAIL p-values become unreliable under strong imbalance, which is
    exactly where significance is decided.
    """
    A = np.asarray(A, dtype=np.float64)
    mu, W, P, X = null["mu"], null["W"], null["P"], null["X"]
    resid = null.get("y") - mu if "y" in null else None
    T = A.T @ resid
    PA = P @ A
    V = np.einsum("ij,ij->j", A, PA)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = T / np.sqrt(V)
    p = 2 * stats.norm.sf(np.abs(z))
    bad = ~np.isfinite(z) | (V <= 1e-12) | (A.std(axis=0) == 0)
    z[bad] = np.nan
    p[bad] = np.nan

    # covariate-adjusted dosage under the no-random-effect weights, for the SPA
    XtWX_inv = _safe_inv(X.T @ (W[:, None] * X))
    for j in (np.flatnonzero(~bad & (np.abs(z) > spa_threshold)) if spa else []):
        g = A[:, j] - X @ (XtWX_inv @ (X.T @ (W * A[:, j])))
        var0 = np.sum(W * g * g)
        if var0 <= 0:
            continue
        # Feed the SPA a score with the SAME z as the GLMM test, so the
        # kinship-adjusted variance is preserved while the Bernoulli CGF
        # supplies the skewness (the variance-ratio device used by SAIGE).
        s = z[j] * np.sqrt(var0)
        ps = _spa_pvalue(s, g, mu)
        if np.isfinite(ps):
            p[j] = ps
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = T / V
    beta[bad] = np.nan
    return beta, z, p


def to_binary(y):
    """Map a phenotype to 0/1. Accepts {0,1} or PLINK {1,2}; else None."""
    y = np.asarray(y, dtype=np.float64)
    vals = set(np.unique(y[np.isfinite(y)]).tolist())
    if vals <= {0.0, 1.0}:
        return y
    if vals <= {1.0, 2.0}:
        return y - 1.0
    return None


def run_glmm_am(dosage, y, X, kinship_patterns=None, spa=True):
    """Convenience wrapper: fit the null once, score-test all columns.

    dosage           : (n_samples, n_tests) local-ancestry dosage to test
    y                : (n_samples,) binary phenotype, 0/1
    X                : (n_samples, p) covariates incl. intercept
    kinship_patterns : (n_samples, n_patterns) dosage used to build K_k;
                       defaults to `dosage`
    """
    K = ancestry_kinship(dosage if kinship_patterns is None else kinship_patterns)
    null = fit_null_glmm(y, X, K)
    null["y"] = np.asarray(y, dtype=np.float64)
    beta, z, p = score_test(dosage, null, spa=spa)
    return beta, p, null


def genotype_grm(G, max_snps=50000, maf_min=0.01, seed=0):
    """Genomic relationship matrix on the TESTED cohort, for the GWAS GLMM.

    G : (n_tested, n_snps) genotype dosages. SNPs below maf_min are dropped,
    and at most max_snps evenly spaced SNPs are used -- tens of thousands of
    markers estimate genome-wide relatedness accurately, and this avoids a
    second full pass over the panel. Built on tested samples only, so
    reference-panel genotypes cannot shape the relatedness structure.
    """
    G = np.asarray(G, dtype=np.float64)
    p = np.nanmean(G, axis=0) / 2.0
    maf = np.minimum(p, 1 - p)
    keep = np.flatnonzero(maf >= maf_min)
    if len(keep) > max_snps:
        keep = keep[np.linspace(0, len(keep) - 1, max_snps).astype(int)]
    if not len(keep):
        return None
    return ancestry_kinship(G[:, keep])
