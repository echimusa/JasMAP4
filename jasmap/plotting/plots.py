"""
jasmap.plotting.plots
---------------------
Publication-quality matplotlib figures for all JasMAP pipeline modes.
All public functions accept data directly and write PNG/PDF to the output
directory.  matplotlib is imported lazily so non-plotting code paths do not
require it at import time.
"""

import os
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Style helpers
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = ["#E41A1C","#377EB8","#4DAF4A","#984EA3",
           "#FF7F00","#A65628","#F781BF","#999999"]

# Global DPI setting — overridden by JasMAP4.py via set_plot_dpi()
_PLOT_DPI = 300

def set_plot_dpi(dpi: int):
    """Set the DPI used by all _save() and _fig() calls in this module."""
    global _PLOT_DPI
    _PLOT_DPI = int(dpi)

def _mpl():
    """Lazy matplotlib import."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

def _fig(w=10, h=6, dpi=None):
    dpi = dpi or _PLOT_DPI
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)
    ax.spines[["top","right"]].set_visible(False)
    return fig, ax

def _save(fig, path, dpi=None):
    dpi = dpi or _PLOT_DPI
    plt = _mpl()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {os.path.basename(path)}")


def _norm_chrom(value):
    """
    Normalise any chromosome representation to a clean string.

    Handles every form the pipeline can produce:
      "1", 1, 1.0, np.float64(1.0), "chr1", "CHR1", "X", "chrX", np.nan, None

    Returns a plain string such as "1", "X", or "NA".

    Bug fix: several plotting functions called `.isdigit()` directly on values
    taken from a DataFrame column. When that column has float dtype (which
    happens whenever the upstream tool output had no chromosome field and it
    was filled with np.nan), the element is a numpy float and
    `.isdigit()` raises `AttributeError: 'float' object has no attribute
    'isdigit'`. A float such as 1.0 also stringifies to "1.0", which is not
    `.isdigit()`, so it silently fell through to the "unknown" branch and was
    mis-assigned to chromosome 23. Routing every chromosome value through this
    helper fixes both the crash and the silent mis-assignment.
    """
    if value is None:
        return "NA"
    try:
        if isinstance(value, float) and np.isnan(value):
            return "NA"
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    if s.lower() in ("nan", "none", ""):
        return "NA"
    # Strip a leading "chr"/"CHR" prefix
    if s.lower().startswith("chr"):
        s = s[3:]
    # Collapse float-like representations: "1.0" -> "1"
    if s.replace(".", "", 1).isdigit() and s.endswith(".0"):
        s = s[:-2]
    return s or "NA"


def _chrom_sort_key(value):
    """Sort key for chromosomes: numeric first in order, then X, Y, MT, then rest."""
    s = _norm_chrom(value)
    if s.isdigit():
        return (0, int(s), "")
    order = {"X": 23, "Y": 24, "MT": 25, "M": 25}
    if s.upper() in order:
        return (1, order[s.upper()], "")
    return (2, 999, s)


def _chrom_to_int(value, default=23):
    """
    Convert a chromosome value to its integer index for karyogram row placement.
    X -> 23, Y -> 24, MT -> 25. Unrecognised values return `default`.
    Accepts float/int/str/'chr1'/'1.0'/NaN via _norm_chrom().
    """
    s = _norm_chrom(value)
    if s.isdigit():
        return int(s)
    return {"X": 23, "Y": 24, "MT": 25, "M": 25}.get(s.upper(), default)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Manhattan plot  (GWAS / admixture mapping / GIFT p-values)
# ─────────────────────────────────────────────────────────────────────────────

def manhattan(df, pval_col="P", chrom_col="CHR", bp_col="BP",
              snp_col="SNP", title="Manhattan Plot",
              sig_line=5e-8, sugg_line=1e-5,
              highlight_top=10, out_path="manhattan.png"):
    """
    Parameters
    ----------
    df : DataFrame with at least CHR, BP, P columns.
    highlight_top : colour the N most-significant SNPs red.
    """
    plt = _mpl()
    df = df.copy().dropna(subset=[pval_col])
    df[pval_col] = pd.to_numeric(df[pval_col], errors="coerce")
    df = df[df[pval_col] > 0].copy()
    df["log10p"] = -np.log10(df[pval_col])

    # Chromosome offsets
    df["_chrom_norm"] = df[chrom_col].map(_norm_chrom)
    chroms = sorted(df["_chrom_norm"].unique(), key=_chrom_sort_key)
    offsets = {}; running = 0
    for c in chroms:
        sub = df[df["_chrom_norm"] == c]
        offsets[c] = running
        running += int(sub[bp_col].max()) + int(5e6)
    df["x"] = df.apply(
        lambda r: r[bp_col] + offsets.get(_norm_chrom(r[chrom_col]), 0), axis=1)

    fig, ax = plt.subplots(figsize=(14,5), dpi=300)
    colors = ["#4393C3","#2166AC"]
    for i, c in enumerate(chroms):
        sub = df[df["_chrom_norm"] == c]
        ax.scatter(sub["x"], sub["log10p"],
                   c=colors[i%2], s=4, alpha=0.75, linewidths=0)
    if highlight_top > 0:
        top = df.nlargest(highlight_top, "log10p")
        ax.scatter(top["x"], top["log10p"], c="#E31A1C", s=18, zorder=5)

    if sig_line:
        ax.axhline(-np.log10(sig_line),  color="#E31A1C", lw=0.8, ls="--", label=f"p={sig_line:.0e}")
    if sugg_line:
        ax.axhline(-np.log10(sugg_line), color="#FF7F00", lw=0.8, ls=":", label=f"p={sugg_line:.0e}")

    tick_pos = [offsets[c] + (df[df["_chrom_norm"]==c][bp_col].max()//2)
                for c in chroms]
    ax.set_xticks(tick_pos); ax.set_xticklabels(chroms, fontsize=7)
    ax.set_xlabel("Chromosome"); ax.set_ylabel("-log₁₀(p)")
    ax.set_title(title, fontweight="bold"); ax.legend(fontsize=8)
    ax.spines[["top","right"]].set_visible(False)
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 2. QQ plot
# ─────────────────────────────────────────────────────────────────────────────

def qq_plot(pvals, title="QQ Plot", out_path="qq.png"):
    plt = _mpl()
    pvals = np.array(pvals, dtype=float)
    pvals = pvals[np.isfinite(pvals) & (pvals > 0)]
    n = len(pvals)
    # Bug fix: when EVERY p-value is NaN/non-positive (e.g. an association
    # test that degenerated entirely -- a constant phenotype makes every
    # SNP's test statistic undefined), the filter above leaves pvals
    # completely empty. observed/expected then also end up empty, and
    # max() on an empty array raises "zero-size array to reduction
    # operation maximum which has no identity" -- a cryptic numpy
    # internals message that gives no hint of the actual, simple cause.
    # Found via a real run with a genuinely constant phenotype (see
    # AUC=nan/r=nan's own root cause) collapsing an entire GWAS's worth
    # of p-values to NaN. Raising a clear, specific message here instead
    # still gets caught by the caller's own try/except (so the pipeline
    # doesn't crash), but now says exactly what happened.
    if n == 0:
        raise ValueError(
            "No finite, positive p-values to plot -- every association "
            "test result was NaN or non-positive. This usually means the "
            "phenotype has no usable variation within the tested cohort "
            "(e.g. a constant value) -- verify this before trusting any "
            "other result from this run.")
    observed = np.sort(-np.log10(pvals))[::-1]
    expected = -np.log10(np.arange(1, n+1) / (n+1))

    fig, ax = plt.subplots(figsize=(6,6), dpi=300)
    ax.scatter(expected, observed, s=8, alpha=0.7, color="#377EB8")
    lim = max(expected.max(), observed.max()) * 1.05
    ax.plot([0,lim],[0,lim], "r--", lw=0.8)

    # Genomic inflation factor λ
    chisq = np.array([x**2 for x in pvals])
    lam = np.median(chisq) / 0.4549  # ~chi2(df=1) median
    ax.text(0.05, 0.95, f"lambda_GC = {lam:.3f}", transform=ax.transAxes,
            va="top", fontsize=10)
    ax.set_xlabel("Expected -log₁₀(p)"); ax.set_ylabel("Observed -log₁₀(p)")
    ax.set_title(title, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 3. GIFT field potential / SNP-importance bar chart
# ─────────────────────────────────────────────────────────────────────────────

def gift_snp_importance(bim, scores, top_n=20, title="GIFT SNP Importance",
                        out_path="gift_snp_importance.png"):
    plt = _mpl()
    # Bug fix: top_n was used directly for range(top_n)/set_yticks(range(top_n))
    # while idx/vals were correctly truncated by scores[idx]'s slice -- when
    # len(scores) < top_n (e.g. --chr restricting to a chromosome with few
    # SNPs), idx/vals end up SHORTER than top_n (numpy slicing past the end
    # is not an error), but range(top_n) stays at the original, larger
    # value -- ax.barh() then tries to broadcast a shorter values array
    # against a longer positions array and crashes. Found via a real
    # --chr + --mode gift_admix run. Clamping top_n to the actual number
    # of scores available fixes this for every caller, not just the ones
    # that happen to pass an already-clamped value explicitly.
    top_n = min(top_n, len(scores))
    idx = np.argsort(scores)[::-1][:top_n]
    labels = [f"{bim.iloc[i]['CHR']}:{bim.iloc[i]['SNP']}" for i in idx]
    vals   = scores[idx]

    fig, ax = plt.subplots(figsize=(10,4), dpi=300)
    ax.barh(range(top_n), vals[::-1], color=PALETTE[1], alpha=0.85)
    ax.set_yticks(range(top_n)); ax.set_yticklabels(labels[::-1], fontsize=8)
    ax.set_xlabel("Normalised GIFT Score"); ax.set_title(title, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    _save(fig, out_path)


def gift_field_potential(U, samples_df, title="GIFT Field Potential Energy",
                         out_path="gift_field_potential.png"):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8,4), dpi=300)
    ax.hist(U, bins=40, color=PALETTE[0], alpha=0.8, edgecolor="white")
    ax.axvline(U.mean(), color="k", lw=1.5, ls="--", label=f"mean={U.mean():.3f}")
    ax.set_xlabel("Field Potential Energy H"); ax.set_ylabel("Count")
    ax.set_title(title, fontweight="bold"); ax.legend()
    ax.spines[["top","right"]].set_visible(False)
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Global ancestry bar chart
# ─────────────────────────────────────────────────────────────────────────────

def global_ancestry_bar(Q, sample_ids=None, pop_labels=None,
                        title="Global Ancestry Proportions",
                        out_path="global_ancestry.png"):
    plt = _mpl()
    n, K = Q.shape
    pop_labels = pop_labels or [f"Anc{k+1}" for k in range(K)]
    colors = PALETTE[:K]

    # Sort by dominant ancestry
    order = np.argsort(-Q[:, 0])
    Qs = Q[order]

    # Same unbounded-figsize class of bug as lai_heatmap()'s (fixed there
    # after a real run hit matplotlib's hard 65,536 px limit). This one
    # scales at n//5 rather than n//3, so it survived at that cohort size
    # (1363 samples -> 272 in -> 81,600 px would ALSO have exceeded the
    # limit; smaller cohorts stayed under it by luck rather than design).
    # Capped proactively on the same reasoning: past a certain width the
    # extra inches add nothing readable, they just risk an unrenderable
    # canvas.
    _MAX_FIG_IN = 60
    fig, ax = plt.subplots(figsize=(min(max(8, n // 5), _MAX_FIG_IN), 3.5), dpi=300)
    bottoms = np.zeros(n)
    for k in range(K):
        ax.bar(range(n), Qs[:,k], bottom=bottoms, color=colors[k],
               width=1.0, label=pop_labels[k], linewidth=0)
        bottoms += Qs[:,k]
    ax.set_xlim(-0.5, n-0.5); ax.set_ylim(0,1)
    ax.set_xlabel("Individuals"); ax.set_ylabel("Ancestry proportion")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.6)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.set_yticks([0,0.25,0.5,0.75,1.0])
    if sample_ids is not None and n <= 40:
        ax.set_xticks(range(n)); ax.set_xticklabels(
            np.array(sample_ids)[order], rotation=90, fontsize=6)
    else:
        ax.set_xticks([])
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Karyogram (local ancestry)
# ─────────────────────────────────────────────────────────────────────────────

def karyogram(bed_a_df, bed_b_df, ind_name="Sample",
              pop_order=None, out_path="karyogram.png"):
    """
    bed_a_df / bed_b_df: DataFrames with columns CHR, START, STOP, ANC
    pop_order: list of ancestry labels
    """
    plt = _mpl()
    import matplotlib.patches as mpatches

    ancs = sorted(set(bed_a_df["ANC"].tolist() + bed_b_df["ANC"].tolist()))
    pop_order = pop_order or ancs
    cmap = {a: PALETTE[i % len(PALETTE)] for i,a in enumerate(pop_order)}
    cmap.setdefault("UNK", "#999999")

    fig, ax = plt.subplots(figsize=(12,7), dpi=300)
    chroms = list(range(1,23))
    ax.set_xlim(-5, 300); ax.set_ylim(len(chroms)+0.6, 0.4)
    ax.set_yticks(chroms)
    ax.set_xlabel("Genetic / physical position (Mb or cM)")
    ax.set_ylabel("Chromosome"); ax.set_title(ind_name, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)

    def _draw(df, hap_offset):
        for _, row in df.iterrows():
            try:
                chrom = int(str(row["CHR"]).replace("chr",""))
            except ValueError:
                chrom = 23
            y = chrom + hap_offset
            col = cmap.get(str(row["ANC"]), "#999999")
            rect = plt.Rectangle(
                (float(row["START"])/1e6, y - 0.35),
                (float(row["STOP"]) - float(row["START"]))/1e6,
                0.35, color=col, lw=0)
            ax.add_patch(rect)

    _draw(bed_a_df, 0)     # top half
    _draw(bed_b_df, -0.35) # bottom half

    handles = [mpatches.Patch(color=cmap.get(p,"#999999"), label=p)
               for p in pop_order + (["UNK"] if "UNK" not in pop_order else [])]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.6)
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Local ancestry heatmap (window × individual)
# ─────────────────────────────────────────────────────────────────────────────

def lai_heatmap(LA, sample_ids=None, ancestry_idx=0,
                title="Local Ancestry (Ancestry 1 proportion)",
                out_path="lai_heatmap.png"):
    """
    LA: np.ndarray (n_samples, n_windows, K)
    ancestry_idx: which ancestry to visualise
    """
    plt = _mpl()

    data = LA[:, :, ancestry_idx].T   # (windows, samples)
    # Bug fix: figsize width was max(8, n_samples//3) with NO upper bound,
    # so at real cohort sizes it demanded a canvas matplotlib flatly
    # refuses to allocate -- "Image size of 81600x1050 pixels is too
    # large. It must be less than 2^16 (65536) in each direction."
    # Confirmed exactly: 816 samples -> 816//3 = 272 inches -> 272*300dpi
    # = 81,600 px, matching a real run's error message precisely. The
    # per-sample width scaling only ever mattered for readable x-axis tick
    # LABELS, which this function already restricts to <= 30 samples
    # anyway (see the sample_ids block below) -- beyond that the extra
    # width buys nothing but an unrenderable canvas. Capped at 60 inches
    # (18,000 px at dpi=300), comfortably inside the limit while still
    # giving a wide, detailed heatmap. Same class of unbounded-figsize
    # bug already fixed in lai_region_plot(); this one sat in a different
    # function and was only reachable at larger cohort sizes.
    _MAX_FIG_IN = 60
    fig_w = min(max(8, data.shape[1] // 3), _MAX_FIG_IN)
    fig, ax = plt.subplots(figsize=(fig_w, 5), dpi=300)
    im = ax.imshow(data, aspect="auto", cmap="RdYlBu_r",
                   vmin=0, vmax=1, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Proportion")
    ax.set_xlabel("Individuals"); ax.set_ylabel("Genomic window")
    ax.set_title(title, fontweight="bold")
    if sample_ids is not None and data.shape[1] <= 30:
        ax.set_xticks(range(data.shape[1]))
        ax.set_xticklabels(sample_ids, rotation=90, fontsize=6)
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 7. PCA scatter
# ─────────────────────────────────────────────────────────────────────────────

def pca_scatter(Z, labels=None, var_explained=None,
                pc_x=0, pc_y=1, title="PCA",
                out_path="pca.png"):
    """
    Z: (n, n_components) PCA scores
    labels: list of population/group labels (str) per sample
    """
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7,6), dpi=300)

    if labels is None:
        ax.scatter(Z[:,pc_x], Z[:,pc_y], s=15, alpha=0.7, color=PALETTE[0])
    else:
        unique = sorted(set(labels))
        for i, lab in enumerate(unique):
            idx = [j for j,l in enumerate(labels) if l == lab]
            ax.scatter(Z[idx,pc_x], Z[idx,pc_y], s=18, alpha=0.75,
                       label=lab, color=PALETTE[i%len(PALETTE)], linewidths=0)
        ax.legend(fontsize=8, markerscale=1.5, framealpha=0.6)

    xlab = f"PC{pc_x+1}" + (f" ({var_explained[pc_x]:.1f}%)" if var_explained is not None else "")
    ylab = f"PC{pc_y+1}" + (f" ({var_explained[pc_y]:.1f}%)" if var_explained is not None else "")
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.set_title(title, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 8. JasMAP joint association (Manhattan-style for JointPPA)
# ─────────────────────────────────────────────────────────────────────────────

def joint_ppa_plot(df, ppa_col="Joint_PPA", pval_col="SNP_Pval",
                   title="JasMAP Joint Posterior Probability of Association",
                   out_path="joint_ppa.png"):
    plt = _mpl()
    df = df.copy().dropna(subset=[ppa_col])
    df[ppa_col] = pd.to_numeric(df[ppa_col], errors="coerce")

    fig, axes = plt.subplots(2, 1, figsize=(12,8), dpi=300,
                             sharex=True,
                             gridspec_kw={"height_ratios":[2,1]})

    axes[0].scatter(range(len(df)), df[ppa_col],
                    c=PALETTE[1], s=6, alpha=0.7, linewidths=0)
    axes[0].axhline(0.5, color="red", lw=0.8, ls="--", label="PPA=0.5")
    axes[0].set_ylabel("Joint PPA"); axes[0].legend(fontsize=8)
    axes[0].set_title(title, fontweight="bold")
    axes[0].spines[["top","right"]].set_visible(False)

    if pval_col in df.columns:
        pvals = pd.to_numeric(df[pval_col], errors="coerce").fillna(1)
        pvals = np.clip(pvals, 1e-300, 1)
        axes[1].scatter(range(len(df)), -np.log10(pvals),
                        c=PALETTE[0], s=4, alpha=0.6, linewidths=0)
        axes[1].axhline(-np.log10(5e-8), color="red", lw=0.8, ls="--")
        axes[1].set_ylabel("-log₁₀(p_SNP)")
        axes[1].spines[["top","right"]].set_visible(False)

    axes[-1].set_xlabel("SNP rank")
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 9. PRS distribution (cases vs. controls)
# ─────────────────────────────────────────────────────────────────────────────

def prs_distribution(prs, y, title="PRS Distribution",
                     out_path="prs_distribution.png"):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7,5), dpi=300)
    for lbl, col, val in [(0,"#4393C3","Controls"),(1,"#D6604D","Cases")]:
        sub = prs[y == lbl]
        if len(sub): ax.hist(sub, bins=35, alpha=0.65, color=col, label=val,
                             density=True, edgecolor="white")
    ax.set_xlabel("PRS (standardised)"); ax.set_ylabel("Density")
    ax.set_title(title, fontweight="bold"); ax.legend()
    ax.spines[["top","right"]].set_visible(False)
    _save(fig, out_path)


def prs_roc(y_true, y_score, title="ROC Curve", out_path="prs_roc.png"):
    plt = _mpl()
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6,6), dpi=300)
    ax.plot(fpr, tpr, color=PALETTE[0], lw=1.5, label=f"AUC = {auc:.3f}")
    ax.plot([0,1],[0,1], "k--", lw=0.8)
    ax.set_xlabel("FPR (1-Specificity)"); ax.set_ylabel("TPR (Sensitivity)")
    ax.set_title(title, fontweight="bold"); ax.legend()
    ax.spines[["top","right"]].set_visible(False)
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Admixture-mapping scan (per-ancestry per-window p-values)
# ─────────────────────────────────────────────────────────────────────────────

def admixture_mapping_plot(scan_df, pop_labels=None,
                           sig_line=1e-6, title="Admixture Mapping Scan",
                           out_path="admixture_mapping.png"):
    plt = _mpl()
    ks = sorted(scan_df["ANCESTRY_K"].unique())
    pop_labels = pop_labels or [f"Anc{k+1}" for k in ks]
    fig, axes = plt.subplots(len(ks), 1, figsize=(12, 3*len(ks)), dpi=300,
                             sharex=True)
    if len(ks) == 1: axes = [axes]

    for i, k in enumerate(ks):
        sub = scan_df[scan_df["ANCESTRY_K"]==k].dropna(subset=["P"])
        sub = sub[sub["P"]>0]
        y = -np.log10(sub["P"])
        axes[i].scatter(range(len(sub)), y, s=6, alpha=0.7,
                        color=PALETTE[i%len(PALETTE)], linewidths=0)
        if sig_line:
            axes[i].axhline(-np.log10(sig_line), color="red",
                            lw=0.8, ls="--", label=f"p={sig_line:.0e}")
        axes[i].set_ylabel(f"-log₁₀(p)\n{pop_labels[i]}", fontsize=9)
        axes[i].spines[["top","right"]].set_visible(False)
        if i==0: axes[i].set_title(title, fontweight="bold")

    axes[-1].set_xlabel("Genomic window")
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 11 & 12  – Karyogram and LAI-region plot
#
#  These are modern Python 3 reimplementations of armartin's original
#  plot_karyogram.py and plot_lai_region.py, adapted to work directly from
#  JasMAP4's ConsensusLAI outputs (lanc_array, L_, BIM positions) without
#  requiring brewer2mpl, bedparser, or an external centromeres_hg19.bed file.
# ─────────────────────────────────────────────────────────────────────────────

# ── Bundled hg19 centromere positions (Mb) ─────────────────────────────────
# Source: UCSC hg19 genome browser, gaps track (centromere type).
# Stored as {chrom_int: (p_end_Mb, q_start_Mb)} for chroms 1-22.
_HG19_CENTRO = {
    1:(120.6,128.2),  2:(90.5,96.8),   3:(87.9,93.9),   4:(48.2,52.7),
    5:(46.1,50.7),    6:(58.7,63.3),   7:(58.0,61.1),   8:(43.1,48.1),
    9:(47.7,50.7),    10:(39.2,42.3),  11:(51.6,55.7),  12:(33.3,38.2),
    13:(16.3,19.5),   14:(16.1,19.1),  15:(15.8,20.7),  16:(34.6,38.2),
    17:(22.2,25.8),   18:(15.4,19.0),  19:(24.4,28.6),  20:(26.3,30.0),
    21:(11.1,13.0),   22:(13.7,17.9),
}


def _extract_ancestry_tracts(lanc_array, bim, sample_idx, pop_labels):
    """
    Convert per-SNP ancestry dosage into a list of ancestry tracts
    (contiguous runs of the same dominant ancestry) for one sample.

    Parameters
    ----------
    lanc_array : np.ndarray (K, n_snps, n_samples)  dosage {0,1,2}
    bim        : pd.DataFrame with columns CHR, BP (physical positions in bp)
                 or None (falls back to SNP rank * 100 kb as position)
    sample_idx : int
    pop_labels : list of K ancestry names

    Returns
    -------
    list of dicts { chrom, start_mb, end_mb, ancestry_label }
    Each entry is one contiguous ancestry tract on one chromosome.
    """
    K, n_snps, n_samples = lanc_array.shape
    dosage  = lanc_array[:, :, sample_idx].astype(np.float32)  # (K, n_snps)
    hard    = dosage.argmax(axis=0)  # (n_snps,) index 0..K-1

    if bim is not None:
        # Normalise column names (handle both uppercase and lowercase)
        bim_cols = {c.upper(): c for c in bim.columns}
        chr_col  = bim_cols.get("CHR", bim_cols.get("CHROM", None))
        bp_col   = bim_cols.get("BP",  bim_cols.get("POS",   None))
        if chr_col and bp_col:
            chroms = bim[chr_col].values
            bp     = bim[bp_col].astype(float).values / 1e6  # -> Mb
        else:
            chroms = np.ones(n_snps, dtype=int)
            bp     = np.arange(n_snps, dtype=float) * 0.1
    else:
        chroms = np.ones(n_snps, dtype=int)
        bp     = np.arange(n_snps, dtype=float) * 0.1

    tracts = []
    i = 0
    while i < n_snps:
        j = i + 1
        while j < n_snps and hard[j] == hard[i] and chroms[j] == chroms[i]:
            j += 1
        tracts.append(dict(
            chrom    = _chrom_to_int(chroms[i]),
            start_mb = float(bp[i]),
            end_mb   = float(bp[j-1]),
            ancestry = pop_labels[int(hard[i])] if int(hard[i]) < len(pop_labels)
                       else "UNK"
        ))
        i = j
    return tracts


def karyogram_from_lanc(lanc_array, bim, sample_ids, pop_labels,
                        sample_idx=0,
                        chrX=False,
                        title=None,
                        out_path="karyogram.png"):
    """
    Draw a chromosome karyogram (ideogram with coloured ancestry tracts)
    directly from JasMAP4's lanc_array.

    Reimplements armartin's plot_karyogram.py in pure Python 3 + matplotlib:
    - No brewer2mpl required (uses PALETTE colour cycle instead)
    - No external centromeres_hg19.bed required (bundled hg19 coordinates)
    - Adapted for the (K, n_snps, n_samples) dosage array format

    Parameters
    ----------
    lanc_array  : np.ndarray (K, n_snps, n_samples)
    bim         : pd.DataFrame with CHR and BP columns (or None)
    sample_ids  : list of sample ID strings
    pop_labels  : list of K ancestry names, e.g. ["AFR","EUR","SAN"]
    sample_idx  : which sample to draw (0-indexed, default 0)
    chrX        : include chromosome 23 / X? (default False)
    title       : plot title (defaults to sample ID)
    out_path    : output PNG/PDF path
    """
    plt = _mpl()
    import matplotlib.patches as mpatches

    n_chrom = 23 if chrX else 22
    ind_name = sample_ids[sample_idx] if sample_ids else f"Sample_{sample_idx}"
    title    = title or f"Ancestry Karyogram - {ind_name}"

    tracts = _extract_ancestry_tracts(lanc_array, bim, sample_idx, pop_labels)

    # Colour map: one colour per ancestry + black for UNK
    anc_colors = {}
    for i, lbl in enumerate(pop_labels):
        anc_colors[lbl] = PALETTE[i % len(PALETTE)]
    anc_colors["UNK"] = "#333333"

    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_xlim(-2, 260)
    ax.set_ylim(n_chrom + 0.8, 0.2)
    ax.set_xlabel("Physical position (Mb)", fontsize=10)
    ax.set_ylabel("Chromosome",            fontsize=10)
    ax.set_title(title, fontweight="bold",  fontsize=12)
    ax.set_yticks(range(1, n_chrom + 1))
    ax.set_yticklabels([str(c) for c in range(1, n_chrom + 1)], fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)

    # Draw chromosome outlines (light grey bars)
    for c in range(1, n_chrom + 1):
        ax.barh(c, 250, left=0, height=0.75, color="#e8e8e8",
                edgecolor="#aaaaaa", linewidth=0.3, align="center")
        # Centromere notch
        if c in _HG19_CENTRO:
            p_end, q_start = _HG19_CENTRO[c]
            ax.barh(c, q_start - p_end, left=p_end,
                    height=0.75, color="#cccccc",
                    edgecolor="#888888", linewidth=0.3, align="center")

    # Draw ancestry tracts (top half = haplotype A proxy)
    for tr in tracts:
        c = tr["chrom"]
        if c > n_chrom:
            continue
        col = anc_colors.get(tr["ancestry"], "#333333")
        width = max(tr["end_mb"] - tr["start_mb"], 0.3)
        ax.barh(c, width, left=tr["start_mb"],
                height=0.35, color=col, alpha=0.85,
                edgecolor="none", align="center")

    # Legend
    handles = [mpatches.Patch(color=anc_colors.get(p, "#333333"), label=p)
               for p in list(pop_labels) + ["UNK"]]
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              framealpha=0.7, title="Ancestry")

    _save(fig, out_path)


def lai_region_plot(lanc_array, bim, phenotype, sample_ids, pop_labels,
                    chrom, start_mb, end_mb,
                    title=None,
                    out_path="lai_region.png",
                    max_samples=100):
    """
    Per-individual ancestry tract plot for a specific genomic region,
    sorted by phenotype value — directly inspired by armartin's
    plot_lai_region.py, reimplemented in Python 3 without bedparser,
    brewer2mpl, or any external files.

    Each row = one sample. Top half of row = haplotype A proxy (dominant
    ancestry call). Bottom half = haplotype B proxy (second-most-likely
    ancestry). Samples are sorted by phenotype (lowest to highest, bottom
    to top — same visual convention as the original script).

    Parameters
    ----------
    lanc_array  : np.ndarray (K, n_snps, n_samples)
    bim         : pd.DataFrame with CHR and BP columns (positions in bp)
    phenotype   : np.ndarray (n_samples,) phenotype values (NaN = missing)
    sample_ids  : list of sample ID strings
    pop_labels  : list of K ancestry names
    chrom       : chromosome number (int) to display
    start_mb    : region start in Mb
    end_mb      : region end in Mb
    title       : plot title
    max_samples : cap on rows actually drawn (default 100). This is an
                  illustrative/exploratory plot, not a complete cohort
                  visualisation -- for a real, large cohort (hundreds to
                  thousands of samples), the per-sample drawing loop
                  itself (independent of figure/canvas size) becomes the
                  bottleneck, found via a real run where even a properly
                  memory-capped figure still took >90s to render for
                  ~1,363 samples. When the cohort exceeds max_samples,
                  a phenotype-stratified subsample (evenly spaced across
                  the sorted phenotype range, so both extremes and the
                  middle of the distribution stay represented) is drawn
                  instead of silently truncating to just the first N.
    out_path    : output path
    """
    plt = _mpl()

    K, n_snps, n_samples = lanc_array.shape

    # ── Filter SNPs to the requested region ──────────────────────────────────
    if bim is not None:
        bim_cols = {c.upper(): c for c in bim.columns}
        chr_col  = bim_cols.get("CHR", bim_cols.get("CHROM", None))
        bp_col   = bim_cols.get("BP",  bim_cols.get("POS", None))
        if chr_col and bp_col:
            chrom_arr = bim[chr_col].astype(str).values
            bp_mb     = bim[bp_col].astype(float).values / 1e6
            chrom_str = str(chrom)
            mask = ((chrom_arr == chrom_str) &
                    (bp_mb   >= start_mb) &
                    (bp_mb   <= end_mb))
        else:
            mask = np.ones(n_snps, dtype=bool)
            bp_mb = np.linspace(start_mb, end_mb, n_snps)
    else:
        mask  = np.ones(n_snps, dtype=bool)
        bp_mb = np.linspace(start_mb, end_mb, n_snps)

    if not mask.any():
        import warnings
        warnings.warn(f"[lai_region_plot] No SNPs found on chr{chrom} "
                      f"{start_mb}-{end_mb} Mb - plotting all available SNPs")
        mask  = np.ones(n_snps, dtype=bool)
        bp_mb = np.linspace(start_mb, end_mb, n_snps)

    regional_dosage = lanc_array[:, mask, :]      # (K, n_reg, n_samples)
    regional_bp     = bp_mb[mask]
    n_reg           = regional_dosage.shape[1]

    # ── Sort samples by phenotype (ascending) ───────────────────────────────
    valid = np.isfinite(phenotype)
    order = np.argsort(phenotype)                  # ascending (lowest bottom)

    # Cap the number of rows actually drawn -- see max_samples' docstring
    # above for why. Filter to valid samples FIRST (so the max_samples
    # budget isn't wasted on entries that would just get skipped anyway),
    # then subsample evenly across the phenotype-sorted order rather than
    # truncating to the first N, so both extremes and the middle of the
    # phenotype distribution stay represented in the plot.
    order = order[valid[order]]
    if len(order) > max_samples:
        pick = np.linspace(0, len(order) - 1, max_samples).round().astype(int)
        order = order[np.unique(pick)]
    valid = np.zeros_like(valid)
    valid[order] = True

    # ── Colour map ───────────────────────────────────────────────────────────
    anc_colors = {}
    for i, lbl in enumerate(pop_labels):
        anc_colors[lbl] = PALETTE[i % len(PALETTE)]
    anc_colors["UNK"] = "#333333"

    n_valid    = int(valid.sum())
    row_height = 0.8
    # Defensive cap: figsize scales directly with (end_mb - start_mb) and
    # n_valid, both of which can be arbitrarily large for real data (a
    # full chromosome span, or a biobank-scale sample count) -- at
    # dpi=300, an uncapped figure can demand tens or hundreds of GB for
    # the raster canvas alone, crashing with a memory allocation error
    # rather than a clear message. Capping width/height independently
    # (rather than the Mb range itself, which callers may have good
    # reason to want plotted in full) keeps this function safe to call
    # with any input, at the cost of a lower-resolution render for very
    # large regions/cohorts -- reasonable for what is fundamentally a
    # illustrative/exploratory plot, not a publication-critical one.
    fig_w = min(max(8, (end_mb - start_mb) * 4), 60)
    fig_h = min(max(6, n_valid * 0.25), 60)
    fig, ax    = plt.subplots(
        figsize=(fig_w, fig_h),
        dpi=300)
    ax.set_xlim(start_mb - 0.05, end_mb + 0.05)
    ax.set_ylim(-0.5, n_valid * 2 + 0.5)
    ax.set_xlabel(f"Physical position chr{chrom} (Mb)", fontsize=10)
    ax.set_ylabel("Individual (sorted by phenotype)", fontsize=10)
    title = title or f"Local Ancestry - chr{chrom}:{start_mb:.1f}-{end_mb:.1f} Mb"
    ax.set_title(title, fontweight="bold")
    ax.set_yticks([])
    ax.spines[["top","right","left"]].set_visible(False)

    # ── Draw one row per sample ───────────────────────────────────────────────
    _tract_rects = []  # accumulated across all samples, drawn in one batch below
    for row_i, si in enumerate(order):
        if not valid[si]:
            continue
        sample_dosage = regional_dosage[:, :, si]  # (K, n_reg)
        # haplotype A proxy: dominant ancestry per SNP
        hap_a = sample_dosage.argmax(axis=0)        # (n_reg,)
        # haplotype B proxy: second-most-likely ancestry
        tmp = sample_dosage.copy()
        tmp[hap_a, np.arange(n_reg)] = -1
        hap_b = tmp.argmax(axis=0)

        y_base_a = row_i * 2 + 1        # top half of row
        y_base_b = row_i * 2            # bottom half of row

        # Performance fix: this used to call ax.barh() once per tract
        # SEGMENT (once per contiguous run of identical ancestry calls),
        # for every haplotype, for every sample -- individually adding a
        # bar/patch to a matplotlib Axes has real per-call overhead, and
        # at real cohort scale (hundreds to thousands of samples, each
        # with its own tract boundaries) this accumulates into tens of
        # thousands of separate add-patch calls. Measured directly: a
        # tight loop of 50,000 individual ax.add_patch() calls alone
        # exceeded a 30s budget before even reaching the (fast, <1s)
        # figure-save step -- confirming the per-call overhead, not
        # figure size or data volume, was the actual bottleneck. Fix:
        # collect every tract as a plain (x, y, width, height, color)
        # tuple here, in the loop, but defer all actual drawing to a
        # single batched PatchCollection after the loop -- matplotlib
        # renders one collection of N patches far faster than N
        # individual add_patch calls.
        for hap_calls, y_base in [(hap_a, y_base_a), (hap_b, y_base_b)]:
            j = 0
            while j < n_reg:
                k_end = j + 1
                while k_end < n_reg and hap_calls[k_end] == hap_calls[j]:
                    k_end += 1
                lbl   = (pop_labels[hap_calls[j]]
                         if hap_calls[j] < len(pop_labels) else "UNK")
                col   = anc_colors.get(lbl, "#333333")
                left  = regional_bp[j]
                right = regional_bp[k_end - 1]
                _tract_rects.append((left, y_base, right - left,
                                     row_height * 0.45, col))
                j = k_end

    if _tract_rects:
        from matplotlib.patches import Rectangle as _Rect
        from matplotlib.collections import PatchCollection
        patches = [_Rect((x, y), w, h) for x, y, w, h, _c in _tract_rects]
        colors  = [c for *_, c in _tract_rects]
        pc = PatchCollection(patches, facecolors=colors, edgecolors="none",
                            alpha=0.85)
        ax.add_collection(pc)

    # Phenotype colour bar (right side)
    pheno_sorted = phenotype[order][valid[order]]
    if len(pheno_sorted) > 1:
        sm = plt.cm.ScalarMappable(
            cmap="RdYlBu_r",
            norm=plt.Normalize(pheno_sorted.min(), pheno_sorted.max()))
        for row_i, val in enumerate(pheno_sorted):
            c = sm.to_rgba(val)
            ax.barh(row_i * 2 + 0.5, 0.02 * (end_mb - start_mb),
                    left=end_mb + 0.01,
                    height=1.6, color=c, edgecolor="none", align="center")
        plt.colorbar(sm, ax=ax, label="Phenotype", shrink=0.4,
                     pad=0.12, aspect=30)

    # Legend
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=anc_colors.get(p, "#333333"), label=p)
               for p in list(pop_labels) + ["UNK"]]
    ax.legend(handles=handles, loc="upper left", fontsize=7,
              framealpha=0.7, title="Ancestry")

    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 13-16  Gene-level rare-variant (RVAT) publication plots
# ─────────────────────────────────────────────────────────────────────────────

def rvat_gene_manhattan(gene_df, pval_col="pval_consensus",
                        chrom_col="chr", gene_col="snp",
                        test_cols=None,
                        sig_line=2.5e-6, sugg_line=1e-4,
                        title="Gene-Level Association (RVAT)",
                        out_path="rvat_gene_manhattan.png"):
    """
    Gene-level Manhattan plot.

    x-axis = genomic position of each gene (chromosome offset + midpoint).
    y-axis = -log10(consensus p-value).
    Points are sized by number of rare variants in the gene when available,
    and coloured by the most significant contributing test type (burden /
    SKAT / SKAT-O).

    Parameters
    ----------
    gene_df   : DataFrame with at least gene/snp, chr, pos, and a p-value column.
    test_cols : list of tool p-value column names for colour assignment.
    """
    plt = _mpl()
    df = gene_df.copy().dropna(subset=[pval_col])
    df[pval_col] = pd.to_numeric(df[pval_col], errors="coerce")
    df = df[df[pval_col] > 0]
    df["log10p"] = -np.log10(df[pval_col])

    # Build x-axis with chromosome offsets (genes, not SNPs)
    df["_chrom_norm"] = df[chrom_col].map(_norm_chrom)
    chroms = sorted(df["_chrom_norm"].unique(), key=_chrom_sort_key)
    offsets = {}; running = 0
    for c in chroms:
        sub = df[df["_chrom_norm"] == c]
        offsets[c] = running
        running += 300_000_000  # fixed 300 Mb spacing per chromosome
    df["x"] = df.apply(
        lambda r: float(r.get("pos", 0) or 0) + offsets.get(_norm_chrom(r[chrom_col]), 0),
        axis=1)

    # Colour by most significant test when multi-tool columns available
    test_palette = {"burden":"#E41A1C", "skat_approx":"#377EB8",
                    "firth_approx":"#4DAF4A", "ols":"#984EA3"}
    if test_cols:
        def _best_test(row):
            best_p, best_t = 1.0, "other"
            for tc in test_cols:
                v = pd.to_numeric(row.get(tc), errors="coerce")
                if pd.notna(v) and v < best_p:
                    best_p, best_t = float(v), tc.replace("pval_","")
            return best_t
        df["_best_test"] = df.apply(_best_test, axis=1)
    else:
        df["_best_test"] = "gene"

    fig, ax = plt.subplots(figsize=(14, 6), dpi=_PLOT_DPI)
    uniq_tests = df["_best_test"].unique()
    for i, test in enumerate(uniq_tests):
        sub = df[df["_best_test"] == test]
        col = test_palette.get(test, PALETTE[i % len(PALETTE)])
        sizes = np.clip(sub["n_tools_significant"].values * 20
                        if "n_tools_significant" in sub.columns
                        else np.full(len(sub), 30), 10, 120)
        ax.scatter(sub["x"], sub["log10p"], s=sizes, c=col, alpha=0.8,
                   label=test, linewidths=0, zorder=3)

    # Reference lines
    if sig_line:
        ax.axhline(-np.log10(sig_line), color="#D62728", lw=1.0, ls="--",
                   label=f"Exome-wide p={sig_line:.1e}")
    if sugg_line:
        ax.axhline(-np.log10(sugg_line), color="#FF7F0E", lw=0.8, ls=":",
                   label=f"Suggestive p={sugg_line:.1e}")

    # Annotate top 5 genes
    top5 = df.nlargest(5, "log10p")
    for _, row in top5.iterrows():
        ax.annotate(str(row[gene_col]),
                    xy=(row["x"], row["log10p"]),
                    xytext=(0, 6), textcoords="offset points",
                    fontsize=6, ha="center", va="bottom",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="grey"))

    # Chromosome tick marks
    tick_pos = []
    for c in chroms:
        sub = df[df["_chrom_norm"] == c]
        if len(sub):
            tick_pos.append((offsets[c] + 150_000_000, c))
    ax.set_xticks([t[0] for t in tick_pos])
    ax.set_xticklabels([t[1] for t in tick_pos], fontsize=7)
    ax.set_xlabel("Chromosome"); ax.set_ylabel("-log\u2081\u2080(p)")
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.legend(fontsize=7, markerscale=1.2, framealpha=0.7,
              title="Leading test", title_fontsize=7)
    ax.spines[["top","right"]].set_visible(False)
    # Set y-axis to always show data range, not just threshold lines
    data_max = df["log10p"].max()
    ax.set_ylim(0, max(data_max * 1.15, -np.log10(sig_line or 5e-8) * 1.1))
    _save(fig, out_path)


def rvat_test_comparison(gene_df, top_n=20, test_cols=None,
                          title="Per-Gene Multi-Test Comparison",
                          out_path="rvat_test_comparison.png"):
    """
    Dot-plot showing p-values from all test types for the top N genes.

    Each row = one gene; each column = one test (burden, SKAT, SKAT-O).
    Dot size = -log10(p), colour = test type.
    Connected by a thin line to show variation across tests for the same gene.
    Directly inspired by the comparison dot plots in SAIGE-GENE+ and STAAR papers.
    """
    plt = _mpl()
    if test_cols is None:
        test_cols = [c for c in gene_df.columns
                     if c.startswith("pval_") and c != "pval_consensus"
                     and c != "pval_heterogeneity"]
    if not test_cols:
        return

    # Select top N genes by consensus p-value
    df = gene_df.copy().dropna(subset=["pval_consensus"])
    df["pval_consensus"] = pd.to_numeric(df["pval_consensus"], errors="coerce")
    top = df.nsmallest(min(top_n, len(df)), "pval_consensus").reset_index(drop=True)

    gene_names = top["snp"].astype(str).tolist()
    n_genes    = len(gene_names)
    n_tests    = len(test_cols)
    test_colors = [PALETTE[i % len(PALETTE)] for i in range(n_tests)]
    test_labels = [tc.replace("pval_","") for tc in test_cols]

    fig, ax = plt.subplots(figsize=(max(8, n_tests * 1.5), max(5, n_genes * 0.5)),
                           dpi=_PLOT_DPI)

    for gi, row in top.iterrows():
        vals = []
        for ti, tc in enumerate(test_cols):
            p = pd.to_numeric(row.get(tc), errors="coerce")
            if pd.notna(p) and 0 < p <= 1:
                lp = -np.log10(p)
                ax.scatter(ti, gi, s=lp * 15, c=test_colors[ti],
                           alpha=0.75, zorder=3, linewidths=0)
                vals.append((ti, lp))
        if len(vals) >= 2:
            xs = [v[0] for v in vals]; ys = [gi] * len(vals)
            ax.plot(xs, ys, color="#aaaaaa", lw=0.6, zorder=1)
        # Consensus dot
        cp = pd.to_numeric(row.get("pval_consensus"), errors="coerce")
        if pd.notna(cp) and cp > 0:
            ax.scatter(n_tests, gi, s=-np.log10(cp) * 20, c="black",
                       marker="D", zorder=5, linewidths=0)

    ax.set_yticks(range(n_genes))
    ax.set_yticklabels(gene_names, fontsize=8)
    ax.set_xticks(list(range(n_tests)) + [n_tests])
    ax.set_xticklabels(test_labels + ["consensus"], fontsize=8, rotation=25, ha="right")
    ax.set_xlabel("Association test"); ax.set_ylabel("Gene (ranked by consensus p)")
    ax.set_title(title, fontweight="bold")
    ax.axvline(n_tests - 0.5, color="#888888", lw=0.7, ls="--")
    ax.invert_yaxis()

    # Size legend
    for lp_ref, label in [(1, "p=0.1"), (3, "p=0.001"), (6, "p=1e-6")]:
        ax.scatter([], [], s=lp_ref * 15, c="#888888", alpha=0.7,
                   label=label, linewidths=0)
    ax.legend(title="-log\u2081\u2080(p) scale", fontsize=7, title_fontsize=7,
              markerscale=1, framealpha=0.7)
    ax.spines[["top","right"]].set_visible(False)
    _save(fig, out_path)


def rvat_lollipop(gene_df, top_n=20, pval_col="pval_consensus",
                   gene_col="snp", consequence_col=None,
                   sig_line=2.5e-6,
                   title="Top Gene Associations (RVAT)",
                   out_path="rvat_lollipop.png"):
    """
    Lollipop (stem-and-circle) plot of the top N gene associations.

    Stick height = -log10(p).  Circle coloured by dominant consequence
    category when a variant-annotation file is provided.
    Horizontal dashed lines at exome-wide and suggestive thresholds.
    Gene labels printed at the top of each lollipop.
    """
    plt = _mpl()
    df = gene_df.copy().dropna(subset=[pval_col])
    df[pval_col] = pd.to_numeric(df[pval_col], errors="coerce")
    df = df[df[pval_col] > 0]
    top = df.nsmallest(min(top_n, len(df)), pval_col).reset_index(drop=True)

    csq_palette = {"missense":"#E41A1C", "nonsense":"#984EA3",
                   "splice_region":"#FF7F00", "synonymous":"#377EB8",
                   "intergenic":"#999999", "intronic":"#A65628",
                   "UTR3":"#4DAF4A", "UTR5":"#4DAF4A", "other":"#888888"}

    heights = (-np.log10(top[pval_col])).values
    cols    = [csq_palette.get(str(top[consequence_col].iloc[i]).lower()
                                if consequence_col and consequence_col in top.columns
                                else "other", "#888888")
               for i in range(len(top))]

    fig, ax = plt.subplots(figsize=(max(10, len(top)*0.65), 6), dpi=_PLOT_DPI)
    x = np.arange(len(top))
    for xi, (h, c) in enumerate(zip(heights, cols)):
        ax.plot([xi, xi], [0, h], color="#bbbbbb", lw=1.5, zorder=1)
        ax.scatter(xi, h, s=80, color=c, zorder=4, linewidths=0.5,
                   edgecolors="white")
        ax.text(xi, h + 0.05, top[gene_col].iloc[xi], ha="center",
                va="bottom", fontsize=6, rotation=60)

    if sig_line and -np.log10(sig_line) <= heights.max() * 1.1:
        ax.axhline(-np.log10(sig_line), color="#D62728", lw=1.0, ls="--",
                   label=f"Exome-wide ({sig_line:.1e})")
    ax.axhline(-np.log10(1e-4), color="#FF7F0E", lw=0.8, ls=":",
               label="Suggestive (1e-4)")

    ax.set_xticks(x); ax.set_xticklabels(top[gene_col].tolist(),
                                           rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("-log\u2081\u2080(consensus p)"); ax.set_title(title, fontweight="bold")
    ax.set_xlim(-0.7, len(top) - 0.3)
    ax.legend(fontsize=8, framealpha=0.7)
    ax.spines[["top","right"]].set_visible(False)

    # Consequence colour legend
    import matplotlib.patches as mpatches
    present_csqs = set(str(c) for c in
                       (top[consequence_col].tolist()
                        if consequence_col and consequence_col in top.columns
                        else []))
    handles = [mpatches.Patch(color=csq_palette.get(c, "#888"), label=c)
               for c in csq_palette if c in present_csqs]
    if handles:
        ax.legend(handles=handles + ax.get_legend_handles_labels()[0],
                  fontsize=7, framealpha=0.7, title="Consequence",
                  title_fontsize=7)
    _save(fig, out_path)


def rvat_regional_gene(snp_df, gene_name, chrom=None,
                        pval_col="pval", maf_col="MAF",
                        consequence_col="CONSEQUENCE",
                        title=None,
                        out_path="rvat_regional.png"):
    """
    Regional association plot zoomed into a single gene.

    Top panel   : -log10(p) per SNP, sized by 1/MAF (rarer = larger),
                  coloured by consequence category.
    Bottom panel: MAF per SNP (bar chart), with rare/common boundary line.

    Suitable as Figure 2/3 in a RVAT manuscript: shows which specific
    rare variants in a gene are driving the association signal.
    """
    plt = _mpl()
    df = snp_df.copy()
    if pval_col in df.columns:
        df[pval_col] = pd.to_numeric(df[pval_col], errors="coerce")
    if maf_col in df.columns:
        df[maf_col] = pd.to_numeric(df[maf_col], errors="coerce")
    if "POS" in df.columns:
        df["POS"] = pd.to_numeric(df["POS"], errors="coerce")
    elif "pos" in df.columns:
        df["POS"] = pd.to_numeric(df["pos"], errors="coerce")
    else:
        df["POS"] = range(len(df))

    title = title or f"Regional association - {gene_name}"
    csq_palette = {"missense":"#E41A1C", "nonsense":"#984EA3",
                   "splice_region":"#FF7F00", "synonymous":"#377EB8",
                   "intergenic":"#999999", "intronic":"#aaaaaa",
                   "UTR3":"#4DAF4A", "UTR5":"#66C2A5",
                   "other":"#888888"}

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12, 7), dpi=_PLOT_DPI,
        gridspec_kw={"height_ratios":[3, 1]}, sharex=True)

    positions = df["POS"].values
    if pval_col in df.columns:
        lp = -np.log10(df[pval_col].clip(1e-300, 1)).values
    else:
        lp = np.zeros(len(df))

    mafs = df[maf_col].values if maf_col in df.columns else np.full(len(df), 0.1)
    sizes = np.clip(1 / (mafs + 0.001) * 5, 10, 400)  # rarer -> larger dot

    csqs = (df[consequence_col].fillna("other").str.lower()
            if consequence_col in df.columns
            else pd.Series(["other"] * len(df)))
    colors = [csq_palette.get(c, "#888") for c in csqs]

    ax_top.scatter(positions, lp, s=sizes, c=colors,
                   alpha=0.85, linewidths=0.4, edgecolors="white", zorder=3)
    ax_top.axhline(-np.log10(0.05 / max(len(df), 1)), color="#D62728",
                   lw=0.9, ls="--", label=f"Bonferroni (n={len(df)})")
    ax_top.set_ylabel("-log\u2081\u2080(p)", fontsize=10)
    ax_top.set_title(title, fontweight="bold", fontsize=12)
    ax_top.spines[["top","right"]].set_visible(False)

    # Annotate the most significant SNP
    if lp.max() > 0:
        bi = np.argmax(lp)
        snp_name = str(df["SNP"].iloc[bi]) if "SNP" in df.columns else f"rs{bi}"
        ax_top.annotate(snp_name, xy=(positions[bi], lp[bi]),
                        xytext=(10, 0), textcoords="offset points",
                        fontsize=7, color="#D62728",
                        arrowprops=dict(arrowstyle="-", lw=0.7, color="grey"))

    # MAF panel
    maf_colors = ["#D62728" if m < 0.05 else "#ABD9E9" for m in mafs]
    ax_bot.bar(positions, mafs, width=50, color=maf_colors, alpha=0.8)
    ax_bot.axhline(0.05, color="#D62728", lw=0.7, ls="--", label="MAF=0.05")
    ax_bot.set_ylabel("MAF", fontsize=9); ax_bot.set_xlabel("Position (bp)", fontsize=10)
    ax_bot.spines[["top","right"]].set_visible(False)

    # Consequence legend
    import matplotlib.patches as mpatches
    present = set(csqs.unique())
    handles = [mpatches.Patch(color=csq_palette.get(c,"#888"), label=c)
               for c in csq_palette if c in present]
    ax_top.legend(handles=handles, fontsize=7, loc="upper right",
                  framealpha=0.7, title="Consequence", title_fontsize=7)

    # Shared size legend
    for maf_ref, lbl in [(0.01,"MAF 1%"),(0.001,"MAF 0.1%")]:
        ax_top.scatter([], [], s=1/(maf_ref+0.001)*5, c="#888",
                       alpha=0.7, label=lbl, linewidths=0)
    ax_top.legend(handles=handles +
                  [plt.scatter([], [], s=1/(m+0.001)*5, c="#888",
                               alpha=0.7, label=lbl, linewidths=0)
                   for m, lbl in [(0.01,"MAF 1%"),(0.001,"MAF 0.1%")]],
                  fontsize=7, loc="upper right",
                  framealpha=0.7, title="Consequence / MAF",
                  title_fontsize=7)
    _save(fig, out_path)
