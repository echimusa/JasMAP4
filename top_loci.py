#!/usr/bin/env python3
"""
top_loci.py — find the strongest admixture-mapping loci and emit
              ready-to-run --lai-region commands
==============================================================================

The two ancestry modes produce DIFFERENT things, and they need reading
differently:

  --mode gift_admix   Window-based admixture mapping. Writes
                      <out>/<prefix>_gift_admix_results/admixture_mapping.tsv
                      with WINDOW, CHR, START_BP, END_BP, ANCESTRY_K, BETA,
                      SE, Z, P -- one row per (window, ancestry). Genomic
                      intervals are explicit, so a locus maps straight onto
                      --lai-region.

  --mode admix        SNP-level per-ancestry tests. Writes one file per
                      ancestry, <prefix>_Jasmap.ANC_<POP>.AM.assoc, with
                      CHR, SNP, POS, beta, Pval -- one row per SNP. There
                      are no windows here, so a "locus" is built by padding
                      a flanking region around the top SNP.

Both are handled below, and each reported locus comes with the exact
--lai-region argument to inspect it.

IMPORTANT on interpretation
---------------------------
Bonferroni is a poor correction for admixture mapping: local ancestry is
strongly autocorrelated along a chromosome, so adjacent windows/SNPs are far
from independent and the effective number of tests is much smaller than the
raw count. This script therefore reports the RANKING and the raw p-values
rather than declaring significance -- it is for choosing where to look, not
for calling a finding. A permutation-based threshold is the right tool for
that.

Usage
-----
    # gift_admix window output
    python3 top_loci.py --gift-admix-dir EMILE2/SAC_gift_admix_results

    # admix per-ancestry SNP output (point at the folder + prefix)
    python3 top_loci.py --admix-dir EMILE --prefix SAC

    # both, showing 10 per source
    python3 top_loci.py --gift-admix-dir EMILE2/SAC_gift_admix_results \\
                        --admix-dir EMILE --prefix SAC --top 10
"""
import argparse
import glob
import os
import sys

import pandas as pd


def _clean_chrom(chrom):
    """Render a chromosome label so --lai-region can parse it.

    pandas reads a numeric CHR column as float, so a chromosome arrives as
    13.0 rather than 13 -- and "13.0:0.0-1.0" fails --lai-region's parser
    (int("13.0") raises). Found by running this against real output. Strips
    a trailing .0 while leaving genuinely non-numeric labels (X, Y, MT)
    untouched."""
    t = str(chrom).strip()
    if t.endswith(".0"):
        t = t[:-2]
    return t.replace("chr", "")


def _fmt_region(chrom, start_bp, end_bp, pad_mb=0.0):
    s = max(0.0, start_bp / 1e6 - pad_mb)
    e = end_bp / 1e6 + pad_mb
    return f"{_clean_chrom(chrom)}:{s:.4f}-{e:.4f}"


def report_gift_admix(d, top, pad_mb):
    path = os.path.join(d, "admixture_mapping.tsv")
    if not os.path.exists(path):
        print(f"  [skip] not found: {path}")
        return
    df = pd.read_csv(path, sep="\t")
    need = {"CHR", "START_BP", "END_BP", "ANCESTRY_K", "P"}
    if not need.issubset(df.columns):
        print(f"  [skip] {path} lacks expected columns "
              f"(missing {sorted(need - set(df.columns))})")
        return
    df["P"] = pd.to_numeric(df["P"], errors="coerce")
    df = df.dropna(subset=["P"])
    n_win = df["WINDOW"].nunique() if "WINDOW" in df.columns else len(df)
    n_anc = df["ANCESTRY_K"].nunique()

    print(f"  source        : {path}")
    print(f"  tests         : {len(df):,} rows = {n_win:,} window(s) x "
          f"{n_anc} ancestry(ies)")
    print(f"  best p-value  : {df['P'].min():.3e}")
    print(f"  reference     : expected minimum under the null with "
          f"{n_win*max(n_anc-1,1):,} effective tests is "
          f"~{1.0/max(n_win*max(n_anc-1,1),1):.2e}")
    print()
    best = df.nsmallest(top, "P")
    print(f"  Top {len(best)} window(s) by p-value:")
    print(f"  {'rank':>4}  {'chr':>4}  {'start_Mb':>10}  {'end_Mb':>10}  "
          f"{'anc_k':>5}  {'P':>11}   --lai-region")
    for i, (_, r) in enumerate(best.iterrows(), 1):
        reg = _fmt_region(r["CHR"], r["START_BP"], r["END_BP"], pad_mb)
        print(f"  {i:>4}  {_clean_chrom(r["CHR"]):>4}  {r['START_BP']/1e6:>10.4f}  "
              f"{r['END_BP']/1e6:>10.4f}  {_clean_chrom(r["ANCESTRY_K"]):>5}  "
              f"{r['P']:>11.3e}   --lai-region {reg}")


def report_admix(d, prefix, top, pad_mb):
    pat = os.path.join(d, f"{prefix}*Jasmap.ANC*.AM.assoc")
    files = sorted(glob.glob(pat))
    if not files:
        print(f"  [skip] no files matched: {pat}")
        return
    print(f"  source        : {len(files)} per-ancestry file(s) in {d}")
    rows = []
    for f in files:
        anc = os.path.basename(f).split("Jasmap.")[-1].replace(".AM.assoc", "")
        try:
            df = pd.read_csv(f, sep="\t")
        except Exception as e:
            print(f"  [skip] {f}: {e}")
            continue
        if not {"CHR", "POS", "Pval"}.issubset(df.columns):
            print(f"  [skip] {f}: unexpected columns {list(df.columns)}")
            continue
        df["Pval"] = pd.to_numeric(df["Pval"], errors="coerce")
        df = df.dropna(subset=["Pval"])
        if not len(df):
            continue
        df["ancestry"] = anc
        rows.append(df)
    if not rows:
        print("  [skip] no usable per-ancestry results")
        return
    allr = pd.concat(rows, ignore_index=True)
    print(f"  tests         : {len(allr):,} SNP x ancestry rows")
    print(f"  best p-value  : {allr['Pval'].min():.3e}")
    print(f"  note          : these are SNP-level, not windows -- each locus "
          f"below is the top SNP with +/-{pad_mb}Mb flanking added so there "
          f"is something to see in a region plot.")
    print()
    best = allr.nsmallest(top, "Pval")
    print(f"  Top {len(best)} SNP(s) by p-value:")
    print(f"  {'rank':>4}  {'chr':>4}  {'pos_Mb':>10}  {'ancestry':>10}  "
          f"{'SNP':>14}  {'Pval':>11}   --lai-region")
    for i, (_, r) in enumerate(best.iterrows(), 1):
        reg = _fmt_region(r["CHR"], r["POS"], r["POS"], max(pad_mb, 0.5))
        snp = str(r["SNP"]) if "SNP" in best.columns else "-"
        print(f"  {i:>4}  {_clean_chrom(r["CHR"]):>4}  {r['POS']/1e6:>10.4f}  "
              f"{r['ancestry']:>10}  {snp:>14}  {r['Pval']:>11.3e}   "
              f"--lai-region {reg}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gift-admix-dir",
                   help="Path to <prefix>_gift_admix_results/ "
                        "(contains admixture_mapping.tsv)")
    p.add_argument("--admix-dir",
                   help="Folder holding <prefix>_Jasmap.ANC_*.AM.assoc files")
    p.add_argument("--prefix", default="",
                   help="--out-prefix used for the admix run (e.g. SAC)")
    p.add_argument("--top", type=int, default=5,
                   help="How many loci to list per source (default: 5)")
    p.add_argument("--pad-mb", type=float, default=1.0,
                   help="Flanking Mb added around each locus in the emitted "
                        "--lai-region (default: 1.0). A few Mb total keeps a "
                        "region plot readable.")
    a = p.parse_args()

    if not a.gift_admix_dir and not a.admix_dir:
        p.error("give --gift-admix-dir and/or --admix-dir")

    if a.gift_admix_dir:
        print("=" * 78)
        print("gift_admix -- WINDOW-based admixture mapping")
        print("=" * 78)
        report_gift_admix(a.gift_admix_dir, a.top, a.pad_mb)
        print()

    if a.admix_dir:
        print("=" * 78)
        print("admix -- SNP-level per-ancestry admixture mapping")
        print("=" * 78)
        report_admix(a.admix_dir, a.prefix, a.top, a.pad_mb)
        print()

    print("Next step: paste one of the --lai-region arguments above into a")
    print("--mode LAI run to see local ancestry across that locus, e.g.")
    print("  python3 JasMAP4.py --bfile ... --mode LAI --yes-anc \\")
    print("      --lai-region <REGION> --out <dir> --out-prefix <prefix>")
    print()
    print("Reminder: the rankings above are for deciding WHERE to look.")
    print("Adjacent windows/SNPs share ancestry tracts and are not")
    print("independent, so a raw p-value here is not a significance claim --")
    print("use a permutation threshold for that.")


if __name__ == "__main__":
    main()
