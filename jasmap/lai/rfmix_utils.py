"""
jasmap.lai.rfmix_utils
----------------------
Utility functions that bridge external LAI tools (SHAPEIT2, RFMix) to JasMAP4.

The complete external LAI workflow is:

  1. Phase genotypes with SHAPEIT2
       shapeit --input-vcf cohort.vcf ... --output-max phased

  2. Convert phased output to RFMix input
       python3 shapeit2rfmix.py \
           --shapeit_hap_ref    ref_AFR.haps,ref_EUR.haps,ref_SAN.haps \
           --shapeit_hap_admixed  admixed.haps \
           --shapeit_sample_ref   ref_AFR.sample,ref_EUR.sample,ref_SAN.sample \
           --shapeit_sample_admixed  admixed.sample \
           --ref_keep     ref_ids.txt \
           --admixed_keep admixed_ids.txt \
           --chr          22 \
           --genetic_map  genetic_map_chr22.txt \
           --out          rfmix_input/

  3. Run RFMix
       rfmix -f rfmix_input/_chr22.alleles \
             -s rfmix_input/_chr22.snp_locations \
             --num-haplotypes-admixed=N_admixed \
             --num-haplotypes-reference=N_ref \
             -o rfmix_output/chr22

  4. Convert RFMix output to JasMAP4 ancfile   ← THIS MODULE
       python3 -m jasmap.lai.rfmix_utils \
           --rfmix  rfmix_output/chr{chrom}.Viterbi.txt \
           --fam    cohort.fam \
           --pop_labels AFR,EUR,SAN \
           --out    jasmap_ancfile.txt

     OR as a Python call:
       from jasmap.lai.rfmix_utils import rfmix_to_ancfile
       rfmix_to_ancfile('rfmix_output/chr{chrom}.Viterbi.txt',
                        'cohort.fam', ['AFR','EUR','SAN'],
                        'jasmap_ancfile.txt')

  5. Run JasMAP4 with the real ancfile
       python3 JasMAP4.py \
           --bfile cohort \
           --pheno pheno.txt \
           --ancfile jasmap_ancfile.txt \
           --nbanc 3 \
           --mode admix \
           --out results/

RFMix output formats supported
-------------------------------
  Viterbi (v1):   {chr}.Viterbi.txt
                  One line per SNP, 2*N_admixed columns of integer ancestry labels (1..K).
                  This is the same format JasMAP4's ancfile expects.

  msp.tsv (v2):   {chr}.msp.tsv
                  Tab-separated with columns: chrom, spos, epos, snp_id,
                  n_snps, IND0.0, IND0.1, IND1.0, IND1.1, ...
                  where .0 and .1 are haplotype 0 and 1 of each individual.

JasMAP4 ancfile format
-----------------------
  One line per SNP.
  Each line contains 2 * N_samples integers separated by spaces or
  concatenated without spaces (compact format).
  Integers are 1-indexed ancestry labels (1, 2, ..., K).
  Example (3 samples, K=3):
      1 1 2 1 3 2
      2 2 1 1 2 3
      ...
"""

import argparse
import re
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# RFMix output parsers
# ─────────────────────────────────────────────────────────────────────────────

def _read_rfmix_viterbi(rfmix_pattern, chroms=None):
    """
    Read RFMix v1 Viterbi output (one file per chromosome).

    Returns
    -------
    np.ndarray  shape (n_snps_total, 2 * n_samples)  dtype int8
    """
    if chroms is None:
        chroms = [str(c) for c in range(1, 23)]
    rows = []
    for c in chroms:
        path = re.sub(r'chr[X0-9]+', f'chr{c}', rfmix_pattern)
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    rows.append([int(x) for x in line.split()])
        except FileNotFoundError:
            warnings.warn(f"[rfmix_utils] chr{c} not found at {path} - skipping")
    if not rows:
        raise ValueError("No RFMix Viterbi data loaded. Check --rfmix path pattern.")
    return np.array(rows, dtype=np.int8)


def _read_rfmix_v2_msp(msp_pattern, chroms=None):
    """
    Read RFMix v2 .msp.tsv output (one file per chromosome or a single file).

    Returns
    -------
    np.ndarray  shape (n_snps, 2 * n_samples)  dtype int8
    list        sample IDs (in the order they appear in the msp file)
    """
    if chroms is None:
        chroms = [str(c) for c in range(1, 23)]

    rows = []; sample_ids = None
    paths_tried = set()

    for c in chroms:
        path = re.sub(r'chr[X0-9]+', f'chr{c}', msp_pattern)
        if path in paths_tried:
            continue
        paths_tried.add(path)
        try:
            df = pd.read_csv(path, sep="\t", comment="##")
        except FileNotFoundError:
            continue

        # Detect header — columns after n_snps are haplotype calls
        hap_cols = [col for col in df.columns
                    if col not in ("#chrom","spos","epos","snp_id","n_snps")]
        if sample_ids is None:
            # Derive sample IDs from haplotype column names: IND.0, IND.1
            sample_ids = sorted(set(c.rsplit(".",1)[0] for c in hap_cols))

        # Expand each window to per-SNP rows (repeat by n_snps)
        for _, row in df.iterrows():
            n = int(row.get("n_snps", 1))
            hap_vals = [int(row[c]) + 1 for c in hap_cols]  # convert 0-based to 1-based
            for _ in range(n):
                rows.append(hap_vals)

    if not rows:
        raise ValueError("No RFMix v2 .msp data loaded. Check --rfmix path pattern.")
    return np.array(rows, dtype=np.int8), sample_ids or []


# ─────────────────────────────────────────────────────────────────────────────
# Core converter
# ─────────────────────────────────────────────────────────────────────────────

def rfmix_to_ancfile(rfmix_pattern, fam_path, pop_labels, out_path,
                     chroms=None, fmt="viterbi"):
    """
    Convert RFMix ancestry output directly to JasMAP4 ancfile format.

    Parameters
    ----------
    rfmix_pattern : str   path with 'chr[X0-9]+' placeholder
                          e.g. 'rfmix_output/chr1.Viterbi.txt'
                          or   'rfmix_output/chr1.msp.tsv'
    fam_path      : str   PLINK .fam file (provides sample order)
    pop_labels    : list  ancestry population names, e.g. ['AFR','EUR','SAN']
    out_path      : str   output ancfile path
    chroms        : list  chromosomes to include (default: 1-22)
    fmt           : str   'viterbi' (RFMix v1) or 'msp' (RFMix v2)

    Output
    ------
    ancfile compatible with JasMAP4 --ancfile:
      - space-delimited
      - one line per SNP
      - 2 * n_samples integers per line
      - values are 1-indexed ancestry labels (1..K)
    """
    logger.info("[rfmix_utils] Converting RFMix output -> JasMAP4 ancfile")
    logger.info(f"  Format  : {fmt}")
    logger.info(f"  Pattern : {rfmix_pattern}")
    logger.info(f"  Pops    : {pop_labels}")

    # Load ancestry calls
    if fmt == "viterbi":
        ancestry = _read_rfmix_viterbi(rfmix_pattern, chroms)
    elif fmt == "msp":
        ancestry, _ = _read_rfmix_v2_msp(rfmix_pattern, chroms)
    else:
        raise ValueError(f"Unknown fmt '{fmt}'. Use 'viterbi' or 'msp'.")

    n_snps, n_haps = ancestry.shape
    n_samples = n_haps // 2
    logger.info(f"  Loaded  : {n_snps} SNPs x {n_samples} samples (x2 haplotypes)")

    # Validate ancestry label range
    min_a, max_a = int(ancestry.min()), int(ancestry.max())
    K = len(pop_labels)
    if max_a > K:
        warnings.warn(
            f"[rfmix_utils] Max ancestry label in file = {max_a} but "
            f"only {K} pop_labels provided ({pop_labels}). "
            f"Labels beyond {K} will appear as integers in the output.")
    if min_a < 1:
        warnings.warn(
            "[rfmix_utils] Ancestry labels include values < 1 "
            f"(found {min_a}). RFMix v2 uses 0-based labelling; "
            "use fmt='msp' which converts automatically.")

    # Write ancfile
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for snp_row in ancestry:
            fh.write(" ".join(str(int(v)) for v in snp_row) + "\n")

    logger.info(f"  Written : {out_path}  ({n_snps} lines)")
    return str(out_path)


def rfmix_to_global_ancestry(rfmix_pattern, fam_path, pop_labels,
                              out_path=None, chroms=None, fmt="viterbi"):
    """
    Compute per-sample global ancestry proportions from RFMix output.

    Returns a DataFrame with one row per sample and K ancestry columns,
    and optionally writes it as a TSV.  Compatible with the global ancestry
    TSV produced by ConsensusLAI.
    """
    if fmt == "viterbi":
        ancestry = _read_rfmix_viterbi(rfmix_pattern, chroms)
    else:
        ancestry, _ = _read_rfmix_v2_msp(rfmix_pattern, chroms)

    n_snps, n_haps = ancestry.shape
    n_samples = n_haps // 2
    K = len(pop_labels)

    # Count proportion of haplotype-sites assigned to each ancestry per sample
    Q = np.zeros((n_samples, K), dtype=np.float32)
    for s in range(n_samples):
        for k in range(K):
            label = k + 1  # 1-indexed
            Q[s, k] = float(np.sum(ancestry[:, 2*s:2*s+2] == label)) / (2 * n_snps)

    # Load sample IDs from FAM
    try:
        fam = pd.read_csv(fam_path, sep=r"\s+", header=None,
                          names=["FID","IID","PAT","MAT","SEX","PHENO"],
                          engine="python")
        sample_ids = fam["IID"].astype(str).tolist()[:n_samples]
    except Exception:
        sample_ids = [f"S{i}" for i in range(n_samples)]

    df = pd.DataFrame(Q, columns=pop_labels)
    df.insert(0, "ID", sample_ids)

    if out_path:
        df.to_csv(out_path, sep="\t", index=False)
        logger.info(f"[rfmix_utils] Global ancestry written -> {out_path}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Command-line interface
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    parser = argparse.ArgumentParser(
        description="Convert RFMix output to JasMAP4 ancfile format.")
    parser.add_argument("--rfmix",      required=True,
        help="RFMix output path with 'chr[X0-9]+' placeholder")
    parser.add_argument("--fam",        required=True,
        help="PLINK .fam file (sample order)")
    parser.add_argument("--pop_labels", required=True,
        help="Comma-separated ancestry labels matching RFMix population order")
    parser.add_argument("--out",        required=True,
        help="Output JasMAP4 ancfile path")
    parser.add_argument("--chroms",     default=None,
        help="Comma-separated chromosomes (default: 1-22)")
    parser.add_argument("--fmt",        default="viterbi",
        choices=["viterbi","msp"],
        help="RFMix output format (default: viterbi)")
    parser.add_argument("--global-ancestry", default=None,
        help="Optional: path to write global ancestry proportions TSV")
    args = parser.parse_args()

    chroms = args.chroms.split(",") if args.chroms else None
    pops   = args.pop_labels.split(",")

    rfmix_to_ancfile(args.rfmix, args.fam, pops, args.out,
                     chroms=chroms, fmt=args.fmt)

    if args.global_ancestry:
        rfmix_to_global_ancestry(args.rfmix, args.fam, pops,
                                 out_path=args.global_ancestry,
                                 chroms=chroms, fmt=args.fmt)
