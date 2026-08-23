"""
rfmix2plink_tped.py  —  Python 3 port of armartin's original script
--------------------------------------------------------------------
Converts RFMix local-ancestry output to per-population PLINK tped files
for use in admixture mapping.

Original author: armartin  (Python 2)
Python 3 port  : JasMAP4 team

Usage (command-line)
--------------------
python3 rfmix2plink_tped.py \
    --rfmix   path/to/chr{1-22}.rfmix.txt \
    --fam     cohort.fam \
    --snp_map path/to/chr{1-22}.snp_map.txt \
    --pop_labels AFR,EUR,SAN \
    --out     output_prefix

Python API
----------
from jasmap.lai.rfmix2plink_tped import convert_rfmix_to_tped
convert_rfmix_to_tped(rfmix_pattern, fam_path, snp_map_pattern,
                      pop_labels, out_prefix)
"""

import argparse
import re


def convert_rfmix_to_tped(rfmix_pattern, fam_path, snp_map_pattern,
                           pop_labels, out_prefix):
    """
    Convert RFMix per-chromosome ancestry output to per-population PLINK tped.

    Parameters
    ----------
    rfmix_pattern  : str  path with 'chr[X0-9]+' placeholder, e.g.
                          'results/chr1.rfmix.txt'
    fam_path       : str  PLINK .fam file (sample order)
    snp_map_pattern: str  SNP map path with same 'chr[X0-9]+' placeholder
    pop_labels     : list of str  ancestry population names, e.g. ['AFR','EUR']
    out_prefix     : str  output file prefix

    Output
    ------
    One {out_prefix}_{POP}.tped + {out_prefix}_{POP}.tfam per population.
    Genotype coding: G G = 0 copies (absent), G A = 1 copy, A A = 2 copies.
    """
    chrs = [str(i) for i in range(1, 23)]

    with open(fam_path, encoding="utf-8") as fh:
        ind_order = [line.strip() for line in fh if line.strip()]

    # Open output files
    out_tfam  = [open(f"{out_prefix}_{pop}.tfam",  "w", encoding="utf-8") for pop in pop_labels]
    out_tped  = [open(f"{out_prefix}_{pop}.tped",  "w", encoding="utf-8") for pop in pop_labels]

    # Write tfam (same content for every population)
    for ind in ind_order:
        for fh in out_tfam:
            fh.write(ind + "\n")
    for fh in out_tfam:
        fh.close()

    # Process each chromosome
    for chrom in chrs:
        rfmix_path   = re.sub(r'chr[X0-9]+', f'chr{chrom}', rfmix_pattern)
        snp_map_path = re.sub(r'chr[X0-9]+', f'chr{chrom}', snp_map_pattern)
        try:
            rfmix_fh   = open(rfmix_path,   encoding="utf-8")
            snp_map_fh = open(snp_map_path, encoding="utf-8")
        except FileNotFoundError as e:
            import warnings
            warnings.warn(f"[rfmix2plink_tped] Skipping chr{chrom}: {e}")
            continue

        for snp_line in snp_map_fh:
            rf_line  = rfmix_fh.readline().strip().split()
            snp_cols = snp_line.strip().split()
            prefix   = f"{chrom} {snp_cols[2]} {snp_cols[1]} {snp_cols[0]} "
            for fh in out_tped:
                fh.write(prefix)

            n_inds = len(rf_line) // 2
            for j in range(n_inds):
                anc_pair = [rf_line[2*j], rf_line[2*j+1]]
                for pi, pop in enumerate(pop_labels):
                    count = anc_pair.count(str(pi + 1))
                    geno  = ["G G", "G A", "A A"][count]
                    out_tped[pi].write(geno + " ")

            for fh in out_tped:
                fh.write("\n")

        rfmix_fh.close()
        snp_map_fh.close()

    for fh in out_tped:
        fh.close()


def main(args):
    convert_rfmix_to_tped(
        rfmix_pattern   = args.rfmix,
        fam_path        = args.fam,
        snp_map_pattern = args.snp_map,
        pop_labels      = args.pop_labels.split(","),
        out_prefix      = args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert RFMix output to per-population PLINK tped files.")
    parser.add_argument("--rfmix",      required=True,
                        help="RFMix output, path containing 'chr[X0-9]+' placeholder")
    parser.add_argument("--fam",        required=True,
                        help="PLINK .fam file (sample IDs in genotype order)")
    parser.add_argument("--pop_labels", required=True,
                        help="Comma-separated ancestry population labels")
    parser.add_argument("--snp_map",    required=True,
                        help="SNP map file, path containing 'chr[X0-9]+' placeholder")
    parser.add_argument("--out",        required=True,
                        help="Output prefix")
    main(parser.parse_args())
