"""
shapeit2rfmix.py  —  Python 3 port of armartin's original script
-----------------------------------------------------------------
Converts SHAPEIT2 phased haplotype files to RFMix input format:
  {out}_chr{N}.alleles
  {out}_chr{N}.snp_locations
  {out}_chr{N}.map
  {out}.classes
  {out}.sample

This is the UPSTREAM step before running RFMix externally:
  SHAPEIT2  ->  shapeit2rfmix.py  ->  RFMix  ->  rfmix2plink_tped.py / rfmix_to_jasmap_ancfile()

Original author: armartin  (Python 2)
Python 3 port  : JasMAP4 team

Usage (command-line)
--------------------
python3 shapeit2rfmix.py \
    --shapeit_hap_ref      ref1.haps,ref2.haps \
    --shapeit_hap_admixed  admixed.haps \
    --shapeit_sample_ref   ref1.sample,ref2.sample \
    --shapeit_sample_admixed admixed.sample \
    --ref_keep     ref_keep.txt \
    --admixed_keep admixed_keep.txt \
    --chr          22 \
    --genetic_map  genetic_map_chr22.txt \
    --out          output_prefix

Python API
----------
from jasmap.lai.shapeit2rfmix import shapeit2rfmix
shapeit2rfmix(shapeit_hap_ref, shapeit_hap_admixed,
              shapeit_sample_ref, shapeit_sample_admixed,
              ref_keep, admixed_keep, chrom, genetic_map, out)
"""

import argparse
import gzip


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _open(filename):
    """Open plain or gzip-compressed text files."""
    if str(filename).endswith("gz"):
        return gzip.open(filename, "rt", encoding="utf-8")
    return open(filename, encoding="utf-8")


def _fill_genos(hap_path, pos_id):
    """Parse a SHAPEIT .haps file into {position: [allele0, allele1]} dict."""
    genos = {}
    with _open(hap_path) as fh:
        for line in fh:
            parts = line.strip().split()
            pos   = int(parts[2])
            pos_id[pos] = parts[1]
            genos[pos]  = [parts[3], parts[4]]
    return genos, pos_id


def _find_indices(keep_set, sample_path, out_sample_fh):
    """
    Return haplotype column indices for samples in keep_set,
    writing their IDs to out_sample_fh.
    """
    EXPECTED_HEADER = ["ID_1","ID_2","missing","father","mother","sex","plink_pheno"]
    with _open(sample_path) as fh:
        header0 = fh.readline().strip().split()
        fh.readline()  # discard second header line
        if header0 != EXPECTED_HEADER:
            raise RuntimeError(
                f"SHAPEIT sample file has unexpected header: {header0}")
        ind_order = [line.strip().split()[1] for line in fh if line.strip()]

    indices = []
    for ind in keep_set:
        if ind in ind_order:
            idx = ind_order.index(ind)
            indices.extend([5 + 2*idx, 6 + 2*idx])
            out_sample_fh.write(ind + "\n")
    return indices


def _get_haps(hap_path, indices, full_intersection):
    """Extract haplotype columns at `indices` for variants in full_intersection."""
    hap_dict = {}
    with _open(hap_path) as fh:
        for line in fh:
            parts = line.strip().split()
            pos   = int(parts[2])
            if pos in full_intersection:
                hap_dict[pos] = [parts[i] for i in indices]
    return hap_dict


def _write_or_flip(snp_alleles, snp_haps, allele_order, out_fh):
    """
    Write haplotype alleles, flipping coding if reference allele order
    differs between cohorts.
    """
    flip = lambda haps: ["0" if h == "1" else "1" for h in haps]

    if len(allele_order) == 1:
        out_fh.write("".join(snp_haps))
        return

    alleles = list(snp_alleles)
    if "0" in alleles:
        missing_idx = alleles.index("0")
        other_idx   = 1 - missing_idx
        alleles[missing_idx] = allele_order[1 - allele_order.index(alleles[other_idx])]

    if alleles == allele_order:
        out_fh.write("".join(snp_haps))
    else:
        out_fh.write("".join(flip(snp_haps)))


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def shapeit2rfmix(shapeit_hap_ref, shapeit_hap_admixed,
                  shapeit_sample_ref, shapeit_sample_admixed,
                  ref_keep_path, admixed_keep_path,
                  chrom, genetic_map_path, out_prefix):
    """
    Convert SHAPEIT2 output to RFMix input files.

    Parameters
    ----------
    shapeit_hap_ref       : str or list  SHAPEIT .haps file(s) for reference panel
    shapeit_hap_admixed   : str          SHAPEIT .haps file for admixed cohort
    shapeit_sample_ref    : str or list  SHAPEIT .sample file(s) for reference
    shapeit_sample_admixed: str          SHAPEIT .sample file for admixed cohort
    ref_keep_path         : str          File listing reference individual IDs to keep
    admixed_keep_path     : str or None  File listing admixed individual IDs to keep
    chrom                 : str          Chromosome identifier (e.g. "22")
    genetic_map_path      : str          3-column genetic map (pos, rate, cM)
    out_prefix            : str          Output file prefix

    Output files
    ------------
    {out_prefix}_chr{chrom}.alleles
    {out_prefix}_chr{chrom}.snp_locations
    {out_prefix}_chr{chrom}.map
    {out_prefix}.classes
    {out_prefix}.sample
    """
    from datetime import datetime

    def _ts():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if isinstance(shapeit_hap_ref,    str): shapeit_hap_ref    = shapeit_hap_ref.split(",")
    if isinstance(shapeit_sample_ref, str): shapeit_sample_ref = shapeit_sample_ref.split(",")

    # ── Load ref_keep and admixed_keep lists ──────────────────────────────────
    with open(ref_keep_path, encoding="utf-8") as fh:
        ref_set = [l.strip() for l in fh if l.strip()]

    admixed_set = []
    if admixed_keep_path:
        with open(admixed_keep_path, encoding="utf-8") as fh:
            admixed_set = [l.strip() for l in fh if l.strip()]

    # ── Fill genos and pos_id from all hap files ──────────────────────────────
    print(f"[shapeit2rfmix] Filling genotypes from SHAPEIT hap files [{_ts()}]")
    pos_id = {}
    genos  = []
    for hap_file in shapeit_hap_ref:
        g, pos_id = _fill_genos(hap_file, pos_id)
        genos.append(g)
    g_admix, pos_id = _fill_genos(shapeit_hap_admixed, pos_id)
    genos.append(g_admix)

    # ── Find sample indices ───────────────────────────────────────────────────
    out_sample_path = out_prefix + ".sample"
    with open(out_sample_path, "w", encoding="utf-8") as out_sample_fh:
        sample_indices = []
        for sample_file in shapeit_sample_ref:
            sample_indices.append(_find_indices(ref_set, sample_file, out_sample_fh))
        admixed_indices = (_find_indices(admixed_set, shapeit_sample_admixed,
                                         out_sample_fh)
                           if admixed_keep_path else [])

    # ── Compute intersection of variant sites across all cohorts ─────────────
    print(f"[shapeit2rfmix] Finding site intersection [{_ts()}]")
    full_intersection = set(genos[0].keys())
    for g in genos[1:]:
        full_intersection &= set(g.keys())
    intersection_ordered = sorted(full_intersection)

    # ── Load genetic map and interpolate positions ────────────────────────────
    print(f"[shapeit2rfmix] Loading genetic map and interpolating [{_ts()}]")
    pos_gen_map = {}
    with open(genetic_map_path, encoding="utf-8") as fh:
        header = fh.readline().strip().split()
        if len(header) != 3:
            raise RuntimeError("Genetic map must have exactly 3 columns: "
                               "physical_pos, recombination_rate, genetic_pos")
        for line in fh:
            parts = line.strip().split()
            pos_gen_map[int(parts[0])] = float(parts[2])

    union_pos = sorted(full_intersection | set(pos_gen_map.keys()))
    for snp in full_intersection - set(pos_gen_map.keys()):
        pos_gen_map[snp] = None

    # Interpolate missing genetic positions
    for i, pos in enumerate(union_pos):
        if pos_gen_map[pos] is not None:
            continue
        try:
            # Find next known position
            j = i + 1
            while j < len(union_pos) and pos_gen_map[union_pos[j]] is None:
                j += 1
            if j >= len(union_pos):
                raise IndexError
            if i == 0:
                interp = (pos_gen_map[union_pos[j]] * pos) / union_pos[j]
            else:
                prev_p = union_pos[i-1]; next_p = union_pos[j]
                interp = (pos_gen_map[prev_p] +
                          (pos - prev_p) * (pos_gen_map[next_p] - pos_gen_map[prev_p])
                          / (next_p - prev_p))
            pos_gen_map[pos] = interp
        except IndexError:
            # Near end of chromosome — extrapolate from last known
            k = i - 1
            while k >= 0 and pos_gen_map[union_pos[k]] is None:
                k -= 1
            if k >= 0 and union_pos[k] > 0:
                pos_gen_map[pos] = (pos_gen_map[union_pos[k]] * pos) / union_pos[k]
            else:
                pos_gen_map[pos] = 0.0

    # ── Write output files ────────────────────────────────────────────────────
    alleles_path = f"{out_prefix}_chr{chrom}.alleles"
    snps_path    = f"{out_prefix}_chr{chrom}.snp_locations"
    map_path     = f"{out_prefix}_chr{chrom}.map"
    classes_path = f"{out_prefix}.classes"

    print(f"[shapeit2rfmix] Writing snp_locations [{_ts()}]")
    with open(snps_path, "w", encoding="utf-8") as snps_fh, \
         open(map_path,  "w", encoding="utf-8") as map_fh:
        for snp in intersection_ordered:
            snps_fh.write(f"{pos_gen_map[snp]}\n")
            map_fh.write(f"{snp}\t{pos_gen_map[snp]}\t{pos_id[snp]}\n")

    print(f"[shapeit2rfmix] Writing classes [{_ts()}]")
    with open(classes_path, "w", encoding="utf-8") as cls_fh:
        for i, indices in enumerate(sample_indices):
            for _ in indices:
                cls_fh.write(f"{i+1} ")
        if admixed_keep_path is None:
            for _ in admixed_set:
                cls_fh.write("0 0 ")
        else:
            for _ in admixed_indices:
                cls_fh.write("0 ")
        cls_fh.write("\n")

    print(f"[shapeit2rfmix] Getting haplotypes of interest [{_ts()}]")
    all_haps = []
    for i, hap in enumerate(shapeit_hap_ref):
        all_haps.append(_get_haps(hap, sample_indices[i], full_intersection))
    all_haps.append(_get_haps(shapeit_hap_admixed, admixed_indices, full_intersection))

    expected_n = sum(len(x) for x in sample_indices) + len(admixed_indices)
    if intersection_ordered:
        actual_n = sum(len(h.get(intersection_ordered[0], [])) for h in all_haps)
        assert actual_n == expected_n, \
            f"Haplotype count mismatch: expected {expected_n}, got {actual_n}"

    print(f"[shapeit2rfmix] Writing alleles (n={len(intersection_ordered)}) [{_ts()}]")
    with open(alleles_path, "w", encoding="utf-8") as alleles_fh:
        for snp in intersection_ordered:
            allele_pairs = [tuple(g[snp]) for g in genos]
            # Determine allele order from the first non-monomorphic cohort
            if len(set(allele_pairs)) == 1:
                allele_order = list(set(allele_pairs[0]) - {"0"}) or list(allele_pairs[0])
            else:
                mono = ["0" in ap for ap in allele_pairs]
                if all(mono):
                    allele_order = list(allele_pairs[0])
                else:
                    allele_order = list(allele_pairs[mono.index(False)])

            for i, g in enumerate(genos):
                _write_or_flip(list(g[snp]), all_haps[i][snp],
                               allele_order, alleles_fh)
            alleles_fh.write("\n")

    print(f"[shapeit2rfmix] Done! [{_ts()}]")


def main(args):
    shapeit2rfmix(
        shapeit_hap_ref        = args.shapeit_hap_ref,
        shapeit_hap_admixed    = args.shapeit_hap_admixed,
        shapeit_sample_ref     = args.shapeit_sample_ref,
        shapeit_sample_admixed = args.shapeit_sample_admixed,
        ref_keep_path          = args.ref_keep,
        admixed_keep_path      = args.admixed_keep,
        chrom                  = args.chr,
        genetic_map_path       = args.genetic_map,
        out_prefix             = args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert SHAPEIT2 phased output to RFMix input files.")
    parser.add_argument("--shapeit_hap_ref",
        help="SHAPEIT .haps file(s) for reference panel (comma-separated if multiple)",
        required=True)
    parser.add_argument("--shapeit_hap_admixed",
        help="SHAPEIT .haps file for admixed cohort", required=True)
    parser.add_argument("--shapeit_sample_ref",
        help="SHAPEIT .sample file(s) for reference (comma-separated)", required=True)
    parser.add_argument("--shapeit_sample_admixed",
        help="SHAPEIT .sample file for admixed cohort", required=True)
    parser.add_argument("--ref_keep",
        help="File listing reference individual IDs to keep", required=True)
    parser.add_argument("--admixed_keep",
        help="File listing admixed individual IDs to keep (optional)", default=None)
    parser.add_argument("--chr",
        help="Chromosome number (e.g. 22)", required=True)
    parser.add_argument("--genetic_map",
        help="3-column genetic map: physical_pos, recombination_rate, genetic_pos",
        required=True)
    parser.add_argument("--out", help="Output file prefix", required=True)
    main(parser.parse_args())
