"""
make_test_data.py  —  JasMAP4 synthetic test dataset generator
===============================================================
Generates a self-contained test data folder covering every input path
JasMAP4.py supports, including the files required for --yes-anc (automatic
local ancestry computation).

Files generated
---------------
Genotype inputs (same underlying dosage matrix, three formats):
  geno_test / geno_test.snp / geno_test.ind  -- EIGENSTRAT (--geno)
  plink_test.bed / .bim / .fam               -- PLINK binary (--bfile)
  vcf_test.vcf                               -- VCF (--vcf)

Phenotype / covariates:
  pheno.txt     FID IID PHENO  (1/2 case-control coding)
  covar.txt     FID IID PC1 PC2 ...

For --mode admix / --mode gift_admix (pre-computed local ancestry):
  ancestry_space.txt    per-SNP haplotype labels, space-delimited
  ancestry_compact.txt  same data, compact single-digit format

For --yes-anc (automatic local ancestry computation):
  pop_labels.txt        sample_ID  population_label (no header)
                        REQUIRED by --pop-file when using --yes-anc.
                        Contains >= n_ancestries distinct labels so
                        JasMAP4.py's validation always passes.
                        Admixed samples are labelled "ADMIX";
                        reference samples are labelled "Anc1", "Anc2", ...

A human-readable README.txt is written to the output folder listing every
file and the exact JasMAP4.py command that uses it.

Usage
-----
  python3 make_test_data.py --outdir test_data --n-samples 60 \
      --n-snps 300 --n-ancestries 3

  # Small smoke-test dataset (fast):
  python3 make_test_data.py --outdir test_small --n-samples 40 \
      --n-snps 100 --n-ancestries 3 --seed 7

  # Benchmark dataset (for timing comparisons):
  python3 make_test_data.py --outdir test_large --n-samples 300 \
      --n-snps 5000 --n-ancestries 3 --seed 42
"""

import argparse
import os

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Phenotype and covariates
# ─────────────────────────────────────────────────────────────────────────────

def write_pheno(path, iids, seed, pop_labels=None):
    """1/2 case-control coding (JasMAP subtracts 1 internally -> 0/1).

    Bug fix: previously assigned a random, valid 1/2 phenotype to EVERY
    sample regardless of population -- but the real --yes-anc workflow
    expects reference/ancestral population samples (used only to inform
    local ancestry inference) to have NO phenotype at all, with valid
    phenotype restricted to the ADMIX-labelled samples being actually
    studied. Found via a real user report: --pop-file already correctly
    distinguishes "ADMIX" from reference population labels, but every
    generated test dataset gave reference samples a valid phenotype too,
    meaning no test run ever exercised the actual expected data shape.
    When pop_labels is given, any sample NOT labelled "ADMIX" gets -9
    (PLINK's missing-phenotype sentinel) instead of a random 1/2 value.
    """
    rng  = np.random.default_rng(seed)
    fids = [f"FAM{i}" for i in range(len(iids))]
    pheno = rng.integers(1, 3, size=len(iids))
    if pop_labels is not None:
        pheno = np.where(np.array(pop_labels) == "ADMIX", pheno, -9)
    with open(path, "w", encoding="utf-8") as fh:
        for fid, iid, p in zip(fids, iids, pheno):
            fh.write(f"{fid}\t{iid}\t{p}\n")
    return iids, pheno


def write_covar(path, iids, n_covar, seed):
    """FID IID PC1 PC2 ... (no header, to match JasMAP4's covar reader)."""
    rng  = np.random.default_rng(seed)
    covs = rng.standard_normal((len(iids), n_covar))
    with open(path, "w", encoding="utf-8") as fh:
        for i, iid in enumerate(iids):
            row = "\t".join(f"{x:.6f}" for x in covs[i])
            fh.write(f"FAM{i}\t{iid}\t{row}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Population labels  (for --yes-anc --pop-file)
# ─────────────────────────────────────────────────────────────────────────────

def write_pop_labels(path, iids, n_ancestries, admix_frac, seed,
                     ref_pop_names=None):
    """
    Two-column file (sample_ID  population_label, no header) required by
    JasMAP4.py's --pop-file option when using --yes-anc.

    Labels
    ------
    ADMIX        : the admixed target population -- these are the individuals
                   whose local ancestry will be inferred. They are NOT one of
                   the K ancestral sources; they are the end-product of
                   historical admixture between those K sources.
    AFR / EUR / SAN (or more generally the names in ref_pop_names)
                 : the K reference populations representing the pure ancestral
                   sources. In a real study, these would be unrelated
                   individuals from each ancestral group (e.g. YRI for AFR,
                   CEU for EUR, etc.).

    Design
    ------
    JasMAP4.py validates that the number of distinct population labels in this
    file is >= --nbanc.  This generator assigns:
      - admix_frac  of samples to label 'ADMIX'
      - the remainder evenly across the K reference populations

    That means the file always contains exactly K + 1 distinct labels
    (K reference + 'ADMIX'), which is always > K = --nbanc.

    Parameters
    ----------
    ref_pop_names : optional list of K reference population names to use
                    instead of the built-in default (AFR, EUR, SAN, ...).

    Returns
    -------
    pop_labels : list of str  (one per sample, same order as iids)
    ref_pops   : list of str  (the K reference population names)
    """
    # Default reference-population names: AFR, EUR, SAN, ASIAN, AMR, ...
    # These are short but biologically interpretable names that make it
    # immediately clear which group is admixed (ADMIX) and which are reference.
    DEFAULT_REF_POPS = ["AFR","EUR","SAN","ASIAN","AMR","EAS","SAS","MID"]
    rng      = np.random.default_rng(seed)
    n        = len(iids)
    if ref_pop_names:
        ref_pops = list(ref_pop_names)[:n_ancestries]
    else:
        ref_pops = DEFAULT_REF_POPS[:n_ancestries]

    n_admix          = max(1, int(n * admix_frac))
    n_ref            = n - n_admix
    ref_labels_pool  = [ref_pops[i % n_ancestries] for i in range(n_ref)]
    rng.shuffle(ref_labels_pool)
    labels = ["ADMIX"] * n_admix + ref_labels_pool

    perm   = rng.permutation(n)
    labels = [labels[perm[i]] for i in range(n)]

    with open(path, "w", encoding="utf-8") as fh:
        for iid, lbl in zip(iids, labels):
            fh.write(f"{iid}\t{lbl}\n")

    return labels, ref_pops


# ─────────────────────────────────────────────────────────────────────────────
# Pre-computed local ancestry files  (for --ancfile in admix mode)
# ─────────────────────────────────────────────────────────────────────────────

def write_ancestry_file(path, n_snps, n_samples, n_ancestries,
                        space_delimited, seed):
    """
    Per-SNP haplotype ancestry label file.  Each line encodes n_samples
    haplotype pairs as integer ancestry labels (1 .. K).

    This file is used as the input for --ancfile when --yes-anc is NOT set.
    When --yes-anc IS set, JasMAP4 computes and writes this file itself.
    """
    rng = np.random.default_rng(seed)
    with open(path, "w", encoding="utf-8") as fh:
        for _ in range(n_snps):
            labels = rng.integers(1, n_ancestries + 1, size=2 * n_samples)
            if space_delimited:
                fh.write(" ".join(str(x) for x in labels) + "\n")
            else:
                fh.write("".join(str(x) for x in labels) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Genotype files
# ─────────────────────────────────────────────────────────────────────────────

def make_geno_dosage(n_snps, n_samples, seed):
    """(n_snps, n_samples) uint8 dosage matrix with values in {0,1,2}."""
    return np.random.default_rng(seed).integers(0, 3, size=(n_snps, n_samples))


def write_eigenstrat(outdir, prefix, dosage, iids, seed):
    """EIGENSTRAT .geno (no extension) / .snp / .ind trio."""
    n_snps, n_samples = dosage.shape
    rng = np.random.default_rng(seed)

    snp_path  = os.path.join(outdir, prefix + ".snp")
    ind_path  = os.path.join(outdir, prefix + ".ind")
    geno_path = os.path.join(outdir, prefix)  # no extension — JasMAP convention

    with open(snp_path, "w", encoding="utf-8") as fh:
        for j in range(n_snps):
            chrom = 1 + (j % 22)
            pos   = 1000 + j * 100
            ref, alt = rng.choice(["A","C","G","T"], size=2, replace=False)
            fh.write(f"rs{j+1}\t{chrom}\t0.0\t{pos}\t{ref}\t{alt}\n")

    with open(ind_path, "w", encoding="utf-8") as fh:
        for iid in iids:
            sex = rng.choice(["M","F"])
            fh.write(f"{iid}\t{sex}\tPOP1\n")

    with open(geno_path, "w", encoding="utf-8") as fh:
        for j in range(n_snps):
            fh.write("".join(str(int(x)) for x in dosage[j]) + "\n")


def write_plink_bed(outdir, prefix, dosage, iids, pheno=None):
    """
    Hand-rolled PLINK .bed/.bim/.fam writer.

    Encoding (SNP-major, little-endian):
        0b00 = hom A1 (dosage 0)
        0b10 = het   (dosage 1)
        0b11 = hom A2 (dosage 2)
        0b01 = missing

    Parameters
    ----------
    pheno : optional array-like of per-sample phenotype values (1/2 coding,
           matching pheno.txt) to write into the .fam file's 6th column.
           Defaults to -9 (PLINK's standard "missing phenotype" sentinel)
           when not provided.

           Writing the SAME phenotype values already used for pheno.txt
           into the .fam file means JasMAP4.py's --bfile "no --pheno given
           -> fall back to the .fam file's PHENOTYPE column" behaviour is
           actually demonstrable with this generator's own output, rather
           than every sample showing as missing (-9) and leaving nothing
           for the LMM to fit on.
    """
    n_snps, n_samples = dosage.shape
    code_map = {0: 0b00, 1: 0b10, 2: 0b11}

    with open(os.path.join(outdir, prefix + ".fam"), "w", encoding="utf-8") as fh:
        for i, iid in enumerate(iids):
            p = int(pheno[i]) if pheno is not None else -9
            fh.write(f"FAM{i} {iid} 0 0 0 {p}\n")

    with open(os.path.join(outdir, prefix + ".bim"), "w", encoding="utf-8") as fh:
        for j in range(n_snps):
            chrom = 1 + (j % 22)
            pos   = 1000 + j * 100
            fh.write(f"{chrom}\trs{j+1}\t0\t{pos}\tA\tG\n")

    n_bytes = (n_samples + 3) // 4
    with open(os.path.join(outdir, prefix + ".bed"), "wb") as fh:
        fh.write(bytes([0x6c, 0x1b, 0x01]))          # magic + SNP-major
        for j in range(n_snps):
            buf = bytearray(n_bytes)
            for i, val in enumerate(dosage[j]):
                code    = code_map.get(int(val), 0b01)
                buf[i // 4] |= code << ((i % 4) * 2)
            fh.write(bytes(buf))


def write_vcf(outdir, prefix, dosage, iids):
    """Minimal VCF (FORMAT=GT only)."""
    n_snps, n_samples = dosage.shape
    gt_map = {0: "0/0", 1: "0/1", 2: "1/1"}
    with open(os.path.join(outdir, prefix + ".vcf"), "w", encoding="utf-8") as fh:
        fh.write("##fileformat=VCFv4.2\n")
        fh.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        fh.write("\t".join(
            ["#CHROM","POS","ID","REF","ALT","QUAL","FILTER","INFO","FORMAT"]
            + list(iids)) + "\n")
        for j in range(n_snps):
            chrom = str(1 + (j % 22))
            pos   = str(1000 + j * 100)
            gts   = "\t".join(gt_map[int(v)] for v in dosage[j])
            fh.write(f"{chrom}\t{pos}\trs{j+1}\tA\tG\t.\t.\t.\tGT\t{gts}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_bed_roundtrip(outdir, prefix, expected_dosage):
    """
    Read back the .bed file with bed-reader (count_A1=False) and confirm
    the dosage matrix is exactly what was written.

    Returns
    -------
    True   : verified correct
    False  : verified WRONG (a real bug in the .bed writer -- should abort)
    None   : could not verify (bed-reader not installed) -- the .bed/.bim/.fam
             files were still written successfully by the hand-rolled writer
             above; this is only a missing OPTIONAL sanity check, not a
             failure of test-data generation itself.

    Bug fix: this function previously let ModuleNotFoundError propagate
    uncaught out of main(), which aborted the ENTIRE script -- including the
    gene annotation, variant annotation, and README.txt writes that happen
    AFTER this call -- over a single missing optional dependency used only
    for this one verification step. A user who forgot
    `pip install -r requirements.txt` (or just `pip install bed-reader`) lost
    every other file the script was about to generate, with a bare traceback
    and no indication that the fix was a one-line pip install.
    """
    try:
        from bed_reader import open_bed
    except ImportError:
        print("  [WARNING] bed-reader not installed -- skipping .bed "
              "round-trip check.")
        print("             The .bed/.bim/.fam files were still written "
              "successfully.")
        print("             To enable this check: pip install bed-reader")
        return None

    bed = open_bed(os.path.join(outdir, prefix + ".bed"), count_A1=False)
    G   = bed.read()          # (n_samples, n_snps)
    ok  = np.array_equal(
        np.nan_to_num(G.T, nan=9).astype(int),
        expected_dosage)
    return ok


# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Gene annotation files for RVAT testing
# ─────────────────────────────────────────────────────────────────────────────

_CONSEQUENCES = ["synonymous","missense","nonsense","splice_region",
                 "UTR3","UTR5","intergenic","intronic"]
_RARE_CONSEQUENCES = ["missense","nonsense","splice_region"]


def _snp_positions(n_snps):
    chroms = [1 + (j % 22) for j in range(n_snps)]
    bps    = [1000 + j * 100 for j in range(n_snps)]
    return chroms, bps


def write_gene_annotation_bed(path, n_snps, seed):
    """BED6 gene annotation: CHR START STOP GENE_NAME SCORE STRAND.
    Each gene covers a window of 3-7 consecutive SNPs on the same chromosome.
    Returns {gene_name: (chrom, start, stop)} for cross-referencing."""
    from collections import defaultdict
    rng    = np.random.default_rng(seed)
    chroms, bps = _snp_positions(n_snps)
    chr_snps = defaultdict(list)
    for j, (c, b) in enumerate(zip(chroms, bps)):
        chr_snps[c].append((j, b))
    genes = []; gene_id = 1; padding = 200
    for c in sorted(chr_snps.keys()):
        snps_on_chr = chr_snps[c]
        i = 0
        while i < len(snps_on_chr):
            ws     = int(rng.integers(3, 8))
            window = snps_on_chr[i:i+ws]
            start  = max(0, window[0][1]  - padding)
            stop   = window[-1][1] + padding
            strand = rng.choice(["+", "-"])
            gname  = f"GENE{gene_id:04d}"
            genes.append((c, start, stop, gname, 0, strand))
            gene_id += 1; i += ws
    with open(path, "w", encoding="utf-8") as fh:
        for c, s, e, name, sc, st in genes:
            fh.write(f"chr{c}\t{s}\t{e}\t{name}\t{sc}\t{st}\n")
    return {name: (c, s, e) for c, s, e, name, sc, st in genes}


def write_gene_annotation_gff3(path, gene_dict):
    """GFF3 gene annotation (gene + mRNA + 2 exons per gene)."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("##gff-version 3\n")
        for gname, (chrom, start, stop) in gene_dict.items():
            mid = (start + stop) // 2
            fh.write(f"chr{chrom}\tJasMAP4\tgene\t{start+1}\t{stop}\t.\t+\t.\tID={gname};Name={gname}\n")
            fh.write(f"chr{chrom}\tJasMAP4\tmRNA\t{start+1}\t{stop}\t.\t+\t.\tID={gname}.1;Parent={gname}\n")
            fh.write(f"chr{chrom}\tJasMAP4\texon\t{start+1}\t{mid}\t.\t+\t.\tParent={gname}.1\n")
            fh.write(f"chr{chrom}\tJasMAP4\texon\t{mid+1}\t{stop}\t.\t+\t.\tParent={gname}.1\n")


def write_variant_annotation(path, n_snps, gene_dict, seed):
    """Per-SNP annotation TSV for RVAT.
    Columns: SNP CHR POS GENE MAF CONSEQUENCE IS_RARE
    MAF is log-uniform so most variants are rare (MAF < 0.05)."""
    rng = np.random.default_rng(seed)
    chroms, bps = _snp_positions(n_snps)
    pos_to_gene = {}
    for gname, (gc, gs, ge) in gene_dict.items():
        for j, (c, b) in enumerate(zip(chroms, bps)):
            if c == gc and gs <= b <= ge:
                pos_to_gene[j] = gname
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("SNP\tCHR\tPOS\tGENE\tMAF\tCONSEQUENCE\tIS_RARE\n")
        for j in range(n_snps):
            maf     = float(np.clip(np.exp(rng.uniform(-6, -0.3)), 0, 0.5))
            is_rare = maf < 0.05
            csq = str(rng.choice(_RARE_CONSEQUENCES, p=[0.5,0.3,0.2])) if is_rare else str(rng.choice(_CONSEQUENCES))
            gene = pos_to_gene.get(j, "intergenic")
            fh.write(f"rs{j+1}\t{chroms[j]}\t{bps[j]}\t{gene}\t{maf:.6f}\t{csq}\t{'yes' if is_rare else 'no'}\n")


# README for the test data folder
# ─────────────────────────────────────────────────────────────────────────────

def write_readme(outdir, args, iids, pop_labels, ref_pops, gene_dict=None):
    n  = args.n_samples
    K  = args.n_ancestries
    s  = ("_" + args.label) if args.label else ""
    n_ref   = sum(1 for l in pop_labels if l != "ADMIX")
    n_admix = sum(1 for l in pop_labels if l == "ADMIX")

    lines = [
        "JasMAP4 synthetic test dataset",
        "=" * 60,
        f"n_samples   = {n}",
        f"n_snps      = {args.n_snps}",
        f"n_ancestries= {K}",
        f"seed        = {args.seed}",
        "",
        "Files and their purpose",
        "-" * 60,
        "",
        "Genotype inputs (same dosage matrix, three formats)",
        f"  geno_test{s}            --geno geno_test{s}",
        f"  geno_test{s}.snp        (SNP metadata for --geno)",
        f"  geno_test{s}.ind        (sample IDs for --geno)",
        f"  plink_test{s}.bed/.bim/.fam  --bfile plink_test{s}",
        f"  vcf_test{s}.vcf         --vcf vcf_test{s}.vcf",
        "",
        "Phenotype / covariates",
        f"  pheno{s}.txt    --pheno pheno{s}.txt",
        "                  FID IID PHENO (1=control, 2=case)",
        f"  covar{s}.txt    --covfile covar{s}.txt",
        "                  FID IID PC1 PC2",
        "",
        "Pre-computed local ancestry (for --ancfile without --yes-anc)",
        f"  ancestry_space{s}.txt    --ancfile ancestry_space{s}.txt",
        f"  ancestry_compact{s}.txt  --ancfile ancestry_compact{s}.txt",
        "  Format: one line per SNP, 2*n_samples integers (haplotype labels)",
        f"  Values: 1..{K}  (matching --nbanc {K})",
        "",
        "Population labels (REQUIRED for --yes-anc --pop-file)",
        f"  pop_labels{s}.txt  --pop-file pop_labels{s}.txt",
        "  Two columns, no header:  sample_ID  population_label",
        f"  {n_admix} admixed samples  -> labelled 'ADMIX'",
        f"  {n_ref} reference samples -> labelled "
        + ", ".join(f"'{p}'" for p in ref_pops),
        f"  Distinct reference populations = {K} = --nbanc -> validation passes",
        "",
        "Gene / variant annotation for RVAT (--consgwas rare-variant tests)",
        f"  gene_annotation{s}.bed   BED6: CHR START STOP GENE_NAME SCORE STRAND",
        f"  gene_annotation{s}.gff3  GFF3: gene + mRNA + exon features",
        f"  variant_annotation{s}.tsv  SNP CHR POS GENE MAF CONSEQUENCE IS_RARE",
        "  Pass these to --consgwas via --gene-annot and --var-annot (future flag)",
        f"  n_genes = {len(gene_dict) if gene_dict else 'N/A'}  "
        "  most SNPs have MAF < 5% (rare) for gene-based tests",
        "",
        "Example commands",
        "-" * 60,
        "",
        "# admix mode (pre-computed ancestry)",
        f"python3 JasMAP4.py --bfile plink_test{s} --pheno pheno{s}.txt "
        f"--covfile covar{s}.txt \\",
        f"  --ancfile ancestry_space{s}.txt --nbanc {K} --mode admix "
        f"--ntest {args.n_snps} --out out_admix/",
        "",
        "# GWAS mode",
        f"python3 JasMAP4.py --bfile plink_test{s} --pheno pheno{s}.txt "
        f"--covfile covar{s}.txt \\",
        "  --mode GWAS --out out_gwas/",
        "",
        "# GWAS + consensus GWAS",
        f"python3 JasMAP4.py --bfile plink_test{s} --pheno pheno{s}.txt "
        f"--covfile covar{s}.txt \\",
        "  --mode GWAS --consgwas --out out_gwas_cons/",
        "",
        "# --yes-anc: compute local ancestry automatically, then run admix",
        f"python3 JasMAP4.py --bfile plink_test{s} --pheno pheno{s}.txt "
        f"--covfile covar{s}.txt \\",
        f"  --nbanc {K} --yes-anc --pop-file pop_labels{s}.txt \\",
        f"  --pop-labels {','.join(ref_pops)} \\",
        f"  --mode admix --ntest {args.n_snps} --out out_yes_anc/",
        "",
        "# GIFT mode",
        f"python3 JasMAP4.py --bfile plink_test{s} --pheno pheno{s}.txt "
        f"--covfile covar{s}.txt \\",
        "  --mode gift --out out_gift/",
        "",
        "# GIFT + admixture mapping",
        f"python3 JasMAP4.py --bfile plink_test{s} --pheno pheno{s}.txt "
        f"--covfile covar{s}.txt \\",
        f"  --ancfile ancestry_space{s}.txt --nbanc {K} --mode gift_admix \\",
        f"  --pop-labels {','.join(ref_pops)} --out out_gift_admix/",
        "",
        "# Any mode + PRS",
        f"python3 JasMAP4.py --bfile plink_test{s} --pheno pheno{s}.txt "
        f"--covfile covar{s}.txt \\",
        "  --mode GWAS --PRS Yes --out out_prs/",
    ]

    with open(os.path.join(outdir, "README.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate JasMAP4 synthetic test datasets for all pipeline paths.")
    ap.add_argument("--outdir",         default="test_data")
    ap.add_argument("--n-samples",      type=int,   default=60)
    ap.add_argument("--n-snps",         type=int,   default=300)
    ap.add_argument("--n-ancestries",   type=int,   default=3)
    ap.add_argument("--n-covar",        type=int,   default=2)
    ap.add_argument("--admix-frac",     type=float, default=0.4,
                    help="Fraction of samples labelled as ADMIX in pop_labels.txt "
                         "(default 0.4 -> 40%% admixed, 60%% reference)")
    ap.add_argument("--ref-pops",       type=str,   default=None,
                    help="Comma-separated reference population names for pop_labels.txt "
                         "(default: AFR,EUR,SAN,... - first K from a built-in list). "
                         "Must provide exactly --n-ancestries names. "
                         "Example: --ref-pops YRI,CEU,SAN")
    ap.add_argument("--seed",           type=int,   default=1)
    ap.add_argument("--label",          default="",
                    help="Optional suffix appended to every filename.")
    args = ap.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    s = ("_" + args.label) if args.label else ""

    # ── Sample IDs + population labels for --yes-anc ─────────────────────────
    # Computed BEFORE phenotype now (see write_pheno()'s docstring for why --
    # phenotype masking needs to know which samples are ADMIX vs reference).
    iids = [f"IND{i}" for i in range(args.n_samples)]
    ref_pop_names = args.ref_pops.split(",") if args.ref_pops else None
    pop_labels, ref_pops = write_pop_labels(
        os.path.join(outdir, f"pop_labels{s}.txt"),
        iids, args.n_ancestries, args.admix_frac, args.seed + 5,
        ref_pop_names=ref_pop_names)

    # ── Phenotype + covariates ──────────────────────────────────────────────
    iids, pheno = write_pheno(
        os.path.join(outdir, f"pheno{s}.txt"), iids, args.seed,
        pop_labels=pop_labels)
    write_covar(
        os.path.join(outdir, f"covar{s}.txt"), iids, args.n_covar, args.seed + 1)

    # ── Pre-computed ancestry files for --ancfile ───────────────────────────
    write_ancestry_file(
        os.path.join(outdir, f"ancestry_space{s}.txt"),
        args.n_snps, args.n_samples, args.n_ancestries,
        space_delimited=True,  seed=args.seed + 2)
    write_ancestry_file(
        os.path.join(outdir, f"ancestry_compact{s}.txt"),
        args.n_snps, args.n_samples, args.n_ancestries,
        space_delimited=False, seed=args.seed + 2)

    # ── Genotype matrices ───────────────────────────────────────────────────
    dosage = make_geno_dosage(args.n_snps, args.n_samples, args.seed + 3)
    write_eigenstrat(outdir, f"geno_test{s}", dosage, iids, args.seed + 4)
    write_plink_bed( outdir, f"plink_test{s}", dosage, iids, pheno=pheno)
    write_vcf(       outdir, f"vcf_test{s}",   dosage, iids)
    bed_ok = validate_bed_roundtrip(outdir, f"plink_test{s}", dosage)

    # ── Gene and variant annotation for RVAT testing ─────────────────────────
    gene_dict = write_gene_annotation_bed(
        os.path.join(outdir, f"gene_annotation{s}.bed"),
        args.n_snps, args.seed + 6)
    write_gene_annotation_gff3(
        os.path.join(outdir, f"gene_annotation{s}.gff3"),
        gene_dict)
    write_variant_annotation(
        os.path.join(outdir, f"variant_annotation{s}.tsv"),
        args.n_snps, gene_dict, args.seed + 7)

    write_readme(outdir, args, iids, pop_labels, ref_pops, gene_dict)

    # ── Summary ─────────────────────────────────────────────────────────────
    n_admix = sum(1 for l in pop_labels if l == "ADMIX")
    pop_counts = {p: sum(1 for l in pop_labels if l == p) for p in ref_pops}

    print(f"\nGenerated test data in: {outdir}")
    print(f"  n_samples={args.n_samples}  n_snps={args.n_snps}  n_ancestries={args.n_ancestries}")
    print("\nGenotype inputs (three formats, same underlying data):")
    print(f"  .geno/.snp/.ind prefix  : geno_test{s}     (use with --geno)")
    print(f"  .bed/.bim/.fam prefix   : plink_test{s}    (use with --bfile)  "
          f"[bed round-trip: {'OK' if bed_ok else ('FAILED' if bed_ok is False else 'not verified')}]")
    print(f"  .vcf file               : vcf_test{s}.vcf  (use with --vcf)")
    print("\nPhenotype / covariate files:")
    print(f"  pheno{s}.txt    (use with --pheno)")
    print(f"  covar{s}.txt    (use with --covfile)")
    print("\nPre-computed local ancestry (use with --ancfile, no --yes-anc):")
    print(f"  ancestry_space{s}.txt   (space-delimited haplotype labels 1..{args.n_ancestries})")
    print(f"  ancestry_compact{s}.txt (compact single-digit format)")
    print("\nPopulation labels for --yes-anc:")
    print(f"  pop_labels{s}.txt")
    print(f"    {n_admix} admixed samples  -> 'ADMIX'")
    for p, c in pop_counts.items():
        print(f"    {c} reference samples -> '{p}'")
    print(f"    Distinct ref populations = {args.n_ancestries} = --nbanc  OK")
    print(f"\nUsage guide written to: {os.path.join(outdir, 'README.txt')}")

    if bed_ok is False:
        # Only abort for a CONFIRMED mismatch -- a real bug in the .bed
        # writer. bed_ok is None (verification skipped, bed-reader missing)
        # is not an error condition; the files were still written correctly.
        raise SystemExit("\nERROR: BED round-trip validation FAILED - "
                         "do not trust the generated .bed file.")


if __name__ == "__main__":
    main()

# Note: annotation generator functions are defined above write_readme()
# and called from main() below.
