'''
jasmap.parse
------------

Support functions for reading JasMAP's input files and converting them into
the tab-separated, 6-metadata-column "tped-style" format consumed by makeG().

This module originated as a fork of Brendan Bulik-Sullivan and Hilary
Finucane's ldsc `parse.py` (2014) with a handful of JasMAP-specific functions
appended. As of this cleanup pass, everything not actually used by JasMAP2.py
has been removed -- this dropped `torch`, `h5py`, and `bitarray` as hard
dependencies (they were previously required just to `import jasmap.parse`,
even though the functionality that needed them -- ldsc's LD-score/ID-list
machinery and an unused GenomeDataset helper -- is never called by JasMAP).

Public functions used by JasMAP2.py (imported as `ps`):
    FilterFile              -- ID-list container (used by jasMapUtils.__filter__)
    read_pheno_file(path)   -- -> (phenotypes: np.ndarray, sample_ids: list)
    load_vcf_samples_in_map(vcf_file, samples_list, out_path)
    makeG(...)              -- kinship/GRM computation from a tped-style file
    read_covar_file(path)   -- -> np.ndarray of covariates (PCs etc.)
'''

from __future__ import division
import gc
import gzip

import numpy as np
import pandas as pd
import polars as pl


def read_vcf_lightweight(vcf_file):
    """
    Pure-Python VCF reader — replaces scikit-allel for --vcf input.

    Why: scikit-allel's speed comes from a compiled Cython/C extension
    (allel.opt.model). On Windows, that extension has patchy PyPI wheel
    coverage across Python versions, so pip falls back to compiling it from
    source, which requires the MSVC Build Tools -- the exact same failure
    mode that pandas-plink had for --bfile (fixed by switching to bed-reader).
    Since VCF is a simple, well-documented text format, a dependency-free
    pure-Python parser is a more durable fix than chasing wheel availability,
    at the cost of some speed on very large VCFs (see note in requirements.txt
    and the README: --bfile via bed-reader remains the recommended path for
    biobank-scale data; this parser is for --vcf correctness/portability).

    Supports plain-text and gzip-compressed (.vcf.gz) files, standard VCF 4.x
    with a FORMAT column containing GT (genotype), diploid calls only
    (matching JasMAP's existing assumptions throughout the codebase).

    Returns
    -------
    vcf_data : dict with keys 'samples', 'variants/CHROM', 'variants/ID',
               'variants/POS', 'variants/REF', 'variants/ALT'
               ('variants/ALT' is a list of lists, one per variant, matching
               scikit-allel's multi-allelic-capable shape so the one call
               site in vcf_to_dataframe that does
               `[i[0] for i in vcf_data['variants/ALT']]` keeps working
               unchanged.)
    geno     : np.ndarray int16, shape (n_variants, n_samples, 2)
               Allele calls per haplotype: 0 = REF, 1/2/3.. = ALT allele
               index, -1 = missing. This is the exact convention create_mat()
               already expects (anything other than the explicit 0/0, 0/1,
               1/0, 1/1 combinations it checks is already treated as missing,
               so multi-allelic sites behave identically to the previous
               scikit-allel-based implementation).
    """
    def _open(path):
        if str(path).endswith(".gz"):
            return gzip.open(path, "rt", encoding="utf-8")
        return open(path, "r", encoding="utf-8")

    chroms, ids, positions, refs, alts = [], [], [], [], []
    genotype_rows = []   # list of (n_samples, 2) int16 arrays, one per variant
    samples = []
    gt_field_idx = 0     # position of "GT" within the colon-delimited FORMAT

    with _open(vcf_file) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                header = line.lstrip("#").split("\t")
                samples = header[9:]
                continue

            fields = line.split("\t")
            chroms.append(fields[0])
            positions.append(int(fields[1]))
            ids.append(fields[2])
            refs.append(fields[3])
            alts.append(fields[4].split(","))

            fmt = fields[8].split(":")
            gt_field_idx = fmt.index("GT") if "GT" in fmt else 0

            row = np.full((len(samples), 2), -1, dtype=np.int16)
            for si, sample_field in enumerate(fields[9:]):
                gt_str = sample_field.split(":")[gt_field_idx]
                sep = "|" if "|" in gt_str else "/"
                alleles = gt_str.split(sep)
                for ai in range(min(2, len(alleles))):
                    a = alleles[ai]
                    row[si, ai] = -1 if a in (".", "") else int(a)
                if len(alleles) == 1 and alleles[0] not in (".", ""):
                    # Haploid call written as a single allele: treat as
                    # homozygous for that allele (standard VCF convention).
                    row[si, 1] = row[si, 0]
            genotype_rows.append(row)

    geno = (np.stack(genotype_rows, axis=0) if genotype_rows
            else np.zeros((0, len(samples), 2), dtype=np.int16))

    vcf_data = {
        "samples": samples,
        "variants/CHROM": chroms,
        "variants/ID": ids,
        "variants/POS": positions,
        "variants/REF": refs,
        "variants/ALT": alts,
    }
    return vcf_data, geno


def read_pheno_file(pheno_file):
    '''Read a phenotype file. Two accepted formats:
      - single column of 1/2 (or any int) codes, one per sample, OR
      - whitespace-delimited FID IID ... PHENO (phenotype in the last column,
        e.g. a standard PLINK .fam file, whose 6th/last column IS a
        phenotype under the same convention).

    Bug fix: PLINK's standard missing-phenotype sentinel is -9 (some older
    files also use 0). Previously this function returned a plain int array,
    which can never represent NaN at all -- so a missing-phenotype sample
    silently became the literal value -9 (or -10 after the "-1" conversion
    used throughout JasMAP4 to convert 1/2 coding to 0/1), rather than being
    excluded. LMM.__init__ already calls np.isnan(Y) expecting to filter
    missing samples -- that filtering could never actually fire before this
    fix, since an int array cannot contain NaN in the first place. This
    matters much more now that --bfile input can default its phenotype
    straight from the .fam file's own PHENOTYPE column (see JasMAP4.py's
    --pheno fallback), where -9 (no phenotype collected -- common for
    reference-panel genotyping data with no associated trait) is a
    realistic, not just theoretical, value to encounter.

    Returns (phenotypes: np.ndarray[float64], sample_ids: list[str]).
    '''
    with open(pheno_file) as fh:
        lines = [l.rstrip() for l in fh if l.rstrip() != ""]

    if not lines:
        return np.array([], dtype=float), []

    MISSING_CODES = {-9}  # standard PLINK missing-phenotype sentinel

    def _to_float_or_nan(v):
        iv = int(v)
        return np.nan if iv in MISSING_CODES else float(iv)

    if len(lines[0].split()) <= 1:
        # Single-column format: no sample IDs available in this file.
        phenotypes = np.array([_to_float_or_nan(l) for l in lines], dtype=float)
        return phenotypes, []

    fields = [l.split() for l in lines]
    phenotypes = np.array([_to_float_or_nan(f[-1]) for f in fields], dtype=float)
    sample_ids = [f[1] for f in fields]
    return phenotypes, sample_ids


def read_covar_file(covar_file):
    '''Read a covariate file: FID IID cov1 cov2 ... (NO header line -- see
    make_test_data.py's write_covar(), which explicitly documents this as
    "no header, to match JasMAP4's covar reader").

    Bug fix: this used header=0, which tells pandas to treat the FIRST LINE
    as column names rather than data -- since covariate files never
    actually have a header, this silently dropped the first sample's
    covariate row in EVERY analysis that used --covfile, project-wide,
    throughout every prior session. Never caught because: (a) most unit
    validation built covariate arrays directly in Python, bypassing this
    function entirely; (b) end-to-end CLI tests checked that a run
    completed and produced correctly-SHAPED output, but never checked that
    the sample COUNT read from --covfile exactly matched the genotype
    file's sample count. Found via a direct comparison against the exact
    LMM class that raised a hard shape-mismatch error (59 covariate rows
    vs 60 samples) while debugging an unrelated kinship-formula issue --
    the shape check inside LMMLowRank's construction is what surfaced it;
    LMM.__init__ apparently tolerates the same mismatch silently in the
    "exact" path (see JasMAP4.py's run_joint_analysis() bug fix note),
    which is why this was never noticed there.

    Returns an (n_samples x n_covariates) np.ndarray.
    '''
    df = pd.read_csv(covar_file, sep=r"\s+", header=None, engine="python")
    return df.iloc[:, 2:].to_numpy(dtype=float)


def create_mat(geno):
    '''Vectorized replacement for the previous nested-Python-loop version.
    geno: array of shape (n_variants, n_samples, 2) of allele calls (0/1/-1).
    Returns an (n_variants, n_samples) uint8 matrix using the SAME additive
    dosage convention as the --geno/--bfile paths (count of the ALT/coded
    allele: 0 = hom ref, 1 = het, 2 = hom alt, 9 = missing).

    Bug fix (#13): the previous version (faithfully carried over from the
    original nested-loop create_mat, which used the exact same mapping) used
    the OPPOSITE convention -- 2 = hom ref, 0 = hom alt. This didn't corrupt
    significance testing (p-values are symmetric under a sign flip), but it
    silently flipped the SIGN of every SNP_beta effect-size estimate for
    --vcf input relative to the --geno/--bfile paths on the exact same
    underlying genotypes, discovered by cross-checking all three input paths
    against each other on identical synthetic data end-to-end.
    '''
    geno = np.asarray(geno)
    a0 = geno[:, :, 0]
    a1 = geno[:, :, 1]
    missing = (a0 < 0) | (a1 < 0)

    out = np.full(a0.shape, 9, dtype=np.uint8)
    out[(a0 == 0) & (a1 == 0)] = 0   # hom REF -> 0 ALT-allele copies
    out[((a0 == 0) & (a1 == 1)) | ((a0 == 1) & (a1 == 0))] = 1
    out[(a0 == 1) & (a1 == 1)] = 2   # hom ALT -> 2 ALT-allele copies
    out[missing] = 9
    return out


def load_vcf_samples_in_map(vcf_file, samples_list, file, chr_arg=None):
    """Convert a VCF to the same 6-metadata-column tped-style format produced by
    the --bfile and --geno read paths (CHR, SNP/ID, CM, POS, A1/REF, A2/ALT,
    then one column per sample).

    Bug fix: the previous version emitted only 5 metadata columns
    (CHR, POS, ID, REF, ALT), but makeG() unconditionally slices genotype
    columns starting at index 6 (`all_gen[:, 6:]`), matching the --bfile/--geno
    convention. That mismatch silently shifted every sample's genotype by one
    column for any VCF-derived kinship computation. A placeholder "CM"
    (genetic distance) column has been inserted, in the same column position
    PLINK's own .bim format uses, to restore the expected 6-column layout.

    Second, more severe bug fix: `vcf_data['samples'] = samples_list` used to
    unconditionally OVERWRITE the VCF's own, actually-read sample list with
    whatever samples_list was passed in (typically --pheno's sample IDs) --
    with no ID matching or validation at all. If samples_list had FEWER
    entries than the VCF's real sample count, this crashed outright
    ("Shape of passed values is (2, 6), indices imply (2, 3)", reproduced
    directly). If the counts happened to match by coincidence, it would have
    silently mislabelled every genotype column with the wrong sample ID
    instead, since samples_list was never actually matched against the
    VCF's real header -- just blindly substituted in.

    Design note, revised after further consideration: this function ALWAYS
    writes every sample actually present in the VCF, using the VCF's own,
    real sample order -- it does NOT filter down to samples_list, even
    when samples_list (from --pheno) lists fewer samples. This matters
    for the --yes-anc data model a real user described: reference/
    ancestral population samples are genuinely needed in the GENOTYPE
    data (for local ancestry inference), even though --pheno intentionally
    omits them (they have no phenotype at all, not being part of the
    study cohort). Dropping them here would silently break --yes-anc for
    VCF input. Phenotype-to-genotype-sample alignment (giving reference
    samples a missing phenotype rather than requiring their removal) is
    handled separately, by _align_pheno_to_genotype_samples() in
    JasMAP4.py, using this function's own actual_samples as the reference
    order -- the same shared mechanism --bfile and --geno now use.
    samples_list is still validated here (any ID not found in the VCF
    raises a clear, early error), just no longer used to filter/reorder
    the genotype output.
    """
    # Bug fix: previously called allel.read_vcf() twice -- once for metadata,
    # once for calldata/GT -- meaning two full file passes. read_vcf_lightweight()
    # does both in a single pass, which is both a scikit-allel-free replacement
    # AND a performance improvement over the code it replaces.
    vcf_data, geno = read_vcf_lightweight(vcf_file)
    actual_samples = list(vcf_data['samples'])   # the VCF's own, real sample order
    mat = create_mat(geno)                       # (n_variants, n_samples_in_vcf)

    if samples_list:
        actual_set = set(actual_samples)
        missing = [s for s in samples_list if s not in actual_set]
        if missing:
            raise ValueError(
                f"--pheno lists {len(missing)} sample ID(s) not found in the "
                f"VCF's own header: {missing[:10]}"
                f"{' ... (' + str(len(missing)-10) + ' more)' if len(missing) > 10 else ''}. "
                f"The VCF has {len(actual_samples)} samples; --pheno must "
                f"list a subset of (or all of) these, by matching ID.")

    vcf_data['samples'] = actual_samples
    vcf_data['variants/ALT'] = [i[0] for i in vcf_data['variants/ALT']]

    n_variants = len(vcf_data['variants/CHROM'])
    bim = pd.DataFrame({
        'CHR': vcf_data['variants/CHROM'],
        'SNP': vcf_data['variants/ID'],
        'CM': np.zeros(n_variants, dtype=np.int8),   # placeholder genetic distance
        'POS': vcf_data['variants/POS'],
        'A1': vcf_data['variants/REF'],
        'A2': vcf_data['variants/ALT'],
    })

    # Bug fix: --chr was silently ignored for --vcf input (self.chr_snp_idx
    # was only ever set inside read_bed_file(), --bfile-specific) -- found
    # via a real run that processed the entire panel with --chr given, no
    # filtering, no warning at all. Resolved here from the VCF's own
    # #CHROM column, applied before the .tmp.tped file is written --
    # RVAT/PRS/GIFT's own --vcf fallback paths all read from this SAME
    # shared .tmp.tped file, so filtering here once propagates correctly
    # to all of them. Duplicated (rather than shared via an import) from
    # JasMAP4.py's _resolve_chr_indices_generic() to avoid a circular
    # import (JasMAP4.py already imports this module as `ps`) -- same
    # matching/error semantics.
    chr_snp_idx = None
    if chr_arg is not None:
        chrom_arr = bim['CHR'].values.astype(str)
        target = chr_arg.strip()
        if target.lower().startswith("chr"):
            target = target[3:]
        chr_snp_idx = np.where(chrom_arr == target)[0]
        if len(chr_snp_idx) == 0:
            available = sorted(set(chrom_arr.tolist()), key=lambda x: (len(x), x))
            preview = available[:30]
            raise ValueError(
                f"--chr {chr_arg!r} matched 0 SNPs in the VCF. Available "
                f"chromosome codes: {preview}"
                f"{' ... (' + str(len(available)-30) + ' more)' if len(available) > 30 else ''}")
        # Restriction notice logged by the caller (JasMAP4.py's
        # vcf_to_dataframe(), via the standard `logger`) instead of a
        # print() here -- this module's own print()-based messages don't
        # appear in the jasmap-*.log file at all (a pre-existing
        # characteristic of every print() call in this module, confirmed
        # directly: even the unrelated, existing "Converting Matrix..."
        # message doesn't show up there either), so logging this
        # user-facing, --chr-validation-relevant notice from the caller
        # keeps it visible where every other input format's equivalent
        # message already is.
        bim = bim.iloc[chr_snp_idx].reset_index(drop=True)
        mat = mat[chr_snp_idx]

    print("\nConverting Matrix to Pandas DataFrame....")
    snps = pd.DataFrame(data=mat, columns=vcf_data['samples'])
    snps = bim.join(snps)
    print('\nWriting the converted vcf file into tped format')
    snps.to_csv(file, sep='\t', encoding='utf-8', index=False, header=True)
    return actual_samples, chr_snp_idx, n_variants


class FilterFile(object):
    '''Minimal ID-list container: reads a single-column file of IDs to keep.
    Kept for API compatibility with jasMapUtils.__filter__ (not currently
    called anywhere in JasMAP2.py, but part of the public `ps.` surface).
    '''

    def __init__(self, fname):
        self.df = pd.read_csv(fname, header=None, sep=r"\s+", engine="python")
        self.df.columns = ['ID']
        self.IDList = self.df[['ID']].astype('object')

    def loj(self, external_df):
        '''Returns indices of IDList entries that also appear in external_df.'''
        r = external_df.columns[0]
        merge_df = external_df.iloc[:, [0]].copy()
        merge_df['keep'] = True
        z = pd.merge(self.IDList, merge_df, how='left', left_on='ID', right_on=r, sort=False)
        return np.nonzero((z['keep'] == True).to_numpy())[0]


def _impute_and_standardize_missing(mat):
    '''Vectorized (no Python per-column loop) missing-value imputation used
    by makeG(): NaNs are replaced by that variant's mean across samples; if a
    variant is entirely missing it is imputed with 0.
    '''
    if not np.isnan(mat).any():
        return mat
    row_mean = np.nanmean(mat, axis=1)
    row_mean = np.nan_to_num(row_mean, nan=0.0)
    nan_rows, nan_cols = np.where(np.isnan(mat))
    mat[nan_rows, nan_cols] = row_mean[nan_rows]
    return mat


def makeG(File, method, File_list=False, n1=0, n2=0, maf_min=0.01):
    '''Calculate the additive kinship/GRM matrix from genotype data in
    tped-style format (>=6 metadata columns followed by one column per sample,
    tab-separated, as produced by the --geno/--bfile/--vcf read paths).

    File: filename of a single tped-style file, or (if File_list=True) a file
        listing one such filename per chromosome.
    method: int in [0,1,2,3]:
        0 - Yang et al. 2010 Nat Genet standardized GRM
        1 - unstandardized sum[(x-2p)(x-2p)] / sum[2p(1-p)]
        2 - "G_new": two-group formulation
        3 - "G_chen": two-group formulation (Chen et al. variant)
    n1, n2: sample counts for the two groups (only used by methods 2 and 3).
    maf_min: minimum minor allele frequency for a SNP to be INCLUDED in the
        kinship/GRM computation (default 0.01, matching GCTA/PLINK
        convention for GRM construction). SNPs below this threshold are
        excluded entirely, not just for statistical-power reasons but for
        numerical stability.

        Bug fix: the existing `dii[~np.isfinite(dii)] = 0.0` guard below only
        catches EXACTLY monomorphic sites (p=0 or p=1, where 1/(2pq) is
        literally infinite) -- it does nothing for a rare variant at, say,
        p=0.0001, where dii = 1/(2*0.0001*0.9999) is approximately 5000: a
        large but perfectly FINITE number that sails straight through the
        np.isfinite() check and contributes a wildly disproportionate weight
        to that one variant in the summed kinship matrix. On real data with
        many rare variants (unlike this project's synthetic test generator,
        which produces no such extreme allele frequencies), this compounds
        across many rare SNPs into astronomically wrong kinship values --
        found via an actual run against a real cohort, where auto-generated
        PCA covariates showed trace(K) on the order of 10^19 instead of the
        expected ~n. A MAF floor is the standard, established fix (this is
        exactly why GCTA and PLINK both default to excluding rare variants
        from GRM/kinship construction), applied identically here and in
        every one of the biobank-scale modules (lowrank_kinship.py,
        cg_kinship.py, streaming_io.py) that reproduce this same formula.
    '''
    method_list = [0, 1, 2, 3]
    if not isinstance(method, int):
        raise ValueError("Parameter 'method' should be int, one of %s" % method_list)
    if method not in method_list:
        raise ValueError("Parameter 'method' should be one of %s" % method_list)

    if File_list:
        genofile = pd.read_csv(File, header=None)
        bad = genofile[0][~genofile[0].str.endswith('.traw')]
        if len(bad) > 0:
            raise ValueError("genotype file type should be .traw of PLINK")
    else:
        genofile = pd.DataFrame([File])

    with open(genofile.iloc[0, 0], "r") as fh:
        line1 = fh.readline().split()
    geno_id = pd.Series(line1[6:])
    N = len(line1) - 6

    if method in (2, 3):
        if not isinstance(n1, int) or not isinstance(n2, int) or n1 <= 0 or n2 <= 0:
            raise ValueError("Parameters n1 and n2 must be positive integers for method 2/3.")
        if n1 + n2 != N:
            raise ValueError("n1 + n2 (%d) does not match the number of individuals (%d)." % (n1 + n2, N))

    G = np.zeros((N, N)) if method in (0, 1) else None
    G11 = np.zeros((n1, n1)) if method in (2, 3) else None
    G12 = np.zeros((n1, n2)) if method in (2, 3) else None
    G22 = np.zeros((n2, n2)) if method in (2, 3) else None
    M = 0
    sum2pq = 0.0
    sum2pq_xx = 0.0
    sum2pq_tb = 0.0
    xxbytb = 0.0

    for i in range(genofile.shape[0]):
        chr_file = genofile.iloc[i, 0]
        all_gen = pl.read_csv(chr_file, separator="\t", null_values="NA").with_columns(
            pl.all().cast(pl.Int8, strict=False))

        if method in (0, 1):
            gen = all_gen[:, 6:].to_numpy().astype(np.float64)
            del all_gen
            gc.collect()
            gen = _impute_and_standardize_missing(gen)

            p = gen.mean(axis=1) / 2.0
            q = 1.0 - p
            maf = np.minimum(p, q)
            keep = maf >= maf_min
            if not keep.all():
                gen = gen[keep]
                p = p[keep]
                q = q[keep]
            Z = gen - 2 * p[:, None]

            if method == 0:
                dii = 1.0 / (2 * p * q)
                dii[~np.isfinite(dii)] = 0.0  # guard against fixed (p==0 or 1) sites
                G += (Z.T * dii) @ Z
                M += gen.shape[0]
            else:  # method == 1
                sum2pq += float((2 * p * q).sum())
                G += Z.T @ Z
            del gen, Z
            gc.collect()
        else:
            xx_gen = all_gen[:, 6:(n1 + 6)].to_numpy().astype(np.float64)
            tb_gen = all_gen[:, (n1 + 6):].to_numpy().astype(np.float64)
            del all_gen
            gc.collect()
            xx_gen = _impute_and_standardize_missing(xx_gen)
            tb_gen = _impute_and_standardize_missing(tb_gen)

            p_xx = xx_gen.mean(axis=1) / 2.0
            p_tb = tb_gen.mean(axis=1) / 2.0
            q_xx = 1.0 - p_xx
            q_tb = 1.0 - p_tb

            # Same MAF floor as methods 0/1 above, applied here for
            # consistency (methods 2/3 sum 2pq across all SNPs before a
            # single final division rather than per-SNP inverse-variance
            # weighting, so they lack method 0's specific blow-up
            # mechanism, but excluding rare variants from GRM construction
            # is standard practice regardless of which formula variant).
            keep_xx = np.minimum(p_xx, q_xx) >= maf_min
            keep_tb = np.minimum(p_tb, q_tb) >= maf_min
            keep_both = keep_xx & keep_tb
            if not keep_both.all():
                xx_gen = xx_gen[keep_both]; tb_gen = tb_gen[keep_both]
                p_xx = p_xx[keep_both]; p_tb = p_tb[keep_both]
                q_xx = q_xx[keep_both]; q_tb = q_tb[keep_both]

            z_xx = xx_gen - 2 * p_xx[:, None]
            z_tb = tb_gen - 2 * p_tb[:, None]

            G11 += z_xx.T @ z_xx
            G12 += z_xx.T @ z_tb
            G22 += z_tb.T @ z_tb

            sum2pq_xx += float((2 * p_xx * q_xx).sum())
            sum2pq_tb += float((2 * p_tb * q_tb).sum())
            if method == 3:
                xxbytb += float(np.sqrt(2 * p_xx * q_xx * 2 * p_tb * q_tb).sum())

            del xx_gen, tb_gen, z_xx, z_tb
            gc.collect()

    if method == 0:
        return [G / M, geno_id]
    if method == 1:
        return [G / sum2pq, geno_id]
    if method == 2:
        G11n = G11 / sum2pq_xx
        G12n = G12 / (np.sqrt(sum2pq_tb) * np.sqrt(sum2pq_xx))
        G22n = G22 / sum2pq_tb
        gmat = np.vstack((np.hstack((G11n, G12n)), np.hstack((G12n.T, G22n))))
        return [gmat, geno_id]
    # method == 3
    G11c = G11 / sum2pq_xx
    G12c = G12 / xxbytb
    G22c = G22 / sum2pq_tb
    gmat = np.vstack((np.hstack((G11c, G12c)), np.hstack((G12c.T, G22c))))
    return [gmat, geno_id]
