#!/usr/bin/python
# -*- coding: utf8 -*-

from __future__ import print_function
import os
import time, sys, argparse
from functools import reduce
from collections.abc import Mapping
from pathlib import Path
import logging, warnings
import scipy.stats as sts
from scipy import linalg,stats
from scipy import optimize
import numpy as np
import pandas as pd
import polars as pl
import jasmap.parse as ps
# Bug fix (#7): jasmap.jasmap_bed (as `ld`) and the stdlib `pickle`/`traceback`
# modules were imported but never referenced anywhere in this file. Removing
# them avoids requiring `bitarray` (jasmap_bed's own dependency) just to launch
# this script. Re-add `import jasmap.jasmap_bed as ld` here if a future feature
# actually needs it.
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def _align_pheno_to_genotype_samples(phenotypes, pheno_sample_ids, genotype_sample_ids):
    """
    Aligns a --pheno file's (phenotypes, sample_ids) pair against the
    genotype file's ACTUAL sample order, by ID -- rather than assuming
    positional alignment, which was the previous, unconditional behaviour
    across all three input paths (--bfile with an explicit --pheno,
    --geno, and --vcf).

    Why this exists
    -----------------
    Supports --pheno listing FEWER samples than the genotype file -- e.g.
    only the admixed study cohort, with reference/ancestral population
    samples present in the genotype file (needed for --yes-anc's local
    ancestry inference) but intentionally absent from --pheno, exactly
    the data model a real user described. Without this, --pheno was
    required to have EXACTLY the same sample count as the genotype file,
    in the SAME order, with no validation at all -- found to crash
    outright for --vcf (a direct shape-mismatch error, reproduced
    directly: "Shape of passed values is (2, 6), indices imply (2, 3)"),
    and to silently misalign for --bfile/--geno (no check ever compared
    --pheno's sample IDs against the genotype file's own sample list, so
    a --pheno subset would have been read positionally against the WRONG
    samples with no warning at all).

    Any genotype-file sample not listed in --pheno gets a missing (NaN)
    phenotype, matching what --bfile's own .fam-column-6 fallback already
    does for -9-coded samples -- consistent behaviour regardless of
    whether the phenotype came from the .fam file directly or a separate
    --pheno file.

    Parameters
    ----------
    phenotypes         : ndarray  from read_pheno_file()
    pheno_sample_ids   : list[str] or None/empty -- from read_pheno_file();
                        empty/None means the single-column --pheno format
                        was used, which has no sample IDs at all
    genotype_sample_ids: list[str]  the genotype file's OWN, actual sample
                        order (bed.iid for --bfile, the .ind file's IDs
                        for --geno, the VCF header's sample list for --vcf)

    Returns
    -------
    ndarray, length len(genotype_sample_ids), aligned to genotype_sample_ids'
    order, with NaN for any genotype-file sample not found in --pheno.
    """
    n_geno = len(genotype_sample_ids)

    if not pheno_sample_ids:
        # Single-column format: no IDs to match on at all. The only
        # possible behaviour is positional alignment -- but only when the
        # lengths actually agree; otherwise this would silently misalign
        # every downstream sample, so raise instead.
        if len(phenotypes) != n_geno:
            raise ValueError(
                f"--pheno has {len(phenotypes)} value(s) (single-column "
                f"format, no sample ID column) but the genotype file has "
                f"{n_geno} samples -- these must match exactly when "
                f"--pheno has no sample IDs to match on. Use the "
                f"'FID IID ... PHENO' format instead if --pheno should "
                f"list only a subset of the genotype file's samples.")
        return np.asarray(phenotypes, dtype=np.float64)

    pheno_id_set = set(pheno_sample_ids)
    geno_id_set = set(genotype_sample_ids)
    missing_from_geno = pheno_id_set - geno_id_set
    if missing_from_geno:
        preview = sorted(missing_from_geno)[:10]
        raise ValueError(
            f"--pheno lists {len(missing_from_geno)} sample ID(s) not "
            f"found in the genotype file: {preview}"
            f"{' ... (' + str(len(missing_from_geno)-10) + ' more)' if len(missing_from_geno) > 10 else ''}. "
            f"Check for a typo or a mismatched --pheno/genotype file pair.")

    pheno_map = dict(zip(pheno_sample_ids, phenotypes))
    aligned = np.array([pheno_map.get(s, np.nan) for s in genotype_sample_ids], dtype=np.float64)
    n_matched = sum(1 for s in genotype_sample_ids if s in pheno_map)
    if n_matched < n_geno:
        logger.info(
            f"[INFO] --pheno lists {n_matched} of {n_geno} genotype-file "
            f"samples by ID; the remaining {n_geno - n_matched} (e.g. "
            f"reference/ancestral population samples used only for local "
            f"ancestry inference) get a missing phenotype.\n")
    return aligned


def _resolve_chr_indices_generic(chrom_values, chr_arg, source_desc):
    """
    Generalized chromosome-index resolver, usable for ANY source of
    per-SNP chromosome labels -- not just --bfile's .bim (which
    _resolve_chr_snp_indices() already handles specifically via
    bed-reader's cheap, genotype-data-free .chromosome property).

    Why this exists
    -----------------
    --chr was only ever wired up for --bfile input: self.chr_snp_idx is
    set in exactly one place in the whole file, inside read_bed_file().
    For --geno/--vcf, --chr was silently accepted and completely ignored
    -- found via a real run ("--geno ... --chr 22" processed the entire
    220-SNP panel with zero warning or error of any kind). Same matching
    semantics as _resolve_chr_snp_indices() (case-insensitive "chr"
    prefix stripped, stops with a clear error listing available codes if
    zero SNPs match), just parameterised over a plain array of chromosome
    labels rather than requiring a bed-reader object specifically.

    Parameters
    ----------
    chrom_values : array-like of per-SNP chromosome labels, in SNP order
    chr_arg      : str or None -- e.g. "22" or "chr22"; None means no
                  filtering (returns None)
    source_desc  : str -- describes the source, for the error message
                  (e.g. the .snp file path, or "the VCF")

    Returns
    -------
    snp_idx : ndarray of matching indices, or None if chr_arg is None
    """
    if chr_arg is None:
        return None
    chrom_arr = np.asarray(chrom_values).astype(str)
    target = chr_arg.strip()
    if target.lower().startswith("chr"):
        target = target[3:]
    snp_idx = np.where(chrom_arr == target)[0]
    if len(snp_idx) == 0:
        available = sorted(set(chrom_arr.tolist()), key=lambda x: (len(x), x))
        preview = available[:30]
        raise ValueError(
            f"--chr {chr_arg!r} matched 0 SNPs in {source_desc}. Available "
            f"chromosome codes: {preview}"
            f"{' ... (' + str(len(available)-30) + ' more)' if len(available) > 30 else ''}")
    return snp_idx


def _resolve_chr_snp_indices(bfile_path, chr_arg, verbose=True):
    """
    Resolves --chr (e.g. "22" or "chr22") to the matching SNP indices in a
    .bed file, reading ONLY the cheap, already-lazy-loaded .bim-derived
    chromosome array (bed-reader's own `.chromosome` property) -- never
    touching genotype data itself.

    Why this exists
    -----------------
    --chr previously existed as a CLI argument (with a docstring
    describing exactly this behaviour) but was never actually wired to
    filter anything -- found by grepping the whole file for any other
    reference to it and finding none. Restricting to one chromosome is a
    practical, standard way to make biobank-scale analysis (hundreds of
    GB to TB for the whole genome at n=500,000+) fit in memory: a single
    chromosome is roughly 1/22 to 1/8 of the genome (smaller for chr21/22,
    larger for chr1), turning an infeasible whole-genome read into
    something that fits comfortably, and gives a natural way to
    parallelise across chromosomes (the same pattern real biobank GWAS
    pipelines use), rather than requiring a full streaming-architecture
    rewrite of the association-testing loop itself to reach biobank scale.

    Parameters
    ----------
    bfile_path : str  path to the .bed file (with or without extension)
    chr_arg    : str or None  e.g. "22" or "chr22" (case-insensitive
                "chr" prefix is stripped); None means no filtering

    Returns
    -------
    snp_idx : ndarray of matching SNP indices, or None if chr_arg is None
    """
    if chr_arg is None:
        return None
    from bed_reader import open_bed
    bed_path = bfile_path if bfile_path.endswith(".bed") else bfile_path + ".bed"
    bed = open_bed(bed_path, count_A1=False)
    chrom_arr = bed.chromosome

    target = chr_arg.strip()
    if target.lower().startswith("chr"):
        target = target[3:]

    snp_idx = np.where(chrom_arr == target)[0]
    if len(snp_idx) == 0:
        available = sorted(set(chrom_arr.tolist()), key=lambda x: (len(x), x))
        preview = available[:30]
        raise ValueError(
            f"--chr {chr_arg!r} matched 0 SNPs in {bed_path}. Available "
            f"chromosome codes in this file: {preview}"
            f"{' ... (' + str(len(available)-30) + ' more)' if len(available) > 30 else ''}")

    if verbose:
        logger.info(f"[INFO] --chr {chr_arg!r}: restricting analysis to "
                    f"{len(snp_idx)} of {len(chrom_arr)} SNPs "
                    f"({100*len(snp_idx)/len(chrom_arr):.2f}% of the panel).\n")
    return snp_idx


def _assert_covariates_used(model, expected_X0, context):
    """
    Permanent safeguard against the class of bug found in this project where
    covariates (typically per-individual PCA eigenvectors, the standard
    mechanism for correcting population stratification -- critical for a
    tool whose entire purpose is analysing ADMIXED populations, where
    stratification is the dominant confounder) were silently dropped due to
    a positional-argument mistake in the LMM(...) constructor call
    (X0s landed in the Kva parameter slot instead of X0, defaulting to
    intercept-only fitting with no error, no warning -- found only via a
    direct numerical comparison against an independent code path, not by
    any unit test).

    Checks that the constructed model's actual covariate dimensionality
    (model.X0.shape[1]) matches what was intended (expected_X0.shape[1]).
    Raises immediately if they disagree, rather than allowing an analysis
    to silently proceed with the wrong statistical model -- this is a
    correctness issue serious enough to fail loudly, not degrade quietly.

    Called after EVERY LMM/LMMLowRank/CGLMM construction in this file that
    is meant to use covariates, so this exact bug class cannot recur
    silently even through a future refactor that reintroduces a similar
    positional-argument mistake at a new call site.
    """
    actual_q = model.X0.shape[1]
    expected_q = expected_X0.shape[1]
    if actual_q != expected_q:
        msg = (f"[FATAL] Covariate dimensionality mismatch in {context}: "
              f"expected {expected_q} covariate column(s) but the "
              f"constructed model actually has {actual_q}. This is the "
              f"exact failure mode of a known prior bug (X0 passed "
              f"positionally instead of via X0=, silently landing in the "
              f"wrong constructor parameter and falling back to "
              f"intercept-only fitting). Refusing to proceed with an "
              f"analysis that would silently drop population-stratification "
              f"covariates.")
        logger.info(msg + "\n")
        raise RuntimeError(msg)


# Windows fix: Python's default StreamHandler writes to sys.stderr using the
# system locale codec (cp1252 on most Windows installs), which cannot encode
# many Unicode characters even in plain logger.info() messages. Re-wrapping
# sys.stderr with UTF-8 reconfiguration prevents UnicodeEncodeError on
# Windows while remaining a no-op on macOS/Linux (which default to UTF-8).
import sys as _sys, io as _io
try:
    if hasattr(_sys.stderr, 'reconfigure'):
        _sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    elif _sys.stderr.encoding and _sys.stderr.encoding.lower() not in ('utf-8','utf8'):
        _sys.stderr = _io.TextIOWrapper(
            _sys.stderr.buffer, encoding='utf-8', errors='replace')
    if hasattr(_sys.stdout, 'reconfigure'):
        _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    elif _sys.stdout.encoding and _sys.stdout.encoding.lower() not in ('utf-8','utf8'):
        _sys.stdout = _io.TextIOWrapper(
            _sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass  # If reconfigure fails (e.g. stdout is not a real stream), continue
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)
warnings.filterwarnings("ignore")

# ro quiet splash $vt_handoff
__version__ = "v4.0"
__author__ = "Jacquiline Mugo, N. Mulder, Emile .R Chimusa"
__email__ = "jacquilinemugo@aims.ac.za and emile.chimusa@northumbria.ac.uk"
__status__ = "Production"

JHEAD ="***********************************************************************************************************************\n"
JHEAD += "** ******** JASMAP: A Joint Ancestry and SNP Mapping Method for Multi-way Admixed Populations   *********************\n"
JHEAD += "** Version: {V}\n".format( V=(__version__).center(71))
JHEAD += '** Developed by %s \n' % (__author__).center(71)
JHEAD += '** Emails: %s \n' % (__email__).center(71)
JHEAD +="***********************************************************************************************************************\n"

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
#pd.set_option('precision', 4)
pd.set_option('max_colwidth',1000)
np.set_printoptions(linewidth=1000)
np.set_printoptions(precision=4)


try:
    x = pd.DataFrame({'A': [1, 2, 3]})
    x.sort_values(by='A')
except AttributeError:
    raise ImportError('JasMAP 4 requires pandas version >= 0.17.0')


class jasMapUtils ():
    '''     Useful functions    '''
    def __init__(self):
        self._x = None

    def logical(self, Value):
        '''Convert a string to logical variable 
        Value: a string, NO or YES
        Return a logical variable'''

        if Value == "YES":
                return True
        else:
                return False

    def safe_open(self,filename):
        '''Opens a file in a safe manner or raise an exception if the file doesn't exist filename: a string, name of a file
        Return: file object'''

        if os.path.exists(filename):
                return open(filename)
        else:
                #self.printHead()
                sys.stderr.write("ERROR ==> No such file or directory\"%s\"\n"%format(filename))
                return False

    def check_files(self,params):
        '''Check existed of files in a dict or a list of  params: a dictionary, of option:value '''
        Fil_N = 0; Row_F = []
        for des in params:
                if not os.path.exists(des):
                    Row_F.append(des)
                    Fil_N = Fil_N +1
                else:
                    pass
        if Fil_N >= 1:
            for k in Row_F:
                sys.stderr.write("ERROR: No such file or directory \"%s\"\n" % k)
            print("Existing with Fatal Error....!\n")
            sys.exit(1)
        else:
            pass


    def sec_to_str(self, t):
        '''Convert seconds to days:hours:minutes:seconds'''
        [d, h, m, s, n] = reduce(lambda ll, b: divmod(ll[0], b) + ll[1:], [(t, 1), 60, 60, 24])
        f = ''
        if d > 0:
            f += '{D}d:'.format(D=d)
        if h > 0:
            f += '{H}h:'.format(H=h)
        if m > 0:
            f += '{M}m:'.format(M=m)

        f += '{S}s'.format(S=s)
        return f

    def _remove_dtype(self, x):
        '''Removes dtype: float64 and dtype: int64 from pandas printouts'''
        x = str(x)
        x = x.replace('\ndtype: int64', '')
        x = x.replace('\ndtype: float64', '')
        return x

    def __filter__(self, fname, noun, verb, merge_obj):
        merged_list = None
        if fname:
            f = lambda x, n: x.format(noun=noun, verb=verb, fname=fname, num=n)
            x = ps.FilterFile(fname)
            c = 'Read list of {num} {noun} to {verb} from {fname}'
            print (f(c, len(x.IDList)))
            merged_list = merge_obj.loj(x.IDList)
            len_merged_list = len(merged_list)
            if len_merged_list > 0:
                c = 'After merging, {num} {noun} remain'
                print (f(c, len_merged_list))
            else:
                error_msg = 'No {noun} retained for analysis'
                raise ValueError(f(error_msg, 0))

            return merged_list
    def terminate(self, log):
        '''Terminate the process'''
        #log = self.outfolder+self.logFile
        logger.info('\nLog file generated in '+log+'\n Existing with Fatal Error \n Have a nice day!\n')
    def _timer(self,ctime,msg):
        time_elapsed = round(time.time() - ctime, 2)
        logger.info(msg)
        logger.info('Total time elapsed : {T}'.format(T=self.sec_to_str(time_elapsed)))

class _LazyLancDict(Mapping):
    """
    Read-only, dict-compatible view over a dense lanc_array, replacing the
    fully-materialised {snp: {ancestry: [per_sample_values]}} dict.

    Why this exists (performance fix, found on a real run):
    building that dict eagerly for a genome-wide panel means allocating
    n_snps * n_ancestries * n_individuals boxed Python integers -- for a
    real 272,791-SNP x 1,363-sample x 5-ancestry cohort that is over 1.8
    BILLION Python objects, which took 738 seconds (12.3 minutes, ~22% of
    that run's total 57-minute wall time) purely to construct. The exact
    same data already exists as self.lanc_array, a dense numpy int8 array
    (~1.9 GB vs. ~52 GB if the ints weren't small-int-cached).

    Critically, the pipeline's hot path never reads those values -- the
    main analysis loop uses lanc_dict only for its KEYS (len() and
    sorted(...keys())), and the one place that does read values is a
    fallback that only runs when lanc_array is unavailable in the first
    place. So the entire 738-second materialisation was building data
    nothing subsequently consumed.

    This class keeps the exact same read interface (d[snp][anc] -> list of
    per-sample values, len(d), iteration, .keys(), .get()) so every
    existing consumer works unchanged, but computes each per-SNP row from
    lanc_array only when actually asked for. Keys are 1-based line numbers,
    matching the original structure.
    """
    __slots__ = ("_arr", "_labels", "_n_snps")

    def __init__(self, lanc_array, lanc_labels_list):
        self._arr = lanc_array                # (K, n_snps, n_individuals)
        self._labels = list(lanc_labels_list)
        self._n_snps = lanc_array.shape[1]

    def __getitem__(self, lineno):
        if not (1 <= lineno <= self._n_snps):
            raise KeyError(lineno)
        idx = lineno - 1
        return {anc: self._arr[anc - 1, idx].tolist() for anc in self._labels}

    def __iter__(self):
        return iter(range(1, self._n_snps + 1))

    def __len__(self):
        return self._n_snps


class jasMapInit(jasMapUtils):
    #logger.removeHandler(logger.handlers[1])
    logger.setLevel(logging.INFO)
    logFile = 'jasmap-' + str(time.time()).split('.')[0] + '.log'
    def __init__(self, argv, logFile=logFile):
        '''Initializing JasMAP by reading the parameter file'''
        self.argv = [argv]
        self.logFile = logFile
        self.started = time.time()
        # Read in the parameter file and set logical parameters true/false
        parser = argparse.ArgumentParser(
            description="JasMAP v4: Joint Ancestry and SNP Mapping Method - "
                        "extended with GIFT, consensus LAI, PRS and publication plots")
        parser.add_argument('--out', default=None, type=str,
            help='Output folder (default: jasmap_results/).')
        parser.add_argument('--out-prefix', default='JasMAP', type=str,
            help='Filename prefix applied to every result, log, and plot '
                 'file written into --out (default: "JasMAP", e.g. '
                 'JasMAP_Jasmap.Joint.assoc). Pass your own, e.g. '
                 '--out-prefix chr22_run, to make runs easier to tell '
                 'apart -- helps guarantee filenames never collide across '
                 'runs, including two PARALLEL runs sharing the same --out '
                 'folder or the same underlying genotype file (e.g. one '
                 'process per chromosome). When --chr is also given, the '
                 'requested chromosome is automatically folded into the '
                 'prefix too (e.g. --out-prefix chr22_run --chr 22 writes '
                 'chr22_run_chr22_Jasmap.Joint.assoc; the default prefix '
                 'gets the same treatment: JasMAP_chr22_Jasmap.Joint.assoc), '
                 'so this happens whether or not --out-prefix is explicitly '
                 'set. Has no effect on temporary/intermediate files, which '
                 'are already named uniquely per run (chromosome-aware) and '
                 'deleted automatically after a successful run regardless.')
        parser.add_argument('--bfile', default=None, type=str,
            help='Prefix for PLINK .bed/.bim/.fam file.')
        parser.add_argument('--vcf', default=None, type=str,
            help='Full VCF filename (e.g. cohort.vcf or cohort.vcf.gz).')
        parser.add_argument('--geno', default=None, type=str,
            help='Prefix for EIGENSTRAT .geno/.snp/.ind files.')
        parser.add_argument('--pheno', default=None, type=str,
            help='Phenotype file (FID IID PHENO, one row per sample).')
        parser.add_argument('--covfile', default=None, type=str,
            help='Covariate file (FID IID COV1 COV2 ...).')
        parser.add_argument('--ancfile', default=None, type=str,
            help='Local ancestry file - used as input if --yes-anc=No, '
                 'or as output filename when --yes-anc is set.')
        parser.add_argument('--ntest', default=10, type=int,
            help='Effective number of tests (for significance correction).')
        parser.add_argument('--nbanc', default=None, type=int,
            help='Number of ancestral populations K.')
        parser.add_argument('--pop-labels', default=None, type=str,
            help='Comma-separated ancestral population labels (K values), '
                 'e.g. "AFR,EUR,SAN". Length must equal --nbanc.')
        parser.add_argument('--pop-file', default=None, type=str,
            help='Two-column file (sample_ID, population_label) '
                 'distinguishing admixed samples from reference populations. '
                 'Required when --yes-anc is used. Number of unique reference '
                 'populations must be >= --nbanc.')
        parser.add_argument('--admix-label', default='ADMIX', type=str,
            help='The population_label (in --pop-file) that identifies '
                 'admixed study-cohort samples, as opposed to reference/'
                 'ancestral population samples. Default "ADMIX", matching '
                 'this project\'s own examples and test data -- change '
                 'this if your --pop-file uses a different label (e.g. '
                 '"STUDY", "CASES", "TARGET"). When --pop-file is given, '
                 'this becomes the PRIMARY signal for identifying the study '
                 'cohort in RVAT/auto-PCA/PRS-output restriction (rather '
                 'than inferring it from phenotype validity alone, which '
                 'would incorrectly exclude an admixed sample that happens '
                 'to have a missing phenotype for an unrelated reason, or '
                 'vice versa) -- any disagreement between the two signals '
                 'is logged as a warning rather than silently resolved.')
        parser.add_argument('--yes-anc', default=False, action='store_true',
            help='Compute local ancestry from scratch using the consensus '
                 'LAI pipeline (Steps 1-8, Document 1). Requires --nbanc and '
                 '--pop-file. If --ancfile is also given it is used as the '
                 'output filename prefix.')
        parser.add_argument('--lai-genome-wide', default=False, action='store_true',
            help='When --yes-anc is used together with --chr, this makes '
                 'the LOCAL ANCESTRY INFERENCE step use the FULL genome '
                 '(ignoring --chr for that step specifically), while --chr '
                 'still restricts the downstream --mode admix/gift_admix '
                 'association analysis to the requested chromosome. '
                 'Default (off): --chr restricts both LAI and the '
                 'downstream analysis to the same chromosome, as before. '
                 'Has no effect without both --yes-anc and --chr.')
        parser.add_argument('--mode', default=None, type=str,
            choices=['admix', 'GWAS', 'gift', 'gift_admix', 'rvat', 'LAI'],
            help='Analysis mode: '
                 'admix=JasMAP ancestry+SNP Bayesian joint association; '
                 'GWAS=LMM SNP-only association; '
                 'gift=Genomic Informational Field Theory; '
                 'gift_admix=GIFT with multi-way admixture mapping; '
                 'rvat=Rare Variant Association Testing (gene-based Burden/'
                 'SKAT/SKAT-O/ACAT); '
                 'LAI=standalone local + global ancestry inference only '
                 '(requires --yes-anc; no downstream association test).')
        parser.add_argument('--chr', default=None, type=str,
            help='Chromosome to analyse (e.g. "22" or "chr22").')
        parser.add_argument('--kinship', default=0, type=int,
            help='Kinship matrix method: 0=Yang GRM, 1=unnorm, 2=two-group, 3=Chen.')
        parser.add_argument('--kinship-maf-min', default=0.01, type=float,
            help='Minimum minor allele frequency for a SNP to be included in '
                 'kinship/GRM construction (default: 0.01, matching GCTA/'
                 'PLINK convention). SNPs below this are excluded entirely -- '
                 'not just for statistical power, but because the Yang GRM '
                 'formula (--kinship 0, the default) divides each SNP by '
                 '2*p*(1-p): a rare variant near p=0 or p=1 produces a huge '
                 '(though finite, so not caught by the existing all-zero-'
                 'variance guard) weight that can dominate the whole kinship '
                 'matrix. Found via a real run where this silently inflated '
                 'trace(K) to ~10^19 instead of the expected ~n. Set to 0 to '
                 'disable filtering entirely (not recommended for --kinship 0 '
                 'on real sequencing/array data with rare variants).')
        parser.add_argument('--kinship-memory-limit-gb', default=2.0, type=float,
            help='Memory ceiling (GB) for the DENSE kinship matrix, used to '
                 'decide when to switch from the exact eigendecomposition to '
                 'the size-aware streaming dispatcher (jasmap.biobank.auto_lmm: '
                 'exact / low-rank / conjugate-gradient, chosen automatically). '
                 'Default 2.0 GB -> exact mode for n <~ 16,000 samples. '
                 'Only affects --bfile input (the only format with a chunked-'
                 'read mechanism available). Lower this to force streaming on '
                 'smaller datasets (e.g. for testing), or raise it if more '
                 'memory is available and exact mode is preferred for larger n.')
        parser.add_argument('--auto-pca', default=True, action='store_true',
            help='When --covfile is NOT provided, automatically compute '
                 'genotype-based PCA covariates and use them (plus an '
                 'intercept) for population stratification correction '
                 '(default: on). Per-individual PCA eigenvectors are the '
                 'standard mechanism for this correction, and JasMAP4 exists '
                 'specifically to analyse admixed populations, where '
                 'stratification is the dominant confounder -- running with '
                 'no correction at all (the previous default: intercept-only) '
                 'risks false-positive associations driven by ancestry '
                 'differences rather than true genetic effects. Only affects '
                 '--bfile/--geno/--vcf input when --covfile is absent; has no '
                 'effect if --covfile is given.')
        parser.add_argument('--no-auto-pca', dest='auto_pca', action='store_false',
            help='Disable automatic PCA covariate generation; fall back to '
                 'intercept-only fitting when --covfile is not provided '
                 '(the previous, unconditional default). Use this only when '
                 'stratification correction is deliberately not wanted, e.g. '
                 'for debugging or a homogeneous reference panel with no '
                 'population structure.')
        parser.add_argument('--auto-pca-k', default=10, type=int,
            help='Number of genotype-based principal components to '
                 'auto-generate as covariates when --auto-pca is active and '
                 'no --covfile is provided (default: 10, matching common '
                 'practice, e.g. PLINK --pca 10 / EIGENSTRAT typical usage).')
        parser.add_argument('--samplesize', default=None, type=str,
            help='Cases,controls for kinship methods 2/3 (e.g. "200,300").')
        parser.add_argument('--PRS', default='No',
            choices=['Yes','No'],
            help='Run consensus PRS pipeline on GIFT/admix outputs (default: No).')
        parser.add_argument('--sumstats', default=None, type=str,
            help='GWAS summary statistics file for PRS (SNP A1 BETA P columns).')
        parser.add_argument('--prs-methods', default='ridge,lasso', type=str,
            help='Comma-separated PRS methods: ridge,lasso,ct (default: '
                 'ridge,lasso). Lasso is meaningfully slower than Ridge on '
                 'large SNP counts (coordinate descent scales with feature '
                 'count, unlike Ridge which can use a kernel trick when '
                 'n_samples << n_snps) -- use --prs-methods ridge (or '
                 'ridge,ct) to skip it entirely when speed matters more '
                 'than Lasso\'s specific sparse-selection behaviour.')
        parser.add_argument('--prs-max-snps', default=20000, type=int,
            help='Pre-filter Ridge/Lasso PRS fitting to at most this many '
                 'SNPs, kept by univariate |correlation| with phenotype, '
                 'before the expensive CV-based fitting step (default: '
                 '20000). Found to matter a lot in practice: LassoCV\'s '
                 'coordinate descent scales roughly linearly with feature '
                 'count, so fitting on hundreds of thousands of SNPs '
                 'directly can take tens of minutes. Only applied when the '
                 'SNP count actually exceeds this value (no effect, no '
                 'wasted computation, on smaller panels). Set to 0 to '
                 'disable pre-filtering entirely and fit on every SNP.')
        parser.add_argument('--consgwas', default=False, action='store_true',
            help='Run the 8-step Consensus GWAS/RVAT pipeline after --mode completes. '
                 'Integrates results from the primary --mode with additional internal '
                 'tools (OLS, Firth-approx, burden, SKAT-approx) via IVW meta-analysis '
                 '(common variants) or ACAT/Cauchy combination (windows/genes). '
                 'Outputs land in {--out}/consensus_gwas/. '
                 'Compatible with all --mode values.')
        parser.add_argument('--consgwas-perms', default=50, type=int,
            help='Permutations for empirical type I error in --consgwas Step 6 '
                 '(0 = skip calibration; default: 50).')
        parser.add_argument('--consgwas-sig', default=None, type=float,
            help='Significance threshold override for --consgwas '
                 '(default: auto - 5e-8 for SNP, 1e-6 for windows).')
        parser.add_argument('--var-annot', default=None, type=str,
            help='Variant annotation TSV for RVAT gene-level plots '
                 '(columns: SNP CHR POS GENE MAF CONSEQUENCE IS_RARE). '
                 'Generated by make_test_data.py as variant_annotation.tsv.')
        parser.add_argument('--gene-annot', default=None, type=str,
            help='Gene annotation BED or GFF3 file (optional; used for '
                 'gene-level labelling in --consgwas RVAT plots).')
        parser.add_argument('--annot', action='store_true',
            help='Automatically construct a gene-annotation BED file '
                 'scoped to this study\'s SNPs (via a cached or freshly '
                 'downloaded reference annotation) and use it for --mode '
                 'rvat gene-based tests, instead of the default 50-SNP '
                 'sliding-window proxies. Ignored if --gene-annot or '
                 '--var-annot is already given explicitly. Requires '
                 'internet access on first use for a given --genome-build '
                 '(cached afterward); if unavailable, falls back to the '
                 'default sliding-window behavior with a clear warning '
                 'rather than failing the run.')
        parser.add_argument('--genome-build', default='hg38', choices=['hg19', 'hg38'],
            help='Genome build the genotype positions use (default: hg38). '
                 'Only relevant with --annot. Getting this wrong silently '
                 'assigns SNPs to the wrong genes -- verify against how '
                 'your genotypes were actually called, don\'t guess.')
        parser.add_argument('--maf-max', default=0.01, type=float,
            help='MAF threshold defining rare variants for --mode rvat '
                 '(default: 0.01 = 1%%). Variants with MAF <= this value '
                 'are included in gene-based tests.')
        parser.add_argument('--rvat-methods', default='burden,skat,skato,acat_v',
            type=str,
            help='Comma-separated RVAT tests to run with --mode rvat '
                 '(default: burden,skat,skato,acat_v).')
        parser.add_argument('--rvat-min-variants', default=2, type=int,
            help='Minimum number of rare variants per gene to test '
                 '(default: 2). Genes with fewer are skipped.')
        parser.add_argument('--plots', default=True, action='store_true',
            help='Generate publication-quality figures (default: on).')
        parser.add_argument('--no-plots', dest='plots', action='store_false',
            help='Disable plot generation.')
        parser.add_argument('--plot-dpi', default=300, type=int,
            help='Resolution for all output figures in dots per inch. '
                 '300 = publication quality (default), 150 = screen/draft, '
                 '600 = high-resolution submission.')
        parser.add_argument('--gift-n-components', default=20, type=int,
            help='PCA components for GIFT field compression (default: 20).')
        parser.add_argument('--gift-lambda', default=1e-3, type=float,
            help='L2 regularisation for GIFT ridge regression (default: 1e-3).')
        parser.add_argument('--gift-mi-chunk', default=512, type=int,
            help='SNPs per MI computation chunk for GIFT (default: 512).')
        parser.add_argument('--gift-max-snps', default=5000, type=int,
            help='GIFT computes a dense (n_snps x n_snps) mutual-information '
                 'matrix and field propagator (the latter also needing a '
                 'matrix exponential) -- O(n_snps^2) memory and O(n_snps^3) '
                 'compute, fundamentally different scaling from GWAS/RVAT/PRS '
                 '(which scale with sample count). Found via a real run: at '
                 '338,320 SNPs this crashed outright needing 853 GiB, not just '
                 'ran slowly. Above this cap (default 5000), a quick '
                 'univariate pre-filter keeps only the SNPs most correlated '
                 'with phenotype (same method already used for PRS pre-'
                 'filtering) before GIFT runs. Set to 0 to disable (not '
                 'recommended above a few thousand SNPs -- consider --chr to '
                 'restrict to one chromosome instead, which stays exact '
                 'rather than pre-filtering).')
        parser.add_argument('--gift-ancestry-mode', default='stratified',
            choices=['stratified','covariate','interaction','all'],
            help='GIFT ancestry coupling mode for gift_admix (default: stratified).')

        #create global variable options
        self.args = parser.parse_args()
        self.start_time = time.time()

        # Apply user-specified plot DPI globally before any plotting occurs
        try:
            from jasmap.plotting.plots import set_plot_dpi
            set_plot_dpi(self.args.plot_dpi)
        except Exception:
            pass

        # Pop labels — validate count against --nbanc immediately
        self._pop_labels = (
            self.args.pop_labels.split(",")
            if self.args.pop_labels else None
        )
        # Conflict #2: --pop-labels count must equal --nbanc
        if self._pop_labels is not None and self.args.nbanc is not None:
            if len(self._pop_labels) != self.args.nbanc:
                print(f"Error: --pop-labels has {len(self._pop_labels)} label(s) "
                      f"but --nbanc={self.args.nbanc}. "
                      f"Supply exactly --nbanc labels (e.g. --pop-labels AFR,EUR,SAN "
                      f"with --nbanc 3).")
                sys.exit(1)
        # PRS methods
        self._prs_methods = tuple(
            m.strip() for m in self.args.prs_methods.split(",")
        )

        if self.args.out is not None:
            self.outfolder = self.args.out
        else:
            self.outfolder = os.path.join(os.getcwd(), "jasmap_results")
        # Bug fix (#12): when --out was supplied by the user, self.outfolder
        # was set with NO trailing path separator (unlike the default case,
        # which explicitly appended "/"), but every output file is written
        # via plain string concatenation (`self.outfolder + "Jasmap...assoc"`,
        # no os.path.join). This silently produced garbled filenames like
        # "out_admix_genoJasmap.Joint.assoc" sitting in the *parent* of the
        # intended output directory -- the requested folder would exist
        # (os.makedirs succeeded) but end up completely empty, with all
        # results actually written one level up under a mangled name. Only
        # found by actually inspecting the output of an end-to-end run, not
        # by code review or by the run's own "success" exit status. Fixed
        # once, here, for both the user-supplied and default cases, rather
        # than at every individual open() call site.
        self.outfolder = os.path.join(self.outfolder, "")  # ensures exactly one trailing "/"
        if not os.path.exists(self.outfolder):
            os.makedirs(self.outfolder)

        # --out-prefix: applied AFTER makedirs (which must operate on the
        # real directory, not a directory+prefix string) by appending it
        # directly onto self.outfolder. Every existing "self.outfolder +
        # filename" call site across this file (there are many) picks up
        # the prefix automatically this way, with no changes needed at
        # any of them individually.
        #
        # Defaults to "JasMAP" (rather than being mandatory, as an earlier
        # version of this required) so a run never stops just because the
        # user omitted --out-prefix -- `or "JasMAP"` is a further, defensive
        # fallback in case an explicitly-empty string ever gets passed.
        #
        # When --chr is also given, its value is automatically folded into
        # the prefix too -- whether the prefix came from the user or from
        # this default -- since that's exactly the scenario --out-prefix
        # exists for: multiple parallel runs (one per chromosome) sharing
        # the same --out folder or the same underlying genotype file, which
        # otherwise cannot be told apart from their filenames alone. The
        # "chr" prefix is stripped case-insensitively first (matching
        # _resolve_chr_indices_generic()'s own normalisation), so --chr 22
        # and --chr chr22 both produce the same, single "_chr22" tag rather
        # than one of them awkwardly duplicating it.
        _prefix = self.args.out_prefix or "JasMAP"
        _chr_tag = ""
        if self.args.chr is not None:
            _chr_clean = str(self.args.chr).strip()
            if _chr_clean.lower().startswith("chr"):
                _chr_clean = _chr_clean[3:]
            _chr_tag = f"_chr{_chr_clean}"
        self.outfolder = self.outfolder + _prefix + _chr_tag + "_"

        # Stale-output detection: warn if this folder already holds results from
        # a previous run. Re-using an --out folder across modes was silently
        # mixing analyses (stale plots, and stale tools entering --consgwas).
        self._stale_outputs = []
        try:
            import glob as _glob
            # Bug fix: os.path.join(self.outfolder, pattern) is wrong once
            # --out-prefix is set -- self.outfolder then ends in the
            # prefix itself (e.g. "results/chr22_run_"), not a bare
            # directory, so os.path.join would insert an extra "/" that
            # doesn't match how output files are actually named (plain
            # string concatenation, self.outfolder + filename, everywhere
            # else in this file). Plain concatenation here too keeps this
            # glob consistent with where files are actually written,
            # prefixed or not.
            self._stale_outputs = sorted(
                [os.path.basename(_p) for _p in
                 _glob.glob(self.outfolder + "Jasmap.*.assoc")] +
                [os.path.basename(_p) for _p in
                 _glob.glob(self.outfolder + "plot_*.png")])
        except Exception:
            pass

        # Bug fix: logging.FileHandler with no explicit encoding opens the log
        # file using the platform default codec -- cp1252 on most Windows
        # installs. Any non-ASCII character in a log message (lambda, em-dash,
        # >=, etc.) then raises UnicodeEncodeError inside StreamHandler.emit,
        # producing a "--- Logging error ---" traceback for EVERY affected line
        # while the analysis itself continues. Reconfiguring sys.stderr (done at
        # the top of this file) only fixes the CONSOLE handler, not this file
        # handler -- both need to be UTF-8.
        fh = logging.FileHandler(self.outfolder + self.logFile, mode='w',
                                 encoding='utf-8')
        logger.addHandler(fh)
        try:
            defaults = vars(parser.parse_args(''))
            opts = vars(self.args)
            non_defaults = [x for x in opts.keys() if opts[x] != defaults[x]]
            header = JHEAD
            header += "Call: \n"
            header += './JasMAP4.py \\\n'
            options = ['--' + x.replace('_', '-') + ' ' + str(opts[x]) + ' \\' for x in non_defaults]
            header += '\n'.join(options).replace('True', '').replace('False', '')
            header = header[0:-1] + '\n'
            logger.info(header)
            logger.info('Beginning analysis at {T}'.format(T=time.ctime()))

            # Surface stale outputs from a previous run in this folder.
            if getattr(self, '_stale_outputs', None):
                _n = len(self._stale_outputs)
                _show = ", ".join(self._stale_outputs[:6])
                if _n > 6:
                    _show += f", ... (+{_n-6} more)"
                logger.info(
                    f'[WARNING] Output folder "{self.outfolder}" already contains '
                    f'{_n} result file(s) from a previous run: {_show}\n'
                    f'          Files not overwritten by this run will remain and '
                    f'may be mistaken for current results. Stale JasMAP.*.assoc '
                    f'files are NO LONGER fed into --consgwas (fixed), but using a '
                    f'fresh --out folder per analysis is strongly recommended.\n')

            # Global requirement: EVERY mode and EVERY genotype input path
            # (--bfile/--vcf/--geno) unconditionally calls
            # ps.read_pheno_file(self.args.pheno) later in the pipeline --
            # there is no mode or input type that can proceed without it.
            # Bug fix: this was never validated upfront. The --bfile branch
            # had NO --pheno check at all; the --vcf/--geno branches only
            # checked "pheno OR covfile" (satisfied by covfile alone, which
            # does not prevent the later unconditional read_pheno_file(None)
            # call from crashing anyway). In practice this meant a missing
            # --pheno surfaced as a bare TypeError deep inside
            # read_bed_file()/read_geno_file()/vcf_to_dataframe() -- AFTER
            # the (potentially very large) genotype file had already been
            # fully read. On a real HapMap3-scale .bed file this wasted
            # over 500 seconds of file I/O before failing on something
            # checkable in a fraction of a second. Checked here, before any
            # file-existence checks or genotype loading happens at all.
            #
            # --bfile-specific fallback: a standard PLINK .fam file's 6th
            # (last) column IS a phenotype (1=control, 2=case, or -9/0 for
            # missing) -- the exact same convention JasMAP4 already expects
            # internally. ps.read_pheno_file() already handles the generic
            # "FID IID ... PHENO" whitespace-delimited format (reading the
            # 2nd column as sample ID and the LAST column as phenotype), so
            # a .fam file needs no special-case parser -- pointing it
            # directly at the .fam path works unmodified. This fallback is
            # --bfile-only: VCF has no standard phenotype field at all, and
            # EIGENSTRAT's .ind file holds sex/population labels, not
            # phenotype, so --pheno remains a hard requirement for --vcf
            # and --geno.
            if self.args.pheno is None and self.args.bfile is not None:
                fam_path = self.args.bfile + ".fam"
                logger.info(
                    f'[INFO] --pheno not specified for --bfile input; '
                    f'using the 6th column (PHENOTYPE) of {fam_path} as '
                    f'the phenotype (standard PLINK convention: 1=control, '
                    f'2=case). Pass --pheno explicitly to use a separate '
                    f'phenotype file instead.\n')
                self.args.pheno = fam_path
            elif self.args.pheno is None:
                logger.info('Error: --pheno is required (every mode needs a '
                            'phenotype file). Provide --pheno path/to/pheno.txt. '
                            '(The .fam-column fallback only applies to --bfile '
                            'input; --vcf and --geno have no equivalent '
                            'phenotype field and always need --pheno explicitly.)\n')
                logger.info('JasMAP4.py -h describes options.\n')
                self.terminate(self.outfolder + self.logFile)
                sys.exit(1)

            # Default --mode to GWAS (the standard kinship-corrected LMM
            # association test) when the user doesn't specify one, instead
            # of erroring out. GWAS is the natural default: it is the only
            # mode requiring nothing beyond genotypes + phenotype (no
            # --nbanc/--ancfile like admix, no --gene-annot like rvat), so
            # it is the one mode guaranteed to be runnable with whatever
            # input the user has already provided. The choice is always
            # logged explicitly -- never a silent default -- consistent
            # with every other fallback/default decision in this file.
            if self.args.mode is None:
                logger.info('[INFO] --mode not specified; defaulting to GWAS '
                            '(standard kinship-corrected LMM association). '
                            'Pass --mode explicitly (admix, GWAS, gift, '
                            'gift_admix, or rvat) to choose a different '
                            'analysis.\n')
                self.args.mode = 'GWAS'

            if self.args.mode is not None:
                if self.args.mode == "admix":
                    if self.args.nbanc is None:
                        logger.info('Must provide the number of ancestral population with --nbanc \n')
                        logger.info('JasMAP4.py -h describes options.\n')
                        self.terminate(self.outfolder + self.logFile)
                        sys.exit(1)
                    else:
                        self.no_of_ancs = self.args.nbanc
                    # If --yes-anc: LAI will be computed; ancfile check is deferred
                    if not self.args.yes_anc:
                        if self.args.ancfile is None:
                            logger.info('Must provide --ancfile OR use --yes-anc to compute local ancestry.\n')
                            logger.info('JasMAP4.py -h describes options.\n')
                            self.terminate(self.outfolder + self.logFile)
                            sys.exit(1)
                        else:
                            self.lanc_file = self.args.ancfile
                            # Conflict #5: verify ancfile exists and has >= nbanc ancestry columns
                            try:
                                with open(self.lanc_file) as _f:
                                    first_line = _f.readline().strip()
                                # Space-delimited: 2*n_samples integers; compact: string of digits
                                if " " in first_line:
                                    max_anc = max(int(x) for x in first_line.split())
                                else:
                                    max_anc = max(int(c) for c in first_line if c.isdigit())
                                if max_anc < self.args.nbanc:
                                    logger.info(
                                        f'[WARNING] --ancfile appears to contain '
                                        f'{max_anc} ancestr{"y" if max_anc==1 else "ies"} '
                                        f'(max label found = {max_anc}) but '
                                        f'--nbanc={self.args.nbanc}. '
                                        f'Results will be wrong if the file has '
                                        f'fewer ancestral components than --nbanc.\n')
                            except Exception:
                                pass  # check is advisory; file-not-found is caught later
                    else:
                        # Defer ancfile assignment — will be set after LAI computation
                        self.lanc_file = None
                    self.lanc_labels_list = [i + 1 for i in range(self.args.nbanc)]

                elif self.args.mode == "GWAS":
                    self.no_of_ancs = 0
                    self.lanc_file = None
                    self.lanc_labels_list = []
                    # Conflict #1: --yes-anc is meaningless for GWAS mode
                    if self.args.yes_anc:
                        logger.info('[WARNING] --yes-anc has no effect with '
                                    '--mode GWAS (local ancestry is not used '
                                    'in GWAS-mode association). '
                                    'Use --mode admix or --mode gift_admix if '
                                    'you want ancestry-informed analysis.\n')
                    # Conflict #6: --gift-* options are silently ignored for GWAS
                    if self.args.gift_n_components != 20 or \
                       self.args.gift_lambda != 1e-3:
                        logger.info('[WARNING] --gift-n-components / '
                                    '--gift-lambda have no effect with --mode GWAS.\n')

                elif self.args.mode == "gift":
                    # GIFT mode: no ancestry file required
                    self.no_of_ancs = 0
                    self.lanc_file = None
                    self.lanc_labels_list = []
                    # Conflict #1: --yes-anc is meaningless for gift mode
                    if self.args.yes_anc:
                        logger.info('[WARNING] --yes-anc has no effect with '
                                    '--mode gift. Use --mode gift_admix to '
                                    'combine GIFT with ancestry inference.\n')
                    # Conflict #6: --gift-ancestry-mode irrelevant for gift
                    if self.args.gift_ancestry_mode != "stratified":
                        logger.info('[WARNING] --gift-ancestry-mode has no '
                                    'effect with --mode gift (only applies to '
                                    '--mode gift_admix).\n')

                elif self.args.mode == "gift_admix":
                    # GIFT + Admixture mode: needs K
                    if self.args.nbanc is None:
                        logger.info('Error: --mode gift_admix requires --nbanc.\n')
                        self.terminate(self.outfolder + self.logFile)
                        sys.exit(1)
                    self.no_of_ancs = self.args.nbanc
                    self.lanc_labels_list = [i + 1 for i in range(self.args.nbanc)]
                    if not self.args.yes_anc:
                        if self.args.ancfile is not None:
                            self.lanc_file = self.args.ancfile
                        else:
                            # No ancfile — will run without LAI weighting
                            self.lanc_file = None
                            logger.info('[GIFT-ADMIX] No --ancfile; will use global-only ancestry. '
                                        'Use --yes-anc to compute local ancestry automatically.\n')
                    else:
                        self.lanc_file = None  # will be set after LAI computation

                elif self.args.mode == "rvat":
                    # Rare Variant Association Testing — no ancestry required
                    self.no_of_ancs = 0
                    self.lanc_file = None
                    self.lanc_labels_list = []
                    # Advisory warnings for incompatible options
                    if self.args.yes_anc:
                        logger.info('[WARNING] --yes-anc has no effect with '
                                    '--mode rvat. Local ancestry is not used '
                                    'in gene-based rare variant tests.\n')
                    if self.args.nbanc is not None:
                        logger.info('[WARNING] --nbanc has no effect with '
                                    '--mode rvat.\n')
                    # Recommendation: ensure annotation is provided
                    if self.args.var_annot is None and self.args.gene_annot is None:
                        logger.info('[INFO] --mode rvat works best with '
                                    '--var-annot variant_annotation.tsv and/or '
                                    '--gene-annot gene_annotation.bed. '
                                    'Without annotation, 50-SNP sliding windows '
                                    'will be used as gene proxies.\n')

                elif self.args.mode == "LAI":
                    # Standalone local + global ancestry inference only --
                    # runs the SAME consensus LAI pipeline --yes-anc already
                    # triggers as a prerequisite step for --mode admix/
                    # gift_admix, but stops there instead of continuing into
                    # a downstream association test. Useful when a user only
                    # wants ancestry estimates (global proportions, local
                    # ancestry tracts, karyograms) without committing to a
                    # specific association analysis yet.
                    self.no_of_ancs = self.args.nbanc if self.args.nbanc else 0
                    self.lanc_file = None
                    self.lanc_labels_list = ([i + 1 for i in range(self.args.nbanc)]
                                             if self.args.nbanc else [])
                    if not self.args.yes_anc:
                        logger.info('Error: --mode LAI requires --yes-anc '
                                    '(there is nothing else for this mode to '
                                    'compute -- it exists specifically to run '
                                    'the consensus local/global ancestry '
                                    'pipeline standalone).\n')
                        self.terminate(self.outfolder + self.logFile)
                        sys.exit(1)
                    if self.args.nbanc is None:
                        logger.info('Error: --mode LAI requires --nbanc (the '
                                    'number of ancestral populations).\n')
                        self.terminate(self.outfolder + self.logFile)
                        sys.exit(1)
                    # Advisory warnings for flags that only make sense once a
                    # downstream association/PRS step actually runs -- none
                    # of which happens for --mode LAI.
                    if self.args.PRS == "Yes":
                        logger.info('[WARNING] --PRS has no effect with '
                                    '--mode LAI. PRS needs an association '
                                    'test or a downstream mode to score '
                                    'against -- use --mode admix/gift_admix/'
                                    'GWAS/rvat with --PRS Yes instead.\n')
                    if self.args.consgwas:
                        logger.info('[WARNING] --consgwas has no effect with '
                                    '--mode LAI. There is no association '
                                    'output for it to build a consensus '
                                    'from.\n')
                    if self.args.gift_n_components != 20 or \
                       self.args.gift_lambda != 1e-3:
                        logger.info('[WARNING] --gift-n-components / '
                                    '--gift-lambda have no effect with '
                                    '--mode LAI.\n')
                    if self.args.auto_pca_k != 10:
                        logger.info('[WARNING] --auto-pca-k has no effect '
                                    'with --mode LAI. Genotype-PCA '
                                    'covariates are only computed for a '
                                    'downstream association test/PRS step, '
                                    'neither of which runs in this mode.\n')

                else:
                    logger.info('Error: --mode must be admix, GWAS, gift, gift_admix, rvat, or LAI (got "%s").\n'
                                % self.args.mode)
                    logger.info('JasMAP4.py -h describes options.\n')
                    self.terminate(self.outfolder + self.logFile)
                    sys.exit(1)
            else:
                logger.info('Error: no analysis mode selected.\n')
                logger.info('JasMAP4.py -h describes options.\n')
                self.terminate(self.outfolder+self.logFile)
                sys.exit()

            # --yes-anc validation: require --nbanc and --pop-file
            if self.args.yes_anc:
                if self.args.nbanc is None:
                    logger.info('Error: --yes-anc requires --nbanc.\n')
                    self.terminate(self.outfolder + self.logFile)
                    sys.exit(1)
                if self.args.pop_file is None:
                    logger.info('Error: --yes-anc requires --pop-file '
                                '(two-column file: sample_ID, population_label).\n')
                    self.terminate(self.outfolder + self.logFile)
                    sys.exit(1)
                if not os.path.exists(self.args.pop_file):
                    logger.info(f'Error: --pop-file "{self.args.pop_file}" not found.\n')
                    self.terminate(self.outfolder + self.logFile)
                    sys.exit(1)
                # Validate reference-population count >= nbanc
                try:
                    pf = pd.read_csv(self.args.pop_file, sep=r"\s+", header=None,
                                     names=["ID","POP"], engine="python")
                    ref_pops = pf["POP"].nunique()
                    if ref_pops < self.args.nbanc:
                        logger.info(f'Error: --pop-file has {ref_pops} unique populations '
                                    f'but --nbanc={self.args.nbanc}. '
                                    f'Number of reference populations must be >= nbanc.\n')
                        self.terminate(self.outfolder + self.logFile)
                        sys.exit(1)
                    logger.info(f'[LAI] pop-file validated: {ref_pops} populations, '
                                f'K={self.args.nbanc}\n')
                    # Store per-sample labels now so PCA plots can use them
                    # to distinguish ADMIX from reference populations.
                    self._sample_pop_labels_for_lai = dict(
                        zip(pf["ID"].astype(str), pf["POP"].astype(str)))
                except Exception as e:
                    logger.info(f'Error reading --pop-file: {e}\n')
                    self.terminate(self.outfolder + self.logFile)
                    sys.exit(1)

            # --admix-label validation: whenever --pop-file is given (with
            # or without --yes-anc -- --pop-file can be used on its own,
            # alongside a pre-computed --ancfile, purely for the study-
            # cohort identification _get_study_cohort_mask() does), stop
            # immediately if --admix-label's value isn't actually one of
            # the labels present in the file, rather than silently
            # proceeding with a mask that matches zero samples. Found via
            # a real run with --admix-label ADZ (a typo/mismatch for the
            # file's actual "ADMIX" label) that crashed deep inside
            # run_joint_analysis() with a confusing IndexError, far from
            # the actual, easily-checked-upfront cause.
            if self.args.pop_file and os.path.exists(self.args.pop_file):
                try:
                    pf_labels = pd.read_csv(self.args.pop_file, sep=r"\s+", header=None,
                                            names=["ID","POP"], engine="python")
                    unique_labels = sorted(pf_labels["POP"].astype(str).unique().tolist())
                    if self.args.admix_label not in unique_labels:
                        logger.info(
                            f'Error: --admix-label {self.args.admix_label!r} does not '
                            f'match any population label actually present in --pop-file '
                            f'"{self.args.pop_file}". Labels found there: {unique_labels}. '
                            f'Pass --admix-label with one of these values (matching '
                            f'whichever label identifies your admixed study-cohort '
                            f'samples), or fix a typo in --pop-file/--admix-label.\n')
                        self.terminate(self.outfolder + self.logFile)
                        sys.exit(1)
                except SystemExit:
                    raise
                except Exception as e:
                    logger.info(f'Error validating --admix-label against --pop-file: {e}\n')
                    self.terminate(self.outfolder + self.logFile)
                    sys.exit(1)
            elif self.args.admix_label != 'ADMIX':
                # Bug fix: --admix-label only ever does anything by being
                # matched against --pop-file's own population labels
                # (_get_study_cohort_mask()'s first priority) -- without
                # --pop-file, it is silently, completely inert, and the
                # study-cohort/reference-population distinction falls
                # back to phenotype validity alone. Found via a real run
                # (--admix-label SAC, no --pop-file at all): with zero
                # warning of any kind, the entire analysis silently ran
                # on the FULL, pooled cohort -- all reference populations
                # plus the admixed samples together, not just the
                # intended study cohort -- because the user's own .pheno
                # file, it turned out, has non-missing phenotype values
                # for the reference samples too (phenotype validity alone
                # can't tell them apart from real study samples without
                # --pop-file's explicit labels). The symptoms were
                # severe but easy to miss as a configuration problem:
                # zero excluded samples logged anywhere, the full sample
                # count everywhere in the log, and a catastrophic
                # lambda_GC (674) from testing five genetically distinct
                # ancestral populations pooled with the admixed cohort as
                # if they were one homogeneous sample. An explicit,
                # non-default --admix-label is a clear, unambiguous
                # signal the user intended this feature to do something
                # -- so this warns immediately, rather than letting a
                # 15+ minute real-data run complete and only reveal the
                # misconfiguration through implausible downstream
                # results.
                logger.info(
                    f"[WARNING] --admix-label {self.args.admix_label!r} was given, but "
                    f"--pop-file was not -- --admix-label has NO EFFECT without --pop-file "
                    f"(it works by matching against --pop-file's own population labels). "
                    f"The study-cohort/reference-population distinction will fall back to "
                    f"phenotype validity alone, which only works correctly if your --pheno "
                    f"file itself gives reference/ancestral population samples a missing "
                    f"phenotype -- verify this is actually the case for your data, or pass "
                    f"--pop-file to make the distinction explicit and reliable.\n")

            # --pop-file is genuinely OPTIONAL for --mode GWAS/gift/rvat (unlike
            # admix/gift_admix, which require it for --yes-anc). When it's
            # absent for one of these three modes, --pheno's own missing-value
            # (-9) coding becomes the ONLY signal distinguishing the study
            # cohort from any reference/ancestral population samples also
            # present in the genotype file -- this already works correctly
            # (every mode's own exclusion/imputation logic falls back to
            # phenotype validity on its own, confirmed directly), but the
            # user gets no confirmation that this is what's actually
            # happening, or a nudge to double-check it's coded correctly,
            # until results look implausible. Found via a real, ~30-minute
            # run where --pheno's -9 coding for reference samples turned out
            # to be wrong, and the result was a fully silent failure -- no
            # warning, no error, just an entire cohort of 5 distinct
            # ancestral populations pooled in with the admixed study
            # samples as if undifferentiated. This one, upfront message
            # gives the same visibility --pop-file-based runs already get
            # (their own "N sample(s) excluded"/"N study-cohort and M
            # reference-population" messages), for the --pheno-only path too.
            if self.args.mode in ("GWAS", "gift", "rvat") and not self.args.pop_file:
                logger.info(
                    f"[INFO] --mode {self.args.mode}: no --pop-file given -- "
                    f"relying entirely on --pheno's own missing-value (-9) "
                    f"coding to identify the study cohort. This is fully "
                    f"supported and already validated. If your genotype file "
                    f"also includes reference/ancestral population samples "
                    f"(not part of the actual study), double-check they are "
                    f"correctly coded -9 in --pheno before trusting the "
                    f"results -- an incorrectly-coded reference sample will "
                    f"be silently included with no warning otherwise. The "
                    f"check_pheno_coding.py utility can verify this in "
                    f"seconds against --pop-file if one is available, even "
                    f"if you don't intend to pass it to this run.\n")

            self.AM_eff_no_tests = float(self.args.ntest)

            # Conflict #8: warn if multiple genotype inputs are supplied
            n_inputs = sum(1 for x in [self.args.bfile, self.args.vcf,
                                        self.args.geno] if x is not None)
            if n_inputs > 1:
                active = ("--bfile" if self.args.bfile else
                          "--vcf"   if self.args.vcf   else "--geno")
                logger.info(f'[WARNING] More than one genotype input supplied. '
                            f'Priority order is --bfile > --vcf > --geno. '
                            f'Using {active} and ignoring the others.\n')

            if self.args.bfile is not None:
                self.bed_files = [self.args.bfile + ".bed", self.args.bfile + ".bim", self.args.bfile + ".fam"]
                self.check_files(self.bed_files)
                # --pheno's presence is already guaranteed by the global
                # check above; still verify the FILE actually exists here,
                # matching the --vcf/--geno branches below (this branch
                # previously had no pheno file-existence check at all).
                self.check_files([self.args.pheno])
                if self.args.covfile is not None:
                    self.bed = [self.args.covfile]
                    self.check_files(self.bed)
                    self.covar_file = self.args.covfile
                self.geno_file = self.args.bfile
            elif self.args.vcf is not None:
                self.bed = [self.args.vcf]
                self.check_files(self.bed)
                if self.args.covfile is not None:
                    self.bed = [self.args.covfile]
                    self.check_files(self.bed)
                    # Bug fix (#10): self.covar_file was never assigned on
                    # this path (only the --bfile branch set it), so
                    # read_covar_file() would raise AttributeError the moment
                    # it tried to use it -- found via an actual end-to-end run.
                    self.covar_file = self.args.covfile
                # Simplified: --pheno's presence is already guaranteed by the
                # global check above (previously this branch only required
                # "pheno OR covfile", which covfile alone could satisfy while
                # leaving the later unconditional read_pheno_file(None) call
                # to crash anyway). Just verify the file exists.
                self.check_files([self.args.pheno])
                self.geno_file = self.args.vcf
            elif self.args.geno is not None:
                # Bug fix (#9): --geno's value is the genotype data file
                # itself (matching the convention used throughout the rest of
                # this script -- read_geno_file() reads directly from
                # self.geno_file with no extension appended, while only
                # .snp/.ind get suffixed), but this check was verifying that
                # a file literally named exactly `--geno`'s value exists --
                # which WAS already correct for the geno data file itself,
                # but never checked that the accompanying .snp/.ind files
                # exist too, so a missing .snp/.ind would instead surface as
                # a confusing crash deep inside read_geno_file(). Discovered
                # via an actual end-to-end run, not just code review.
                self.bed = [self.args.geno, self.args.geno + ".snp", self.args.geno + ".ind"]
                self.check_files(self.bed)
                if self.args.covfile is not None:
                    self.bed = [self.args.covfile]
                    self.check_files(self.bed)
                    # Bug fix (#10): same missing self.covar_file assignment
                    # as the --vcf branch above.
                    self.covar_file = self.args.covfile
                # Simplified: same reasoning as the --vcf branch above.
                self.check_files([self.args.pheno])
                self.geno_file = self.args.geno
            else:
                logger.info('Error: Raw data for the association must be provided.\n')
                self.terminate(self.outfolder + self.logFile)
                sys.exit()
            if self.args.kinship in [2,3]:
                if self.args.samplesize is None:
                    logger.info('Error: --kinship 2 or 3 requires '
                                '--samplesize N_cases,N_controls '
                                '(e.g. --samplesize 200,300).\n')
                    self.terminate(self.outfolder + self.logFile)
                    sys.exit(1)
                else:
                    self.n1 = int(self.args.samplesize.split(',')[0])
                    self.n2 = int(self.args.samplesize.split(',')[1])
            else:
                self.n1 = 0
                self.n2 = 0
                if self.args.kinship not in [0, 1]:
                    logger.info(f'[WARNING] Unrecognised --kinship value '
                                f'{self.args.kinship}; defaulting to 0.\n')
                self.args.kinship = max(0, min(1, self.args.kinship))
        except Exception as ex:
            logger.info(ex)
            sys.exit(1)

    def _tped_path(self):
        """
        Returns the path of the intermediate .tped file this run should
        write to / read from, given self.args.chr's CURRENT value.

        Why this exists
        -----------------
        Every one of the ~9 places that create or read this intermediate
        file previously built its path directly from the INPUT genotype
        file (self.geno_file + ".tped", or self.args.vcf + ".tped") --
        meaning two SEPARATE jasmap processes analysing different
        chromosomes of the SAME underlying genotype file (a natural way
        to parallelise a whole-genome analysis, one process per
        chromosome) would collide on the exact same intermediate file
        path, each overwriting the other mid-run. Found via a real user
        request describing exactly this parallel-chromosome workflow.

        Placing this file inside self.outfolder instead (unique per run,
        assuming the already-recommended practice of a fresh --out per
        run) resolves the common case outright. Naming it with the
        CURRENTLY-active --chr value on top of that (rather than a fixed
        name within a run) is what --lai-genome-wide specifically needs:
        that flag makes _run_lai_pipeline() temporarily clear
        self.args.chr to read the full genome for local ancestry
        inference, then restore it before the main analysis re-reads
        genotypes at the requested chromosome -- since this method reads
        self.args.chr's value fresh every time it's called, those two
        reads, happening sequentially within the same run, naturally
        land on two DIFFERENT filenames (tmp_all.tped vs tmp_chr22.tped)
        instead of one silently overwriting the other's intermediate
        file with a different scope than intended.

        Every file this method's path points to is tracked in
        self._temp_files_created and deleted after a successful run --
        see jasmap()'s cleanup step.
        """
        chr_tag = f"chr{self.args.chr}" if self.args.chr is not None else "all"
        path = self.outfolder + f"tmp_{chr_tag}.tped"
        if not hasattr(self, "_temp_files_created"):
            self._temp_files_created = set()
        self._temp_files_created.add(path)
        return path


class LMM:
    """
         The LMM class is from pylmm by Nicholas A. Furlotte, used and modified under the terms of the GNU Affero General Public License
         # pylmm is a python-based linear mixed-model solver with
         # applications to GWAS Copyright (C) 2015 Nicholas A. Furlotte
         # (nick.furlotte@gmail.com)

         #    This program is free software: you can redistribute it and/or modify
         #    it under the terms of the GNU Affero General Public License as published by
         #    the Free Software Foundation, either version 3 of the License, or
         #    (at your option) any later version.

         #    This program is distributed in the hope that it will be useful,
         #    but WITHOUT ANY WARRANTY; without even the implied warranty of
         #    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
         #    GNU General Public License for more details.

         #    You should have received a copy of the GNU General Public License
         #    along with this program.  If not, see <http://www.gnu.org/licenses/>.

         This is a simple version of EMMA/fastLMM.
         The main purpose of this module is to take a phenotype vector (Y), a set of covariates (X) and a kinship matrix (K)
         and to optimize this model by finding the maximum-likelihood estimates for the model parameters.
         There are three model parameters: heritability (h), covariate coefficients (beta) and the total
         phenotypic variance (sigma).
         Heritability as defined here is the proportion of the total variance (sigma) that is attributed to
         the kinship matrix.

         For simplicity, we assume that everything being input is a numpy array.
         If this is not the case, the module may throw an error as conversion from list to numpy array
         is not done consistently.

    """

    def __init__(self, Y, K, Kva=None, Kve=None, X0=None, verbose=False):

        """
            The constructor takes a phenotype vector or array of size n.
            It takes a kinship matrix of size n x n.  Kva and Kve can be computed as Kva,Kve = linalg.eigh(K) and cached.
            If they are not provided, the constructor will calculate them.
            X0 is an optional covariate matrix of size n x q, where there are q covariates.
            When this parameter is not provided, the constructor will set X0 to an n x 1 matrix of all ones to represent a mean effect.
        """
        # Precision fix: Kva/Kve previously defaulted to a shared mutable
        # list `[]`. No current caller ever passes these explicitly (every
        # LMM(...) call site in this file omits them), and the code only
        # ever *reads* len(Kva) before *reassigning* it via
        # `Kva, Kve = linalg.eigh(K)` -- never mutating the default in
        # place -- so no cross-instance state leakage occurs today. But
        # admix mode constructs a fresh LMM once PER ANCESTRY in the same
        # process, so if this were ever changed to mutate Kva/Kve in place
        # (e.g. an in-place cache update) instead of reassigning, the shared
        # default list would silently leak one ancestry's eigendecomposition
        # into another's. None is the safe, standard-idiom default.
        if Kva is None: Kva = []
        if Kve is None: Kve = []

        if str(X0) == str(None):
            X0 = np.ones(len(Y)).reshape(len(Y), 1)
        self.verbose = verbose

        x = True ^ np.isnan(Y)
        x = x.reshape(-1, )
        if not x.sum() == len(Y):
            # Always surfaced (not gated behind verbose=True): silently
            # dropping samples from an analysis is exactly the kind of
            # decision that should be visible by default, matching every
            # other such decision in this project (stale-output warnings,
            # --mode/--pheno fallback notices, etc.).
            logger.info(f'[INFO] {(True ^ x).sum()} sample(s) excluded: '
                       f'missing (NaN) phenotype.\n')
            Y = Y[x]
            K = K[x, :][:, x]
            X0 = X0[x, :]
            Kva = []
            Kve = []
        self.nonmissing = x

        if len(Kva) == 0 or len(Kve) == 0:
            if self.verbose: sys.stderr.write(
                "Obtaining eigendecomposition for %dx%d matrix\n" % (K.shape[0], K.shape[1]))
            begin = time.time()
            if np.isnan(K).any(): K = np.nan_to_num(K)
            Kva, Kve = linalg.eigh(K)
            end = time.time()
            if self.verbose: sys.stderr.write("Total time: %0.3f\n" % (end - begin))

        self.K = K
        self.Kva = Kva
        self.Kve = Kve
        self.N = self.K.shape[0]
        self.Y = Y.reshape((self.N, 1))
        self.X0 = X0

        if sum(self.Kva < 1e-6):
            if self.verbose: sys.stderr.write("Cleaning %d eigen values\n" % (sum(self.Kva < 0)))
            self.Kva[self.Kva < 1e-6] = 1e-6
        self.transform()

    def transform(self):

        """
            Computes a transformation on the phenotype vector and the covariate matrix.
            The transformation is obtained by left multiplying each parameter by the transpose of the
            eigenvector matrix of K (the kinship).
        """

        self.Yt = np.matmul(self.Kve.T, self.Y)
        self.X0t = np.matmul(self.Kve.T, self.X0)
        self.X0t_stack = np.hstack([self.X0t, np.ones((self.N, 1))])
        self.q = self.X0t.shape[1]

    def getMLSoln(self, h, X, beta):
        """
            Obtains the maximum-likelihood estimates for the covariate coefficients (beta),
            the total variance of the trait (sigma) and also passes intermediates that can
            be utilized in other functions. The input parameter h is a value between 0 and 1 and represents
            the heritability or the proportion of the total variance attributed to genetics.  The X is the
            covariate matrix.
        """

        S = 1.0 / (h * self.Kva + (1.0 - h))
        Xt = X.T * S
        XX = np.matmul(Xt, X)
        if np.isnan(XX).any(): XX = np.nan_to_num(XX)
        try:
            XX_i = linalg.inv(XX)
        except linalg.LinAlgError:
            # Bug fix (#7 cleanup, robustness): singular XX (e.g., collinear
            # covariates) now signaled with None instead of the bare string
            # "pass". The old sentinel relied on `LL()` re-returning that
            # same bare string on failure, which Python's tuple-unpacking at
            # one call site (association()) happened to split into individual
            # characters ('p','a','s','s'), and the caller checked
            # `if beta == "a":` to detect it -- fragile, and it would have
            # raised a hard TypeError at the *other* two call sites
            # (LL_brent's `-self.LL(...)[0]` and fit()'s grid-search list
            # comprehension) had a singular matrix ever been hit there
            # instead. None is checked explicitly and consistently below.
            return None
        if beta is None:
            beta = np.matmul(np.matmul(XX_i, Xt), self.Yt)
        else:
            Xf = self.X0t
            Xft = Xf.T * S
            XXf = np.matmul(Xft, Xf)
            XXf_i = linalg.inv(XXf)
            betaf = np.matmul(np.matmul(XXf_i, Xft), self.Yt)
            beta = np.append(betaf, beta, axis=0)
        Yt = self.Yt - np.matmul(X, beta)
        Q = np.dot(Yt.T * S, Yt)
        sigma = Q * 1.0 / (float(self.N) - float(X.shape[1]))
        return beta, sigma, Q, XX_i, XX

    def LL_brent(self, h, X=None, REML=False):
        if h < 0: return 1e6
        ll = self.LL(h, X, stack=False, REML=REML)[0]
        # A singular XX at this h is treated as a very poor fit (a large
        # positive value for this *minimizer*, since LL_brent returns -LL),
        # steering Brent's search away from it rather than crashing.
        return 1e6 if not np.isfinite(ll) else -ll

    def LL(self, h, X=None, beta=None, stack=True, REML=False):
        """
            Computes the log-likelihood for a given heritability (h).  If X==None, then the
            default X0t will be used.  If X is set and stack=True, then X0t will be matrix concatenated with
            the input X.  If stack is false, then X is used in place of X0t in the LL calculation.
            REML is computed by adding additional terms to the standard LL and can be computed by setting REML=True.
        """

        if str(X) == str(None):
            X = self.X0t
        elif stack:
            self.X0t_stack[:, (self.q)] = np.matmul(self.Kve.T, X)[:, 0]
            X = self.X0t_stack
        n = float(self.N)
        q = float(X.shape[1])
        MLSoln = self.getMLSoln(h, X, beta)
        if MLSoln is None:
            return -np.inf, None, None, None
        else:
            beta, sigma, Q, XX_i, XX = MLSoln

        LL = n * np.log(2 * np.pi) + np.log(h * self.Kva + (1.0 - h)).sum() + n + n * np.log(1.0 / n * Q)
        LL = -0.5 * LL

        if REML:
            LL_REML_part = q * np.log(2.0 * np.pi * sigma) + np.log(linalg.det(np.matmul(X.T, X))) - np.log(
                linalg.det(XX))
            LL = LL + 0.5 * LL_REML_part
        # Bug fix (#7 cleanup): removed a no-op `X = []; h=[]; ...` block here --
        # these locals go out of scope (and are garbage-collected normally) the
        # moment this function returns; explicitly clearing them first achieved
        # nothing. Same applies to the other such statements removed elsewhere
        # in this file.
        LL = LL.sum()

        return LL, beta, sigma, XX_i

    def getMax(self, H, X=None, REML=False):

        """
              Helper functions for .fit(...).
              This function takes a set of LLs computed over a grid and finds possible regions
              containing a maximum.  Within these regions, a Brent search is performed to find the
              optimum.
        """
        n = len(self.LLs)
        HOpt = []
        for i in range(1, n - 2):
            if self.LLs[i - 1] < self.LLs[i] and self.LLs[i] > self.LLs[i + 1]:
                # Bug fix: optimize.brent's own internal bracket search can
                # fail outright (raising BracketError, not returning a NaN
                # or a poor-but-valid answer) when the log-likelihood
                # surface is noisy/near-degenerate -- found via a real run
                # with extreme phenotype missingness (12 of 60 samples
                # valid), where scipy's bracket() could not find a valid
                # bracket at all and crashed the entire pipeline several
                # layers up from any error handling. H[i] itself is
                # already a genuine local maximum on the coarse grid
                # (that is exactly the condition just checked above) --
                # falling back to it when the REFINEMENT step fails keeps
                # the fit usable (a grid-resolution answer) instead of an
                # unrecoverable crash, and is a strictly smaller change
                # than the existing np.isnan(...) fallback right below,
                # which already handles brent() returning a bad value
                # rather than raising -- this covers the raising case too.
                try:
                    HOpt.append(optimize.brent(self.LL_brent, args=(X, REML), brack=(H[i - 1], H[i + 1])))
                    if np.isnan(HOpt[-1]): HOpt[-1] = H[i - 1]
                except Exception as e:
                    if self.verbose:
                        sys.stderr.write(f"NOTE: Brent refinement failed at grid point "
                                        f"H={H[i]:.4f} ({type(e).__name__}: {e}) -- "
                                        f"falling back to the grid-resolution estimate.\n")
                    HOpt.append(H[i])

        if len(HOpt) > 1:
            if self.verbose: sys.stderr.write("NOTE: Found multiple optima.  Returning first...\n")
            return HOpt[0]
        elif len(HOpt) == 1:
            return HOpt[0]
        elif self.LLs[0] > self.LLs[n - 1]:
            return H[0]
        else:
            return H[n - 1]

    def fit(self, X=None, ngrids=100, REML=True):

        """
             Finds the maximum-likelihood solution for the heritability (h) given the current parameters.
             X can be passed and will transformed and concatenated to X0t.  Otherwise, X0t is used as
             the covariate matrix.

             This function calculates the LLs over a grid and then uses .getMax(...) to find the optimum.
             Given this optimum, the function computes the LL and associated ML solutions.

        """

        if str(X) == str(None):
            X = self.X0t
        else:
            self.X0t_stack[:, (self.q)] = np.matmul(self.Kve.T, X)[:, 0]
            X = self.X0t_stack
        H = np.array(range(ngrids)) / float(ngrids)
        L = np.array([self.LL(h, X, stack=False, REML=REML)[0] for h in H])
        self.LLs = L

        hmax = self.getMax(H, X, REML)

        L, beta, sigma, betaSTDERR = self.LL(hmax, X, stack=False, REML=REML)
        if beta is None:
            # Extremely unlikely (would mean the optimum itself is singular).
            # Nudge slightly away from the boundary and retry once rather
            # than crash on `sigma.sum()` below with sigma=None.
            hmax = min(max(hmax, 1e-6), 1 - 1e-6)
            L, beta, sigma, betaSTDERR = self.LL(hmax, X, stack=False, REML=REML)

        self.H = H
        self.optH = hmax.sum()
        self.optLL = L
        self.optBeta = beta

        self.optSigma = sigma.sum()
        return hmax, beta, sigma, L

    def association(self, X, h=None, beta=None, stack=True, REML=True, returnBeta=False):

        """
            Calculates association statitics for the SNPs encoded in the vector X of size n.
            If h == None, the optimal h stored in optH is used.

        """
        if stack:
            self.X0t_stack[:, (self.q)] = np.matmul(self.Kve.T, X)[:, 0]
            X = self.X0t_stack
        if h is None:
            h = self.optH

        L, beta, sigma, betaVAR = self.LL(h, X, beta, stack=False, REML=REML)
        if beta is None:
            # Singular XX at this h -- report as a missing association rather
            # than propagating a broken value (see getMLSoln's comment for
            # why this used to rely on unpacking the string "pass").
            return "nan", "nan", "nan", "nan"
        q = len(beta)

        stderr, ts, ps = self.tstat(beta[q - 1], betaVAR[q - 1, q - 1], sigma, q)

        if returnBeta: return beta[q - 1].sum(), stderr, ts, ps

        return beta[q - 1].sum(), stderr, ts, ps

    def tstat(self, beta, var, sigma, q, log=False):

        """
            Calculates a t-statistic and associated p-value given the estimate of beta and its standard error.
            This is actually an F-test, but when only one hypothesis is being performed, it reduces to a t-test.
        """
        stderr = np.sqrt(var * sigma)
        ts = beta / stderr

        if log:
            ps = 2.0 + (stats.t.logsf(np.abs(ts), self.N - q))
        else:
            ps = 2.0 * (stats.t.sf(np.abs(ts), self.N - q))
        if not len(ts) == 1 or not len(ps) == 1:
            raise Exception("Something bad happened :(")
        return stderr, ts.sum(), ps.sum()

    def association_batch(self, X_batch, h=None, REML=True):
        """Vectorized multi-SNP association test (EMMAX/GEMMA-style batching).

        `association()` tests one SNP at a time: for each SNP it rebuilds the
        (q+1) x (q+1) design matrix and inverts it from scratch, even though
        every SNP in a run shares the exact same fixed covariates X0 and
        heritability h (fixed once by `.fit()`). For large numbers of SNPs,
        that means one small-but-nontrivial matrix inversion *per SNP*,
        purely in a Python loop.

        By the Frisch-Waugh-Lovell theorem, testing one variable at a time
        against a shared set of covariates is mathematically equivalent to:
          1. Residualizing the covariates X0 out of Y and out of ALL of the
             SNPs in the batch at once (one shared (q x q) matrix inversion,
             computed only once, then applied to the whole batch via matrix
             multiplication), then
          2. Computing each SNP's beta/se/p-value from a simple weighted
             univariate regression of the residualized Y against that SNP's
             residualized genotype column.

        This turns what was O(n_snps) separate (q+1)x(q+1) inversions into
        one (q x q) inversion plus vectorized elementwise arithmetic across
        the whole batch -- verified in validate_association_batch.py to
        match `association()`'s per-SNP output to within floating-point
        precision.

        Parameters
        ----------
        X_batch : ndarray, shape (n, b)
            b SNPs' (or ancestry-dosage) columns, untransformed (as passed to
            `association()`'s X argument with stack=True).
        h : float, optional. Defaults to self.optH (set by `.fit()`).
        REML : bool. Must match how `.fit()` was called, for consistency.

        Returns
        -------
        betas, stderrs, ts, ps : each an ndarray of shape (b,)
        """
        if h is None:
            h = self.optH

        S = 1.0 / (h * self.Kva + (1.0 - h))          # (n,)
        X0t = self.X0t                                  # (n, q)
        q = X0t.shape[1]
        n = self.N

        A = (X0t.T * S) @ X0t                            # (q, q), shared across the whole batch
        A_inv = linalg.inv(A)

        Yt = self.Yt                                     # (n, 1)
        Y_resid = Yt - X0t @ (A_inv @ ((X0t.T * S) @ Yt))  # (n, 1)

        Xt_batch = self.Kve.T @ X_batch                   # (n, b)
        Xt_resid = Xt_batch - X0t @ (A_inv @ ((X0t.T * S) @ Xt_batch))  # (n, b)

        SY = (S[:, None] * Y_resid)                        # (n, 1)
        numer = (Xt_resid * SY).sum(axis=0)                 # (b,)
        denom = (Xt_resid ** 2 * S[:, None]).sum(axis=0)    # (b,)

        # Robustness: flag SNPs with (near-)zero raw variance up front, the
        # same way calculateKinship() does (`col_var > 0`) -- relying only on
        # the residualized `denom` being exactly 0.0 is not reliable in
        # floating point (a monomorphic/fixed column can residualize to a
        # tiny-but-nonzero value and still produce a wildly unstable,
        # not-actually-meaningful beta rather than a clean divide-by-zero).
        raw_var = X_batch.var(axis=0)
        near_zero_var = raw_var < (1e-8 * max(raw_var.max(), 1.0))

        with np.errstate(divide='ignore', invalid='ignore'):
            betas = numer / denom

        YtWY = float((S[:, None] * Y_resid * Y_resid).sum())
        Q = YtWY - betas * numer                            # (b,) weighted RSS per SNP

        dof = float(n - (q + 1))
        sigmas = Q / dof
        with np.errstate(divide='ignore', invalid='ignore'):
            variances = sigmas / denom
            stderrs = np.sqrt(variances)
            ts = betas / stderrs
        ps = 2.0 * stats.t.sf(np.abs(ts), dof)

        # SNPs with zero/near-zero variance (e.g. a fixed ancestry dosage or
        # monomorphic SNP) get the same "nan" treatment association() intends
        # for a singular single-SNP test (see this method's docstring).
        invalid = ~np.isfinite(betas) | (denom <= 0) | near_zero_var
        betas = np.where(invalid, np.nan, betas)
        stderrs = np.where(invalid, np.nan, stderrs)
        ts = np.where(invalid, np.nan, ts)
        ps = np.where(invalid, np.nan, ps)

        return betas, stderrs, ts, ps

    def calculateKinship(self, K, center=False):
        """
        W is an n x m matrix encoding SNP (or local-ancestry-dosage) minor
        allele counts. Computes the standardized additive kinship matrix.

        Performance fix: the previous implementation looped over columns in
        pure Python (`for i in range(m): ...`) to impute missing values and
        standardize each column -- for large m (many SNPs / many distinct
        local-ancestry patterns) this Python-level loop, not the final matrix
        multiply, was the actual bottleneck (each iteration paying full
        Python + numpy call overhead for a single column at a time). This
        version does the same computation with fully vectorized numpy calls
        across all columns at once, and removes the now-unused custom
        `matrixMult` BLAS-dispatch helper -- modern numpy's `@`/`np.dot`
        already dispatches to BLAS (OpenBLAS/MKL) directly, so the manual
        Fortran-order bookkeeping in `matrixMult` was redundant complexity
        that could add its own extra array copies on large matrices.
        """
        starting_time = time.time()
        W = np.array(K, dtype=np.float64)
        n, m = W.shape

        col_mean = np.nanmean(W, axis=0)
        nan_mask = np.isnan(W)
        if nan_mask.any():
            nan_rows, nan_cols = np.where(nan_mask)
            W[nan_rows, nan_cols] = col_mean[nan_cols]

        col_var = W.var(axis=0)
        keep = col_var > 0
        if not keep.all():
            W = W[:, keep]
            col_mean = col_mean[keep]
            col_var = col_var[keep]

        W = (W - col_mean) / np.sqrt(col_var)
        Kmat = (W @ W.T) / float(m)

        if center:
            # Performance fix: the previous version built an explicit n x n
            # centering matrix P = I - (1/n)*J and computed trace(P @ K @ P)
            # via two full n x n matrix multiplications -- O(n^3) and, for
            # large n, by far the most expensive step in this function.
            # trace(P K P) has a closed form requiring only O(n^2) work:
            #     trace(P K P) = trace(K) - (1/n) * sum(K)
            # (derived from P = I - (1/n) J with J the all-ones matrix; see
            # README for the derivation). This avoids forming P entirely.
            S = np.trace(Kmat) - Kmat.sum() / float(n)
            K_n = (n - 1) * Kmat / S
            logger.info('Obtaining the kinship matrix took :%s' % str(round(time.time() - starting_time, 4)) + 's')
            return K_n

        logger.info('Obtaining the kinship matrix took :%s' % str(round(time.time() - starting_time, 4)) + 's')
        return Kmat


class Joint_Association(jasMapInit, LMM):
    '''This module reads the input files and implements the joint association.'''

    def read_lanc(self):
        '''Read the local ancestry inference input and split into different
        ancestral populations' dosage matrices.

        Performance fix: the previous implementation parsed the file with a
        pure-Python `fileinput` loop, and for every SNP line individually
        looped over haplotype pairs and then over every ancestry label with
        nested if/else branches to compute a 0/1/2 dosage -- for a file with
        many SNPs and/or many ancestries this is an O(n_snps * n_haplotypes *
        K) pure-Python computation, which is exactly the kind of loop that is
        catastrophically slow in Python relative to a vectorized alternative.

        The dosage rule being computed is simply, for ancestry k and
        individual i with haplotype ancestry labels (h0, h1):
            dosage[k, i] = (h0 == k) + (h1 == k)
        which vectorizes trivially over SNPs, individuals, and ancestries at
        once with numpy broadcasting -- no nested loops needed at all.

        Result: self.lanc_array, shape (K, n_snps, n_individuals), dtype
        int8. self.lanc_dict (the original dict-of-dict-of-list structure)
        is still populated from it afterwards, for backward compatibility
        with the rest of the pipeline, using bulk `.tolist()` conversion
        rather than element-by-element Python appends.
        '''
        starting_time = time.time()
        self.genotype_strata = {}; self.phenotype_strata = {}

        with open(self.lanc_file) as fh:
            raw_lines = [l.rstrip('\n').rstrip('\r') for l in fh]
        raw_lines = [l for l in raw_lines if l != '']

        # Bug fix: the local ancestry file is assumed positionally aligned
        # with the ORIGINAL, unfiltered .bed file's SNP order (one line per
        # SNP, same order as the .bim file) -- but --chr filtering (see
        # _resolve_chr_snp_indices()) was applied to the GENOTYPE side only
        # (self.genotypes), never to this ancestry data. That left
        # self.lanc_dict/self.lanc_array keyed 1..(full SNP count) while
        # self.genotypes was keyed 1..(chr-filtered count), causing a
        # KeyError the moment admix mode's per-ancestry loop tried to look
        # up a SNP index beyond the filtered genotype set -- found via a
        # real --chr + --mode admix run. Applying the SAME chr_snp_idx
        # filter here, positionally, keeps both data sources consistent --
        # for a USER-PROVIDED --ancfile, which is at the ORIGINAL,
        # unfiltered scale (that fix's assumption).
        #
        # Second bug fix: with --yes-anc, self.lanc_file gets reassigned
        # to the FRESHLY-COMPUTED local ancestry file (self._lai_obj.
        # anc_path, written after _run_lai_pipeline() runs). Usually that
        # computation already ran on the chr-filtered genotype data, so
        # the file it wrote is ALREADY at the filtered scale, and applying
        # chr_idx again would double-filter it, indexing far past its
        # (already-short) length -- found via a real --yes-anc + --chr +
        # --mode admix run crashing with "list index out of range" at
        # exactly this line.
        #
        # Third bug fix: the above is NOT true when --lai-genome-wide is
        # also given -- that flag makes LAI compute from the FULL genome
        # regardless of --chr, so the ancestry file it writes is genome-
        # wide, not chr-filtered, even though --yes-anc was used. An
        # earlier version of this fix special-cased on self.args.yes_anc
        # directly ("skip filtering whenever --yes-anc was used"), which
        # broke exactly this combination -- found via a real
        # --yes-anc + --lai-genome-wide + --chr + --mode admix run
        # crashing with "index N out of bounds" for self.snp_meta,
        # because self.lanc_dict (built from the genome-wide file) still
        # had every SNP while self.snp_meta (from the SEPARATE, correctly
        # chr-filtered genotype read) did not.
        #
        # Rather than tracking a THIRD flag's worth of special-casing
        # (and risking a fourth scenario breaking the same way again),
        # the bounds-check below is the actual, robust signal: it
        # directly measures whether chr_idx's indices (into the ORIGINAL,
        # full SNP space) actually fit within however long raw_lines
        # really is, and adapts correctly regardless of WHY the file
        # happens to be at whichever scale it's at.
        chr_idx = getattr(self, 'chr_snp_idx', None)
        if chr_idx is not None:
            if max(chr_idx) < len(raw_lines):
                raw_lines = [raw_lines[i] for i in chr_idx]
            else:
                logger.warning(
                    f"[WARNING] --chr filtering skipped for the local "
                    f"ancestry file: it has {len(raw_lines)} line(s), too "
                    f"few to index with --chr's SNP positions (up to "
                    f"{max(chr_idx)}) into the full, unfiltered panel. "
                    f"Proceeding with the file's own {len(raw_lines)} "
                    f"line(s) as-is -- verify this ancestry file is "
                    f"actually aligned with the --chr-restricted SNP set "
                    f"you intended.\n")

        if len(raw_lines) == 0:
            sys.stderr.write('ERROR: local ancestry file "%s" is empty!\n' % self.lanc_file)
            self.terminate(self.outfolder + self.logFile)
            sys.exit(1)

        # Detect format from the first line: space-delimited integer labels,
        # or a compact string of single-digit labels with no delimiter.
        space_delimited = (' ' in raw_lines[0])

        if space_delimited:
            hap_matrix = np.array([line.split() for line in raw_lines], dtype=np.int8)
        else:
            n_hap = len(raw_lines[0])
            hap_matrix = np.empty((len(raw_lines), n_hap), dtype=np.int8)
            for i, line in enumerate(raw_lines):
                # Fast C-level bytes->int8 conversion (digits '0'-'9' only);
                # still one call per line, but no per-character Python loop.
                hap_matrix[i, :] = np.frombuffer(line.encode('ascii'), dtype=np.uint8) - ord('0')

        n_snps, n_hap_total = hap_matrix.shape
        if n_hap_total % 2 != 0:
            sys.stderr.write('ERROR: local ancestry file has an odd number of haplotype columns!\n')
            self.terminate(self.outfolder + self.logFile)
            sys.exit(1)

        bad = (hap_matrix < 1) | (hap_matrix > self.no_of_ancs)
        if bad.any():
            bad_row = int(np.nonzero(bad.any(axis=1))[0][0]) + 1
            sys.stderr.write('ERROR: Unknown ancestral label in line %i!\n\n' % bad_row)
            self.terminate(self.outfolder + self.logFile)
            sys.exit(1)

        hap0 = hap_matrix[:, 0::2]   # (n_snps, n_individuals)
        hap1 = hap_matrix[:, 1::2]

        self.lanc_array = np.stack(
            [((hap0 == anc).astype(np.int8) + (hap1 == anc).astype(np.int8))
             for anc in self.lanc_labels_list],
            axis=0)  # shape: (K, n_snps, n_individuals)

        # Populate self.lanc_dict for backward compatibility with the rest of
        # the pipeline (run_joint_analysis iterates `for snp in self.lanc_dict`).
        # Performance fix: this used to eagerly materialise the ENTIRE
        # dict-of-dict-of-list structure via per_anc_lists = [...tolist()],
        # which for a real genome-wide run (272,791 SNPs x 1,363 samples x 5
        # ancestries) meant allocating >1.8 billion Python ints and took 738
        # seconds -- ~22% of that run's total wall time -- to build data the
        # hot path never actually reads (see _LazyLancDict's docstring).
        # The lazy view below presents the identical read interface at
        # effectively zero construction cost.
        self.lanc_dict = _LazyLancDict(self.lanc_array, self.lanc_labels_list)

        logger.info("\nReading/converting ancestry file took %s time to finish.\n\n"
                    % str(round(time.time() - starting_time, 4)) + 's')

    def get_ancestry_AM_instances(self, X0s):
        '''For each ancestry, deduplicate its SNP x individual dosage matrix,
        compute its kinship matrix, and fit the null LMM (pseudo-heritability)
        for that ancestry.

        Performance fix: the previous implementation rebuilt each ancestry's
        SNP x individual matrix as a pandas DataFrame, called
        `drop_duplicates()`, then converted the *deduplicated* rows back into
        a nested Python list one row at a time via a Python-level loop
        (`[list(df.iloc[i]) for i in range(len(df))]`) before transposing --
        an expensive DataFrame round-trip and row-by-row conversion for what
        is fundamentally just "deduplicate the rows of a numpy array".
        `np.unique(..., axis=0)` does exactly that in optimized C code.
        '''
        self.lai_h_dict = {}; self.lai_L_instances = {}
        for anc in self.lanc_labels_list:
            anc_matrix = self.lanc_array[anc - 1]          # (n_snps, n_individuals)
            unique_rows = np.unique(anc_matrix, axis=0)     # dedup SNP patterns
            lanc_dict_per_anc_T = unique_rows.T.astype(np.float64)  # (n_individuals, n_unique_snps)

            lai_kinship = self.calculateKinship(K=lanc_dict_per_anc_T)
            # Bug fix: this used to redundantly re-read --pheno on every
            # single ancestry iteration (self.phenotypes was already set,
            # once, correctly, by whichever read_*_file() ran earlier).
            # Beyond the wasted repeated file I/O, this actively UNDID that
            # earlier setup: read_pheno_file() on its own returns the raw
            # phenotype array with no genotype-sample alignment applied --
            # only the shared _align_pheno_to_genotype_samples() step
            # (added to each read_*_file() method) does that alignment.
            # Re-reading here would silently overwrite the correctly
            # aligned self.phenotypes with the unaligned version again,
            # on every ancestry, for the whole rest of admix mode's
            # analysis. self.args.pheno never changes between iterations,
            # so there is nothing this re-read could legitimately pick up
            # that self.phenotypes doesn't already correctly have.
            # Bug fix: this previously passed X0s POSITIONALLY as the 3rd
            # argument to LMM(...), but the constructor's signature is
            # __init__(self, Y, K, Kva=None, Kve=None, X0=None, verbose=False)
            # -- the 3rd parameter is Kva (eigenvalues), not X0. X0s was
            # silently landing in Kva (then discarded/overwritten by a
            # fresh eigendecomposition, since Kve was also unset), while
            # the REAL X0 parameter stayed at its default (None), silently
            # falling back to intercept-only fitting. This means every
            # analysis using --covfile in admix mode's per-ancestry
            # admixture-mapping test has been ignoring the user's
            # covariates entirely, project-wide, throughout every prior
            # session -- found via a direct end-to-end comparison against
            # the streaming dispatcher (which correctly used X0= as a
            # keyword argument), not by any isolated unit test.
            lai_L = LMM(self.phenotypes, lai_kinship, X0=X0s)
            _assert_covariates_used(lai_L, X0s, f"admix Part I (ancestry {anc})")
            lai_hmax, lai_beta, lai_sigma, lai_LL_null = lai_L.fit()
            self.lai_h_dict[anc] = lai_hmax
            self.lai_L_instances[anc] = lai_L
            logger.info('\nPseudo-heritability for ancestry %s = %s.\n\n' % (str(anc), str(lai_hmax)))

    def read_covar_file(self):
        '''Reading in the covariate file (FID IID PC1 PC2 ...).

        Performance fix: replaced the pure-Python `fileinput` line-by-line
        loop with a single vectorized pandas read. Note: preserved exactly
        the original's behavior of not skipping a header line (its own
        `fileinput.lineno() > 0` check was always true, since line numbers
        start at 1) -- if your covariate file has a text header row, remove
        it before running, same as with the original script.
        '''
        if self.args.covfile:
            df = pd.read_csv(self.covar_file, sep=r"\s+", header=None, engine="python")
            self.covar_list = df.iloc[:, 2:].to_numpy(dtype=float)
        logger.info("\nReading PCA file done.\n")

    # Bug fix (#7 cleanup): read_geno_snp() and read_geno_ind() are no longer
    # needed -- read_geno_file() now reads the .snp/.ind files directly via
    # vectorized pandas calls -- removed as dead code.

    def _populate_genotypes_from_tped(self, tped_path):
        '''Bug fix (#1 from the earlier review): self.genotypes was
        previously never populated by ANY input path (--geno/--bfile/--vcf
        all wrote a tped-style file for kinship computation but never loaded
        per-SNP genotype arrays into self.genotypes, which the main
        association loop in run_joint_analysis depends on:
        `true_geno = self.genotypes[snp]`). jasmap/parse.py contained a
        commented-out reference implementation of exactly this step,
        confirming it was a leftover gap from an incomplete refactor.

        This single, shared, vectorized implementation reads back the
        tped-style file already written for kinship computation (no need to
        parse the raw input a second time) and populates self.genotypes with
        standardized (mean-centered, unit-variance) per-SNP dosage arrays,
        keyed by 1-based line number to match self.lanc_dict's keys.
        '''
        df = pl.read_csv(tped_path, separator="\t", null_values="NA")
        geno_matrix = df[:, 6:].to_numpy().astype(np.float64)

        # Store SNP metadata (chromosome, real ID, position) indexed by the
        # SAME 1-based line number used as self.genotypes' keys below --
        # needed so run_joint_analysis()'s output-writing code can report
        # real SNP IDs and chromosome (previously just the positional line
        # number) for downstream enrichment analysis. Extracted by COLUMN
        # POSITION (0=chrom, 1=snp, 3=pos), not name, since the tped
        # header's casing differs between input paths (lowercase for
        # --bfile, uppercase for --geno) -- position is consistent across
        # all three (--bfile/--geno/--vcf all write the same 6-column
        # metadata header in the same order). Single, shared location
        # (matching this method's own existing "single, shared,
        # vectorized" design) rather than duplicating extraction logic in
        # each of the three read_*_file() methods separately.
        meta_cols = df.columns[:6]
        self.snp_meta = df.select(meta_cols).to_pandas()
        self.snp_meta.columns = ['chrom', 'snp', 'cm', 'pos', 'a0', 'a1']
        self.snp_meta = self.snp_meta[['chrom', 'snp', 'pos']]

        if np.isnan(geno_matrix).any():
            # Bug fix: this used to impute every missing call using a
            # per-SNP mean computed across ALL samples in geno_matrix --
            # for --yes-anc's data model, that means averaging in
            # reference/ancestral population samples (who can outnumber
            # the actual study cohort) with systematically different
            # allele frequencies from the admixed samples being studied,
            # by the very design reason they were chosen as ancestry
            # references. self.genotypes feeds BOTH GWAS mode's
            # association_batch() AND admix mode's own SNP-only
            # association test, making this the most widespread instance
            # of a contamination pattern already found and fixed for
            # RVAT's own, separate genotype read (see run_rvat()'s
            # equivalent fix) -- found via a real user report describing
            # this exact data model and asking for it to be handled
            # consistently everywhere, not just piecemeal per mode.
            #
            # self.genotypes must stay FULL-LENGTH regardless (every
            # downstream LMM already does its own K=K[x,:][:,x]-style
            # restriction to non-missing-phenotype samples AT THE POINT
            # OF USE, and that logic assumes self.genotypes/self.kinship/
            # self.phenotypes all share the same length going in) -- so
            # unlike RVAT (which permanently restricts G to the study
            # cohort), the fix here is to impute EACH group using ONLY
            # its own group's mean, not to drop anyone.
            #
            # Only takes this path when a genuine split actually exists
            # (0 < study_mask.sum() < len(study_mask)); a regular cohort
            # with no reference/ancestral samples at all (the common
            # case) falls through to the original, single-mean behavior
            # unchanged, since there's no "other group" to separate from.
            genotype_sample_ids = list(df.columns[6:])
            study_mask = self._get_study_cohort_mask(
                self.phenotypes, genotype_sample_ids=genotype_sample_ids)
            if 0 < int(study_mask.sum()) < len(study_mask):
                row_mean_study = np.nan_to_num(
                    np.nanmean(geno_matrix[:, study_mask], axis=1), nan=0.0)
                row_mean_ref = np.nan_to_num(
                    np.nanmean(geno_matrix[:, ~study_mask], axis=1), nan=0.0)
                nan_rows, nan_cols = np.where(np.isnan(geno_matrix))
                fill_vals = np.where(study_mask[nan_cols],
                                     row_mean_study[nan_rows],
                                     row_mean_ref[nan_rows])
                geno_matrix[nan_rows, nan_cols] = fill_vals
                logger.info(
                    f"[INFO] Imputed missing genotype calls separately "
                    f"for {int(study_mask.sum())} study-cohort and "
                    f"{int((~study_mask).sum())} reference-population "
                    f"sample(s), using each group's own per-SNP mean "
                    f"dosage (not a pooled mean across both).")
            else:
                row_mean_raw = np.nanmean(geno_matrix, axis=1)
                nan_rows, nan_cols = np.where(np.isnan(geno_matrix))
                geno_matrix[nan_rows, nan_cols] = row_mean_raw[nan_rows]

        row_mean = geno_matrix.mean(axis=1)
        row_var = geno_matrix.var(axis=1)
        row_var_safe = np.where(row_var == 0, 1.0, row_var)
        standardized = (geno_matrix - row_mean[:, None]) / row_var_safe[:, None]
        self.genotypes = {i + 1: standardized[i] for i in range(standardized.shape[0])}

    def read_geno_file(self):
        '''Reading genotype information provided as an EIGENSTRAT-style
        .geno/.snp/.ind trio ({0,1,2} dosage codes; .geno may be one compact
        digit-string per SNP line, or whitespace-separated -- both supported).

        Streaming/biobank-scale note: unlike read_bed_file(), this path has
        no chunked-read mechanism (EIGENSTRAT's plain-text .geno format has
        no equivalent to bed-reader's indexed access), so the streaming
        dispatcher in jasmap.biobank.auto_lmm is not used here -- always
        the exact LMM class, regardless of sample count. Explicit False
        (not left unset) so run_joint_analysis()'s dispatch check always
        has a value to read, for any input type.

        Performance fix: the previous implementation read the .snp/.ind/.geno
        files with pure-Python `fileinput` loops and wrote the output tped
        file one line at a time with Python string formatting -- replaced
        with vectorized pandas/numpy reads and a single bulk `to_csv` write.
        Also fixes the .ind sample-ID bug noted in _populate_genotypes_from_tped's
        docstring area (column 0, the actual ID, instead of column 1, gender).
        '''
        starting_time = time.time()
        self._use_streaming_lmm = False  # no chunked-read mechanism for this format

        snp_meta = pd.read_csv(self.geno_file + ".snp", sep=r"\s+", header=None,
                               names=["SNP", "CHR", "CM", "POS", "A1", "A2"], engine="python")
        ind_ids = pd.read_csv(self.geno_file + ".ind", sep=r"\s+", header=None, engine="python")[0].tolist()

        with open(self.geno_file) as fh:
            raw_lines = [l.rstrip('\n').rstrip('\r') for l in fh if l.strip() != '']

        n_snps = len(raw_lines)
        n_samples = len(ind_ids)
        geno_matrix = np.empty((n_snps, n_samples), dtype=np.float64)
        for i, line in enumerate(raw_lines):
            data = line.split()
            if len(data) == 1:
                geno_matrix[i, :] = np.frombuffer(data[0].encode('ascii'), dtype=np.uint8) - ord('0')
            else:
                geno_matrix[i, :] = np.array(data, dtype=np.float64)

        tped_meta = snp_meta.iloc[:n_snps][["CHR", "SNP", "CM", "POS", "A1", "A2"]].reset_index(drop=True)

        # Bug fix: --chr was silently ignored for --geno input (self.chr_snp_idx
        # was only ever set inside read_bed_file(), --bfile-specific) --
        # found via a real run that processed the entire panel with --chr
        # given, no filtering, no warning at all. Resolved here from the
        # .snp file's own CHR column, then applied before the .tmp.tped
        # file is written -- RVAT/PRS/GIFT's own --geno fallback paths all
        # read from this SAME shared .tmp.tped file, so filtering here
        # once propagates correctly to all of them, not just this
        # method's own downstream self.genotypes population.
        self.chr_snp_idx = _resolve_chr_indices_generic(
            tped_meta["CHR"].values, self.args.chr,
            source_desc=self.geno_file + ".snp")
        if self.chr_snp_idx is not None:
            logger.info(
                f"[INFO] --chr {self.args.chr!r}: restricting analysis to "
                f"{len(self.chr_snp_idx)} of {n_snps} SNPs "
                f"({100*len(self.chr_snp_idx)/n_snps:.2f}% of the panel).\n")
            tped_meta = tped_meta.iloc[self.chr_snp_idx].reset_index(drop=True)
            geno_matrix = geno_matrix[self.chr_snp_idx]

        tped_df = pd.concat([tped_meta, pd.DataFrame(geno_matrix, columns=ind_ids)], axis=1)
        tped_df.to_csv(self._tped_path(), sep='\t', index=False, header=True)

        logger.info("Reading genotype file done...\n")
        logger.info("\nReading genotypes file took %s time to finish.\n\n"
                    % str(round(time.time() - starting_time, 4)) + 's')

        self.phenotypes, self.samples_list = ps.read_pheno_file(self.args.pheno)
        self.phenotypes = _align_pheno_to_genotype_samples(
            self.phenotypes, self.samples_list, ind_ids)

        kS_time = time.time()
        # Bug fix (#8): ps.makeG() returns [kinship_matrix, geno_id_series] --
        # the previous code assigned the whole 2-element list to self.kinship
        # without indexing [0], which would have broken LMM's eigendecomposition
        # (linalg.eigh cannot operate on a [matrix, Series] list) the moment any
        # input path actually reached that call.
        self.kinship = ps.makeG(self._tped_path(), method=self.args.kinship,
                                File_list=False, n1=self.n1, n2=self.n2,
                                maf_min=self.args.kinship_maf_min)[0]
        logger.info('Obtaining kinship matrix took :%s' % str(round(time.time() - kS_time, 4)) + 's')

        self._populate_genotypes_from_tped(self._tped_path())

    def read_bed_file(self):
        '''Reading genotype information from PLINK .bed/.bim/.fam.

        Uses `bed-reader` instead of `pandas-plink` -- `pandas-plink`'s own
        documentation recommends installing it via conda specifically
        because its PyPI wheel coverage on Windows is inconsistent across
        Python versions, which forces pip to fall back to compiling its
        `bed_reader` C extension from source (requiring the MSVC Build
        Tools). `bed-reader` (a Rust-backed PLINK reader from the same
        FaST-LMM team, used by PySnpTools and FaST-LMM itself) publishes
        genuine prebuilt wheels for Windows/macOS/Linux across Python
        3.10-3.14, needs no compiler on any platform, and depends only on
        numpy. `count_A1=False` is required to match the dosage convention
        used everywhere else in this file -- verified to produce numerically
        identical output to pandas-plink on the same .bed file.

        Biobank-scale note: this method's own genotype read (`bed.read().T`
        below) materialises the ENTIRE dense genotype matrix in memory as
        its first step, regardless of sample count -- at n=500,000 samples
        and m=2,000,000 SNPs that is ~8 TB, actually LARGER than the
        1.86 TB dense kinship matrix the jasmap.biobank.auto_lmm dispatcher
        was built to avoid. Fixing the kinship computation alone therefore
        does not unlock true biobank scale on its own when SNP count is
        also extreme -- that would need this method's own genotype-read/
        tped-write/self.genotypes-dict architecture to ALSO become
        streaming-aware, which is separate, not-yet-completed work (it
        overlaps with what the original priority table called the "GWAS"
        biobank-scale item, not purely "Kinship/GRM"). What IS wired in
        here: when sample count alone is large enough to make the DENSE
        KINSHIP matrix infeasible (independent of SNP count), the kinship/
        LMM construction step below uses the streaming dispatcher
        (jasmap.biobank.auto_lmm.build_lmm_from_bed) instead of the exact
        LMM class -- this covers the realistic case of large sample counts
        with a moderate SNP panel (e.g. an exome chip, not whole-genome
        sequencing), while being explicit via a logged warning about the
        harder case this does not yet solve.
        '''
        logger.info('Reading genotypes from {fname}\n'.format(fname=self.geno_file))
        starting_time = time.time()
        from bed_reader import open_bed
        bed = open_bed(self.geno_file + ".bed", count_A1=False)

        # --chr filtering: resolved once, here, before any genotype data is
        # read -- see _resolve_chr_snp_indices()'s docstring for why this
        # matters at biobank scale (turns an infeasible whole-genome read
        # into one chromosome's worth, which fits comfortably in memory).
        # Stored on self so the downstream auto-PCA/kinship-streaming calls
        # below can apply the SAME filter consistently.
        self.chr_snp_idx = _resolve_chr_snp_indices(self.geno_file, self.args.chr)

        # Biobank-scale size check, using bed-reader's own cheap iid_count/
        # sid_count properties -- no genotype data is read for this check.
        from jasmap.biobank.auto_lmm import recommend_lmm_strategy
        n_bed = bed.iid_count
        m_bed = bed.sid_count if self.chr_snp_idx is None else len(self.chr_snp_idx)
        kinship_tier = recommend_lmm_strategy(
            n_bed, memory_limit_bytes=self.args.kinship_memory_limit_gb * 1024**3)
        self._use_streaming_lmm = (kinship_tier == "approximate")

        if self._use_streaming_lmm:
            dense_genotype_bytes = n_bed * m_bed * 8
            logger.info(
                f'[INFO] n={n_bed} samples: dense kinship matrix '
                f'({n_bed*n_bed*8/1e9:.2f} GB) exceeds the '
                f'{self.args.kinship_memory_limit_gb:.1f} GB exact-mode '
                f'limit. Kinship/LMM construction will use the streaming, '
                f'size-aware dispatcher (jasmap.biobank.auto_lmm) instead '
                f'of the exact eigendecomposition.\n')
            if dense_genotype_bytes > self.args.kinship_memory_limit_gb * 1024**3 * 4:
                logger.info(
                    f'[WARNING] The dense genotype matrix itself '
                    f'({dense_genotype_bytes/1e9:.2f} GB for n={n_bed} x '
                    f'm={m_bed}) is ALSO large. This method still reads '
                    f'the full genotype matrix densely for the '
                    f'association-testing step below (only the kinship/LMM '
                    f'step uses streaming) -- at this SNP count that read '
                    f'may itself be slow or memory-constrained. Streaming '
                    f'genotype access for the association-testing step '
                    f'itself is separate, not-yet-completed work. Consider '
                    f'--chr to restrict to one chromosome at a time if '
                    f'this is still too large.\n')

        read_idx = (None, self.chr_snp_idx) if self.chr_snp_idx is not None else None
        G = pd.DataFrame(bed.read(index=read_idx).T)  # bed.read() is (samples, SNPs); transpose to (SNPs, samples)
        G.columns = list(bed.iid)
        if self.chr_snp_idx is not None:
            bim = pd.DataFrame({
                'chrom': np.asarray(bed.chromosome)[self.chr_snp_idx],
                'snp': np.asarray(bed.sid)[self.chr_snp_idx],
                'cm': np.asarray(bed.cm_position)[self.chr_snp_idx],
                'pos': np.asarray(bed.bp_position)[self.chr_snp_idx],
                'a0': np.asarray(bed.allele_1)[self.chr_snp_idx],
                'a1': np.asarray(bed.allele_2)[self.chr_snp_idx],
            })
        else:
            bim = pd.DataFrame({
                'chrom': list(bed.chromosome),
                'snp': list(bed.sid),
                'cm': list(bed.cm_position),
                'pos': list(bed.bp_position),
                'a0': list(bed.allele_1),
                'a1': list(bed.allele_2),
            })
        G = bim.join(G)
        G.to_csv(self._tped_path(), sep='\t', encoding='utf-8', index=False, header=True)

        logger.info("\nReading genotypes file took %s time to finish.\n\n"
                    % str(round(time.time() - starting_time, 4)) + 's')

        self.phenotypes, self.samples_list = ps.read_pheno_file(self.args.pheno)
        self.phenotypes = _align_pheno_to_genotype_samples(
            self.phenotypes, self.samples_list, list(bed.iid))

        kS_time = time.time()
        if self._use_streaming_lmm:
            # Skip the dense ps.makeG() eigendecomposition entirely -- the
            # streaming dispatcher builds (and fits) the model directly
            # from the .bed file in run_joint_analysis() instead. self.kinship
            # is left unset deliberately (rather than set to a placeholder)
            # so any code path that assumes it exists fails loudly instead
            # of silently using a wrong value.
            logger.info('[INFO] Skipping dense kinship matrix construction '
                        '(using streaming dispatcher instead).\n')
        else:
            # Bug fix (#8): see read_geno_file's note -- makeG() returns a
            # 2-element list; index [0] to get the actual kinship matrix.
            self.kinship = ps.makeG(self._tped_path(), method=self.args.kinship,
                                    File_list=False, n1=self.n1, n2=self.n2,
                                    maf_min=self.args.kinship_maf_min)[0]
            logger.info('Obtaining kinship matrix took :%s' % str(round(time.time() - kS_time, 4)) + 's')

        self._populate_genotypes_from_tped(self._tped_path())

    # Bug fix (#7): `from gzip import open as gzopen` was defined here but never
    # used anywhere in the class -- removed as dead code.

    def vcf_to_dataframe(self):
        """Convert a multi-sample VCF into the tped-style format used for
        kinship computation, via jasmap.parse.load_vcf_samples_in_map (which
        itself now emits the correct 6 metadata columns -- see that
        function's docstring for the column-count bug that was fixed there).
        """
        logger.info('Reading vcf file from {fname}\n'.format(fname=self.args.vcf))
        starting_time = time.time()
        self._use_streaming_lmm = False  # no chunked-read mechanism for VCF input
        self.phenotypes, self.samples_list = ps.read_pheno_file(self.args.pheno)
        vcf_actual_samples, self.chr_snp_idx, _vcf_n_variants = ps.load_vcf_samples_in_map(
            self.args.vcf, self.samples_list, self._tped_path(),
            chr_arg=self.args.chr)
        if self.chr_snp_idx is not None:
            logger.info(
                f"[INFO] --chr {self.args.chr!r}: restricting analysis to "
                f"{len(self.chr_snp_idx)} of {_vcf_n_variants} SNPs "
                f"({100*len(self.chr_snp_idx)/_vcf_n_variants:.2f}% of the panel).\n")
        self.phenotypes = _align_pheno_to_genotype_samples(
            self.phenotypes, self.samples_list, vcf_actual_samples)
        logger.info("\nReading vcf file took %s time to finish.\n\n"
                    % str(round(time.time() - starting_time, 4)) + 's')

        kS_time = time.time()
        # Bug fix (#8): see read_geno_file's note -- makeG() returns a
        # 2-element list; index [0] to get the actual kinship matrix.
        self.kinship = ps.makeG(self._tped_path(), method=self.args.kinship,
                                File_list=False, n1=self.n1, n2=self.n2,
                                maf_min=self.args.kinship_maf_min)[0]
        logger.info('Obtaining kinship matrix took :%s' % str(round(time.time() - kS_time, 4)) + 's')

        self._populate_genotypes_from_tped(self._tped_path())


    # Bug fix (#7 cleanup): get_ll() was defined but never called anywhere in
    # the codebase -- removed as dead code. (It also still used the old,
    # fragile "pass"-string sentinel this cleanup pass replaced elsewhere.)

    def run_joint_analysis(self):
        '''The main function where estimation of joint beta is done and the corresponding information written on the output file'''

        # Bug fix (#6): output files and headers now depend on --mode. "admix" keeps
        # the original joint ancestry+SNP output; "GWAS" (previously unimplemented)
        # writes a SNP-only association file instead, since there is no ancestry
        # prior to report in that mode.
        is_admix_mode = (self.args.mode == "admix")

        if is_admix_mode:
            # Headers are written as part of the bulk vectorized write further
            # below (each file's content is built as a list of lines with the
            # header as line 0, then written in one shot) -- opened here in
            # append mode so that bulk write is the only write to each file.
            joint_fout = open(self.outfolder + "Jasmap.Joint.assoc", "wt")

            anc_prior_fout_dict = {}
            for idx in range(1, self.no_of_ancs + 1):
                # Use the real --pop-labels population name in the filename
                # when available (e.g. "Jasmap.ANC_AFR.AM.assoc") instead of
                # just the bare numeric index -- makes per-ancestry output
                # files self-documenting without needing to separately
                # cross-reference which numeric index maps to which
                # population. Keeps the "ANC" prefix either way so
                # --consgwas's own glob("Jasmap.ANC*.AM.assoc") still finds
                # them (confirmed: its filename parsing is purely
                # positional -- split(".")[1] -- not dependent on the
                # numeric-only form). Falls back to the exact original
                # "ANC{idx}" filename, unchanged, when no --pop-labels was
                # given, so behavior is identical to before for anyone not
                # using --pop-labels.
                anc_tag = (f"ANC_{self._pop_labels[idx-1]}"
                          if getattr(self, "_pop_labels", None) else f"ANC{idx}")
                anc_prior_fout_dict[idx] = open(self.outfolder + f"Jasmap.{anc_tag}.AM.assoc", "wt")
        else:
            joint_fout = open(self.outfolder + "Jasmap.GWAS.assoc", "wt")
            joint_fout.writelines("CHR" + "\t" + "SNP" + "\t" + "POS" + "\t" + "SNP_beta" + "\t" + "SNP_Pval" + "\n")
            anc_prior_fout_dict = {}

        # Read other inputs, genotypes, phenotype and covariates
        if self.args.geno is not None:
            self.read_geno_file()
        elif self.args.bfile is not None:
            self.read_bed_file()
        elif self.args.vcf is not None:
            self.vcf_to_dataframe()
        # self.phenotypes is set by whichever of the three readers above ran;
        # self.inds_size must be available regardless of input format (bug fix:
        # previously only set on the --geno path, causing an AttributeError on
        # --bfile/--vcf runs without --covfile).
        self.inds_size = len(self.phenotypes)
        if self.args.pheno is not None:
            #ps.read_pheno_file(self.args.pheno)
            #self.Pheno, self.samples_list = ps.read_pheno_file(self.args.pheno)
            #self.inds_size = len(self.phenotypes)
            logger.info("Reading phenotype file done...")

        # Read local ancestry file and separate to different ancestries and ancestry matrices
        #
        # Bug fix: this used to check self.args.ancfile specifically -- but
        # --yes-anc without an explicit --ancfile sets self.lanc_dict
        # DIRECTLY from the LAI object's own result (self.lanc_dict =
        # lai.lanc_dict, inside _run_lai_pipeline()), never going through
        # read_lanc() at all. That direct assignment reflects whatever
        # scale LAI itself ran at -- genome-wide when --lai-genome-wide is
        # used -- and, with --ancfile absent, nothing ever re-filtered it
        # to match the SEPARATE, chr-restricted genotype read the main
        # analysis does. Found via a real --vcf + --yes-anc (no --ancfile)
        # + --lai-genome-wide + --chr run: self.lanc_dict stayed genome-
        # wide (matching --lai-genome-wide's 220-SNP scale) while
        # self.snp_meta was correctly chr-restricted to 10, crashing with
        # the same "index N out of bounds" as the --ancfile case this
        # file's OTHER --chr/--yes-anc fixes already handle -- just via a
        # different path that skipped read_lanc() entirely, so those
        # fixes never got a chance to run. self.lanc_file is set either
        # way (to the user's --ancfile path, or to
        # self._lai_obj.anc_path from --yes-anc's own computation) --
        # using it as the condition, rather than self.args.ancfile
        # specifically, means read_lanc() always re-reads and re-filters
        # from the actual written file (already handling both scales
        # correctly via its own bounds-check), regardless of which path
        # produced that file.
        if getattr(self, 'lanc_file', None) is not None:
            self.read_lanc()


        if self.args.covfile is not None:
            self.read_covar_file()
            X0s = self.covar_list
        elif self.args.auto_pca and self.args.bfile is not None:
            # Per-individual PCA eigenvectors are the standard mechanism
            # for correcting population stratification -- and JasMAP4
            # exists specifically to analyse admixed populations, where
            # stratification is the dominant confounder. Previously,
            # omitting --covfile meant X0s = intercept-only: ZERO
            # stratification correction, silently. This is the same risk
            # category as the positional-argument bug fixed earlier in
            # this file's LMM(...) call sites (see _assert_covariates_used's
            # docstring) -- just for "no covariates given at all" rather
            # than "covariates given but silently dropped". Defaults ON
            # (opt-out via --no-auto-pca), since running with no
            # correction at all is the more dangerous default for this
            # tool's actual use case.
            #
            # Only wired for --bfile: it is the only input format with a
            # chunked-read mechanism (jasmap.biobank.streaming_io),
            # needed so this works at both small and large sample counts
            # without a separate code path per size tier. --geno/--vcf
            # fall through to the intercept-only default below, same as
            # before -- extending auto-PCA to those formats is future work.
            # Bug fix: the "dense" branch here previously built G_dense from
            # self.genotypes, which stores ALREADY-STANDARDIZED (mean-
            # centered, variance-scaled) values for the association-testing
            # step, not raw 0/1/2 dosage. compute_auto_pca_dense internally
            # computes allele frequency as p=mean/2 (assuming raw dosage) --
            # fed pre-centered data (mean≈0 for every SNP), this gives
            # maf≈0 for every SNP, so the maf_min filter added earlier
            # excluded literally all of them (m=0, trace(K)=0.00, found via
            # a real run). Before that filter existed, the same mismatch
            # silently produced the earlier astronomical trace(K) blow-up
            # (~10^19) -- dividing by a near-zero *computed* variance,
            # not the "many rare variants" explanation first suspected,
            # which could not be reproduced despite trying several
            # plausible synthetic scenarios. compute_auto_pca_streaming
            # reads raw dosage directly from the .bed file regardless of
            # data size (its own StreamingGenotypeSource auto-caches in
            # memory for small data, behaving like a direct read with no
            # extra overhead) -- using it unconditionally here removes
            # this entire bug class by removing the second, incorrect
            # code path rather than patching it in place.
            from jasmap.biobank.auto_pca import compute_auto_pca_streaming

            # Bug fix: auto-PCA was computed from ALL samples, including
            # any reference/ancestral population samples present purely to
            # inform --yes-anc's local ancestry inference (not part of the
            # phenotyped study cohort). Unlike kinship -- a purely pairwise
            # statistic, mathematically unaffected by which OTHER samples
            # are also included when computing it -- PCA eigenvectors ARE
            # cohort-dependent: with several genuinely distinct reference
            # populations mixed in, the top PCs from the full cohort would
            # likely capture reference-population separation (a large,
            # dominant axis of variation) rather than the finer, within-
            # cohort stratification the association test actually needs
            # corrected for admixed study samples specifically. Found via
            # a real user report describing this data model precisely.
            # Computing PCA from only the phenotyped (non-missing) subset,
            # then embedding the result back into a full-length array (the
            # excluded rows' placeholder values never matter -- they get
            # dropped by LMM.__init__()'s own X0=X0[x,:] slicing moments
            # later, matching how it already drops the corresponding K/Y
            # rows) gives the intended, within-cohort PCA while keeping
            # every downstream shape assumption intact.
            _valid_for_pca = np.where(
                ~np.isnan(np.asarray(self.phenotypes, dtype=np.float64)))[0]
            if 0 < len(_valid_for_pca) < len(self.phenotypes):
                logger.info(
                    f"[auto_pca] Restricting to {len(_valid_for_pca)} of "
                    f"{len(self.phenotypes)} samples with non-missing "
                    f"phenotype for PCA computation (excluding reference/"
                    f"ancestral population samples used only for local "
                    f"ancestry inference, if any).")
                X0s_restricted = compute_auto_pca_streaming(
                    self.geno_file, n_pcs=self.args.auto_pca_k, verbose=True,
                    maf_min=self.args.kinship_maf_min, sample_idx=_valid_for_pca,
                    snp_idx=getattr(self, 'chr_snp_idx', None))
                X0s = np.zeros((len(self.phenotypes), X0s_restricted.shape[1]))
                X0s[_valid_for_pca] = X0s_restricted
            else:
                X0s = compute_auto_pca_streaming(
                    self.geno_file, n_pcs=self.args.auto_pca_k, verbose=True,
                    maf_min=self.args.kinship_maf_min,
                    snp_idx=getattr(self, 'chr_snp_idx', None))
        else:
            if not self.args.auto_pca:
                logger.info('[INFO] --no-auto-pca: proceeding with '
                            'intercept-only fitting (no population '
                            'stratification correction). Not recommended '
                            'unless deliberately testing/debugging or '
                            'analysing a cohort known to have no '
                            'population structure.\n')
            X0s = np.ones((self.inds_size, 1))
        self.get_ancestry_AM_instances(X0s)
        # Call an instance of the LMM class.
        logger.info("Calling instance of LMM ...")

        # Bug fix: LMM.__init__ silently drops any sample with a missing
        # (NaN) phenotype -- K, Y, X0 all get subset to the retained
        # samples internally (see LMM.__init__'s `x = True ^ np.isnan(Y)`
        # filtering). Every X_batch/covariate array built directly from
        # self.genotypes / self.lanc_array below, however, is read
        # independently and still covers ALL samples -- so if any
        # phenotype was actually missing, association_batch() would later
        # crash with a shape mismatch (self.Kve.T is (n_retained, n_retained)
        # but X_batch is still (n_full, b)). This could never actually
        # trigger before read_pheno_file() was fixed to convert PLINK's -9
        # missing-phenotype sentinel to NaN (previously returned as a plain
        # int array that could never hold NaN at all), so this is a
        # latent bug that predates that fix, only just made reachable.
        # Computed here (from self.phenotypes directly, not from whichever
        # model class ends up used) so it applies identically whether the
        # exact LMM class or the streaming dispatcher is used below --
        # CGLMM does not yet implement its own missing-value filtering
        # (a known, documented gap in jasmap.biobank.cg_kinship), so
        # filtering upfront here sidesteps that gap entirely rather than
        # needing every model class to handle it independently.
        nonmissing_mask = ~np.isnan(np.asarray(self.phenotypes, dtype=np.float64))

        if getattr(self, '_use_streaming_lmm', False):
            from jasmap.biobank.auto_lmm import build_lmm_from_bed
            logger.info('[INFO] Using the size-aware streaming dispatcher '
                        '(exact / low-rank / conjugate-gradient) for '
                        'kinship/LMM construction.\n')
            phenotypes_filtered = np.asarray(self.phenotypes, dtype=np.float64)[nonmissing_mask]
            X0s_filtered = X0s[nonmissing_mask, :]
            L, _lmm_strategy = build_lmm_from_bed(
                self.geno_file, phenotypes_filtered, X0=X0s_filtered,
                sample_idx=nonmissing_mask,
                memory_limit_bytes=self.args.kinship_memory_limit_gb * 1024**3,
                verbose=True)
            logger.info(f"[INFO] Kinship/LMM strategy chosen: "
                       f"{_lmm_strategy['chosen']} -- {_lmm_strategy['reason']}\n")
        else:
            # Bug fix: this previously passed X0s POSITIONALLY as the 3rd
            # argument, landing in Kva (eigenvalues) instead of X0 --
            # see the identical fix and full explanation at the
            # get_ancestry_AM_instances() call site above. Found via a
            # direct end-to-end comparison between this exact path and the
            # streaming dispatcher (which used X0= correctly) on the same
            # real data: beta estimates differed by 0.109 and p-values by
            # 0.789 -- far too large for approximation noise -- traced
            # through several other candidates (kinship formula, covariate
            # file parsing, genotype alignment, all independently verified
            # correct) before finding this was the actual cause. This means
            # every GWAS/admix run using --covfile has been silently
            # ignoring the user's covariates and fitting intercept-only,
            # project-wide, throughout every prior session.
            L = LMM(self.phenotypes, self.kinship, X0=X0s)
        _assert_covariates_used(L, X0s, "run_joint_analysis() main LMM")
        # self.kinship  and self.phenotype to be deleted
        # Obtain the pseudo heritability parameter (rho:p) from the function fit() in lmm class
        hmax, beta_null, sigma, LL_null = L.fit()
        ##logger.info('Pseudo-heritability for genotype = %s.\n\n\n'%str(hmax))

        ###########################################################################################
        #                    MAIN ANALYSIS
        ###########################################################################################

        # The analysis runs per SNPs by reading through the LAI dictionary with the LAI inference per SNP per ancestry
        # structure: self.lanc_dict = {SNP_1:{ANC_1:[LAI_1],ANC_2:[LAI_2],...,ANC_K:[LAI_K]},...,SNP_K:{...}}

        logger.info("Main analysis running ...")

        snp_iterable = self.lanc_dict if is_admix_mode else self.genotypes
        if len(snp_iterable) == 0:
            logger.info('Error: no SNPs available to analyze (self.genotypes is empty).\n')
            self.terminate(self.outfolder + self.logFile)
            sys.exit(1)

        # Bug fix (#11): self.no_of_snps was referenced later (as the
        # Bonferroni-style denominator for the SNP-only association's
        # non-centrality parameter, the direct analog of self.AM_eff_no_tests
        # for the admixture-mapping side) but never actually defined anywhere
        # -- an AttributeError that only surfaces once a run gets all the way
        # to the joint-association step, found via an actual end-to-end run.
        self.no_of_snps = len(snp_iterable)

        if not is_admix_mode:
            # Performance fix: GWAS mode no longer loops per SNP calling
            # L.association() one at a time -- all SNPs share the same fixed
            # covariates and heritability (h fixed by L.fit() above), so
            # association_batch() tests all of them in one vectorized pass
            # (see its docstring for the derivation; validated in
            # validate_association_batch.py to match the per-SNP results to
            # within floating-point precision, at 20-500x the speed).
            snp_order = list(self.genotypes.keys())
            X_batch = np.column_stack([self.genotypes[s] for s in snp_order])
            X_batch = X_batch[nonmissing_mask, :]
            betas, stderrs, ts_vals, ps_vals = L.association_batch(X_batch, h=hmax, REML=True)

            # Vectorized metadata lookup (single indexing op, not a per-row
            # .iloc() call) plus a single bulk write at the end -- matching
            # the "build lines, one bulk write" pattern already used for
            # admix mode's own output further below, rather than adding
            # per-row pandas overhead to what was already an unvectorized
            # per-SNP loop at up to hundreds of thousands of SNPs.
            snp_order_arr = np.array(snp_order) - 1  # self.snp_meta is 0-indexed
            chr_vals = self.snp_meta['chrom'].values[snp_order_arr]
            snp_vals = self.snp_meta['snp'].values[snp_order_arr]
            pos_vals = self.snp_meta['pos'].values[snp_order_arr]

            lines = []
            for idx in range(len(snp_order)):
                if np.isnan(ps_vals[idx]):
                    lines.append(f"{chr_vals[idx]}\t{snp_vals[idx]}\t{pos_vals[idx]}\tnan\tnan")
                else:
                    lines.append(f"{chr_vals[idx]}\t{snp_vals[idx]}\t{pos_vals[idx]}\t{betas[idx]}\t{ps_vals[idx]}")
            joint_fout.write("\n".join(lines) + "\n")

            joint_fout.close()
            logger.info('GWAS-mode batched association complete for %d SNPs.' % len(snp_order))
            return

        # Performance fix: admix mode's Part I (per-ancestry admixture mapping)
        # and Part II (joint posterior computation) previously looped over
        # every SNP in pure Python, calling L.association() (and, for each
        # ancestry, lai_L.association()) one SNP at a time -- exactly the
        # kind of per-SNP scalar loop association_batch() was built to
        # replace. This block does the same computation for every SNP and
        # every ancestry in a handful of vectorized calls instead.
        #
        # Validated in validate_admix_vectorized.py to reproduce the original
        # scalar per-SNP loop's output byte-for-byte on the small test
        # dataset before being trusted on larger data; end-to-end wall time
        # on a 300-sample/5000-SNP/3-ancestry benchmark dropped from ~80s to
        # a few seconds (see the README benchmark table).
        snp_order = sorted(self.lanc_dict.keys())
        n_snps = len(snp_order)
        n_samples = self.lanc_array.shape[2]

        # Vectorized metadata lookup (real chromosome/SNP ID/position,
        # instead of the positional line number previously written) --
        # computed once here, reused for both the per-ancestry
        # ANC{k}.AM.assoc files below and Jasmap.Joint.assoc further down,
        # since snp_order is the same set for both. See GWAS mode's
        # identical fix (and self.snp_meta's construction in
        # _populate_genotypes_from_tped()) for why this matters --
        # downstream enrichment analysis needs real genomic
        # coordinates/identifiers, not an arbitrary row index.
        snp_order_arr = np.array(snp_order) - 1  # self.snp_meta is 0-indexed
        chr_vals = self.snp_meta['chrom'].values[snp_order_arr]
        snp_vals = self.snp_meta['snp'].values[snp_order_arr]
        pos_vals = self.snp_meta['pos'].values[snp_order_arr]

        pval_matrix = np.full((n_snps, self.no_of_ancs), np.nan)
        beta_matrix = np.full((n_snps, self.no_of_ancs), np.nan)
        stderr_matrix = np.full((n_snps, self.no_of_ancs), np.nan)

        for anc in range(1, self.no_of_ancs + 1):
            anc_matrix = self.lanc_array[anc - 1].astype(np.float64)  # (n_snps, n_samples)
            lai_ave = anc_matrix.sum(axis=1) / (2.0 * n_samples)
            lai_var = 2 * lai_ave * (1 - lai_ave)
            valid = (lai_ave >= 0.01) & (lai_var != 0.0)

            with np.errstate(divide='ignore', invalid='ignore'):
                norm_anc = (anc_matrix - lai_ave[:, None]) / lai_var[:, None]
            norm_anc_safe = np.where(valid[:, None], norm_anc, 0.0)
            # Same nonmissing filtering as the GWAS-mode X_batch above --
            # lai_L (constructed further up from the SAME self.phenotypes)
            # has already internally dropped any missing-phenotype samples,
            # so the ancestry-dosage covariate needs the same samples-axis
            # filter before being handed to lai_L.association_batch().
            norm_anc_safe = norm_anc_safe[:, nonmissing_mask]

            lai_hmax = self.lai_h_dict[anc]
            lai_L = self.lai_L_instances[anc]
            Abetas, Astderrs, Ats_arr, Aps_arr = lai_L.association_batch(
                norm_anc_safe.T, h=lai_hmax, REML=True)

            Abetas = np.where(valid, Abetas, np.nan)
            Aps_arr = np.where(valid, Aps_arr, np.nan)
            Astderrs = np.where(valid, Astderrs, np.nan)

            beta_matrix[:, anc - 1] = Abetas
            pval_matrix[:, anc - 1] = Aps_arr
            stderr_matrix[:, anc - 1] = Astderrs

            anc_lines = ["CHR\tSNP\tPOS\tbeta\tPval"]
            for i in range(n_snps):
                anc_lines.append(f"{chr_vals[i]}\t{snp_vals[i]}\t{pos_vals[i]}\t{Abetas[i]}\t{Aps_arr[i]}")
            anc_prior_fout_dict[anc].write("\n".join(anc_lines) + "\n")

        # ---- Part II: choose, per SNP, the ancestry with the lowest AM p-value ----
        all_nan_rows = np.all(np.isnan(pval_matrix), axis=1)
        pval_filled = np.where(np.isnan(pval_matrix), np.inf, pval_matrix)
        min_idx = np.argmin(pval_filled, axis=1)
        min_pval = np.where(all_nan_rows, 0.9999, pval_matrix[np.arange(n_snps), min_idx])
        am_beta_chosen = np.where(all_nan_rows, np.nan, beta_matrix[np.arange(n_snps), min_idx])

        am_burden = self.AM_eff_no_tests
        am_lambda = (sts.norm.ppf(1 - 0.05 / float(am_burden) / 2) + sts.norm.ppf(0.8)) ** 2
        am_prior = 1 / float(am_burden)
        am_chisq = sts.chi2.ppf(1 - min_pval, 1)
        am_chisq = np.where(np.isinf(am_chisq), 68.7632522116684, am_chisq)
        am_pdf_lambda = sts.ncx2.pdf(am_chisq, 1, nc=am_lambda)
        am_pdf_0 = sts.chi2.pdf(am_chisq, 1)
        am_post = (am_pdf_lambda * am_prior) / (am_pdf_lambda * am_prior + am_pdf_0 * (1 - am_prior))

        # ---- Part 2.2: SNP-only association, batched across all SNPs at once ----
        X_batch = np.column_stack([self.genotypes[s] for s in snp_order])
        X_batch = X_batch[nonmissing_mask, :]
        betas2, stderrs2, ts2, ps2 = L.association_batch(X_batch, h=hmax, REML=True)

        valid_snp_assoc = ~np.isnan(ps2)
        ps2_capped = np.where(ps2 >= 1.0, 0.99999999, ps2)
        assoc_lambda = (sts.norm.ppf(1 - 0.05 / float(self.no_of_snps) / 2.0) + sts.norm.ppf(0.8)) ** 2
        assoc_chisq = sts.chi2.ppf(1 - ps2_capped, 1)
        assoc_chisq = np.where(np.isinf(assoc_chisq), 68.7632522116684, assoc_chisq)
        joint_pdf_lambda = sts.ncx2.pdf(assoc_chisq, 1, nc=assoc_lambda)
        joint_pdf_0 = sts.chi2.pdf(assoc_chisq, 1)
        joint_post = (joint_pdf_lambda * am_post) / (joint_pdf_lambda * am_post + joint_pdf_0 * (1 - am_post))
        joint_post = np.where(joint_post >= 1.0, 0.99999999, joint_post)

        # Per-SNP dominant ancestry (highest aggregate local-ancestry dosage
        # across the cohort, mapped to the real --pop-labels name when
        # available) -- computed once here rather than per-row, since it's
        # the same lookup for every SNP in this file. Lets a row in this
        # output be read directly as "this SNP is associated, AND this is
        # the ancestral population most enriched at that locus" without
        # needing to separately cross-reference the per-ancestry AM.assoc
        # files.
        dominant_ancestry_map = self._compute_dominant_ancestry_per_snp()
        # Companion "Gene" column, reusing whatever --gene-annot is set
        # (user-provided or auto-built via --annot) -- empty dict (all
        # SNPs fall back to "Intergenic") when no annotation is available
        # at all, matching RVAT's own default terminology for this case.
        gene_per_snp_map = self._compute_gene_per_snp()

        joint_lines = ["CHR\tSNP\tPOS\tanc\tAM_beta\tAM_Pval\tSNP_beta\tSNP_Pval\tJoint_PPA\tAncestry\tGene"]
        for i, snp in enumerate(snp_order):
            if all_nan_rows[i]:
                anc_field, am_beta_field, am_pval_field = "nan", "nan", "nan"
            else:
                anc_field = str(min_idx[i] + 1)
                am_beta_field = str(am_beta_chosen[i])
                am_pval_field = str(min_pval[i])
            if not valid_snp_assoc[i]:
                snp_beta_field, snp_pval_field, joint_field = str(betas2[i]), "nan", "nan"
            else:
                snp_beta_field = str(betas2[i])
                snp_pval_field = str(ps2[i])
                joint_field = str(joint_post[i])
            ancestry_field = dominant_ancestry_map.get(snp, "Unknown")
            gene_field = gene_per_snp_map.get(snp, "Intergenic")
            joint_lines.append("\t".join([str(chr_vals[i]), str(snp_vals[i]), str(pos_vals[i]),
                                          anc_field, am_beta_field, am_pval_field,
                                          snp_beta_field, snp_pval_field, joint_field,
                                          ancestry_field, gene_field]))
        joint_fout.write("\n".join(joint_lines) + "\n")

        # Close the files opened for writting
        joint_fout.close()
        for idx2 in anc_prior_fout_dict:
            anc_prior_fout_dict[idx2].close()


class JasMAP(Joint_Association):
    '''Performs the overrall steps of JasMAP as described in the JasMAP method'''

    def _get_study_cohort_mask(self, y, genotype_sample_ids=None):
        """
        Identifies which samples belong to the phenotyped study cohort (as
        opposed to reference/ancestral population samples included only
        for --yes-anc's local ancestry inference), for RVAT's MAF/gene-
        level filtering, auto-PCA restriction, and PRS output restriction.

        Priority:
          1. --pop-file's --admix-label match, if --pop-file was given --
             an EXPLICIT signal, not inferred from phenotype at all.
             Previously, every one of these restrictions relied SOLELY on
             phenotype validity (~np.isnan(y)) to infer "study cohort" --
             which is only a proxy, not the true signal: an admixed
             sample could have a missing phenotype for an unrelated
             reason (e.g. dropout), and a reference sample could
             accidentally have a non-missing value from a data-entry
             error -- phenotype validity alone would mishandle both.
          2. Phenotype validity as a fallback, when --pop-file wasn't
             given at all (preserves the previous, already-validated
             behaviour for that case exactly).

        When BOTH signals are available, any disagreement between them is
        logged as a warning -- surfacing a likely data issue rather than
        silently picking one signal over the other without telling the user.

        Parameters
        ----------
        y                   : phenotype array
        genotype_sample_ids : the GENOTYPE FILE's own sample order (not
                              necessarily self.samples_list, which is
                              --pheno's own order and may differ after
                              _align_pheno_to_genotype_samples()'s
                              realignment) -- e.g. list(bed.iid), ind_ids,
                              or fam["IID"].values, matching y's own order
                              at the call site. Falls back to phenotype
                              validity alone if not given or mismatched.

        Returns
        -------
        boolean ndarray, length len(y), True for study-cohort samples
        """
        y_valid_mask = ~np.isnan(np.asarray(y, dtype=np.float64))

        if not self.args.pop_file:
            return y_valid_mask

        try:
            pf = pd.read_csv(self.args.pop_file, sep=r"\s+", header=None,
                             names=["ID", "POP"], engine="python")
            id_to_pop = dict(zip(pf["ID"].astype(str), pf["POP"].astype(str)))
        except Exception as e:
            logger.warning(f"[INFO] Could not read --pop-file for study-cohort "
                           f"identification ({e}); falling back to phenotype "
                           f"validity alone.")
            return y_valid_mask

        if genotype_sample_ids is None or len(genotype_sample_ids) != len(y):
            logger.warning(
                f"[INFO] --pop-file given but matching sample IDs were not "
                f"provided for study-cohort identification; falling back "
                f"to phenotype validity alone.")
            return y_valid_mask

        label_mask = np.array([id_to_pop.get(str(s)) == self.args.admix_label
                               for s in genotype_sample_ids])

        n_disagree = int((label_mask != y_valid_mask).sum())
        if n_disagree > 0:
            logger.warning(
                f"[WARNING] {n_disagree} sample(s) disagree between "
                f"--pop-file's {self.args.admix_label!r} label and "
                f"phenotype validity (e.g. labelled {self.args.admix_label!r} "
                f"but missing phenotype, or vice versa) -- using the "
                f"--pop-file label as the authoritative signal for "
                f"study-cohort identification. Verify this is intentional.")

        return label_mask

    def _get_dense_genotype_for_mode(self):
        """
        Reads (and caches) the dense genotype matrix used by GIFT/GIFT-ADMIX
        and --consgwas -- both need the SAME, full-panel dense G (bed_reader
        for --bfile, the shared .tmp.tped file otherwise), with byte-for-byte
        identical read logic that was previously duplicated independently in
        _run_gift_mode() and _run_consgwas().

        Why this exists
        -----------------
        A run combining --mode gift/gift_admix with --consgwas (a common,
        useful combination -- --consgwas picks up the mode's own results
        automatically) read the ENTIRE dense genotype matrix TWICE: once in
        _run_gift_mode(), a second time, completely independently, in
        _run_consgwas(). For a real, 272,791-SNP x 1,363-sample panel, that
        first read alone took over 4 minutes (found directly in a real run's
        log) -- doing it twice for no reason adds several more minutes to
        every combined run for zero benefit, since both consumers want
        EXACTLY the same data (same bed.read() call, same chr_idx, same
        dtype). Caching on self after the first read and reusing it for the
        second consumer removes that duplicate cost entirely.

        Safe to cache and reuse as-is (not copies) because every actual
        consumer already copies before modifying rather than mutating in
        place -- confirmed directly: impute_missing_genotypes() (used by
        both GIFT's own imputation and ConsensusGWAS.run()'s) does
        `G_imputed = G.copy()` before writing into it, never touching its
        input array.

        Returns
        -------
        (G, bim, fam) -- same shapes/columns either read path would have
        produced; G is None (with bim/fam None too) if no genotype input is
        configured at all (should not happen in practice, since --bfile/
        --geno/--vcf is required earlier in argument validation).
        """
        cached = getattr(self, "_dense_G_cache", None)
        if cached is not None:
            logger.info("[INFO] Reusing already-read genotype matrix "
                       "(GIFT/consgwas share the same dense read -- "
                       "avoiding a second, several-minute re-read).")
            return cached

        if self.args.bfile is not None:
            from bed_reader import open_bed
            bed = open_bed(self.args.bfile + ".bed", count_A1=False)
            chr_idx = getattr(self, 'chr_snp_idx', None)
            read_idx = (None, chr_idx) if chr_idx is not None else None
            G = bed.read(index=read_idx).astype(np.float32)
            fam = pd.read_csv(self.args.bfile + ".fam", sep=r"\s+", header=None,
                              names=["FID","IID","PAT","MAT","SEX","PHENO"])
            bim = pd.read_csv(self.args.bfile + ".bim", sep=r"\s+", header=None,
                              names=["CHR","SNP","CM","BP","A1","A2"])
            if chr_idx is not None:
                bim = bim.iloc[chr_idx].reset_index(drop=True)
        elif self.args.geno is not None or self.args.vcf is not None:
            import polars as pl
            tped = self._tped_path()
            if not os.path.exists(tped):
                return None, None, None
            df = pl.read_csv(tped, separator="\t", null_values="NA")
            G   = df[:, 6:].to_numpy().astype(np.float32).T
            bim = df.select(df.columns[:6]).to_pandas()
            bim.columns = ["CHR","SNP","CM","BP","A1","A2"]
            fam_path = (self.geno_file + ".ind" if self.args.geno
                        else self.args.vcf + ".ind")
            try:
                fam = pd.read_csv(fam_path, sep=r"\s+", header=None,
                                  names=["FID","IID","SEX"])
            except Exception:
                fam = pd.DataFrame({"FID": list(df.columns[6:]),
                                    "IID": list(df.columns[6:])})
        else:
            return None, None, None

        self._dense_G_cache = (G, bim, fam)
        return G, bim, fam

    def _ancestry_index_to_label(self, idx):
        """
        Maps a 1-based ancestry index to its real population label from
        --pop-labels, if available, falling back to a plain "ANC{idx}"
        string when no --pop-labels was given -- matching the same
        fallback convention already used elsewhere for identifying
        ancestries without explicit labels.
        """
        pop_labels = getattr(self, "_pop_labels", None)
        if pop_labels is not None and 1 <= idx <= len(pop_labels):
            return pop_labels[idx - 1]
        return f"ANC{idx}"

    def _compute_dominant_ancestry_per_snp(self):
        """
        Returns {snp_lineno: ancestry_label} -- for each SNP, the ancestral
        population with the highest AGGREGATE local-ancestry dosage across
        the cohort (summed haplotype-copy count across all samples, not a
        per-sample or per-test-statistic measure). Used to add an
        "Ancestry" column to admix/gift_admix's association and PRS
        outputs, so a disease-associated SNP's row directly shows which
        ancestral population is most enriched at that locus -- the
        aggregation method requested for this feature specifically.

        Prefers self.lanc_array when already available (a dense
        (K, n_snps, n_individuals) numpy array -- built by the --ancfile
        read path) for one fast, vectorized .sum(axis=2)/.argmax(axis=0)
        pass. Falls back to reconstructing the same computation from
        self.lanc_dict (the {snp: {ancestry: [per_sample_values]}}
        structure that's ALWAYS populated regardless of source -- see
        this method's own investigation: --yes-anc's consensus LAI path
        sets self.lanc_dict directly from the LAI object's own result,
        without also copying its lanc_array onto self) when the fast
        path isn't available, so this works correctly regardless of
        whether ancestry came from --ancfile or --yes-anc.
        """
        if not getattr(self, "lanc_dict", None):
            return {}

        K = self.no_of_ancs
        lineno_list = sorted(self.lanc_dict.keys())

        lanc_array = getattr(self, "lanc_array", None)
        if lanc_array is None:
            lai_obj = getattr(self, "_lai_obj", None)
            if lai_obj is not None:
                lanc_array = getattr(lai_obj, "lanc_array", None)

        if lanc_array is not None and lanc_array.shape[0] == K:
            # Fast path: lanc_array is (K, n_snps, n_individuals), already
            # ordered by SNP index 0..n_snps-1 matching lineno-1.
            agg = lanc_array.sum(axis=2)  # (K, n_snps)
            dominant_idx = agg.argmax(axis=0) + 1  # 1-based
            return {lineno_list[i]: self._ancestry_index_to_label(int(dominant_idx[i]))
                   for i in range(len(lineno_list))}

        # Fallback: reconstruct from lanc_dict directly (correct regardless
        # of source, just not vectorized).
        result = {}
        for lineno in lineno_list:
            per_anc = self.lanc_dict[lineno]
            best_k, best_sum = None, -1
            for k in range(1, K + 1):
                vals = per_anc.get(k)
                if vals is None:
                    continue
                s = sum(vals)
                if s > best_sum:
                    best_sum, best_k = s, k
            result[lineno] = self._ancestry_index_to_label(best_k) if best_k else "Unknown"
        return result

    def _compute_dominant_ancestry_by_snp_name(self):
        """
        Same computation as _compute_dominant_ancestry_per_snp(), keyed by
        SNP NAME instead of positional line number -- for consumers like
        PRS's per-SNP weights file (prs_snp_weights.tsv), which identifies
        rows by SNP ID rather than the 1-based line number self.lanc_dict
        uses internally. Reuses self.snp_meta (populated once, by the same
        _populate_genotypes_from_tped() shared across --bfile/--geno/--vcf,
        so this works regardless of input format) to translate line number
        -> real SNP name.
        """
        per_lineno = self._compute_dominant_ancestry_per_snp()
        if not per_lineno or getattr(self, "snp_meta", None) is None:
            return {}
        snp_names = self.snp_meta["snp"].astype(str).tolist()
        # self.snp_meta's row i (0-based) corresponds to lineno i+1.
        return {snp_names[lineno - 1]: label
               for lineno, label in per_lineno.items()
               if 0 <= lineno - 1 < len(snp_names)}

    def _compute_snp_to_gene_dict(self):
        """
        Core computation shared by _compute_gene_per_snp() (lineno-keyed,
        for Jasmap.Joint.assoc) and _compute_gene_per_snp_by_name()
        (SNP-name-keyed, for PRS's prs_snp_weights.tsv) -- both need the
        same underlying assign_variants_to_genes() call; this avoids
        running it twice. Returns {snp_name: gene_name}, excluding
        "intergenic" SNPs entirely (callers should treat a missing key as
        "no gene", not as an error).
        """
        if not self.args.gene_annot:
            return {}
        if getattr(self, "snp_meta", None) is None:
            return {}
        try:
            from jasmap.rvat.rvat_engine import load_gene_annotation, assign_variants_to_genes
            gene_df = load_gene_annotation(self.args.gene_annot)
            bim_for_match = self.snp_meta.rename(
                columns={"chrom": "CHR", "snp": "SNP", "pos": "BP"})
            assigned = assign_variants_to_genes(bim_for_match, gene_df)
            snp_to_gene = dict(zip(assigned["SNP"].astype(str), assigned["GENE"]))
            return {name: gene for name, gene in snp_to_gene.items()
                   if pd.notna(gene) and gene not in ("intergenic", "Intergenic")}
        except Exception as e:
            logger.warning(f"[INFO] Could not compute per-SNP gene "
                           f"assignments from --gene-annot: {e}")
            return {}

    def _compute_gene_per_snp(self):
        """
        Returns {snp_lineno: gene_name}, for Jasmap.Joint.assoc's own
        loop (which identifies rows by lineno, via snp_order). See
        _compute_snp_to_gene_dict() for the shared underlying computation.
        """
        snp_to_gene = self._compute_snp_to_gene_dict()
        if not snp_to_gene or getattr(self, "snp_meta", None) is None:
            return {}
        snp_names = self.snp_meta["snp"].astype(str).tolist()
        return {i + 1: snp_to_gene[name] for i, name in enumerate(snp_names)
               if name in snp_to_gene}

    def _compute_gene_per_snp_by_name(self):
        """
        Returns {snp_name: gene_name}, for PRS's prs_snp_weights.tsv,
        which identifies rows by SNP ID rather than line number. See
        _compute_snp_to_gene_dict() for the shared underlying computation.
        """
        return self._compute_snp_to_gene_dict()

    def _maybe_auto_build_gene_annotation(self):
        """
        --annot support: when given (and neither --gene-annot nor
        --var-annot was already provided explicitly), automatically
        constructs a gene-annotation BED file scoped to this study's own
        SNPs, and sets self.args.gene_annot to it so RVAT's existing,
        unmodified --gene-annot logic picks it up transparently -- no
        separate code path needed downstream.

        Reuses jasmap.annotation.build_gene_annotation, a standalone
        utility already built and validated independently (real download
        source verified, real hg38 data used to find and fix two
        aggregation bugs, real end-to-end RVAT integration tested) before
        being wired in here -- this function is a thin adapter, not a
        reimplementation.

        On ANY failure (no internet, download error, zero genes matched,
        etc.) this logs a clear reason and leaves self.args.gene_annot
        untouched (None), so RVAT falls through to its existing, already-
        correct default: --var-annot if given, else 50-SNP sliding-window
        proxies. A failed auto-annotation attempt must never be mistaken
        for "the user wants sliding windows" vs. "the user wanted real
        genes but this quietly didn't work" -- hence the explicit log
        line either way.
        """
        if not self.args.annot:
            return
        if self.args.gene_annot is not None or self.args.var_annot is not None:
            logger.info("[INFO] --annot given, but --gene-annot/--var-annot "
                       "was also provided explicitly -- using that instead "
                       "of auto-building a new one.")
            return

        bim_path = (self.args.bfile + ".bim") if self.args.bfile else None
        if bim_path is None or not os.path.exists(bim_path):
            logger.info("[WARNING] --annot requires --bfile (a .bim file to "
                       "read SNP positions from) -- not available for this "
                       "input format. Falling back to the default gene-"
                       "proxy behavior.")
            return

        logger.info(f"[--annot] Attempting to auto-build a gene annotation "
                   f"for genome build {self.args.genome_build!r} ...")
        try:
            from jasmap.annotation.build_gene_annotation import (
                read_bim_chrom_ranges, get_cached_or_download, parse_gtf_genes)

            chrom_ranges = read_bim_chrom_ranges(bim_path)
            ref_path = get_cached_or_download(
                self.args.genome_build,
                os.path.expanduser("~/.jasmap_annotation_cache"),
                timeout=30, force_download=False)
            if ref_path is None:
                logger.info("[--annot] Reference annotation unavailable "
                           "(no internet connection, or the source is "
                           "unreachable) -- falling back to the default "
                           "gene-proxy behavior for this run. A cached copy "
                           "will be used automatically once one is "
                           "successfully downloaded on a connected run.")
                return

            all_genes = parse_gtf_genes(ref_path, max_cluster_gap=500_000)
            kept = [(c, s, e, g) for c, s, e, g in all_genes
                   if c in chrom_ranges and e >= chrom_ranges[c][0]
                   and s <= chrom_ranges[c][1]]
            if not kept:
                logger.info("[--annot] 0 reference genes overlap this "
                           "study's SNPs -- check --genome-build is correct "
                           "for this data. Falling back to the default "
                           "gene-proxy behavior.")
                return

            out_path = self.outfolder + "auto_gene_annotation.bed"
            kept.sort(key=lambda r: (r[0], r[1]))
            with open(out_path, "w") as f:
                for c, s, e, g in kept:
                    f.write(f"{c}\t{s}\t{e}\t{g}\n")

            self.args.gene_annot = out_path
            logger.info(f"[--annot] Auto-built gene annotation: {len(kept)} "
                       f"gene(s) relevant to this study's SNPs -> {out_path}. "
                       f"Using this for --mode rvat's gene-based tests.")
        except Exception as e:
            logger.info(f"[--annot] Auto-build failed ({type(e).__name__}: "
                       f"{e}) -- falling back to the default gene-proxy "
                       f"behavior for this run.")

    # ──────────────────────────────────────────────────────────────────────────
    # Consensus LAI pipeline (--yes-anc)
    # ──────────────────────────────────────────────────────────────────────────

    def _run_lai_pipeline(self):
        """
        Steps 1-8 consensus LAI (Document 1).  Called when --yes-anc is set.
        Populates self.lanc_dict, self.lanc_array, self.global_ancestry_q
        and sets self.lanc_file to the written ancfile path.
        """
        logger.info("\n" + "="*60)
        logger.info("[LAI] --yes-anc: running consensus local ancestry pipeline")
        logger.info("="*60 + "\n")

        from jasmap.lai.consensus_lai import ConsensusLAI

        pop_labels = self._pop_labels  # may be None

        # Bug fix: out_prefix must ALWAYS be inside the --out output folder.
        # Previously, when --ancfile was also provided, out_prefix was derived
        # from the --ancfile path (e.g. ".\test_data\ancestry_space"), which
        # caused all consensus LAI outputs (global ancestry TSV, ancfile, plots)
        # to land in the test-data directory instead of the --out folder.
        # --ancfile when combined with --yes-anc now serves ONLY as a hint for
        # the final ancfile basename; all outputs still go to self.outfolder.
        if self.args.ancfile:
            ancfile_basename = os.path.splitext(
                os.path.basename(self.args.ancfile))[0]
            out_prefix = self.outfolder + ancfile_basename
        else:
            out_prefix = self.outfolder + "consensus_lai"

        # Read per-sample labels from --pop-file so ConsensusLAI can pass them
        # to the PCA plots — this makes admixed samples appear as "ADMIX" in
        # both the ancestry-proportion PCA and the genotype PCA, rather than
        # being absorbed into the nearest reference-population cluster.
        sample_pop_labels = None
        if self.args.pop_file:
            try:
                pf = pd.read_csv(self.args.pop_file, sep=r"\s+", header=None,
                                 names=["ID","POP"], engine="python")
                # Align to sample_ids order after genotype reading
                # (done below after sample_ids is available)
                _pop_file_df = pf
            except Exception:
                _pop_file_df = None
        else:
            _pop_file_df = None

        lai = ConsensusLAI(
            n_ancestries=self.args.nbanc,
            pop_labels=pop_labels,
            out_prefix=out_prefix,
            seed=42)

        # G is available from whichever input path was read; phenotypes loaded
        # We need the dosage matrix in float32 form.  Re-read from the tped.
        import polars as pl
        tped_file = self._tped_path()
        if not os.path.exists(tped_file):
            logger.info("[LAI] No tmp.tped found - re-reading genotypes for LAI ...")
            self.run_joint_analysis.__func__   # ensure read functions exist

            # --lai-genome-wide: temporarily bypass --chr for THIS read only,
            # so local ancestry is inferred from the full genome even when
            # --chr is set. Restored immediately after (try/finally, so an
            # error during the read can't leave --chr permanently cleared)
            # -- the LATER, SEPARATE genotype read that run_joint_analysis()/
            # _run_gift_mode() do for the actual --mode admix/gift_admix
            # analysis happens after this function returns, with --chr back
            # to its real value, so that step still gets correctly
            # restricted to the requested chromosome. Without this, --chr
            # restricted BOTH steps identically -- there was no way to run
            # LAI genome-wide while still testing one chromosome, exactly
            # the workflow a real user asked for.
            _orig_chr = self.args.chr
            if self.args.lai_genome_wide and self.args.chr is not None:
                logger.info(
                    f"[LAI] --lai-genome-wide: computing local ancestry from "
                    f"the FULL genome (ignoring --chr {self.args.chr!r} for "
                    f"this step only) -- --chr will still restrict the "
                    f"downstream --mode {self.args.mode} analysis to that "
                    f"chromosome.\n")
                self.args.chr = None
            try:
                if self.args.geno is not None:
                    self.read_geno_file()
                elif self.args.bfile is not None:
                    self.read_bed_file()
                elif self.args.vcf is not None:
                    self.vcf_to_dataframe()
                # Bug fix: tped_file was computed ONCE, above, before this
                # temporary self.args.chr override -- so when
                # --lai-genome-wide actually changes self.args.chr (to
                # None) for this read, the file that gets WRITTEN lands
                # at a DIFFERENT path (tmp_all.tped) than the STALE
                # tped_file variable still pointed to (tmp_chr22.tped),
                # crashing with FileNotFoundError the moment the read
                # below tried to open a file that was never written.
                # Recomputing here, still inside this try block (so
                # self.args.chr is still at whatever value was ACTUALLY
                # used for the read that just happened), keeps tped_file
                # in sync with reality regardless of whether
                # --lai-genome-wide changed anything.
                tped_file = self._tped_path()
            finally:
                self.args.chr = _orig_chr

        all_gen = pl.read_csv(tped_file, separator="\t", null_values="NA")
        G_lai = all_gen[:, 6:].to_numpy().astype(np.float32).T  # (n_samples, n_snps)

        sample_ids = (list(all_gen.columns)[6:] if len(all_gen.columns) > 6
                      else [f"S{i}" for i in range(G_lai.shape[0])])

        # Align per-sample population labels to sample_ids order
        if _pop_file_df is not None:
            id_to_pop = dict(zip(_pop_file_df["ID"].astype(str),
                                 _pop_file_df["POP"].astype(str)))
            sample_pop_labels = [id_to_pop.get(str(sid), "Unknown")
                                 for sid in sample_ids]
            lai.sample_pop_labels = sample_pop_labels
        elif hasattr(self, '_sample_pop_labels_for_lai'):
            # Use the dict cached during validation (already parsed earlier)
            id_to_pop = self._sample_pop_labels_for_lai
            sample_pop_labels = [id_to_pop.get(str(sid), "Unknown")
                                 for sid in sample_ids]
            lai.sample_pop_labels = sample_pop_labels

        try:
            bim = all_gen.select(all_gen.columns[:6]).to_pandas()
        except Exception:
            bim = None

        lai.fit(G_lai, sample_ids=sample_ids, bim=bim)

        # Wire outputs back into JasMAP's internal state
        self.lanc_dict        = lai.lanc_dict
        self.lanc_array       = lai.lanc_array
        self.global_ancestry_q = lai.Q_
        self.lanc_file        = lai.anc_path
        self._lai_obj         = lai

        logger.info(f"[LAI] Consensus LAI complete. ancfile -> {self.lanc_file}\n")

        # Publication plots
        if self.args.plots:
            # Bug fix: same "one shared try/except for every plot" issue
            # already found and fixed in run_prs()'s and
            # _plot_main_results()' own plotting blocks -- a single plot
            # failing here skipped every REMAINING, independent plot too.
            # Confirmed on a real run: lai_heatmap crashed on matplotlib's
            # canvas-size limit, and the heatmap, BOTH PCA plots, and all
            # karyograms were silently never attempted, even though only
            # one of them was actually broken.
            def _safe_plot(name, fn, *a, **kw):
                try:
                    fn(*a, **kw)
                except Exception as e:
                    logger.warning(f"[LAI] Plot {name!r} failed: {e}")

            _safe_plot("lai_global_ancestry", lai.plot_global,
                      out_path=self.outfolder + "lai_global_ancestry.png")
            _safe_plot("lai_heatmap", lai.plot_lai_heatmap,
                      out_path=self.outfolder + "lai_heatmap.png")
            # Ancestry-proportion PCA (admixed samples shown as "ADMIX")
            _safe_plot("lai_pca", lai.plot_pca,
                      out_path=self.outfolder + "lai_pca.png")
            # Genotype-level PCA (admixed samples between reference clusters)
            _safe_plot("lai_genotype_pca", lai.plot_genotype_pca,
                      out_path=self.outfolder + "lai_genotype_pca.png")
            # Karyogram for the first few samples
            _safe_plot("karyograms", lai.plot_karyograms_all,
                      out_dir=self.outfolder, max_samples=5)
            try:
                # LAI region example: first chromosome found, full span
                if lai._bim is not None:
                    try:
                        bim_cols = {c.upper(): c for c in lai._bim.columns}
                        chr_col  = bim_cols.get("CHR", bim_cols.get("CHROM", None))
                        bp_col   = bim_cols.get("BP",  bim_cols.get("POS", None))
                        if chr_col and bp_col:
                            first_chr = int(str(lai._bim[chr_col].iloc[0]).replace("chr",""))
                            # Bug fix: bp_mb (and therefore start_mb/end_mb)
                            # was computed from the WHOLE, unfiltered _bim
                            # column -- fine when _bim only ever covered one
                            # chromosome, but --lai-genome-wide deliberately
                            # makes _bim span the ENTIRE genome while --chr
                            # still restricts the main analysis to one
                            # chromosome. Without filtering to first_chr
                            # first, start_mb/end_mb spanned every
                            # chromosome's positions at once (each restarting
                            # near 0, so min() stayed small but max() picked
                            # up whichever chromosome happens to have the
                            # largest position) -- plot_lai_region() then
                            # tried to render/allocate a region spanning
                            # hundreds of millions of bp, failing with "bad
                            # allocation". Found via a real run using
                            # --lai-genome-wide + --chr on real, multi-
                            # chromosome genome-wide data.
                            chr_mask  = lai._bim[chr_col].astype(str).str.replace("chr", "", regex=False) == str(first_chr)
                            bp_mb     = lai._bim.loc[chr_mask, bp_col].astype(float) / 1e6
                            start_mb  = float(bp_mb.min())
                            end_mb    = float(bp_mb.max())
                            # Bug fix: even correctly restricted to
                            # first_chr's own SNPs (the fix above this
                            # one), plotting a FULL chromosome's span
                            # (e.g. ~249 Mb for chr1) is still what
                            # crashed with "bad allocation" --
                            # lai_region_plot()'s figsize scales directly
                            # with (end_mb - start_mb), so a full-
                            # chromosome span at dpi=300 demands a canvas
                            # on the order of 100+ GB. That function was
                            # designed for a small, targeted region (its
                            # own name and docstring: "a specific genomic
                            # region"), not an entire chromosome -- this
                            # auto-plot is just meant to be one
                            # illustrative example, so capping it to a
                            # fixed, reasonable window keeps that intent
                            # regardless of how large the actual
                            # chromosome is. Found via a real run: even
                            # after the first fix (restricting to
                            # first_chr only, not the whole genome), the
                            # SAME "bad allocation" error persisted.
                            _region_cap_mb = 5.0
                            if end_mb - start_mb > _region_cap_mb:
                                end_mb = start_mb + _region_cap_mb
                            pheno_arr = (np.asarray(self.phenotypes, dtype=float) - 1
                                         if hasattr(self, 'phenotypes') else None)
                            lai.plot_lai_region(
                                chrom    = first_chr,
                                start_mb = start_mb,
                                end_mb   = end_mb,
                                phenotype= pheno_arr,
                                out_path = self.outfolder + f"lai_region_chr{first_chr}.png")
                    except Exception as e2:
                        logger.warning(f"[LAI] LAI region auto-plot failed: {e2}")
            except Exception as e:
                logger.warning(f"[LAI] Plotting failed: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # GIFT / GIFT-ADMIX dispatch
    # ──────────────────────────────────────────────────────────────────────────

    def _run_gift_mode(self):
        """Dispatch --mode gift or --mode gift_admix."""
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

        from jasmap.gift.gift_engine import run_gift, run_gift_admix

        # Optimization: shared, cached dense-genotype read -- see
        # _get_dense_genotype_for_mode()'s own docstring for why (this used
        # to independently re-read the ENTIRE panel a second time, taking
        # several extra minutes for no reason, whenever --consgwas also ran
        # in the same command).
        G, bim, fam = self._get_dense_genotype_for_mode()

        # Bug fix: G is read directly here via bed_reader (--bfile) or the
        # tped file (--geno/--vcf) -- a SEPARATE code path from
        # self.genotypes' own population (_populate_genotypes_from_tped()),
        # which already does per-SNP mean imputation. This path never did,
        # meaning any missing genotype call fed straight into GIFT/
        # GIFT-ADMIX's internal PCA/NMF steps with no handling at all.
        # Found via a real run: --mode gift_admix on real data with
        # genuine missing calls (a merged multi-reference-panel admixed
        # cohort, where SNP coverage rarely overlaps 100% across distinct
        # genotyping arrays) crashed with sklearn's "Input X contains
        # NaN" inside gift_fullpipeline.py's _feats() -- the third
        # separate location this exact class of bug has surfaced in real
        # data this session (LAI's NMF step, and PRS, were the other
        # two). Imputed once, here, right after G is populated by
        # EITHER branch above, covers both input paths consistently
        # rather than needing a fix inside GIFT's own internals.
        from jasmap.prs.consensus_prs import impute_missing_genotypes
        G = impute_missing_genotypes(G, tag="GIFT")

        y = self.phenotypes.astype(np.float64) - 1  # convert 1/2 -> 0/1

        out_gift = self.outfolder + self.args.mode + "_results"

        if self.args.mode == "gift":
            snp_scores, y_pred, model = run_gift(
                G=G, y=y, bim=bim, fam=fam,
                out_prefix=out_gift,
                n_components=self.args.gift_n_components,
                lambda_reg=self.args.gift_lambda,
                mi_chunk=self.args.gift_mi_chunk,
                max_snps=(self.args.gift_max_snps if self.args.gift_max_snps > 0 else None),
                seed=42,
                plots=self.args.plots)
            self._gift_snp_scores = snp_scores
            self._gift_model      = model

        elif self.args.mode == "gift_admix":
            # Get local / global ancestry
            local_anc  = getattr(self, 'lanc_array', None)
            global_anc = getattr(self, 'global_ancestry_q', None)

            # Prefer the window-level (n_samples, n_windows, K) L_ array from
            # the ConsensusLAI object when available -- run_gift_admix expects
            # (n_samples, n_windows, K), not the (K, n_snps, n_samples) lanc_array.
            if hasattr(self, '_lai_obj') and self._lai_obj is not None:
                local_anc  = self._lai_obj.L_   # (n_samples, n_windows, K)
                global_anc = self._lai_obj.Q_   # (n_samples, K)
            elif local_anc is not None and len(local_anc.shape) == 3 and local_anc.shape[0] == self.no_of_ancs:
                # lanc_array shape is (K, n_snps, n_samples); convert to
                # (n_samples, n_snps, K) for gift_admix compatibility
                local_anc = local_anc.transpose(2, 1, 0)  # (n_samples, n_snps, K)
                # compute simple global ancestry from dosage sums
                if global_anc is None:
                    dos_sum = local_anc.sum(axis=1)  # (n_samples, K)
                    global_anc = dos_sum / (dos_sum.sum(1, keepdims=True) + 1e-12)

            if global_anc is None:
                logger.info("[GIFT-ADMIX] No global ancestry available - "
                            "computing from genotypes ...")
                from jasmap.lai.consensus_lai import ConsensusLAI
                # Pass per-sample population labels so plots include ADMIX group
                spl = getattr(self, '_sample_pop_labels_for_lai', None)
                lai = ConsensusLAI(n_ancestries=self.no_of_ancs, seed=42,
                                   pop_labels=self._pop_labels,
                                   sample_pop_labels=spl,
                                   out_prefix=self.outfolder + "gift_admix_lai")
                lai.fit(G)
                local_anc  = lai.L_
                global_anc = lai.Q_
                self._lai_obj = lai
                if self.args.plots:
                    try:
                        lai.plot_global(out_path=self.outfolder+"lai_global_ancestry.png")
                        lai.plot_pca(out_path=self.outfolder+"lai_pca.png")
                    except Exception: pass

            snp_scores, y_pred, model, admix_df = run_gift_admix(
                G=G, y=y, bim=bim, fam=fam,
                local_ancestry=local_anc,
                global_ancestry=global_anc,
                out_prefix=out_gift,
                n_ancestries=self.no_of_ancs,
                ancestry_mode=self.args.gift_ancestry_mode,
                n_components=self.args.gift_n_components,
                lambda_reg=self.args.gift_lambda,
                mi_chunk=self.args.gift_mi_chunk,
                pop_labels=self._pop_labels,
                max_snps=(self.args.gift_max_snps if self.args.gift_max_snps > 0 else None),
                plots=self.args.plots,
                seed=42)
            self._gift_snp_scores = snp_scores
            self._gift_model      = model

            # Publication Manhattan-style plot for admixture mapping
            if self.args.plots and len(admix_df):
                try:
                    from jasmap.plotting.plots import admixture_mapping_plot
                    admixture_mapping_plot(
                        admix_df,
                        pop_labels=self._pop_labels,
                        out_path=self.outfolder+"gift_admix_mapping.png")
                except Exception as e:
                    logger.warning(f"[GIFT-ADMIX] Admixture mapping plot failed: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # RVAT mode  (--mode rvat)
    # ──────────────────────────────────────────────────────────────────────────

    def _run_rvat_mode(self):
        """
        Gene-based rare variant association testing (--mode rvat).
        Runs Burden, SKAT, SKAT-O, and ACAT-V per gene/region,
        combines them via ACAT-O to a consensus gene p-value,
        and generates publication-quality RVAT plots.
        """
        from jasmap.rvat.rvat_engine import run_rvat

        # Note: --annot's auto-build (if requested) already ran once, early,
        # in jasmap()'s own dispatch -- before any mode-specific work, since
        # it only needs self.args.bfile, not a genotype read. Not repeated
        # here.

        # Load genotype matrix
        if self.args.bfile is not None:
            from bed_reader import open_bed
            bed = open_bed(self.args.bfile + ".bed", count_A1=False)
            chr_idx = getattr(self, 'chr_snp_idx', None)
            read_idx = (None, chr_idx) if chr_idx is not None else None
            G   = bed.read(index=read_idx).astype(np.float32)
            bim = pd.read_csv(self.args.bfile + ".bim", sep=r"\s+", header=None,
                              names=["CHR","SNP","CM","BP","A1","A2"])
            if chr_idx is not None:
                bim = bim.iloc[chr_idx].reset_index(drop=True)
            fam = pd.read_csv(self.args.bfile + ".fam", sep=r"\s+", header=None,
                              names=["FID","IID","PAT","MAT","SEX","PHENO"])
        else:
            import polars as pl
            tped = self._tped_path()
            df_t = pl.read_csv(tped, separator="\t", null_values="NA")
            G    = df_t[:, 6:].to_numpy().astype(np.float32).T
            bim  = df_t.select(df_t.columns[:6]).to_pandas()
            bim.columns = ["CHR","SNP","CM","BP","A1","A2"]
            fam  = pd.DataFrame({
                # Bug fix: same issue as the earlier synthetic-ID fix in
                # this file -- use the tped file's own column headers
                # (the REAL sample IDs load_vcf_samples_in_map()/
                # read_geno_file() wrote them with) instead of inventing
                # new ones that would never match --pop-file.
                "FID": list(df_t.columns[6:]),
                "IID": list(df_t.columns[6:])})

        # Note: G's own missing-genotype imputation for this pipeline is
        # NOT done here anymore -- see rvat_engine.py's run_rvat() for
        # why (imputing here, before the study-cohort restriction, would
        # compute each SNP's mean dosage across the FULL, --yes-anc
        # cohort, including reference/ancestral population samples --
        # exactly the contamination the MAF-restriction fix elsewhere in
        # this pipeline exists to prevent). auto-PCA just below doesn't
        # need G pre-imputed either -- compute_auto_pca_dense() delegates
        # to randomized_kinship_eigendecomp(), which already has its own
        # NaN handling internally (confirmed directly).
        y = np.asarray(self.phenotypes, dtype=np.float64) - 1

        # Population stratification covariates for RVAT null-model fitting.
        # Same priority as run_joint_analysis()/PRS: --covfile if given,
        # else auto-generated PCA (default ON, --no-auto-pca to disable),
        # else no adjustment (intercept-only null, the previous,
        # unconditional default). Built independently here (not assumed
        # already loaded by another mode), same reasoning as PRS's own
        # covariate-loading -- see _fit_null_model()'s docstring in
        # rvat_engine.py for why this matters: every RVAT test previously
        # ran against an unadjusted intercept-only null, the same
        # ancestry-confounding risk already found and fixed for GWAS/admix
        # and PRS.
        X0_for_rvat = None
        if self.args.covfile is not None:
            self.read_covar_file()
            X0_for_rvat = self.covar_list
        elif self.args.auto_pca and self.args.bfile is not None:
            from jasmap.biobank.auto_pca import compute_auto_pca_dense
            # Bug fix: same issue and reasoning as GWAS/admix mode's auto-PCA
            # fix above -- G here is still the FULL, unfiltered cohort (this
            # runs before run_rvat()'s own internal phenotype-based
            # filtering), so computing PCA from it would mix in any
            # reference/ancestral population samples present for --yes-anc.
            # Restrict to phenotyped samples, then embed back into a full-
            # length array so run_rvat()'s own X0=X0[valid_mask] slicing
            # (mirroring LMM.__init__()'s identical pattern) still works.
            _valid_for_pca = np.where(~np.isnan(np.asarray(y, dtype=np.float64)))[0]
            if 0 < len(_valid_for_pca) < len(y):
                logger.info(
                    f"[auto_pca] Restricting to {len(_valid_for_pca)} of "
                    f"{len(y)} samples with non-missing phenotype for PCA "
                    f"computation (excluding reference/ancestral population "
                    f"samples used only for local ancestry inference, if any).")
                X0_restricted = compute_auto_pca_dense(
                    G[_valid_for_pca].astype(np.float64), n_pcs=self.args.auto_pca_k,
                    verbose=True, maf_min=self.args.kinship_maf_min)
                X0_for_rvat = np.zeros((len(y), X0_restricted.shape[1]))
                X0_for_rvat[_valid_for_pca] = X0_restricted
            else:
                X0_for_rvat = compute_auto_pca_dense(
                    G.astype(np.float64), n_pcs=self.args.auto_pca_k, verbose=True,
                    maf_min=self.args.kinship_maf_min)
        elif not self.args.auto_pca:
            logger.info('[INFO] --no-auto-pca: RVAT null models will '
                        'proceed with no population stratification '
                        'correction.\n')

        methods = tuple(m.strip() for m in self.args.rvat_methods.split(","))

        _study_mask = self._get_study_cohort_mask(y, genotype_sample_ids=fam["IID"].values)

        gene_results = run_rvat(
            G              = G,
            y              = y,
            bim            = bim,
            fam            = fam,
            out_prefix     = self.outfolder + "rvat_results",
            gene_file      = self.args.gene_annot,
            var_annot_file = self.args.var_annot,
            maf_max        = self.args.maf_max,
            min_variants   = self.args.rvat_min_variants,
            methods        = methods,
            plots          = self.args.plots,
            X0             = X0_for_rvat,
            study_cohort_mask = _study_mask,
            seed           = 42)

        self._rvat_results = gene_results

        if gene_results is not None and len(gene_results):
            n_bonf = int((gene_results["consensus_p"] <
                          0.05 / max(len(gene_results), 1)).sum())
            logger.info(f"[RVAT] {len(gene_results)} genes tested, "
                        f"{n_bonf} Bonferroni-significant")

    # ──────────────────────────────────────────────────────────────────────────
    # PRS pipeline (--PRS Yes)
    # ──────────────────────────────────────────────────────────────────────────

    def _run_prs_pipeline(self):
        """Consensus PRS (Document 4 Steps 1-8)."""
        logger.info("\n" + "="*60)
        logger.info("[PRS] Running consensus PRS pipeline ...")
        logger.info("="*60 + "\n")

        from jasmap.prs.consensus_prs import run_prs

        y = self.phenotypes.astype(np.float64) - 1

        # Decide whether streaming can be used for the genotype load itself.
        # C+T scores SNPs matching EXTERNAL summary statistics (--sumstats),
        # which may not overlap at all with the univariate-correlation-
        # selected subset Ridge/Lasso use -- so if C+T is requested, the
        # full SNP panel is still needed and streaming is skipped for
        # correctness, not just simplicity. C+T is opt-in (default
        # --prs-methods is 'ridge,lasso', not including 'ct'), so this
        # covers the common, default case cleanly.
        ct_requested = ('ct' in self._prs_methods) and (self.args.sumstats is not None)
        max_snps = self.args.prs_max_snps if self.args.prs_max_snps > 0 else None

        prs_streaming_used = False
        if (self.args.bfile is not None and not ct_requested
                and max_snps is not None):
            from bed_reader import open_bed
            _bed_probe = open_bed(self.args.bfile + ".bed", count_A1=False)
            if _bed_probe.sid_count > max_snps:
                # Build covariates FIRST (via the already-streaming-capable
                # compute_auto_pca_streaming, if applicable), THEN use them
                # to residualize the streaming SNP screen -- same ordering
                # as the dense path, just both pieces now streaming-capable.
                X0_for_screen = None
                if self.args.covfile is not None:
                    self.read_covar_file()
                    X0_for_screen = self.covar_list
                elif self.args.auto_pca:
                    from jasmap.biobank.auto_pca import compute_auto_pca_streaming
                    # Bug fix: same issue and reasoning as GWAS/admix mode's
                    # auto-PCA fix -- restrict to phenotyped samples (see
                    # that fix's comment for the full explanation), then
                    # embed back into a full-length array.
                    _valid_for_pca = np.where(~np.isnan(np.asarray(y, dtype=np.float64)))[0]
                    if 0 < len(_valid_for_pca) < len(y):
                        logger.info(
                            f"[auto_pca] Restricting to {len(_valid_for_pca)} of "
                            f"{len(y)} samples with non-missing phenotype for "
                            f"PCA computation (excluding reference/ancestral "
                            f"population samples used only for local ancestry "
                            f"inference, if any).")
                        X0_restricted = compute_auto_pca_streaming(
                            self.args.bfile, n_pcs=self.args.auto_pca_k, verbose=True,
                            maf_min=self.args.kinship_maf_min, sample_idx=_valid_for_pca,
                            snp_idx=getattr(self, 'chr_snp_idx', None))
                        X0_for_screen = np.zeros((len(y), X0_restricted.shape[1]))
                        X0_for_screen[_valid_for_pca] = X0_restricted
                    else:
                        X0_for_screen = compute_auto_pca_streaming(
                            self.args.bfile, n_pcs=self.args.auto_pca_k, verbose=True,
                            maf_min=self.args.kinship_maf_min,
                            snp_idx=getattr(self, 'chr_snp_idx', None))

                from jasmap.prs.consensus_prs import select_top_snps_streaming
                keep_idx, G = select_top_snps_streaming(
                    self.args.bfile, y, X0=X0_for_screen, max_snps=max_snps,
                    snp_idx=getattr(self, 'chr_snp_idx', None), verbose=True)
                bim_full = pd.read_csv(self.args.bfile + ".bim", sep=r"\s+", header=None,
                                       names=["CHR","snp","CM","pos","A1","A2"])
                bim = bim_full.iloc[keep_idx].reset_index(drop=True)
                fam = pd.read_csv(self.args.bfile + ".fam", sep=r"\s+", header=None,
                                  names=["FID","IID","PAT","MAT","SEX","PHENO"])
                logger.info(f"[PRS] Streaming genotype load: full panel "
                           f"({_bed_probe.sid_count} SNPs) never materialised "
                           f"densely -- only the pre-filtered {len(keep_idx)} "
                           f"SNPs are held in memory.")
                prs_streaming_used = True
                X0_for_prs_precomputed = X0_for_screen

        if not prs_streaming_used:
            # Original dense path: either the panel is small enough that
            # streaming has no benefit, C+T needs the full panel, or
            # --bfile isn't the input format (--geno/--vcf lack the same
            # chunked-read mechanism, same limitation as kinship streaming).
            if self.args.bfile is not None:
                from bed_reader import open_bed
                bed = open_bed(self.args.bfile + ".bed", count_A1=False)
                chr_idx = getattr(self, 'chr_snp_idx', None)
                read_idx = (None, chr_idx) if chr_idx is not None else None
                G   = bed.read(index=read_idx).astype(np.float32)   # (n_samples, n_snps)
                bim = pd.read_csv(self.args.bfile + ".bim", sep=r"\s+", header=None,
                                  names=["CHR","snp","CM","pos","A1","A2"])
                if chr_idx is not None:
                    bim = bim.iloc[chr_idx].reset_index(drop=True)
                fam = pd.read_csv(self.args.bfile + ".fam", sep=r"\s+", header=None,
                                  names=["FID","IID","PAT","MAT","SEX","PHENO"])
            else:
                import polars as pl
                tped = self._tped_path()
                df   = pl.read_csv(tped, separator="\t", null_values="NA")
                G    = df[:, 6:].to_numpy().astype(np.float32).T  # (n_samples, n_snps)
                bim  = df.select(df.columns[:6]).to_pandas()
                bim.columns = ["CHR","snp","CM","pos","A1","A2"]
                fam  = pd.DataFrame({
                    # Bug fix: same synthetic-ID issue fixed earlier in
                    # this file -- see that fix's comment for the full
                    # explanation (found via a real --vcf + --admix-label
                    # run: synthetic IDs never matched --pop-file,
                    # silently restricting PRS's cohort to 0 samples).
                    "FID": list(df.columns[6:]),
                    "IID": list(df.columns[6:])})
            X0_for_prs_precomputed = None

        # Population stratification covariates for PRS weight fitting.
        # Same priority as run_joint_analysis(): --covfile if given, else
        # auto-generated PCA (default ON, --no-auto-pca to disable), else
        # no adjustment. Built independently here (not assumed already
        # loaded by another mode) since --PRS Yes can run alongside ANY
        # --mode, including rvat/gift, which do not necessarily load
        # covariates themselves. See residualize_phenotype()'s docstring
        # in consensus_prs.py for why this matters: PRS weight estimation
        # without stratification correction is a well-known source of
        # ancestry-confounded, poorly-transferable PRS.
        if X0_for_prs_precomputed is not None:
            # Streaming path above already computed this (needed for the
            # SNP-screening step itself) -- reuse it rather than recompute.
            X0_for_prs = X0_for_prs_precomputed
        else:
            X0_for_prs = None
            if self.args.covfile is not None:
                self.read_covar_file()
                X0_for_prs = self.covar_list
            elif self.args.auto_pca and self.args.bfile is not None:
                from jasmap.biobank.auto_pca import compute_auto_pca_dense
                # Bug fix: same issue and reasoning as GWAS/admix mode's
                # auto-PCA fix -- restrict to phenotyped samples (see that
                # fix's comment for the full explanation), then embed back
                # into a full-length array.
                _valid_for_pca = np.where(~np.isnan(np.asarray(y, dtype=np.float64)))[0]
                if 0 < len(_valid_for_pca) < len(y):
                    logger.info(
                        f"[auto_pca] Restricting to {len(_valid_for_pca)} of "
                        f"{len(y)} samples with non-missing phenotype for PCA "
                        f"computation (excluding reference/ancestral population "
                        f"samples used only for local ancestry inference, if any).")
                    X0_restricted = compute_auto_pca_dense(
                        G[_valid_for_pca].astype(np.float64), n_pcs=self.args.auto_pca_k,
                        verbose=True, maf_min=self.args.kinship_maf_min)
                    X0_for_prs = np.zeros((len(y), X0_restricted.shape[1]))
                    X0_for_prs[_valid_for_pca] = X0_restricted
                else:
                    X0_for_prs = compute_auto_pca_dense(
                        G.astype(np.float64), n_pcs=self.args.auto_pca_k, verbose=True,
                        maf_min=self.args.kinship_maf_min)
            elif not self.args.auto_pca:
                logger.info('[INFO] --no-auto-pca: PRS weight fitting will '
                            'proceed with no population stratification '
                            'correction.\n')

        # Ancestry labels for PRS portability analysis.
        # Priority:
        #   1. Per-sample labels from --pop-file (ADMIX, AFR, EUR, SAN) --
        #      the most informative for portability since ADMIX is distinguished
        #      from reference populations, not collapsed into a dominant component.
        #   2. Dominant-component labels from global ancestry Q matrix (fallback
        #      when no --pop-file was supplied but --yes-anc was still run).
        anc_labels = None
        if hasattr(self, '_sample_pop_labels_for_lai'):
            # Use the original per-sample labels from --pop-file, aligned to
            # sample order from the genotype matrix.
            id_to_pop = self._sample_pop_labels_for_lai   # {sample_id: pop_label}
            # fam["IID"] gives us the sample order matching the G matrix
            anc_labels = [str(id_to_pop.get(str(iid), "Unknown"))
                          for iid in fam["IID"].values]
        elif hasattr(self, 'global_ancestry_q') and self.global_ancestry_q is not None:
            Q     = self.global_ancestry_q
            pl_lbl = (self._pop_labels
                      if self._pop_labels and len(self._pop_labels)==Q.shape[1]
                      else [f"Anc{k+1}" for k in range(Q.shape[1])])
            anc_labels = [pl_lbl[np.argmax(Q[i])] for i in range(len(Q))]

        # Restrict PRS's reported output (prs_scores.tsv, validation
        # metrics) to only the phenotyped study cohort when the mode is
        # admix/gift_admix -- see run_prs()'s restrict_output_idx
        # docstring for the full reasoning: in this context specifically,
        # any unphenotyped samples are reference/ancestral population
        # samples included purely for --yes-anc's local ancestry
        # inference, not part of the study population at all, so a PRS
        # percentile/risk-tier spanning both groups would be actively
        # misleading. Deliberately scoped to admix/gift_admix only (not
        # every mode) -- a generically partially-phenotyped cohort under
        # GWAS/GIFT/RVAT is a different situation, where the unphenotyped
        # samples are still the same study population, just missing
        # outcome data, and reporting a score for them remains meaningful.
        _restrict_idx = None
        if self.args.mode in ("admix", "gift_admix"):
            _study_mask = self._get_study_cohort_mask(y, genotype_sample_ids=fam["IID"].values)
            _restrict_idx = np.where(_study_mask)[0]
            if len(_restrict_idx) == len(y):
                _restrict_idx = None  # nothing to restrict, avoid a no-op log line
            else:
                logger.info(
                    f"[PRS] --mode {self.args.mode}: restricting reported "
                    f"PRS scores to {len(_restrict_idx)} of {len(y)} "
                    f"phenotyped study samples (excluding reference/"
                    f"ancestral population samples used only for local "
                    f"ancestry inference).")

        prs_scores, validation = run_prs(
            G=G, y=y, bim=bim, fam=fam,
            out_prefix=self.outfolder + "prs_results",
            sumstats_path=self.args.sumstats,
            methods=self._prs_methods,
            ancestry_labels=anc_labels,
            X0=X0_for_prs,
            max_snps=(self.args.prs_max_snps if self.args.prs_max_snps > 0 else None),
            restrict_output_idx=_restrict_idx,
            plots=self.args.plots,
            seed=42)

        self._prs_scores     = prs_scores
        self._prs_validation = validation

        logger.info(f"[PRS] AUC={validation.get('auc',float('nan')):.3f}  "
                    f"r={validation.get('pearson_r',float('nan')):.3f}")

        # Post-processing: add an "Ancestry" column to the per-SNP weights
        # file for --mode admix/gift_admix, where local ancestry data
        # exists -- same aggregate-dosage method and rationale as
        # Jasmap.Joint.assoc's own "Ancestry" column (see
        # _compute_dominant_ancestry_per_snp()'s docstring). Done as a
        # post-processing read-modify-write step on run_prs()'s own
        # output file, rather than threading ancestry data through
        # run_prs()'s signature, since consensus_prs.py has no reason to
        # know about JasMAP4's local-ancestry structures at all -- this
        # keeps that module's own interface unchanged and mode-agnostic.
        if self.args.mode in ("admix", "gift_admix") and getattr(self, "lanc_dict", None):
            weights_path = Path(self.outfolder + "prs_results") / "prs_snp_weights.tsv"
            if weights_path.exists():
                try:
                    ancestry_by_snp = self._compute_dominant_ancestry_by_snp_name()
                    gene_by_snp = self._compute_gene_per_snp_by_name()
                    wdf = pd.read_csv(weights_path, sep="\t")
                    snp_col = "snp" if "snp" in wdf.columns else wdf.columns[0]
                    wdf["Ancestry"] = wdf[snp_col].astype(str).map(
                        lambda s: ancestry_by_snp.get(s, "Unknown"))
                    wdf["Gene"] = wdf[snp_col].astype(str).map(
                        lambda s: gene_by_snp.get(s, "Intergenic"))
                    wdf.to_csv(weights_path, sep="\t", index=False)
                    logger.info(f"[PRS] Added Ancestry and Gene columns to "
                               f"prs_snp_weights.tsv (dominant ancestral "
                               f"population by aggregate local-ancestry "
                               f"dosage per SNP; known gene, if any, from "
                               f"--gene-annot/--annot).")
                except Exception as e:
                    logger.warning(f"[PRS] Could not add Ancestry/Gene "
                                  f"columns to prs_snp_weights.tsv: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # Main publication plots for admix / GWAS modes
    # ──────────────────────────────────────────────────────────────────────────

    def _plot_main_results(self):
        """
        Auto-generate publication figures from run_joint_analysis() outputs.

        Bug fix: this method previously plotted whatever result files happened
        to exist in the output folder, regardless of which mode produced them.
        Re-using an --out folder across modes therefore produced figures from a
        PREVIOUS run mixed in with the current one -- e.g. running --mode GWAS
        into a folder that still held an earlier admix run emitted a stale
        plot_joint_ppa.png alongside the current GWAS plots, with nothing to
        indicate it was not from this analysis. Plotting is now gated on
        self.args.mode so only the current run's outputs are ever rendered.
        """
        mode = self.args.mode
        # Bug fix: same "one shared try/except for every plot" issue
        # already found and fixed in run_prs()'s own plotting block --
        # a single plot failing here (e.g. qq_plot() on an entirely
        # degenerate, all-NaN p-value column, the exact scenario a real
        # run with a constant phenotype produced) would have skipped
        # every OTHER, independent plot in this block too (e.g. any
        # remaining per-ancestry admixture-mapping plots for --mode
        # admix). Each plot now gets its own try/except via this helper.
        def _safe_plot(name, fn, *a, **kw):
            try:
                fn(*a, **kw)
            except Exception as e:
                logger.warning(f"[PLOT] {name!r} failed: {e}")

        from jasmap.plotting.plots import (
            joint_ppa_plot, manhattan, qq_plot)

        joint_path = self.outfolder + "Jasmap.Joint.assoc"
        gwas_path  = self.outfolder + "Jasmap.GWAS.assoc"

        # Joint PPA plot belongs to admix mode only
        if mode == "admix" and os.path.exists(joint_path):
            df = pd.read_csv(joint_path, sep="\t")
            _safe_plot("plot_joint_ppa", joint_ppa_plot, df,
                      out_path=self.outfolder + "plot_joint_ppa.png")

        # GWAS Manhattan/QQ belong to GWAS mode only
        if mode == "GWAS" and os.path.exists(gwas_path):
            df_gwas = pd.read_csv(gwas_path, sep="\t")
            # Build a minimal CHR/BP pseudo-axis using SNP rank
            df_gwas["CHR"] = "1"
            df_gwas["BP"]  = range(len(df_gwas))
            _safe_plot("plot_gwas_manhattan", manhattan, df_gwas, pval_col="SNP_Pval",
                      title="GWAS Manhattan", sig_line=5e-8,
                      out_path=self.outfolder + "plot_gwas_manhattan.png")
            _safe_plot("plot_gwas_qq", qq_plot, df_gwas["SNP_Pval"].dropna().astype(float),
                      title="QQ - GWAS", out_path=self.outfolder + "plot_gwas_qq.png")

        # Per-ancestry admixture-mapping plots belong to admix mode only
        if mode == "admix":
            for anc in range(1, getattr(self, 'no_of_ancs', 0)+1):
                anc_path = self.outfolder + f"Jasmap.ANC{anc}.AM.assoc"
                if os.path.exists(anc_path):
                    df_anc = pd.read_csv(anc_path, sep="\t")
                    df_anc["CHR"] = "1"; df_anc["BP"] = range(len(df_anc))
                    _safe_plot(f"plot_am_anc{anc}", manhattan, df_anc, pval_col="Pval",
                              title=f"Admixture Mapping - Ancestry {anc}",
                              out_path=self.outfolder + f"plot_am_anc{anc}.png")

    # ──────────────────────────────────────────────────────────────────────────
    # Consensus GWAS/RVAT pipeline  (--consgwas)
    # ──────────────────────────────────────────────────────────────────────────

    def _run_consgwas(self):
        """
        8-Step Consensus GWAS/RVAT pipeline (Section 6 of
        Consensus_GWAS_RareVariant_Approach.docx).

        Integrates results from whichever --mode ran before this call with
        additional internal tools (OLS, Firth-approx, burden, SKAT-approx).
        The output file picker (`collect_mode_results`) intelligently selects
        the correct files from the primary mode's output directory, so the
        user never has to specify them manually.
        """
        from jasmap.consensus_gwas.consensus_gwas import ConsensusGWAS, collect_mode_results

        logger.info("\n" + "="*60)
        logger.info("[--consgwas] Step-by-Step Consensus GWAS/RVAT Pipeline")
        logger.info(f"  Integrating with --mode={self.args.mode}")
        logger.info("="*60 + "\n")

        # Step 1: Intelligent output file picking — pick up the primary mode's
        # results without requiring the user to specify file paths.
        # Load the BIM first so collect_mode_results() can translate JasMAP's
        # positional SNP indices into real variant IDs (see its docstring).
        #
        # Optimization: shared, cached dense-genotype read -- see
        # _get_dense_genotype_for_mode()'s own docstring. When --mode
        # gift/gift_admix already read this exact same panel earlier in the
        # SAME run, this reuses that read instead of independently repeating
        # a multi-minute dense read for identical data (bed.read() with the
        # same chr_idx, same dtype -- there is nothing for a second read to
        # discover that the first one didn't already capture). Preserves
        # this method's own graceful-failure behavior: a read failure still
        # logs a warning and continues with G=None (internal tools skipped)
        # rather than crashing the whole --consgwas step.
        G = None; bim = None; fam = None
        try:
            G, bim, fam = self._get_dense_genotype_for_mode()
        except Exception as e:
            logger.warning(f"[--consgwas] Could not load raw genotypes: {e}. "
                           "Internal tools (OLS/Firth/Burden/SKAT) will be skipped.")

        jasmap_primary = collect_mode_results(self.args.mode, self.outfolder, bim=bim)
        if jasmap_primary:
            logger.info(f"[--consgwas] Auto-picked {len(jasmap_primary)} result set(s) "
                        f"from {self.args.mode} output:")
            for k in jasmap_primary:
                logger.info(f"  . {k}  ({len(jasmap_primary[k])} rows)")
        else:
            # Conflict #4: warn explicitly rather than silently running with no primary tool
            logger.info(
                f"[--consgwas] WARNING: no primary --mode output files found in "
                f"'{self.outfolder}' for mode '{self.args.mode}'. "
                f"Consensus will use only internal tools (OLS, Firth-approx, "
                f"burden, SKAT-approx). For a richer consensus, run JasMAP4 with "
                f"--mode first, or combine both flags in a single command "
                f"(the mode runs first, then --consgwas picks up its results).\n")

        # (genotypes already loaded above)

        # y = phenotype (0/1 for case-control, continuous otherwise)
        y = None
        if hasattr(self, 'phenotypes') and self.phenotypes is not None:
            y = np.asarray(self.phenotypes, dtype=np.float64) - 1
        elif fam is not None and "PHENO" in fam.columns:
            y = fam["PHENO"].values.astype(float)
            y = np.where(y > 0, y - 1, np.nan)

        cons_out = self.outfolder + "consensus_gwas"
        engine = ConsensusGWAS(
            mode=self.args.mode,
            out_dir=cons_out,
            n_permutations=self.args.consgwas_perms,
            sig_thresh=self.args.consgwas_sig,
            seed=42,
            plots=self.args.plots)

        consensus_df = engine.run(
            G=G, y=y, bim=bim,
            jasmap_results=jasmap_primary,
            var_annot_path=self.args.var_annot)

        if consensus_df is not None and len(consensus_df):
            n_sig = int(
                (consensus_df["pval_consensus"] < engine._sig_thresh).sum())
            logger.info(f"\n[--consgwas] Consensus complete - "
                        f"{len(consensus_df)} variants, "
                        f"{n_sig} at threshold {engine._sig_thresh:.2e}")
        else:
            logger.warning("[--consgwas] Consensus produced no results.")

    # ──────────────────────────────────────────────────────────────────────────
    # jasmap() — top-level dispatcher
    # ──────────────────────────────────────────────────────────────────────────

    def jasmap(self):
        '''Running JasMAP - dispatches all modes.'''

        mode = self.args.mode

        # --annot: auto-build a gene annotation once, early, before any
        # mode-specific work -- relevant to --mode rvat (its own gene-based
        # tests) AND admix/gift_admix (this feature's own "Gene" column on
        # Joint.assoc/PRS output, added alongside the "Ancestry" column).
        # Only needs self.args.bfile (a .bim path), not a genotype read, so
        # safe to run this early regardless of which mode follows. Already
        # a no-op if --gene-annot/--var-annot was given explicitly, or if
        # --annot wasn't passed at all (checked inside).
        self._maybe_auto_build_gene_annotation()

        # Step A: compute local ancestry if requested
        if self.args.yes_anc and mode in ("admix", "gift_admix", "LAI"):
            self._run_lai_pipeline()
            # Update lanc_file for the admix path
            if mode == "admix":
                self.lanc_file = getattr(self, 'lanc_file',
                                          self._lai_obj.anc_path)

        # Step B: mode-specific analysis
        if mode == "LAI":
            # Standalone ancestry-only mode: Step A above already ran the
            # full consensus LAI pipeline (global/local ancestry, karyograms,
            # plots) -- nothing else for this mode to do. Skips straight to
            # cleanup, bypassing GWAS/RVAT/GIFT/PRS/consgwas entirely (the
            # validation step above already warned if any of those flags
            # were also given, since none of them have anywhere to attach
            # without a downstream association test).
            pass

        elif mode in ("admix", "GWAS"):
            self.run_joint_analysis()
            if self.args.plots:
                self._plot_main_results()

        elif mode in ("gift", "gift_admix"):
            # Read genotype data (tped format needed for kinship)
            if self.args.geno is not None:
                self.read_geno_file()
            elif self.args.bfile is not None:
                self.read_bed_file()
            elif self.args.vcf is not None:
                self.vcf_to_dataframe()
            self.inds_size = len(self.phenotypes)
            self._run_gift_mode()

        elif mode == "rvat":
            # Read genotype data (needed for _run_rvat_mode)
            if self.args.geno is not None:
                self.read_geno_file()
            elif self.args.bfile is not None:
                self.read_bed_file()
            elif self.args.vcf is not None:
                self.vcf_to_dataframe()
            self._run_rvat_mode()

        # Step C: consensus PRS (if requested)
        # --mode LAI explicitly excluded: PRS needs an association test or
        # downstream mode to score against, which this mode doesn't produce
        # (see the validation warning above for why this is enforced, not
        # just advised).
        if self.args.PRS == "Yes" and mode != "LAI":
            # Need phenotypes loaded for PRS
            if not hasattr(self, 'phenotypes'):
                if self.args.geno is not None:
                    self.read_geno_file()
                elif self.args.bfile is not None:
                    self.read_bed_file()
                elif self.args.vcf is not None:
                    self.vcf_to_dataframe()
            self._run_prs_pipeline()

        # Step D: Consensus GWAS/RVAT pipeline (--consgwas) — runs after
        # --mode and optionally after --PRS, picks up all output files
        # from earlier steps automatically. --mode LAI excluded for the
        # same reason as Step C: no association output exists to consense.
        if self.args.consgwas and mode != "LAI":
            self._run_consgwas()

        # Clean up intermediate .tped file(s) now that every step (including
        # --consgwas, which may also have read from one) has finished
        # successfully. Placed here, at the very end, specifically so nothing
        # gets deleted while it might still be needed by a later step.
        # Tracked in self._temp_files_created by _tped_path() itself, so
        # this covers every scope actually used this run (e.g. both
        # tmp_all.tped and tmp_chr22.tped when --lai-genome-wide was used)
        # without needing to separately enumerate them here. Only runs on
        # a SUCCESSFUL completion (this line only executes if nothing above
        # raised) -- a crashed run leaves its intermediate files in place,
        # which is deliberate: they're useful for debugging what happened,
        # and get cleaned up automatically on the NEXT successful run in
        # the same --out folder regardless (self._tped_path() always
        # overwrites, never appends).
        for _tmp_path in getattr(self, "_temp_files_created", set()):
            try:
                if os.path.exists(_tmp_path):
                    os.remove(_tmp_path)
            except Exception as _e:
                logger.warning(f"[INFO] Could not remove temporary file "
                               f"{_tmp_path!r}: {_e}")

        logger.info('Analysis finished at {T}'.format(T=time.ctime()))
        time_elapsed = round(time.time() - self.start_time, 2)
        logger.info('Total time elapsed: {T}'.format(T=self.sec_to_str(time_elapsed)))

if __name__ == '__main__':

    # Bug fix (#7): `global gpv1` referenced a variable that was never defined or
    # used anywhere in the module -- removed as dead code. sys.argv is always
    # defined (an IndexError here was never actually reachable), so the
    # try/except/finally has been simplified accordingly.
    argv = sys.argv
    run = JasMAP(argv)
    run.jasmap()
