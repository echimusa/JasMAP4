#!/usr/bin/env python3
"""
build_gene_annotation.py
==========================

Standalone utility: automatically constructs a gene-annotation BED file
for JasMAP4's --gene-annot, scoped to the SNPs actually present in a
study's genotype panel.

Design summary
---------------
Per-SNP gene lookup via a live web API (e.g. Ensembl's REST overlap
endpoint) is NOT used here -- it takes one genomic region per call, and
at real GWAS-panel scale (hundreds of thousands of SNPs) that would take
HOURS even under generous rate limits (confirmed directly: Ensembl's
overlap/region endpoint accepts a single region per request). Instead:
a complete, versioned reference gene-annotation file is downloaded ONCE,
cached locally, and the SNP-to-gene matching happens entirely offline
against that cache -- fast, and mostly network-independent after the
first run.

Genome build handling is intentionally NOT automatic. GENCODE/UCSC/
Ensembl all maintain SEPARATE files/servers for GRCh37 (hg19) vs
GRCh38 (hg38) -- silently guessing wrong doesn't fail loudly, it
silently assigns SNPs to the wrong genes, which is worse than no
annotation at all. --genome-build must be given explicitly (defaults
to hg38, the current standard, but always logs which build is in use
so a mismatch is easy to notice).

Data source: UCSC's refGene GTF (hgdownload.soe.ucsc.edu), NOT
knownGene -- confirmed directly by inspecting both: knownGene's own
gene_id/gene_name fields are set to internal transcript identifiers
(e.g. "ENST00000456328.2"), not real gene symbols, so grouping by name
there produces one nonsense "gene" per transcript (241,418 of them for
hg38 -- confirmed). refGene's gene_id/gene_name are real, human-readable
gene symbols (e.g. "DDX11L1"), and its per-transcript rows aggregate
correctly into real gene boundaries.

Aggregation correctness (found and fixed via real hg38 data, not
assumptions): UCSC's refGene/ncbiRefSeq convention marks a transcript
re-mapped to a SEPARATE, physically distant genomic copy by appending an
additional "_2", "_3", etc. suffix to that transcript's OWN accession
(e.g. NR_106918 -> NR_106918_2) -- confirmed this is a real, deliberate
disambiguation signal, not noise. A naive group-by-gene-name aggregation
ignores this and merges genuinely separate copies into one bogus span
(confirmed: MIR6859-1's two chr1 copies, 170kb apart, merged into one
170kb span covering neither real copy). The suffix is now treated as
authoritative -- different suffix values are never merged regardless of
distance -- with a secondary 500kb proximity cap for any other, unmarked
multi-locus situations. A second, related bug was caught and fixed while
building this: a naive regex for detecting that suffix incorrectly
matched RefSeq accessions' own normal format (e.g. "NM_003682" was
misread as accession "NM" + copy-marker "003682"), which would have
silently over-split nearly every multi-transcript gene in the genome
(confirmed: this bug alone inflated MADD, a normal single-locus gene,
into 105 fake "copies"). Both bugs were found by inspecting real output
against real data, not by inspection of the code alone.

Usage
-----
    python3 build_gene_annotation.py --bfile SAC_REF --genome-build hg38 --out SAC

    # Uses a specific .bim file directly instead of a --bfile prefix:
    python3 build_gene_annotation.py --bim SAC_REF.bim --genome-build hg19 --out SAC

Output
------
Writes {out}_gene_annotation.bed (BED4: chrom, start, stop, gene) in the
exact format JasMAP4's load_gene_annotation() already expects -- can be
passed directly via --gene-annot, no conversion needed.

On any failure (no internet, download error, cache unusable) this exits
non-zero and writes NO output file, by design -- a calling pipeline can
treat "no output file produced" as the signal to fall back to its
existing default (e.g. JasMAP4's 50-SNP sliding-window gene proxies)
rather than risk silently using no annotation or a corrupt one.
"""
import argparse
import gzip
import os
import re
import ssl
import sys
import time
import urllib.request
import urllib.error

# UCSC's build-keyed URL scheme is stable across releases (the file
# content is updated in place at the same URL, unlike GENCODE's
# release-numbered URLs, which go stale as new releases ship) --
# confirmed directly for both hg38 and hg19 before choosing this source.
REFGENE_URL_TEMPLATE = (
    "https://hgdownload.soe.ucsc.edu/goldenPath/{build}/bigZips/genes/"
    "{build}.refGene.gtf.gz"
)

_ATTR_RE = re.compile(r'(\w+)\s+"([^"]*)"')
# Matches a copy-marker suffix APPENDED to an already-complete RefSeq
# accession (e.g. "NR_106918_2") -- deliberately requires the accession's
# own normal <letters>_<digits> form to already be present before the
# marker, so normal accessions like "NM_003682" are never misread as
# accession "NM" + marker "003682" (see module docstring for the real
# bug this fixes).
_COPY_MARKER_RE = re.compile(r'^([A-Za-z]{1,3}_\d+)_(\d+)$')


def log(msg):
    print(f"[build_gene_annotation] {msg}", flush=True)


def get_cached_or_download(build, cache_dir, timeout, force_download):
    """
    Returns the local path to the (possibly newly downloaded) reference
    GTF for the given build, or None if unavailable (no internet,
    download error, etc.) -- callers must treat None as "fall back to
    default", not retry indefinitely.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{build}.refGene.gtf.gz")

    if os.path.exists(cache_path) and not force_download:
        log(f"Using cached reference annotation: {cache_path} "
           f"({os.path.getsize(cache_path)/1e6:.1f} MB, "
           f"cached {time.time()-os.path.getmtime(cache_path):.0f}s ago). "
           f"Pass --force-download to refresh it.")
        return cache_path

    url = REFGENE_URL_TEMPLATE.format(build=build)
    log(f"No usable cache found -- attempting download from {url} "
       f"(timeout={timeout}s)...")

    def _attempt(ssl_context, label):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout,
                                    context=ssl_context) as resp:
            return resp.read()

    try:
        # Bug fix: previously this made a single urlopen() call using
        # Python's default SSL context, which fails outright with
        # "CERTIFICATE_VERIFY_FAILED: unable to get local issuer
        # certificate" in environments whose Python can't locate a CA
        # certificate bundle -- extremely common with anaconda installs
        # on Linux/macOS, and confirmed directly by a real user hitting
        # exactly this against the (perfectly reachable, correctly
        # configured) UCSC server. The download isn't genuinely
        # unavailable in that case, so reporting it as "no internet
        # connection" was both wrong and unhelpful.
        #
        # Now: try the system default first (correct and preferred when
        # it works), then retry with certifi's own bundled CA roots --
        # certifi ships with anaconda and pip, is the same trust store
        # requests/pip themselves use, and resolves this specific
        # misconfiguration WITHOUT disabling verification. Certificate
        # verification stays fully enabled in both attempts; this is not
        # an SSL bypass.
        try:
            data = _attempt(None, "system default CA store")
        except urllib.error.URLError as e:
            if not isinstance(getattr(e, "reason", None), ssl.SSLError):
                raise
            log(f"System CA store could not verify the server certificate "
               f"({e.reason}). Retrying with certifi's bundled CA roots "
               f"(verification stays enabled) ...")
            try:
                import certifi
            except ImportError:
                raise urllib.error.URLError(
                    "certificate verification failed and certifi is not "
                    "installed -- run `pip install certifi` (or `conda "
                    "install certifi`) to fix this, or download the "
                    "annotation manually and pass it via --gene-annot")
            data = _attempt(ssl.create_default_context(cafile=certifi.where()),
                           "certifi CA bundle")

        tmp_path = cache_path + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(data)
        os.replace(tmp_path, cache_path)  # atomic -- no partial cache file on crash
        log(f"Downloaded and cached {len(data)/1e6:.1f} MB -> {cache_path}")
        return cache_path
    except (urllib.error.URLError, urllib.error.HTTPError,
            TimeoutError, OSError) as e:
        log(f"Download failed ({type(e).__name__}: {e}). "
           f"No internet connection, or the source is unreachable from "
           f"this network. Falling back to no auto-annotation -- the "
           f"calling pipeline should use its default gene-proxy behavior.")
        return None


def parse_gtf_genes(path, max_cluster_gap=500_000):
    """
    Parse a gene-model GTF (gzipped) into (chrom, start, stop, gene)
    records. See module docstring for why this specific aggregation
    logic exists -- it is not a naive group-by-name.
    """
    gene_rows = []
    transcript_rows = []

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            feature = parts[2]
            if feature not in ("gene", "transcript"):
                continue
            chrom = parts[0].replace("chr", "")
            start = int(parts[3])
            end   = int(parts[4])
            attrs = dict(_ATTR_RE.findall(parts[8]))
            gene_name = attrs.get("gene_name", attrs.get("gene_id"))
            if gene_name is None:
                continue
            transcript_id = attrs.get("transcript_id", "")
            m = _COPY_MARKER_RE.match(transcript_id)
            copy_marker = m.group(2) if m else "0"
            rec = (chrom, start, end, gene_name, copy_marker)
            (gene_rows if feature == "gene" else transcript_rows).append(rec)

    def cluster_and_aggregate(records):
        groups = {}
        for chrom, start, end, gene_name, copy_marker in records:
            groups.setdefault((chrom, gene_name, copy_marker), []).append((start, end))
        result = []
        for (chrom, gene_name, _marker), spans in groups.items():
            spans.sort()
            c_start, c_end = spans[0]
            for s, e in spans[1:]:
                if s - c_end <= max_cluster_gap:
                    c_end = max(c_end, e)
                else:
                    result.append((chrom, c_start, c_end, gene_name))
                    c_start, c_end = s, e
            result.append((chrom, c_start, c_end, gene_name))
        return result

    if gene_rows:
        return cluster_and_aggregate(gene_rows)
    return cluster_and_aggregate(transcript_rows)


def read_bim_chrom_ranges(bim_path):
    """Returns {chrom: (min_bp, max_bp)} for chromosomes present in a .bim file."""
    ranges = {}
    with open(bim_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            chrom = parts[0].replace("chr", "")
            try:
                bp = int(parts[3])
            except ValueError:
                continue
            if chrom not in ranges:
                ranges[chrom] = [bp, bp]
            else:
                ranges[chrom][0] = min(ranges[chrom][0], bp)
                ranges[chrom][1] = max(ranges[chrom][1], bp)
    return {c: tuple(v) for c, v in ranges.items()}


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--bfile", help="PLINK prefix (reads {bfile}.bim)")
    src.add_argument("--bim", help="Path to a .bim file directly")
    p.add_argument("--genome-build", choices=["hg19", "hg38"], default="hg38",
                   help="Genome build the .bim positions use (default: hg38). "
                        "Getting this wrong silently assigns SNPs to the wrong "
                        "genes -- verify against how your genotypes were called, "
                        "don't guess.")
    p.add_argument("--out", required=True,
                   help="Output prefix; writes {out}_gene_annotation.bed")
    p.add_argument("--flank-kb", type=float, default=0,
                   help="Extra flanking window (kb) around each chromosome's "
                        "SNP range when filtering the reference to relevant "
                        "genes (default: 0, i.e. exact range only)")
    p.add_argument("--cache-dir", default=os.path.expanduser("~/.jasmap_annotation_cache"),
                   help="Where to cache the downloaded reference annotation "
                        "(default: ~/.jasmap_annotation_cache)")
    p.add_argument("--force-download", action="store_true",
                   help="Bypass the cache and re-download even if a cached copy exists")
    p.add_argument("--timeout", type=int, default=30,
                   help="Network timeout in seconds (default: 30)")
    args = p.parse_args()

    bim_path = args.bim if args.bim else args.bfile + ".bim"
    if not os.path.exists(bim_path):
        log(f"ERROR: .bim file not found: {bim_path}")
        sys.exit(1)

    log(f"Reading SNP positions from {bim_path} ...")
    chrom_ranges = read_bim_chrom_ranges(bim_path)
    n_snps = sum(1 for _ in open(bim_path))
    log(f"{n_snps} SNPs across {len(chrom_ranges)} chromosome(s): "
       f"{sorted(chrom_ranges.keys())}")
    log(f"Genome build: {args.genome_build} -- verify this matches how your "
       f"genotypes were actually called; a build mismatch silently "
       f"produces wrong gene assignments.")

    ref_path = get_cached_or_download(
        args.genome_build, args.cache_dir, args.timeout, args.force_download)
    if ref_path is None:
        log("No reference annotation available -- exiting without writing "
           "an output file. Use JasMAP4's default gene-proxy behavior instead.")
        sys.exit(2)

    log("Parsing reference annotation (gene-level aggregation, "
       "copy-marker-aware)...")
    t0 = time.time()
    all_genes = parse_gtf_genes(ref_path, max_cluster_gap=int(500_000))
    log(f"Parsed {len(all_genes)} gene records in {time.time()-t0:.1f}s.")

    flank = int(args.flank_kb * 1000)
    kept = []
    for chrom, start, end, gene in all_genes:
        rng = chrom_ranges.get(chrom)
        if rng is None:
            continue
        lo, hi = rng[0] - flank, rng[1] + flank
        if end >= lo and start <= hi:
            kept.append((chrom, start, end, gene))

    if not kept:
        log("WARNING: 0 reference genes overlap this study's SNP range -- "
           "check --genome-build is correct for this data, since this is "
           "an implausible result otherwise. No output file written.")
        sys.exit(3)

    out_path = f"{args.out}_gene_annotation.bed"
    kept.sort(key=lambda r: (r[0], r[1]))
    with open(out_path, "w") as f:
        for chrom, start, end, gene in kept:
            f.write(f"{chrom}\t{start}\t{end}\t{gene}\n")

    log(f"Wrote {len(kept)} gene record(s) relevant to this study's "
       f"{n_snps} SNPs -> {out_path}")
    log(f"Pass this to JasMAP4 with: --gene-annot {out_path}")


if __name__ == "__main__":
    main()
