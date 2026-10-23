#!/usr/bin/env python3
"""
jasmap_parallel.py -- run JasMAP4 per chromosome in parallel, then merge
=======================================================================

Three steps:

  1. prepare  (once)   genome-wide PCs on the tested cohort -> a --covfile
                       shared by every chromosome job
  2. run      (parallel) one JasMAP4 job per chromosome with --chr N and
                       --covfile <the file from step 1>; `slurm` or `pbs`
                       writes a ready-made job-array script for this
  3. merge    (once)   combine every per-chromosome result into genome-wide
                       files, recompute genomic control genome-wide, and draw
                       genome-wide Manhattan and QQ plots

    python3 jasmap_parallel.py prepare --bfile data/COHORT --pheno data/COHORT.pheno \\
        --pop-file data/POP.txt --admix-label SAC --auto-pca-k 4 --out results/
    python3 jasmap_parallel.py slurm --chroms 1-22 --out results/ --out-prefix SAC \\
        -- --bfile data/COHORT --pheno data/COHORT.pheno --mode admix ...
    sbatch results/SAC_array.slurm
        # ...or on PBS (PBS Pro/OpenPBS by default; --pbs-flavour torque for Torque):
    python3 jasmap_parallel.py pbs --chroms 1-22 --out results/ --out-prefix SAC \\
        -- --bfile data/COHORT --pheno data/COHORT.pheno --mode admix ...
    qsub results/SAC_array.pbs
    python3 jasmap_parallel.py merge --out results/ --out-prefix SAC

Why step 1 matters
------------------
A run with --chr N computes its PCs, and its kinship, from chromosome N ALONE.
PCs from one chromosome are noisy and can track chromosome-specific features
(large inversions, long LD blocks) rather than genome-wide ancestry -- worst
on small chromosomes. Computing PCs once genome-wide and sharing them puts
every chromosome on the same footing as a genome-wide run.

The kinship is the remaining difference: each job still builds it from its
own chromosome, which is also the chromosome being tested. In a mixed model
that lets the random effect absorb part of a genuine signal ("proximal
contamination"), so per-chromosome runs are CONSERVATIVE -- they lose power,
they do not create false positives. A leave-one-chromosome-out kinship is
the full remedy and is not yet implemented.

What merges, and what does not
------------------------------
* Association results (GWAS, admixture mapping, joint, consensus statistics,
  local ancestry, GIFT and RVAT tables) are concatenated across chromosomes;
  genomic control is recomputed on the merged p-values. A genome-wide lambda
  is NOT the average of per-chromosome lambdas.
* Global ancestry is averaged across chromosomes, weighted by each
  chromosome's SNP count. Ancestry labels are consistent across chromosome
  jobs because each job anchors its components to the --pop-file reference
  panels -- without that, merging per-chromosome ancestry would be invalid.
* PRS is NOT merged: a polygenic score sums over the whole genome, and 22
  separately fitted per-chromosome models are not one genome-wide model. Run
  --PRS once, genome-wide.
* GIFT importance scores come from a per-run multivariate model, so merged
  per-chromosome GIFT scores rank SNPs within chromosomes, not genome-wide.
* Per-chromosome summary files (per-chromosome genomic control, provenance,
  plots) are superseded by the genome-wide versions written here.
"""
import argparse
import os
import re
import shlex
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


# ==========================================================================
# prepare: genome-wide PCs on the tested cohort
# ==========================================================================
def cmd_prepare(a):
    from jasmap.biobank.auto_pca import compute_auto_pca_streaming

    prefix = a.bfile[:-4] if a.bfile.endswith(".bed") else a.bfile
    fam = pd.read_csv(prefix + ".fam", sep=r"\s+", header=None,
                      usecols=[0, 1], names=["FID", "IID"], dtype=str)
    n = len(fam)

    tested = np.ones(n, dtype=bool)
    if a.pheno:
        ph = pd.read_csv(a.pheno, sep=r"\s+", header=None, dtype=str)
        ph = ph.iloc[:, [1, 2]] if ph.shape[1] >= 3 else None
        if ph is not None:
            vals = dict(zip(ph.iloc[:, 0], pd.to_numeric(ph.iloc[:, 1], errors="coerce")))
            v = np.array([vals.get(i, np.nan) for i in fam["IID"]], dtype=float)
            tested &= np.isfinite(v) & (v != -9)
    if a.pop_file and a.admix_label:
        pf = pd.read_csv(a.pop_file, sep=r"\s+", header=None, names=["ID", "POP"], dtype=str)
        lab = dict(zip(pf["ID"], pf["POP"]))
        tested &= np.array([lab.get(i) == a.admix_label for i in fam["IID"]])
    idx = np.flatnonzero(tested)
    if len(idx) < a.auto_pca_k + 2:
        sys.exit(f"ERROR: only {len(idx)} tested sample(s) -- check --pheno / --pop-file.")

    print(f"[prepare] Genome-wide PCA on {len(idx)} tested sample(s) of {n} "
          f"(all chromosomes, k={a.auto_pca_k}) ...")
    pcs = compute_auto_pca_streaming(prefix, n_pcs=a.auto_pca_k, sample_idx=idx,
                                     maf_min=a.maf_min, verbose=False)
    pcs = np.asarray(pcs, dtype=float)
    if pcs.ndim == 2 and pcs.shape[1] > 1 and np.allclose(pcs[:, 0], pcs[0, 0]):
        pcs = pcs[:, 1:]                       # drop an intercept column if present
    full = np.zeros((n, pcs.shape[1]))
    full[idx] = pcs                            # untested samples get 0, as JasMAP4 does

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.out_prefix}_genomewide_pcs.cov" if a.out_prefix
                        else "genomewide_pcs.cov")
    out = pd.concat([fam, pd.DataFrame(full)], axis=1)
    out.to_csv(path, sep="\t", header=False, index=False, float_format="%.8g")
    print(f"[prepare] Wrote {pcs.shape[1]} genome-wide PC(s) -> {path}")
    print(f"[prepare] Pass it to EVERY chromosome job:  --covfile {path}")
    return path


# ==========================================================================
# slurm: job-array script
# ==========================================================================
def _parse_chroms(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if re.fullmatch(r"\d+-\d+", part):
            lo, hi = map(int, part.split("-"))
            out += [str(c) for c in range(lo, hi + 1)]
        elif part:
            out.append(part)
    return out


def _mem_pbs(mem):
    """'16G' / '16GB' / '500M' -> PBS form ('16gb', '500mb')."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]?)B?\s*", str(mem), re.I)
    if not m:
        return str(mem)
    return f"{m.group(1)}{(m.group(2) or 'M').lower()}b"


def cmd_jobscript(a, passthrough, scheduler):
    """Write a job-array script, one task per chromosome, for SLURM or PBS."""
    chroms = _parse_chroms(a.chroms)
    n = len(chroms)
    out_abs = os.path.abspath(a.out)
    logs = os.path.join(out_abs, "logs")
    os.makedirs(logs, exist_ok=True)
    jasmap = os.path.join(_HERE, "JasMAP4.py")
    merge_cmd = (f"python3 {os.path.join(_HERE, 'jasmap_parallel.py')} merge "
                 f"--out {a.out} --out-prefix {a.out_prefix}")
    args = " ".join(shlex.quote(x) for x in passthrough)
    env = (f'eval "$(conda shell.bash hook)"\nconda activate {a.conda_env}\n'
           if a.conda_env else
           "# Activate your environment here if jobs do not inherit it, e.g.\n"
           "#   eval \"$(conda shell.bash hook)\"; conda activate jasmap\n")
    run = (f'python3 {jasmap} {args} --chr "$CHR" '
           f"--out {a.out} --out-prefix {a.out_prefix}\n")
    chrom_arr = "CHROMS=(" + " ".join(chroms) + ")\n"

    if scheduler == "slurm":
        hdr = ["#!/bin/bash",
               f"#SBATCH --job-name={a.out_prefix}_chr",
               f"#SBATCH --array=0-{n - 1}",
               f"#SBATCH --cpus-per-task={a.cpus}",
               f"#SBATCH --mem={a.mem}",
               f"#SBATCH --time={a.time}",
               f"#SBATCH --output={logs}/%x_%a.out"]
        if a.queue:
            hdr.append(f"#SBATCH --partition={a.queue}")
        if a.account:
            hdr.append(f"#SBATCH --account={a.account}")
        body = (chrom_arr + "CHR=${CHROMS[$SLURM_ARRAY_TASK_ID]}\n"
                "export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}\n")
        ext, submit = "slurm", "sbatch"
    else:
        pro = a.pbs_flavour == "pro"
        # Job names are capped at 15 characters on some PBS installations.
        name = re.sub(r"[^A-Za-z0-9_]", "_", f"{a.out_prefix}_chr")[:15]
        if not name[:1].isalpha():
            name = ("J" + name)[:15]
        hdr = ["#!/bin/bash", f"#PBS -N {name}"]
        if n > 1:   # PBS Pro rejects a one-element array, so omit it then
            hdr.append(f"#PBS -J 1-{n}" if pro else f"#PBS -t 1-{n}")
        if pro:
            hdr.append(f"#PBS -l select=1:ncpus={a.cpus}:mem={_mem_pbs(a.mem)}")
        else:
            hdr += [f"#PBS -l nodes=1:ppn={a.cpus}", f"#PBS -l mem={_mem_pbs(a.mem)}"]
        hdr += [f"#PBS -l walltime={a.time}", "#PBS -j oe", f"#PBS -o {logs}/", "#PBS -V"]
        if a.queue:
            hdr.append(f"#PBS -q {a.queue}")
        if a.account:
            hdr.append(f"#PBS -A {a.account}")
        body = (
            "# PBS starts jobs in $HOME, not the submission folder: move back so\n"
            "# relative paths (results/, data/) resolve as they did at qsub time.\n"
            'cd "${PBS_O_WORKDIR:-.}"\n'
            + chrom_arr +
            "# PBS Pro / OpenPBS set PBS_ARRAY_INDEX; Torque sets PBS_ARRAYID.\n"
            f'IDX="${{PBS_ARRAY_INDEX:-${{PBS_ARRAYID:-{"1" if n == 1 else ""}}}}}"\n'
            'if [ -z "$IDX" ]; then\n'
            '  echo "ERROR: no PBS array index found -- submit with qsub as an array job" >&2\n'
            "  exit 1\n"
            "fi\n"
            'CHR="${CHROMS[$((IDX-1))]}"\n'
            "export OMP_NUM_THREADS=${NCPUS:-${PBS_NUM_PPN:-1}}\n")
        ext, submit = "pbs", "qsub"

    alt = ""
    if scheduler == "pbs":
        alt = (f"# If your site rejects the array directive, delete it and submit with\n"
               f"#   qsub -J 1-{n} <script>   (PBS Pro / OpenPBS)\n"
               f"#   qsub -t 1-{n} <script>   (Torque)\n")
    path = os.path.join(a.out, f"{a.out_prefix}_array.{ext}")
    with open(path, "w") as fh:
        fh.write("\n".join(hdr) + "\n")
        fh.write(f"# Generated by jasmap_parallel.py: one task per chromosome; each\n"
                 f"# writes {a.out_prefix}_chr<N>_* into {a.out} (JasMAP4 adds the\n"
                 f"# chromosome tag, so tasks never overwrite each other).\n"
                 f"# When every task has finished:\n#   {merge_cmd}\n" + alt + "\n"
                 + env + "\n" + body + 'echo "chromosome $CHR on $(hostname)"\n\n' + run)
    print(f"[{scheduler}] Wrote {path}  ({n} tasks: chr {', '.join(chroms)})")
    print(f"[{scheduler}] Submit with:  {submit} {path}")
    if scheduler == "pbs":
        print(f"[pbs] Script uses {'PBS Pro / OpenPBS (-J)' if a.pbs_flavour == 'pro' else 'Torque (-t)'} "
              f"array syntax. Check yours with `qstat --version`: 'pbs_version' "
              f"means PBS Pro/OpenPBS; 'Version: 6.x' or older means Torque "
              f"(then use --pbs-flavour torque).")
    if "--covfile" not in args:
        print(f"[{scheduler}] NOTE: no --covfile given. Each job will then compute PCs "
              f"from its own chromosome only -- run `prepare` first and pass its "
              f"--covfile for genome-wide PCs.")
    if "--PRS" in args:
        print(f"[{scheduler}] NOTE: --PRS is not meaningful per chromosome; run it "
              f"once, genome-wide, instead.")
    return path


# ==========================================================================
# merge
# ==========================================================================
_SKIP_EXT = (".png", ".pdf", ".svg", ".log", ".json", ".bed", ".gff3")
_SKIP_NAMES = ("genomic_control.tsv", "validation_summary.tsv", "provenance.tsv",
               "orchestration_provenance.json")
_PVAL_COLS = ["pval_consensus", "SNP_Pval", "Pval", "P", "pval", "AM_Pval", "P_VALUE"]
_CHR_COLS = ["CHR", "chr", "CHROM", "chrom", "#CHROM"]
_POS_COLS = ["POS", "BP", "pos", "bp", "START_BP", "start"]


def _chrom_key(c):
    s = str(c).lower().replace("chr", "")
    return (0, int(s)) if s.isdigit() else (1, {"x": 23, "y": 24, "mt": 25, "m": 25}.get(s, 99))


def _discover(out_dir, prefix):
    pat = re.compile(rf"^{re.escape(prefix)}_chr([0-9A-Za-z]+)_(.+)$")
    groups = defaultdict(dict)
    for entry in os.listdir(out_dir):
        m = pat.match(entry)
        if not m:
            continue
        chrom, suffix = m.group(1), m.group(2)
        full = os.path.join(out_dir, entry)
        if os.path.isdir(full):
            for root, _d, files in os.walk(full):
                for f in files:
                    rel = os.path.relpath(os.path.join(root, f), full)
                    groups[f"{suffix}/{rel}"][chrom] = os.path.join(root, f)
        else:
            groups[suffix][chrom] = full
    return groups


def _read_table(path):
    with open(path) as fh:
        first = fh.readline()
    sep = "\t" if "\t" in first else r"\s+"
    toks = first.split()
    has_header = bool(toks) and any(not re.fullmatch(r"[-+0-9.eE]+|nan|NA", t) for t in toks)
    return pd.read_csv(path, sep=sep, header=0 if has_header else None,
                       engine="python"), has_header


def _first(cols, options):
    for o in options:
        if o in cols:
            return o
    return None


def _lambda_gc(p):
    from scipy import stats
    p = np.asarray(p, dtype=float)
    p = p[np.isfinite(p) & (p > 0) & (p <= 1)]
    if len(p) < 10:
        return float("nan")
    return float(np.median(stats.chi2.isf(p, 1)) / stats.chi2.ppf(0.5, 1))


def cmd_merge(a):
    from jasmap.plotting import plots

    groups = _discover(a.out, a.out_prefix)
    if not groups:
        sys.exit(f"ERROR: no '{a.out_prefix}_chr<N>_*' outputs found in {a.out}. "
                 f"Check --out and --out-prefix match the chromosome jobs.")
    all_chroms = sorted({c for g in groups.values() for c in g}, key=_chrom_key)
    print(f"[merge] Found results for {len(all_chroms)} chromosome(s): "
          f"{', '.join(all_chroms)}")
    merged_dir = os.path.join(a.out, f"{a.out_prefix}_merged")
    os.makedirs(merged_dir, exist_ok=True)

    summary, warnings = [], []
    for suffix, by_chr in sorted(groups.items()):
        name = os.path.basename(suffix)
        if name.endswith(_SKIP_EXT) or name in _SKIP_NAMES:
            continue
        if "prs_results" in suffix:
            if "prs" not in [w[0] for w in warnings]:
                warnings.append(("prs", "PRS outputs were NOT merged: a polygenic score "
                                 "must be fitted genome-wide. Run --PRS once without --chr."))
            continue
        missing = [c for c in all_chroms if c not in by_chr]
        if missing:
            warnings.append((suffix, f"{suffix}: missing chromosome(s) {', '.join(missing)} "
                             f"-- a job may have failed; merged file is incomplete."))
        chroms = sorted(by_chr, key=_chrom_key)
        try:
            parts = [(c, *_read_table(by_chr[c])) for c in chroms]
        except Exception as e:
            warnings.append((suffix, f"{suffix}: could not read ({e}); skipped."))
            continue

        out_path = os.path.join(merged_dir, suffix.replace("/", "__"))

        # Global ancestry: one row per SAMPLE per chromosome -> weighted mean
        if "global_ancestry" in name:
            frames = [df for _c, df, _h in parts]
            num = frames[0].select_dtypes("number").columns
            w = []
            for c in chroms:
                la = groups.get("consensus_lai_local_ancestry.txt", {}).get(c)
                w.append(sum(1 for _ in open(la)) if la and os.path.exists(la) else 1)
            w = np.asarray(w, dtype=float) / np.sum(w)
            avg = frames[0].copy()
            avg[num] = sum(wi * f[num].to_numpy(dtype=float) for wi, f in zip(w, frames))
            avg.to_csv(out_path, sep="\t", index=False)
            summary.append((suffix, len(avg), len(chroms), "", "", "SNP-weighted mean"))
            continue

        df = pd.concat([d for _c, d, _h in parts], ignore_index=True)
        has_header = parts[0][2]
        chr_c, pos_c = _first(df.columns, _CHR_COLS), _first(df.columns, _POS_COLS)
        if chr_c is not None:
            order = sorted(range(len(df)), key=lambda i: (
                _chrom_key(df[chr_c].iat[i]),
                float(df[pos_c].iat[i]) if pos_c is not None and pd.notna(df[pos_c].iat[i]) else 0.0))
            df = df.iloc[order].reset_index(drop=True)
        df.to_csv(out_path, sep="\t", index=False, header=has_header)

        pcol = _first(df.columns, _PVAL_COLS)
        lam, minp = "", ""
        if pcol is not None:
            pv = pd.to_numeric(df[pcol], errors="coerce")
            lam = _lambda_gc(pv)
            minp = float(np.nanmin(pv)) if pv.notna().any() else float("nan")
            stem = os.path.splitext(suffix.replace("/", "__"))[0]
            if chr_c is not None and pos_c is not None and pv.notna().sum() >= 10:
                pdf = pd.DataFrame({"CHR": df[chr_c], "BP": pd.to_numeric(df[pos_c], errors="coerce"),
                                    "P": pv}).dropna()
                try:
                    plots.manhattan(pdf, pval_col="P", chrom_col="CHR", bp_col="BP",
                                    title=f"{a.out_prefix}: {stem} (genome-wide, merged)",
                                    out_path=os.path.join(merged_dir, f"{stem}_manhattan.png"))
                    plots.qq_plot(pdf["P"].to_numpy(), title=f"{stem}  lambda_GC={lam:.3f}",
                                  out_path=os.path.join(merged_dir, f"{stem}_qq.png"))
                except Exception as e:
                    warnings.append((suffix, f"{suffix}: plot failed ({e})."))
        summary.append((suffix, len(df), len(chroms), lam, minp, pcol or ""))

    rep = pd.DataFrame(summary, columns=["file", "rows", "chromosomes",
                                         "lambda_GC_genomewide", "min_p", "pvalue_column"])
    rep_path = os.path.join(merged_dir, "merge_summary.tsv")
    rep.to_csv(rep_path, sep="\t", index=False)

    print(f"\n[merge] Wrote {len(rep)} merged file(s) to {merged_dir}/")
    for _, r in rep.iterrows():
        lam = r["lambda_GC_genomewide"]
        lam_t = f"lambda_GC={lam:.3f}" if isinstance(lam, float) and np.isfinite(lam) else ""
        print(f"   {r['file']:<52} {r['rows']:>9,} rows  {r['chromosomes']:>2} chr  {lam_t}")
    for _k, w in warnings:
        print(f"[merge] WARNING: {w}")
    print(f"[merge] Summary -> {rep_path}")
    return rep


# ==========================================================================
def main():
    argv = sys.argv[1:]
    passthrough = []
    if "--" in argv:
        i = argv.index("--")
        argv, passthrough = argv[:i], argv[i + 1:]

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare", help="genome-wide PCs on the tested cohort -> --covfile")
    pp.add_argument("--bfile", required=True)
    pp.add_argument("--pheno", default=None)
    pp.add_argument("--pop-file", default=None)
    pp.add_argument("--admix-label", default=None)
    pp.add_argument("--auto-pca-k", type=int, default=10)
    pp.add_argument("--maf-min", type=float, default=0.01)
    pp.add_argument("--out", required=True)
    pp.add_argument("--out-prefix", default="")

    for sched, hlp in [("slurm", "write a SLURM job-array script (sbatch)"),
                       ("pbs", "write a PBS job-array script (qsub)")]:
        ps = sub.add_parser(sched, help=hlp + "; JasMAP4 options go after --")
        ps.add_argument("--chroms", default="1-22", help='e.g. "1-22" or "1-22,X"')
        ps.add_argument("--out", required=True)
        ps.add_argument("--out-prefix", required=True)
        ps.add_argument("--cpus", type=int, default=4)
        ps.add_argument("--mem", default="16G", help='per job, e.g. "16G"')
        ps.add_argument("--time", default="12:00:00", help="walltime HH:MM:SS")
        ps.add_argument("--queue", default=None,
                        help="SLURM partition / PBS queue (-q), if your site requires one")
        ps.add_argument("--account", default=None,
                        help="project/allocation to charge (SLURM --account, PBS -A)")
        ps.add_argument("--conda-env", default=None,
                        help="conda environment to activate inside each job, e.g. jasmap")
        if sched == "pbs":
            ps.add_argument("--pbs-flavour", default="pro", choices=["pro", "torque"],
                            help='"pro" = PBS Pro / OpenPBS (#PBS -J, PBS_ARRAY_INDEX; '
                                 'default); "torque" = Torque (#PBS -t, PBS_ARRAYID)')

    pm = sub.add_parser("merge", help="merge per-chromosome results, genome-wide lambda + plots")
    pm.add_argument("--out", required=True, help="folder holding the per-chromosome outputs")
    pm.add_argument("--out-prefix", required=True, help="the --out-prefix used by the jobs")

    a = p.parse_args(argv)
    if a.cmd == "prepare":
        cmd_prepare(a)
    elif a.cmd in ("slurm", "pbs"):
        cmd_jobscript(a, passthrough, a.cmd)
    else:
        cmd_merge(a)


if __name__ == "__main__":
    main()
