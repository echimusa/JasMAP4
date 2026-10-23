# JasMAP v4 — Joint Ancestry and SNP Mapping Method
### Extended with GIFT · Consensus LAI · Consensus PRS · Publication Plots

---

## Quick-start

```bash
pip install -r requirements.txt

# Admixture mapping (classic JasMAP)
python3 JasMAP4.py --bfile data/cohort --pheno pheno.txt \
  --ancfile ancestry.txt --nbanc 3 --mode admix --out results/

# GWAS (LMM SNP-only)
python3 JasMAP4.py --bfile data/cohort --pheno pheno.txt \
  --mode GWAS --out results/

# GIFT only
python3 JasMAP4.py --bfile data/cohort --pheno pheno.txt \
  --mode gift --out results/

# GIFT + multi-way admixture mapping
python3 JasMAP4.py --bfile data/cohort --pheno pheno.txt \
  --ancfile ancestry.txt --nbanc 3 --mode gift_admix \
  --pop-labels AFR,EUR,SAN --out results/

# Compute local ancestry automatically, then run admixture mapping
python3 JasMAP4.py --bfile data/cohort --pheno pheno.txt \
  --nbanc 3 --yes-anc --pop-file pop_labels.txt \
  --pop-labels AFR,EUR,SAN --mode admix --out results/

# Any mode + 8-step consensus GWAS/RVAT pipeline
python3 JasMAP4.py --bfile data/cohort --pheno pheno.txt \
  --mode GWAS --consgwas --out results/

# Any mode + consensus PRS
python3 JasMAP4.py --bfile data/cohort --pheno pheno.txt \
  --mode GWAS --PRS Yes --sumstats gwas_sumstats.tsv --out results/

# Full pipeline: admix + consensus GWAS + PRS, all with plots
python3 JasMAP4.py --bfile data/cohort --pheno pheno.txt \
  --ancfile ancestry.txt --nbanc 3 --mode admix \
  --consgwas --PRS Yes --pop-labels AFR,EUR,SAN --out results/

# Parallel per-chromosome runs on the SAME genotype file (e.g. one process
# per chromosome, submitted separately) -- --out-prefix defaults to
# "JasMAP" and --chr's value is folded into it automatically, so filenames
# never collide even when sharing the same --out folder:
python3 JasMAP4.py --bfile data/cohort --pheno pheno.txt \
  --mode GWAS --chr 1 --out results/ &
python3 JasMAP4.py --bfile data/cohort --pheno pheno.txt \
  --mode GWAS --chr 2 --out results/ &
# ...one per chromosome... writes JasMAP_chr1_Jasmap.GWAS.assoc,
# JasMAP_chr2_Jasmap.GWAS.assoc, etc. Pass --out-prefix explicitly (e.g.
# --out-prefix cohortA) to distinguish separate COHORTS the same way.
```

---

## Installation

**Supported Python: 3.10–3.12** (3.12 recommended). Always use a dedicated
environment, never a shared or `base` one.

### On HPC clusters — use conda (recommended)

```bash
conda env create -f environment.yml      # or: mamba env create -f environment.yml
conda activate jasmap
```

`environment.yml` installs everything from **conda-forge as prebuilt
binaries** — nothing is compiled — and resolves all packages together, so the
required `numpy<2` holds across the whole stack. Tested by building the
environment from scratch and running the validators and a full GLMM analysis
inside it. Create the environment on a login node if compute nodes have no
internet access.

**Why not pip on a cluster?** pip compiles from source whenever no prebuilt
wheel matches the cluster's Python version, CPU architecture or system C
library — and clusters rarely have a working C/Cython toolchain. Typical
symptoms:

```
meson.build:1:0: ERROR: Compiler cython cannot compile programs.
ERROR: Failed to build 'pandas' when installing build dependencies for pandas
```

If you see `Building wheel for numpy` or `Building wheel for pandas`, pip has
fallen back to compiling. Use conda instead.

### With pip (workstations)

```bash
python3 -m venv jasmap-env && source jasmap-env/bin/activate
pip install --only-binary=:all: -r requirements.txt
```

`--only-binary=:all:` makes pip **fail immediately with a clear message**
instead of attempting a compile. If it fails, check the likely causes:

```bash
python3 --version          # must be 3.10-3.12
ldd --version | head -1    # system C library; very old versions lack wheels
uname -m                   # x86_64 is best supported
```

On a cluster with environment modules, `module load python/3.12` (exact name
varies by site) may provide a suitable interpreter — but conda is more
reliable.

---

## Analysis modes (`--mode`)

| Mode | Description |
|------|-------------|
| `admix` | JasMAP Bayesian joint ancestry + SNP association (original method) |
| `GWAS` | LMM SNP-only association (batched, fast) |
| `gift` | Genomic Informational Field Theory (GIFT): MI field + propagator |
| `gift_admix` | GIFT with multi-way ancestry stratification + admixture mapping scan |
| `rvat` | Gene-based rare-variant association (burden, SKAT, SKAT-O, ACAT-V) |
| `LAI` | Local and global ancestry inference only — no association testing |

`--consgwas` can be added to any of the above to run the 8-step Consensus
GWAS/RVAT pipeline after the primary mode completes (see below).

---

## Association models: LMM, GLMM and saddlepoint approximation

Both `--mode admix` (admixture mapping) and `--mode GWAS` can use either model:

| Option | Applies to | Model |
|--------|------------|-------|
| `--am-model lmm` | admixture mapping | Gaussian linear mixed model **(default)** |
| `--am-model glmm` | admixture mapping | Logistic mixed model |
| `--gwas-model lmm` | GWAS | Gaussian linear mixed model **(default)** |
| `--gwas-model glmm` | GWAS | Logistic mixed model |
| `--spa` | any `glmm` above | Saddlepoint approximation for the score test |

Defaults are unchanged; the GLMM and SPA are opt-in. Both GLMMs are
implemented in `jasmap/admix_glmm.py`.

**The models.** Admixture mapping, for each ancestry *k*:

```
logit P(case_i) = covariates + genotype PCs + q_ik + a_ik(l) + u_i,   u ~ N(0, tau_k K_k)
```

with `q_ik` the global ancestry proportion, `a_ik(l)` the expected
local-ancestry dosage (0–2) and `K_k` an ancestry-specific kinship. GWAS:

```
logit P(case_i) = covariates + genotype PCs + g_i(l) + u_i,           u ~ N(0, tau K)
```

with `g_i(l)` the SNP dosage and `K` a genomic relationship matrix from up to
50,000 evenly spaced SNPs. **All kinships are built on the tested
(phenotyped) cohort only**, so reference panels cannot shape them. Each null
model is fitted once; every locus is then a score test.

**When to use the GLMM.** A Gaussian model on a binary outcome is well
calibrated when cases and controls are balanced, but not otherwise — and the
error runs in *opposite directions* depending on predictor frequency, so one
genomic-control correction cannot fix it. In null simulations at 87.6% cases,
the LMM gave λ<sub>GC</sub> = 0.54 against 1.02 for the GLMM. With balanced
data the two agree closely (Spearman ρ = 0.96 between their p-values in
testing), so the GLMM costs little when it is not needed.

**When to add `--spa`.** The GLMM score test uses a normal approximation,
which is accurate in the body of the distribution but anti-conservative in
the *tails* for low-frequency predictors under imbalance — exactly where
significance is decided. In null simulations at 87.6% cases:

| Predictor | α | Expected | Normal approx. | With `--spa` |
|-----------|---|----------|----------------|--------------|
| MAF 0.30 | 10⁻³ | 20 | 13 | 13 |
| MAF 0.03 | 10⁻³ | 20 | 33 | 26 |
| MAF 0.03 | 10⁻⁴ | 2 | 5 | 1 |

SPA matters for rare variants (GWAS) and minor ancestries (admixture
mapping), and changes nothing for common predictors. It is applied only where
|z| > 2, so p-values elsewhere are identical and the added cost is small. The
run suggests `--spa` when the smaller outcome class is under 20%.

**Covariates.** Logistic models need roughly ten events per covariate in the
*smaller* outcome class, so keep `--auto-pca-k` modest when controls (or
cases) are few — with *K* ancestries, *K*−1 PCs span the ancestry space. If a
null model fails to converge the run says so; reduce `--auto-pca-k` or use the
LMM. Both GLMMs fall back to the LMM, with a warning, if the phenotype is not
binary (0/1 or PLINK 1/2). Cost: about a minute per null fit at n≈700.

```bash
# admixture mapping, imbalanced case-control
python3 JasMAP4.py --bfile data/COHORT --pheno data/COHORT.pheno \
    --nbanc 5 --yes-anc --pop-file data/POP.txt --pop-labels KHS,CEU,YRI,CHB,GIH \
    --admix-label SAC --mode admix --am-model glmm --spa --auto-pca-k 4 \
    --out results/ --out-prefix SAC

# GWAS, imbalanced case-control
python3 JasMAP4.py --bfile data/COHORT --pheno data/COHORT.pheno \
    --pop-file data/POP.txt --admix-label SAC \
    --mode GWAS --gwas-model glmm --spa --auto-pca-k 4 \
    --out results/ --out-prefix SAC_gwas
```

---

## Key options

```
Genotype input (choose one):
  --bfile PREFIX     PLINK .bed/.bim/.fam (recommended; uses bed-reader, no compiler needed)
  --vcf FILE         VCF file
  --geno PREFIX      EIGENSTRAT .geno/.snp/.ind
  --chr N            Restrict analysis to one chromosome (e.g. --chr 22 or
                     --chr chr22, both accepted). Applies consistently
                     across every --mode, --PRS, and --yes-anc; stops with a
                     clear error listing the chromosome codes actually
                     present if N doesn't match anything in the data. Works
                     with all three genotype input formats. See
                     --lai-genome-wide below for combining this with
                     --yes-anc.

Phenotype / covariates:
  --pheno FILE       Phenotype file (FID IID PHENO)
  --covfile FILE     Covariate file (FID IID COV1 COV2 ...)

Ancestry:
  --mode admix|gift_admix requires --nbanc K
  --ancfile FILE     Local ancestry file (input if --yes-anc is off;
                     output filename prefix if --yes-anc is on)
  --nbanc K          Number of ancestral populations
  --pop-labels L1,L2,L3   Comma-separated ancestry names (e.g. AFR,EUR,SAN)

Consensus LAI (--yes-anc):
  --yes-anc          Compute local ancestry automatically before the main
                     analysis (Steps 1-8 of the consensus LAI pipeline).
                     Requires --nbanc and --pop-file.
  --pop-file FILE    Two-column file (sample_ID, population_label).
                     Number of unique reference populations must be >= --nbanc.
  --admix-label L    The population_label in --pop-file that identifies
                     admixed study-cohort samples, as opposed to reference/
                     ancestral population samples (default: "ADMIX"; change
                     this if your --pop-file uses a different label, e.g.
                     "STUDY" or "TARGET"). Stops with a clear error listing
                     the labels actually present in --pop-file if L doesn't
                     match any of them. Used as the authoritative signal
                     (over phenotype validity alone) for restricting RVAT's
                     MAF/gene-level filtering, auto-PCA, and PRS's reported
                     output to the true study cohort.
  --lai-genome-wide  When used together with --chr, computes local ancestry
                     from the FULL genome (ignoring --chr for that step
                     only), while --chr still restricts the downstream
                     --mode admix/gift_admix association analysis to the
                     requested chromosome. Default (off): --chr restricts
                     both LAI and the downstream analysis identically. Has
                     no effect without both --yes-anc and --chr.

Consensus PRS:
  --PRS Yes|No       Run consensus PRS pipeline after the main analysis
                     (default: No)
  --sumstats FILE    GWAS summary statistics for C+T PRS (SNP A1 BETA P)
  --prs-methods M    Comma-separated PRS methods: ridge,lasso,ct
                     (default: ridge,lasso)

GIFT parameters:
  --gift-n-components N      PCA components for field compression (default: 20)
  --gift-lambda L            L2 regularisation (default: 1e-3)
  --gift-mi-chunk C          SNPs per MI chunk — tune to RAM (default: 512)
  --gift-ancestry-mode MODE  stratified|covariate|interaction|all
                             (default: stratified)

Output / plots:
  --out DIR          Output directory (default: jasmap_results/)
  --out-prefix P     Filename prefix applied to every result/log/plot file
                     (default: "JasMAP", e.g. JasMAP_Jasmap.Joint.assoc).
                     Pass your own (e.g. --out-prefix chr22_run) to make runs
                     easier to tell apart. When --chr is also given, the
                     requested chromosome is automatically folded into the
                     prefix too -- e.g. --out-prefix chr22_run --chr 22
                     writes chr22_run_chr22_Jasmap.Joint.assoc; the default
                     prefix gets the same treatment
                     (JasMAP_chr22_Jasmap.Joint.assoc) -- so two parallel
                     runs analysing different chromosomes of the SAME
                     genotype file, even sharing the same --out folder,
                     never collide on filenames. Has no effect on
                     temporary/intermediate files, which are already named
                     uniquely per run and deleted after a successful run.
  --plots            Generate publication figures (default: on)
  --no-plots         Disable plot generation

Consensus GWAS/RVAT (--consgwas):
  --consgwas         Run the 8-step Consensus GWAS/RVAT pipeline after the
                     primary --mode. Integrates primary-mode results with
                     internal pure-Python tools (OLS, Firth-approx, burden
                     collapsing, SKAT-approx) via inverse-variance-weighted
                     meta-analysis (common variants) or ACAT/Cauchy combination
                     (windows/gene-based). Works with all four --mode values.
                     Outputs land in {--out}/consensus_gwas/ and include:
                       consensus_snp_stats.tsv   (full per-SNP consensus table)
                       consensus_significant_hits.tsv
                       genomic_control.tsv        (λ_GC + weight per tool)
                       validation_summary.tsv     (cross-tool concordance)
                       provenance.tsv             (per-step audit trail)
                       consensus_manhattan.png    consensus QQ, per-tool QQ
                       genomic_inflation_barchart.png  tool_concordance_scatter.png
  --consgwas-perms N Number of null permutations for empirical type I error
                     calibration (Step 6; 0 = skip; default: 50).
  --consgwas-sig F   Significance threshold override (default: 5e-8 for SNPs,
                     1e-6 for windows/genes — auto-detected).

LMM / kinship:
  --kinship 0|1|2|3  GRM method (default: 0, Yang et al.)
  --samplesize n,m   Cases,controls for kinship methods 2 and 3
  --ntest N          Effective number of tests for Bonferroni correction
```

---

## Output files

### All modes
| File | Contents |
|------|----------|
| `jasmap-*.log` | Full run log |

### `admix` mode
| File | Contents |
|------|----------|
| `Jasmap.Joint.assoc` | SNP × ancestry joint PPA, AM beta/p, SNP beta/p |
| `Jasmap.ANC_<POP>.AM.assoc` | Per-ancestry admixture-mapping stats, one file per `--pop-labels` entry (e.g. `Jasmap.ANC_CEU.AM.assoc`) |
| `plot_joint_ppa.png` | Joint PPA + SNP p-value scatter |
| `plot_am_anc{k}.png` | Per-ancestry Manhattan plots |

Admixture mapping uses **expected** local-ancestry dosage (posterior
probabilities) rather than hard ancestry calls when `--yes-anc` is used, so
minor ancestries that are rarely anyone's most likely call remain testable.
Ancestry statistics are computed on the tested (phenotyped) cohort only. The
log reports, per ancestry, the mean local-ancestry proportion and the share of
SNPs with usable variation — check it if many tests return `nan`.

### `GWAS` mode
| File | Contents |
|------|----------|
| `Jasmap.GWAS.assoc` | SNP beta and p-values |
| `plot_gwas_manhattan.png` | Manhattan plot |
| `plot_gwas_qq.png` | QQ plot with λ_GC |

### `gift` mode
| File / folder | Contents |
|---------------|----------|
| `gift_results/snp_importance.tsv` | Normalised GIFT score per SNP |
| `gift_results/gift_predictions.tsv` | Per-sample predictions |
| `gift_results/field_potential.tsv` | Per-sample field potential energy |
| `gift_results/gift_snp_importance.png` | Top-N SNP bar chart |
| `gift_results/gift_field_potential.png` | Field potential histogram |
| `gift_results/gift_qq.png` | QQ of GIFT scores |
| `gift_results/gift_field_pca.png` | Field-space PCA scatter |

### `gift_admix` mode
All of the above, plus:
| File / folder | Contents |
|---------------|----------|
| `gift_admix_results/admixture_mapping.tsv` | Per-window per-ancestry scan |
| `gift_admix_results/global_ancestry.tsv` | Per-sample ancestry proportions |
| `gift_admix_results/global_ancestry.png` | Stacked ancestry bar chart |
| `gift_admix_results/lai_heatmap.png` | Local ancestry window heatmap |
| `gift_admix_results/pca_global_ancestry.png` | Ancestry PCA scatter |
| `gift_admix_mapping.png` | Admixture mapping Manhattan (all ancestries) |

### `--yes-anc` (consensus LAI)
| File | Contents |
|------|----------|
| `consensus_lai_global_ancestry.tsv` | Per-sample ancestry proportions (K columns) |
| `consensus_lai_local_ancestry.txt` | Per-SNP haplotype ancestry strings (ancfile format) |
| `lai_global_ancestry.png` | Stacked ancestry bar chart |
| `lai_heatmap.png` | Window × sample local ancestry heatmap |
| `lai_pca.png` | PCA of global ancestry proportions |
| `karyogram_<ID>.png` | Genome-wide local ancestry for the 5 most admixed study-cohort samples |
| `lai_region_chr<N>.png` | Local ancestry across one region (see `--lai-region`) |

**Ancestry labels are anchored to the reference panels.** Inferred ancestry
components are matched to `--pop-labels` using the reference samples in
`--pop-file`; the log reports each panel's loading on its own component. A
**weak anchoring** warning (loading below 0.5) means that population is not
cleanly separated at this *K* — treat results labelled with it cautiously.
Without `--pop-file` the component names are arbitrary.

**`--lai-region`** chooses the region plot:

```bash
--lai-region 8:20-30   # chr8, 20–30 Mb
--lai-region 11        # all of chr11 (dense; a few Mb is more readable)
--lai-region none      # skip it
```

Without it, an arbitrary 5 Mb slice of the first chromosome is drawn as a
format demonstration only.

### `--PRS Yes`
| File / folder | Contents |
|---------------|----------|
| `prs_results/prs_scores.tsv` | Per-sample consensus PRS + per-method scores |
| `prs_results/prs_validation.tsv` | AUC, Pearson r, calibration slope |
| `prs_results/prs_portability.tsv` | Per-ancestry portability ratios (when LAI available) |
| `prs_results/prs_distribution.png` | Cases vs. controls PRS density |
| `prs_results/prs_roc.png` | ROC curve with AUC |

### `--consgwas`
Runs after any `--mode` and produces a separate `consensus_gwas/` folder.
| File | Contents |
|------|----------|
| `consensus_gwas/consensus_snp_stats.tsv` | Per-variant consensus beta, SE, p + per-tool columns |
| `consensus_gwas/consensus_significant_hits.tsv` | Variants below the significance threshold |
| `consensus_gwas/genomic_control.tsv` | λ_GC and adjusted weight per tool |
| `consensus_gwas/validation_summary.tsv` | Cross-tool concordance fraction and hit counts |
| `consensus_gwas/provenance.tsv` | Per-step audit trail (tools used, decisions taken) |
| `consensus_gwas/consensus_manhattan.png` | Manhattan plot of consensus p-values |
| `consensus_gwas/consensus_qq.png` | QQ of consensus p-values |
| `consensus_gwas/per_tool_qq_comparison.png` | Overlaid QQ for all tools |
| `consensus_gwas/genomic_inflation_barchart.png` | λ_GC bar chart with inflation flags |
| `consensus_gwas/tool_concordance_scatter.png` | p-value scatter between the top-2 tools |

---

## Running chromosomes in parallel (HPC)

`--chr N` restricts a run to one chromosome and tags its outputs
(`<out-prefix>_chrN_*`), so jobs can share an `--out` folder without
overwriting each other. Each such run plots only its own chromosome.
`jasmap_parallel.py` turns this into a genome-wide workflow:

```bash
# 1. once: genome-wide PCs on the tested cohort -> a shared covariate file
python3 jasmap_parallel.py prepare --bfile data/COHORT --pheno data/COHORT.pheno \
    --pop-file data/POP.txt --admix-label SAC --auto-pca-k 4 --out results/ --out-prefix SAC

# 2. one job per chromosome (writes a SLURM array script; JasMAP4 options go after --)
python3 jasmap_parallel.py slurm --chroms 1-22 --out results/ --out-prefix SAC \
    --mem 32G --time 24:00:00 -- \
    --bfile data/COHORT --pheno data/COHORT.pheno --mode admix --am-model glmm --spa \
    --nbanc 5 --yes-anc --pop-file data/POP.txt --pop-labels KHS,CEU,YRI,CHB,GIH \
    --admix-label SAC --covfile results/SAC_genomewide_pcs.cov
sbatch results/SAC_array.slurm
#   ...or on PBS: same options, `pbs` instead of `slurm`, then qsub
python3 jasmap_parallel.py pbs --chroms 1-22 --out results/ --out-prefix SAC \\
    --mem 32G --time 24:00:00 -- <same JasMAP4 options as above>
qsub results/SAC_array.pbs

# 3. once all jobs finish: genome-wide files, lambda_GC and plots
python3 jasmap_parallel.py merge --out results/ --out-prefix SAC
```

**Schedulers.** `slurm` writes a script for `sbatch`; `pbs` writes one for
`qsub`. PBS comes in two families with incompatible array syntax, so check
yours with `qstat --version`:

| `qstat --version` shows | Family | Option | Array directive | Index variable |
|---|---|---|---|---|
| `pbs_version = ...` | PBS Pro / OpenPBS | `--pbs-flavour pro` (default) | `#PBS -J 1-22` | `PBS_ARRAY_INDEX` |
| `Version: 6.x` or older | Torque | `--pbs-flavour torque` | `#PBS -t 1-22` | `PBS_ARRAYID` |

The PBS script reads either index variable, and returns to the submission
folder before running — PBS starts jobs in `$HOME`, which would otherwise
break relative paths such as `results/`. If your site rejects the array
directive, delete it and pass the range at submission instead
(`qsub -J 1-22 …` for PBS Pro, `qsub -t 1-22 …` for Torque). Both schedulers
accept `--queue`, `--account`, `--cpus`, `--mem`, `--time` and
`--conda-env jasmap` (activates the environment inside each job — useful
when jobs do not inherit it). Tested by running the generated PBS script as
PBS would, from a different starting directory with PBS's own variables set:
all 22 tasks completed and merged to a result identical to the SLURM-style
run.

`merge` writes to `results/SAC_merged/`: each result type concatenated
across chromosomes and sorted, a genome-wide Manhattan and QQ plot per
association file, and `merge_summary.tsv` with the genome-wide λ<sub>GC</sub>.
It warns if any chromosome is missing from a result type — usually a failed
job. Works for every mode (`GWAS`, `admix`, `gift`, `gift_admix`, `rvat`,
`LAI`) and for `--consgwas` outputs.

**Why step 1 matters.** Without `--covfile`, a `--chr` run computes its PCs
from that one chromosome — noisy, and liable to track chromosome-specific
features such as inversions rather than ancestry. With shared genome-wide PCs,
the merged per-chromosome GWAS reproduced a genome-wide run in testing
(p-value rank correlation 0.998, λ<sub>GC</sub> 0.523 vs 0.526).

**What merges, and what does not.**

| Output | Handling |
|--------|----------|
| GWAS, admixture-mapping, joint, consensus, local-ancestry, GIFT and RVAT tables | Concatenated; λ<sub>GC</sub> recomputed genome-wide (not averaged) |
| Global ancestry | Averaged across chromosomes, weighted by SNP count |
| PRS | **Not merged** — a polygenic score must be fitted genome-wide; run `--PRS` once without `--chr` |
| Per-chromosome summaries and plots | Superseded by the genome-wide versions |

Ancestry labels stay consistent across chromosome jobs because each job
anchors its ancestry components to the `--pop-file` reference panels. GIFT
importance scores come from a per-run model, so merged GIFT scores rank SNPs
within, not across, chromosomes.

**Known limitation.** Each job still builds its kinship from its own
chromosome — the one being tested. In a mixed model this lets the random
effect absorb part of a genuine signal ("proximal contamination"), making
per-chromosome runs **conservative**: some power is lost, but no false
positives are created. The effect grows with the trait's heritability. A
leave-one-chromosome-out kinship is the full remedy and is not yet
implemented.

---

## Input file formats

**Phenotype (`--pheno`).** `FID IID PHENO` (or a single column in sample
order); an optional header row is skipped. Binary or quantitative:

| Values | Read as |
|--------|---------|
| `1` / `2` (PLINK) | binary: control / case |
| `0` / `1` | binary: control / case |
| anything else, e.g. `1.73`, `-0.42` | quantitative, used as given |
| `-9`, `NA` | missing |
| `0` alongside `1` and `2` | missing (PLINK convention for 1/2 coding) |

The GLMMs (`--am-model glmm`, `--gwas-model glmm`) need a binary trait and
otherwise fall back to the LMM. For a quantitative trait, PRS reports
held-out r and R².

**Covariates (`--covfile`).** `FID IID cov1 [cov2 ...]`, optional header.
Rows are **matched by IID**, so the order need not follow the `.fam`. Every
sample with a phenotype must have a row with numeric values.

**Populations (`--pop-file`, `--pop-labels`).** Each `--pop-labels` entry must
match a population in `--pop-file` exactly (case-sensitive), so its ancestry
component can be anchored; otherwise the run stops with the list of
populations found.

---

## Preparing the phenotype file

Reference-panel samples must be coded missing (`-9`), and the study cohort
must carry real case/control values. `check_pheno_coding.py` checks this, and
can rebuild a correct file from the `.fam`'s phenotype column:

```bash
# check an existing --pheno file
python3 check_pheno_coding.py --pop-file POP.txt --pheno COHORT.pheno --admix-label SAC

# rebuild it from the .fam (reference samples forced to -9; 0/9 normalised)
python3 check_pheno_coding.py --fam COHORT.fam --pop-file POP.txt \
    --admix-label SAC --from-fam --out COHORT_corrected.pheno
```

A phenotype with no variation in the study cohort makes every association
test undefined — results will be `nan` throughout. Check before interpreting.

---

## Test data generation

```bash
# Small (smoke test): 40 samples, 100 SNPs, 3 ancestries
python3 make_test_data.py --outdir test_small \
  --n-samples 40 --n-snps 100 --n-ancestries 3 --seed 7

# Benchmark: 300 samples, 5000 SNPs, 3 ancestries
python3 make_test_data.py --outdir test_large \
  --n-samples 300 --n-snps 5000 --n-ancestries 3 --seed 42
```

Generates `geno_test`/`.snp`/`.ind`, `plink_test.bed/.bim/.fam`, `vcf_test.vcf`,
`pheno.txt`, `covar.txt`, `ancestry_space.txt`, `ancestry_compact.txt`.

---

## Package structure

```
JasMAP4.py                    -- entry point (all six modes, all options)
requirements.txt
make_test_data.py             -- test data generator (genotype + gene annotation)
validate_kinship.py           -- unit test: calculateKinship()
validate_read_lanc.py         -- unit test: read_lanc() vectorisation
validate_association_batch.py -- unit test: association_batch()
validate_rvat.py              -- unit test: Burden / SKAT / SKAT-O / ACAT-V

jasmap/
├── parse.py          -- harmonised input reading + kinship (makeG)
├── gift/             -- GIFT engine (ported from uploaded scripts, patched)
├── rvat/
│   └── rvat_engine.py  -- Burden / SKAT / SKAT-O / ACAT-V gene-based tests
├── prs/
│   └── consensus_prs.py -- ConsensusPRS: C+T / Ridge / Lasso + stacked validator
├── consensus_gwas/
│   └── consensus_gwas.py -- ConsensusGWAS 8-step pipeline
├── lai/
│   ├── consensus_lai.py    -- built-in LAI engine (HMM+NMF, Steps 1-8)
│   ├── rfmix_utils.py      -- rfmix_to_ancfile(): RFMix v1/v2 -> JasMAP4 ancfile
│   ├── rfmix2plink_tped.py -- Python 3 port: RFMix -> PLINK tped per population
│   └── shapeit2rfmix.py    -- Python 3 port: SHAPEIT2 -> RFMix input format
└── plotting/
    └── plots.py    -- 18 publication-quality plot functions, 300 DPI default
```

---

## External LAI workflow (SHAPEIT2 + RFMix -> JasMAP4)

Use this when you have phased data and prefer RFMix over the built-in engine:

```bash
# 1. Phase with SHAPEIT2
shapeit --input-vcf cohort.vcf --output-max cohort_phased

# 2. Convert to RFMix input (Python 3 port of armartin's shapeit2rfmix.py)
python3 -m jasmap.lai.shapeit2rfmix \
    --shapeit_hap_ref      ref_AFR.haps,ref_EUR.haps,ref_SAN.haps \
    --shapeit_hap_admixed  admixed.haps \
    --shapeit_sample_ref   ref_AFR.sample,ref_EUR.sample,ref_SAN.sample \
    --shapeit_sample_admixed admixed.sample \
    --ref_keep ref_ids.txt --admixed_keep admixed_ids.txt \
    --chr 22 --genetic_map genetic_map_chr22.txt --out rfmix_input/

# 3. Run RFMix externally
rfmix -f rfmix_input/_chr22.alleles -s rfmix_input/_chr22.snp_locations \
      --num-haplotypes-admixed=N --num-haplotypes-reference=M \
      -o rfmix_output/chr22

# 4. Convert RFMix output to JasMAP4 ancfile (NEW rfmix_utils.py)
python3 -m jasmap.lai.rfmix_utils \
    --rfmix rfmix_output/chr{chrom}.Viterbi.txt \
    --fam cohort.fam --pop_labels AFR,EUR,SAN \
    --out jasmap_ancfile.txt \
    --global-ancestry global_ancestry.tsv   # optional

# 5. Run JasMAP4 with the real ancfile
python3 JasMAP4.py --bfile cohort --pheno pheno.txt \
    --ancfile jasmap_ancfile.txt --nbanc 3 --mode admix --out results/
```

Note: rfmix_utils.py supports both RFMix v1 Viterbi format (--fmt viterbi)
and RFMix v2 .msp.tsv format (--fmt msp).

Alternatively, --yes-anc uses the built-in LAI engine with no external tools.


---

## Performance benchmarks (v3 → v4, unchanged core)

| Scenario | v1 original (estimated) | v4 |
|---|---|---|
| `calculateKinship()` (n=100, m=50k) | ~10s | ~2s (5× speedup) |
| `association_batch()` (300 samples, 5000 SNPs) | ~78s | ~3.5s (22× speedup) |
| End-to-end admix (300×5000×3 anc) | ~80s | ~5s (16× speedup) |

---

## Bugs fixed in this release (extending the v3 log)

All v3 bugs (#1-#14) remain fixed. New issues found and corrected in v4:

**Bug #15** — `np.linalg.norm(matrix, 0)` in `gift_fullpipeline.py` and
`gift_admixture_full.py` raised `ValueError: Invalid norm order for matrices`
because MATLAB-style `norm(A, 0)` is not valid in NumPy for 2-D arrays.
Fixed to `np.linalg.norm(matrix, axis=0)` (column-wise L2 norms).

**Bug #16** — `_run_gift_mode()` called `bed.read().T` on the bed-reader output.
`bed.read()` already returns `(n_samples, n_snps)` (rows = samples); transposing
it produced `(n_snps, n_samples)`, causing an off-by-one `IndexError` in
`run_gift()`'s train/test split. Fixed: removed `.T` from all three
input-path branches in `_run_gift_mode()`, `_run_prs_pipeline()`, and
`_run_lai_pipeline()`.

**Bug #17** — `AdmixtureMapping.scan()` received a window list built from the
GIFT default window size (50 SNPs, 25 step), which could contain more windows
than actually exist in `local_ancestry.shape[1]`, causing an out-of-bounds
`IndexError`. Fixed by capping the window list to `local_ancestry.shape[1]`
before passing it to `scan()`.

**Bug #18** -- `_step3_standardise()` in the ConsensusGWAS module could
produce duplicate column names when `collect_mode_results()` had already
pre-renamed a column (e.g., "SNP" → "snp") before the standardiser
tried to rename the same column again. The duplicate caused `df["snp"]`
to return a DataFrame instead of a Series, triggering `AttributeError:
'DataFrame' object has no attribute 'tolist'` and a related
`TypeError: arg must be a list, tuple, 1-d array, or Series` across all
six `--consgwas` test scenarios. Fixed by dropping any pre-existing target
column before renaming to avoid the conflict.

---

## Recommended next steps

- **Chunked batch PRS at biobank scale**: `association_batch()` already supports
  arbitrary batch sizes. For >1M SNPs pass `--gift-mi-chunk 10000` and loop
  over 50k-SNP batches explicitly.
- **Faster PLINK reading for very large .bed files**: swap `bed-reader` for
  `sgkit` + Dask for out-of-core processing (no API change to JasMAP4.py needed —
  only `read_bed_file()` would change).
- **numba JIT acceleration**: `calculateKinship()` and `_builtin_lai()` are the
  two remaining pure-NumPy loops that would benefit most from numba `@jit`
  when numba is available, with a graceful fallback to the NumPy path when it
  is not.
- **LangGraph orchestration**: all eight pipeline steps (LAI, GWAS, GIFT,
  admixture mapping, PRS, plotting, validation, reporting) are now implemented
  as self-contained, independently callable functions, making them natural nodes
  in a LangGraph state graph — a next step documented in the companion series
  documents.

---

## Publishing a new version

Version history is in `CHANGELOG.md`. To publish an update on top of the
existing GitHub repository — keeping its history and earlier tags — run from
the package folder:

```bash
./release_to_github.sh --tag v4.1 --dry-run     # preview the changes first
./release_to_github.sh --tag v4.1 --message "JasMAP4 v4.1"
```

It clones the repository, syncs this folder into it, and refuses to publish if
anything resembling participant or genotype data, or any file over 5 MB, would
be included. Run it from a clean copy of the package, not from a folder where
analyses have been run. (`push_to_github.sh` is for creating a repository for
the first time; it cannot update one that already has history.)
