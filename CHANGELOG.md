# Changelog

## v4.1

Changes since the `v4.0` baseline. Several fix statistical errors, so
**results produced with v4.0 should be re-run** where noted below.

### Re-run advised

- **Ancestry labels (`--yes-anc`).** Inferred ancestry components were named
  by position in `--pop-labels` with no matching step, so every
  ancestry-labelled output could carry the wrong population name: the
  per-ancestry `Jasmap.ANC_<POP>.AM.assoc` files, the `Ancestry` column,
  karyograms and global ancestry tables. Components are now matched to the
  `--pop-file` reference panels (Hungarian assignment). On five clearly
  separated simulated populations, labels went from 0/5 to 5/5 correct, for
  both global and local ancestry.
- **Admixture mapping.** Ancestry statistics were computed over all samples,
  including reference panels, before restricting to the tested cohort; and
  hard ancestry calls discarded minority ancestries. Now: expected (soft)
  local-ancestry dosage, statistics on the tested cohort only, and empirical
  variance. Test data went from ~0.1% to 100% usable tests.
- **PRS.** The reported AUC was in-sample. A held-out estimate is now
  reported separately, and SNP pre-filtering no longer sees held-out
  phenotypes (feature-selection leakage: on pure noise the leaky filter gave
  a held-out AUC of 1.00 against a truth of 0.5). All PRS inputs are now
  restricted to the phenotyped study cohort, so reference panels no longer
  influence imputation.

### New features

- **Logistic mixed models:** `--am-model glmm` (admixture mapping) and
  `--gwas-model glmm` (GWAS), for binary phenotypes. In null simulations at
  87.6% cases, λ<sub>GC</sub> was 0.54 for the LMM and 1.02 for the GLMM.
- **`--spa`:** saddlepoint approximation for GLMM score tests, correcting
  tail p-values at low-frequency predictors under case-control imbalance.
- **Parallel per-chromosome workflow** (`jasmap_parallel.py`): `prepare`
  (genome-wide PCs once), `slurm` / `pbs` (job-array scripts; PBS Pro,
  OpenPBS and Torque), `merge` (genome-wide results, λ<sub>GC</sub> and
  plots). Merged results reproduced a genome-wide run (rank correlation 0.998).
- **`--lai-region`** to choose the local-ancestry region plot
  (`CHR:START-END`, a whole chromosome, or `none`).
- **`check_pheno_coding.py --from-fam`** rebuilds a correct phenotype file
  from the `.fam`.
- **`top_loci.py`** lists the strongest admixture-mapping loci with
  ready-to-run `--lai-region` arguments.
- **`jasmap_orchestrator.py`**: LangGraph orchestration with checkpoint and
  resume, per-step retry and a provenance trail (optional).
- **`--mode LAI`** documented; karyograms now show admixed study-cohort
  samples rather than reference individuals.

### Scaling

- LAI: global ancestry fitted on a 50,000-SNP subsample; local-ancestry array
  memory-mapped above 4 GB; vectorised construction; inert smoothing skipped.
- Chunked kinship streaming now also covers `--vcf` and `--geno`.
- GWAS association runs in SNP chunks instead of one dense design matrix.
- Manhattan plots downsample the non-significant cloud above 250,000 points,
  keeping every point at or above the suggestive line.

### Reporting

- Cross-tool concordance reports N/A, plus pairwise Spearman ρ, when no tool
  has a hit (it previously reported a misleading 0.000).
- RVAT detects a hard MAF floor (panels pre-filtered upstream).
- Admixture-mapping windows annotate SNP rows in the consensus table without
  entering the consensus statistic.
- Mode-accurate `--annot` messages; per-ancestry diagnostics and weak-anchoring
  warnings in the log.

### Installation

- `pyarrow<17` added to `requirements.txt` (it was missing, and pyarrow 17+
  requires NumPy 2).
- Supported Python documented as 3.10–3.12, with a runtime warning on newer
  versions.
- `environment.yml` for conda-forge: the recommended install on HPC clusters,
  where pip often fails trying to compile NumPy or pandas.

### Known limitations

- Kinship is not leave-one-chromosome-out. In mixed models this costs some
  power (conservative, not false positives); per-chromosome runs are most
  affected.
