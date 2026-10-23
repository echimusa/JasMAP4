#!/usr/bin/env python3
"""
check_pheno_coding.py
======================

Standalone sanity check for the --yes-anc / --admix-label data model:
admixed study-cohort samples should have a DEFINED phenotype, and
reference/ancestral population samples should be coded MISSING (-9,
PLINK's convention) in the --pheno file.

Catches exactly the failure mode from a real run: a --pop-file correctly
labelling samples as e.g. "SAC" (admixed) vs "KHS"/"CEU"/"YRI"/"CHB"/"GIH"
(reference), but a --pheno file that gives reference samples a real,
non-missing phenotype value instead of -9 -- which silently defeats
JasMAP4's phenotype-validity fallback (used whenever --pop-file isn't
ALSO passed on a given command) and lets reference populations leak into
the actual analysis undetected until the results look implausible.

Usage
-----
    python3 check_pheno_coding.py --pop-file SAC_POP.txt --pheno SAC_REF.pheno \\
        --admix-label SAC

    # Also write a corrected --pheno file (reference samples forced to -9):
    python3 check_pheno_coding.py --pop-file SAC_POP.txt --pheno SAC_REF.pheno \\
        --admix-label SAC --fix --out SAC_REF_fixed.pheno

File format assumptions (matching JasMAP4's own conventions)
--------------------------------------------------------------
--pop-file : whitespace-delimited, no header, two columns: sample_ID  POP_LABEL
--pheno    : whitespace-delimited, no header, either:
               FID  IID  PHENO      (3 columns -- the common case)
             or
               PHENO                (1 column, one value per sample, no IDs --
                                      in this format sample order must already
                                      match the genotype file exactly; this
                                      script can still report the numeric
                                      breakdown but cannot cross-check by ID)
"""
import argparse
import sys
import pandas as pd
import numpy as np


def load_pop_file(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["ID", "POP"],
                     engine="python", dtype=str)
    return df


def load_pheno_file(path):
    # Sniff 1 vs 3 columns, matching JasMAP4's own read_pheno_file()
    with open(path) as fh:
        first_line = fh.readline().split()
    if len(first_line) >= 3:
        df = pd.read_csv(path, sep=r"\s+", header=None,
                         names=["FID", "IID", "PHENO"], engine="python",
                         usecols=[0, 1, 2])
        df["PHENO"] = pd.to_numeric(df["PHENO"], errors="coerce")
        return df, True
    else:
        df = pd.read_csv(path, sep=r"\s+", header=None, names=["PHENO"],
                         engine="python")
        df["PHENO"] = pd.to_numeric(df["PHENO"], errors="coerce")
        return df, False


def load_fam_file(path):
    """PLINK .fam: FID IID PAT MAT SEX PHENO (column 6 is the phenotype)."""
    df = pd.read_csv(path, sep=r"\s+", header=None,
                     names=["FID", "IID", "PAT", "MAT", "SEX", "PHENO"],
                     engine="python", dtype={"FID": str, "IID": str})
    df["PHENO"] = pd.to_numeric(df["PHENO"], errors="coerce")
    return df


def build_pheno_from_fam(args):
    """
    Build a correct --pheno from the .fam's own phenotype column.

    Why this mode exists
    --------------------
    The original --fix mode could only CORRECT an existing --pheno: it forced
    reference samples to the missing code and left everything else alone. That
    is useless when the --pheno file itself carries no information -- found on
    real data, where every one of 733 admixed samples was coded 0, so --fix
    faithfully preserved 733 zeros and the "corrected" file was as
    uninformative as the original. The real case/control status was sitting
    in the .fam's column 6 the whole time.

    Three normalisations are applied, each reported rather than silent:

      1. Reference-population samples (any label in --pop-file other than
         --admix-label) are set to the missing code, regardless of what the
         .fam says. This catches genotype panels where reference samples
         carry a stray phenotype value -- confirmed on real data, where 8
         CEU reference samples were coded 2 (case), which is exactly the
         kind of leak that inflates a study cohort with samples that were
         never phenotyped.

      2. Non-standard missing codes in the .fam are normalised. PLINK's
         convention is -9, but 0 is also conventionally missing for
         case/control, and real files use a bare 9 (confirmed: 622 samples).
         All of {0, 9, -9, blank} become --missing-code.

      3. Everything else is passed through untouched. Genuine 1/2
         (control/case) values are never modified.
    """
    if not args.fam:
        print("ERROR: --from-fam requires --fam <path to .fam>", file=sys.stderr)
        sys.exit(1)
    if not args.out:
        print("ERROR: --from-fam requires --out <output .pheno path>", file=sys.stderr)
        sys.exit(1)

    pop_df = load_pop_file(args.pop_file)
    fam_df = load_fam_file(args.fam)

    unique_labels = sorted(pop_df["POP"].unique().tolist())
    if args.admix_label not in unique_labels:
        print(f"ERROR: --admix-label {args.admix_label!r} does not match any "
              f"label in --pop-file. Labels found: {unique_labels}",
              file=sys.stderr)
        sys.exit(1)

    print(f".fam:       {len(fam_df)} samples ({args.fam})")
    print(f"--pop-file: {len(pop_df)} samples ({args.pop_file})")

    # load_pop_file() names its columns ID/POP, not IID/POP -- merge on the
    # actual column names rather than assuming they match.
    merged = fam_df.merge(pop_df, left_on="IID", right_on="ID",
                          how="left", indicator=True)
    unlabelled = merged[merged["_merge"] == "left_only"]
    if len(unlabelled):
        print(f"\n[WARNING] {len(unlabelled)} sample(s) in the .fam have no "
              f"--pop-file label and will be treated as reference (set to "
              f"{args.missing_code:g}): "
              f"{unlabelled['IID'].tolist()[:10]}"
              f"{' ...' if len(unlabelled) > 10 else ''}")

    is_admix = (merged["POP"] == args.admix_label).fillna(False)
    raw = merged["PHENO"]

    MISSING_LIKE = {0, 9, -9}
    looks_missing = raw.isna() | raw.isin(MISSING_LIKE)

    print(f"\n{'='*70}")
    print("SOURCE: .fam column 6, by population")
    print(f"{'='*70}")
    tab = pd.crosstab(merged["POP"].fillna("(unlabelled)"), raw.fillna("NA"))
    print(tab.to_string())

    out = pd.DataFrame({
        "FID": merged["FID"], "IID": merged["IID"],
        "PHENO": raw.copy()})

    n_ref_overridden = int((~is_admix & ~looks_missing).sum())
    ref_detail = (merged[~is_admix & ~looks_missing]
                  .groupby("POP")["IID"].apply(list))
    out.loc[~is_admix, "PHENO"] = args.missing_code

    n_norm = int((is_admix & looks_missing).sum())
    out.loc[is_admix & looks_missing, "PHENO"] = args.missing_code

    kept = out.loc[is_admix & ~looks_missing, "PHENO"]

    print(f"\n{'='*70}")
    print("ACTIONS TAKEN")
    print(f"{'='*70}")
    if n_ref_overridden:
        print(f"[FIXED] {n_ref_overridden} reference-population sample(s) had a "
              f"real phenotype in the .fam and were set to "
              f"{args.missing_code:g} (they are not part of the phenotyped "
              f"study cohort):")
        for lab, ids in ref_detail.items():
            print(f"    {lab}: {len(ids)} -- e.g. {ids[:5]}"
                  f"{' ...' if len(ids) > 5 else ''}")
    else:
        print("[OK] No reference-population sample carried a stray phenotype.")
    if n_norm:
        print(f"[NORMALISED] {n_norm} admixed sample(s) had a missing-like code "
              f"(0/9/-9/blank) normalised to {args.missing_code:g}.")

    print(f"\n{'='*70}")
    print("RESULT")
    print(f"{'='*70}")
    if len(kept) == 0:
        print("[PROBLEM] After correction, NO admixed sample has a usable "
              "phenotype. Association testing is impossible on this data -- "
              "check that the .fam really contains case/control status.")
    else:
        vc = kept.value_counts().sort_index()
        print(f"{len(kept)} admixed sample(s) with a usable phenotype:")
        for val, n in vc.items():
            label = {1: "control", 2: "case"}.get(int(val), "value")
            print(f"    {int(val)} ({label}): {n}")
        if len(vc) < 2:
            print("\n[PROBLEM] Only ONE distinct phenotype value. Association "
                  "tests, AUC and lambda_GC are all undefined without "
                  "variation -- this is what produces nan results.")
        else:
            frac = vc.max() / vc.sum()
            print(f"\n[OK] Genuine phenotype variation present.")
            if frac > 0.75:
                print(f"[NOTE] Heavily imbalanced ({100*frac:.1f}% in the "
                      f"largest class). Valid, but expect wide confidence "
                      f"intervals, and interpret AUC with that in mind.")
    print(f"{int((out['PHENO'] == args.missing_code).sum())} sample(s) coded "
          f"missing ({args.missing_code:g}) in total.")

    out["PHENO"] = out["PHENO"].astype(float).round().astype(int)
    out.to_csv(args.out, sep="\t", header=False, index=False)
    print(f"\nWrote corrected --pheno -> {args.out}")
    print(f"Use it with:  --pheno {args.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pop-file", required=True, help="Path to --pop-file (sample_ID, POP_LABEL)")
    p.add_argument("--pheno", required=False, default=None,
                  help="Path to an existing --pheno file to CHECK. Optional "
                       "when --from-fam is used to build one from scratch.")
    p.add_argument("--fam", default=None,
                  help="Path to the PLINK .fam file. Column 6 is the "
                       "phenotype; used as the authoritative source with "
                       "--from-fam.")
    p.add_argument("--from-fam", action="store_true",
                  help="BUILD a corrected --pheno from --fam's column 6 "
                       "rather than checking an existing --pheno. Reference-"
                       "population samples (per --pop-file) are forced to "
                       "--missing-code, and any non-standard missing codes "
                       "in the .fam (0, 9) are normalised to --missing-code. "
                       "Requires --fam and --out.")
    p.add_argument("--admix-label", default="ADMIX",
                  help="The population label identifying admixed study-cohort "
                       "samples (default: ADMIX, matching JasMAP4's own default)")
    p.add_argument("--missing-code", type=float, default=-9,
                  help="The missing-phenotype sentinel value (default: -9, "
                       "PLINK's convention)")
    p.add_argument("--fix", action="store_true",
                  help="Write a corrected --pheno file: forces every "
                       "reference-population sample's phenotype to "
                       "--missing-code, leaving admixed samples untouched.")
    p.add_argument("--out", default=None,
                  help="Output path for the corrected file (required with --fix)")
    args = p.parse_args()

    if args.from_fam:
        build_pheno_from_fam(args)
        return

    if not args.pheno:
        print("ERROR: --pheno is required unless --from-fam is given.",
              file=sys.stderr)
        sys.exit(1)

    pop_df = load_pop_file(args.pop_file)
    pheno_df, has_ids = load_pheno_file(args.pheno)

    unique_labels = sorted(pop_df["POP"].unique().tolist())
    if args.admix_label not in unique_labels:
        print(f"ERROR: --admix-label {args.admix_label!r} does not match any label "
              f"in --pop-file. Labels found: {unique_labels}", file=sys.stderr)
        sys.exit(1)

    n_admix_pop = int((pop_df["POP"] == args.admix_label).sum())
    n_ref_pop = len(pop_df) - n_admix_pop
    print(f"--pop-file: {len(pop_df)} samples total "
          f"({n_admix_pop} labelled {args.admix_label!r}, {n_ref_pop} reference)")
    print(f"--pheno:    {len(pheno_df)} samples total "
          f"({'FID IID PHENO format' if has_ids else 'single-column format, no IDs'})")

    if not has_ids:
        n_missing = int((pheno_df["PHENO"] == args.missing_code).sum())
        n_defined = len(pheno_df) - n_missing
        print(f"\nNo sample IDs in --pheno (single-column format) -- cannot cross-check "
              f"against --pop-file by ID. Numeric breakdown only:")
        print(f"  {n_missing} sample(s) coded missing ({args.missing_code})")
        print(f"  {n_defined} sample(s) have a defined phenotype value")
        if n_defined != n_admix_pop:
            print(f"\n[WARNING] {n_defined} samples have a defined phenotype, but "
                 f"--pop-file lists {n_admix_pop} as {args.admix_label!r} -- these "
                 f"numbers should match. Cannot identify WHICH samples differ "
                 f"without IDs in --pheno; consider switching to the FID IID PHENO "
                 f"format so this can be checked precisely.")
        else:
            print(f"\nCounts match ({n_defined} defined == {n_admix_pop} labelled "
                 f"{args.admix_label!r}) -- looks consistent, though sample ORDER "
                 f"must also match the genotype file exactly for this format.")
        return

    merged = pop_df.merge(pheno_df, left_on="ID", right_on="IID", how="outer", indicator=True)

    not_in_pheno = merged[merged["_merge"] == "left_only"]
    not_in_pop = merged[merged["_merge"] == "right_only"]
    if len(not_in_pheno):
        print(f"\n[WARNING] {len(not_in_pheno)} sample(s) in --pop-file are missing "
             f"from --pheno entirely: {not_in_pheno['ID'].tolist()[:10]}"
             f"{' ...' if len(not_in_pheno) > 10 else ''}")
    if len(not_in_pop):
        print(f"\n[WARNING] {len(not_in_pop)} sample(s) in --pheno are missing "
             f"from --pop-file entirely: {not_in_pop['IID'].tolist()[:10]}"
             f"{' ...' if len(not_in_pop) > 10 else ''}")

    matched = merged[merged["_merge"] == "both"].copy()
    is_admix = matched["POP"] == args.admix_label
    is_missing_pheno = matched["PHENO"] == args.missing_code

    # The two failure directions that matter
    admix_but_missing = matched[is_admix & is_missing_pheno]
    ref_but_defined = matched[~is_admix & ~is_missing_pheno]

    print(f"\n{'='*70}")
    print("CROSS-CHECK RESULTS")
    print(f"{'='*70}")
    print(f"Admixed ({args.admix_label!r}) samples with a DEFINED phenotype (correct): "
         f"{int((is_admix & ~is_missing_pheno).sum())}")
    print(f"Reference samples correctly coded missing ({args.missing_code}): "
         f"{int((~is_admix & is_missing_pheno).sum())}")

    problems = 0
    if len(admix_but_missing):
        problems += len(admix_but_missing)
        print(f"\n[PROBLEM] {len(admix_but_missing)} sample(s) labelled "
             f"{args.admix_label!r} but coded MISSING in --pheno (should have a "
             f"real phenotype value):")
        print(f"  {admix_but_missing['IID'].tolist()[:20]}"
             f"{' ...' if len(admix_but_missing) > 20 else ''}")

    if len(ref_but_defined):
        problems += len(ref_but_defined)
        print(f"\n[PROBLEM] {len(ref_but_defined)} reference-population sample(s) "
             f"have a DEFINED phenotype in --pheno (should be coded "
             f"{args.missing_code} -- this is the exact issue that let reference "
             f"populations silently leak into an analysis run without --pop-file):")
        by_pop = ref_but_defined.groupby("POP")["IID"].apply(list)
        for pop_label, iids in by_pop.items():
            print(f"  {pop_label}: {len(iids)} sample(s) -- e.g. {iids[:5]}"
                 f"{' ...' if len(iids) > 5 else ''}")

    print(f"\n{'='*70}")
    if problems == 0:
        if len(not_in_pheno) or len(not_in_pop):
            print("RESULT: No CONTRADICTORY codings found. Note the ID-mismatch "
                 "warning(s) above though -- samples entirely absent from one file "
                 "are treated as missing-phenotype by JasMAP4 (a valid encoding, "
                 "equivalent to -9), but double check that absence is intentional "
                 "and not a typo'd/dropped sample ID.")
        else:
            print("RESULT: No problems found -- --pheno's coding is fully consistent "
                 "with --pop-file's admix/reference labels.")
    else:
        print(f"RESULT: {problems} sample(s) have inconsistent phenotype coding. "
             f"Fix these before running JasMAP4 without --pop-file (or always pass "
             f"--pop-file, which overrides phenotype validity and works correctly "
             f"regardless of this file's own coding).")
    print(f"{'='*70}")

    if args.fix:
        if not args.out:
            print("\nERROR: --fix requires --out <path>", file=sys.stderr)
            sys.exit(1)
        fixed = pheno_df.copy()
        ref_ids = set(pop_df.loc[pop_df["POP"] != args.admix_label, "ID"])
        fix_mask = fixed["IID"].isin(ref_ids)
        n_fixed = int((fix_mask & (fixed["PHENO"] != args.missing_code)).sum())
        fixed.loc[fix_mask, "PHENO"] = args.missing_code
        fixed["PHENO"] = fixed["PHENO"].astype(int)
        fixed.to_csv(args.out, sep="\t", header=False, index=False)
        print(f"\nWrote corrected --pheno to {args.out} "
             f"({n_fixed} reference sample(s) changed to {args.missing_code}, "
             f"admixed samples left untouched).")


if __name__ == "__main__":
    main()
