# Legacy Python 2 reference files

These three files are **not part of the running JasMAP4 pipeline**. They are
kept only as provenance/reference copies of the original scripts that later
modules in `jasmap/` were ported or rewritten from. None of them are
imported anywhere in the codebase (verified: no `import` statement anywhere
in the project references them), and none of them are valid Python 3 --
running `python3 -m py_compile` on any of them fails with a SyntaxError or
TabError, which is expected and harmless since nothing ever executes them.

| File | Superseded by |
|------|----------------|
| `input.py` | Not needed -- the functionality it provided (ldsc-style ID-list/LD-score machinery) was never used by JasMAP2/JasMAP4 in the first place; kept only for historical reference to the original ldsc fork `jasmap/parse.py` was derived from. |
| `plot_pca_orig.py` | `jasmap/plotting/plots.py :: pca_scatter()` -- the Python 3 rewrite used throughout the pipeline (LAI global-ancestry PCA, genotype PCA, GIFT field PCA). |
| `plot_karyogram_orig.py` | `jasmap/plotting/plots.py :: karyogram_from_lanc()` -- the Python 3 rewrite with bundled hg19 centromere coordinates, no external `centromeres_hg19.bed` file or `brewer2mpl` dependency required. |

They were moved out of the `jasmap/` package directory (where they used to
sit as `*_orig.py` files) specifically so that:
  - they are never accidentally importable as part of the `jasmap` package
  - a project-wide `python3 -m py_compile` / `compileall` sweep does not
    report false-positive errors from files that were never meant to run
    under Python 3 in the first place

Safe to delete entirely if you don't need the historical reference.
