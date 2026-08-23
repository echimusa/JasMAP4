"""
Genomic Informational Field Theory (GIFT) for Genotype-Phenotype Association
=============================================================================
Input  : PLINK binary format  (.bed / .bim / .fam)
Output : Phenotype predictions, SNP importance scores, field potential energy

Dependencies
------------
  pip install numpy scipy scikit-learn pandas

Usage
-----
  # CLI
  python gift_genomics.py --bfile data/mydata --phenotype 1 --out results/

  # Python API
  from gift_genomics import PlinkReader, GenomicInformationalFieldTheory
  G, samples, snps = PlinkReader("data/mydata").load()
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# 1. PLINK BINARY FILE READER
# ─────────────────────────────────────────────────────────────────────────────

class PlinkReader:
    """
    Read PLINK binary format (.bed / .bim / .fam) into a NumPy dosage matrix.

    PLINK .bed format
    -----------------
    Byte 0: magic 0x6c
    Byte 1: magic 0x1b
    Byte 2: mode byte  0x01 = SNP-major (standard), 0x00 = individual-major
    Bytes 3+: packed genotype data

    SNP-major layout (standard):
      Each SNP occupies ceil(n_samples / 4) bytes.
      Each byte holds 4 genotypes, 2 bits each, LSB-first:
        00 -> homozygous A1/A1  (dosage 0)
        01 -> missing           (dosage NaN)
        10 -> heterozygous      (dosage 1)
        11 -> homozygous A2/A2  (dosage 2)

    Parameters
    ----------
    bfile      : path prefix (without extension)
    pheno_col  : 1-based index of phenotype column in .fam (5=sex, 6=pheno)
                 or a path to an external phenotype file (space/tab delimited,
                 columns: FID IID PHENO1 [PHENO2 ...])
    pheno_name : column name if external phenotype file is provided
    missing_geno_strategy : 'mean' | 'zero' | 'drop'
                 How to handle missing genotypes (coded NaN after reading).
    """

    MAGIC = bytes([0x6c, 0x1b])
    GENO_MAP = np.array([0, np.nan, 1, 2], dtype=np.float32)  # 2-bit -> dosage

    def __init__(self, bfile, pheno_col=6,
                 pheno_name=None, missing_geno_strategy="mean"):
        self.bfile    = Path(bfile)
        self.bed_path = self.bfile.with_suffix(".bed")
        self.bim_path = self.bfile.with_suffix(".bim")
        self.fam_path = self.bfile.with_suffix(".fam")
        self.pheno_col             = pheno_col
        self.pheno_name            = pheno_name
        self.missing_geno_strategy = missing_geno_strategy

        for p in [self.bed_path, self.bim_path, self.fam_path]:
            if not p.exists():
                raise FileNotFoundError(f"PLINK file not found: {p}")

    # ── .fam reader ──────────────────────────────────────────────────────────

    def read_fam(self):
        """
        Returns DataFrame with columns:
          FID, IID, PAT, MAT, SEX, PHENO
        Missing phenotype coded as -9 or 0 in PLINK convention -> mapped to NaN.
        """
        fam = pd.read_csv(
            self.fam_path, sep=r"\s+", header=None,
            names=["FID", "IID", "PAT", "MAT", "SEX", "PHENO"],
            dtype={"FID": str, "IID": str}
        )
        fam["PHENO"] = pd.to_numeric(fam["PHENO"], errors="coerce")
        fam.loc[fam["PHENO"].isin([-9, 0]), "PHENO"] = np.nan
        return fam

    # ── .bim reader ──────────────────────────────────────────────────────────

    def read_bim(self):
        """
        Returns DataFrame with columns:
          CHR, SNP, CM, BP, A1, A2
        """
        bim = pd.read_csv(
            self.bim_path, sep=r"\s+", header=None,
            names=["CHR", "SNP", "CM", "BP", "A1", "A2"],
            dtype={"CHR": str, "SNP": str, "A1": str, "A2": str}
        )
        return bim

    # ── .bed reader ──────────────────────────────────────────────────────────

    def read_bed(self, n_samples, n_snps):
        """
        Parse SNP-major .bed file into float32 dosage matrix (n_samples × n_snps).
        Uses vectorised bit-unpacking — no Python loops over SNPs or samples.
        """
        raw = np.fromfile(self.bed_path, dtype=np.uint8)

        # Validate magic bytes and mode
        if tuple(raw[:2]) != (0x6c, 0x1b):
            raise ValueError("Invalid .bed magic bytes - not a PLINK binary file.")
        if raw[2] not in (0x01,):
            raise ValueError(
                f"Unsupported .bed mode byte: 0x{raw[2]:02x}. "
                "Only SNP-major (0x01) is supported."
            )

        data = raw[3:]                              # skip 3-byte header
        bytes_per_snp = (n_samples + 3) // 4       # ceil(n/4)
        expected = bytes_per_snp * n_snps
        if len(data) != expected:
            raise ValueError(
                f".bed data length mismatch: got {len(data)} bytes, "
                f"expected {expected} ({n_snps} SNPs x {bytes_per_snp} bytes/SNP)."
            )

        # Reshape to (n_snps, bytes_per_snp)
        data = data.reshape(n_snps, bytes_per_snp)

        # Unpack 2-bit genotypes: each byte -> 4 genotypes
        # bits[:, :, k] = (byte >> 2k) & 0x03  for k=0,1,2,3
        shifts   = np.array([0, 2, 4, 6], dtype=np.uint8)
        unpacked = (data[:, :, None] >> shifts) & 0x03  # (n_snps, bytes/snp, 4)

        # Flatten sample axis and trim padding
        unpacked = unpacked.reshape(n_snps, -1)[:, :n_samples]  # (n_snps, n_samples)

        # Map 2-bit codes -> dosage using lookup table
        G = self.GENO_MAP[unpacked].T               # (n_samples, n_snps)
        return G

    # ── missing genotype imputation ───────────────────────────────────────────

    def _handle_missing(self, G):
        """Impute or drop missing genotype calls (NaN entries)."""
        if self.missing_geno_strategy == "mean":
            col_means = np.nanmean(G, axis=0)
            nan_mask  = np.isnan(G)
            G[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        elif self.missing_geno_strategy == "zero":
            G = np.nan_to_num(G, nan=0.0)
        elif self.missing_geno_strategy == "drop":
            snp_miss   = np.isnan(G).mean(axis=0)
            keep_snps  = snp_miss < 0.05            # drop SNPs with >5% missingness
            samp_miss  = np.isnan(G).mean(axis=1)
            keep_samps = samp_miss < 0.10           # drop samples >10% missing
            G = G[np.ix_(keep_samps, keep_snps)]
            return G, keep_snps, keep_samps
        else:
            raise ValueError(f"Unknown missing_geno_strategy: {self.missing_geno_strategy}")
        return G, np.ones(G.shape[1], dtype=bool), np.ones(G.shape[0], dtype=bool)

    # ── external phenotype file ───────────────────────────────────────────────

    def _load_external_pheno(self, fam, pheno_path):
        """
        Merge an external phenotype file (FID IID PHENO ...) onto .fam order.
        Missing coded as -9 -> NaN.
        """
        ext = pd.read_csv(pheno_path, sep=r"\s+",
                          dtype={"FID": str, "IID": str})
        ext.replace(-9, np.nan, inplace=True)
        col = self.pheno_name or ext.columns[2]
        merged = fam[["FID","IID"]].merge(
            ext[["FID","IID", col]], on=["FID","IID"], how="left"
        )
        return merged[col].values.astype(float)

    # ── public load method ────────────────────────────────────────────────────

    def load(self, pheno_file=None):
        """
        Load .bed/.bim/.fam and return:
            G       : np.ndarray float32  (n_samples, n_snps) dosage matrix
            samples : pd.DataFrame        .fam rows (filtered if drop strategy)
            snps    : pd.DataFrame        .bim rows (filtered if drop strategy)
            y       : np.ndarray float64  phenotype vector (NaN for missing)

        Parameters
        ----------
        pheno_file : optional path to external phenotype file
        """
        t0 = time.perf_counter()
        print("[PLINK] Reading .fam ...", end=" ", flush=True)
        fam = self.read_fam()
        print(f"{len(fam)} samples")

        print("[PLINK] Reading .bim ...", end=" ", flush=True)
        bim = self.read_bim()
        print(f"{len(bim)} SNPs")

        print("[PLINK] Parsing .bed ...", end=" ", flush=True)
        G = self.read_bed(len(fam), len(bim))
        print(f"done  ({time.perf_counter()-t0:.2f}s)")

        miss_rate = np.isnan(G).mean()
        print(f"[PLINK] Overall missingness: {miss_rate*100:.2f}%")

        print(f"[PLINK] Imputing missing genotypes (strategy='{self.missing_geno_strategy}') ...")
        G, keep_snps, keep_samps = self._handle_missing(G)
        bim = bim.iloc[keep_snps].reset_index(drop=True)
        fam = fam.iloc[keep_samps].reset_index(drop=True)

        # Phenotype
        if pheno_file is not None:
            y = self._load_external_pheno(fam, pheno_file)
        elif isinstance(self.pheno_col, int):
            y = fam["PHENO"].values.astype(float)
        else:
            raise ValueError("Provide pheno_col (int) or pheno_file (path).")

        n_miss_y = np.isnan(y).sum()
        print(f"[PLINK] Phenotype: {(~np.isnan(y)).sum()} valid, {n_miss_y} missing\n")

        print(f"[PLINK] Final matrix: {G.shape[0]} samples x {G.shape[1]} SNPs")
        return G, fam, bim, y


# ─────────────────────────────────────────────────────────────────────────────
# 2. GIFT CORE
# ─────────────────────────────────────────────────────────────────────────────

class GenomicInformationalFieldTheory:
    """
    GIFT Model
    ----------
    Treats each SNP as a node in an information field. The field state vector
    Φ ∈ R^n_snps encodes allelic dosage information. Interactions propagate
    through a field propagator K derived from the mutual information structure
    of the genotype matrix.

    Phenotype prediction:
        ŷ = w^T · K · Φ_encoded + bias

    Parameters
    ----------
    n_components    : PCA components for field compression
    lambda_reg      : L2 regularisation on field weights
    field_temp      : Temperature parameter for field encoding
    propagator_tau  : Diffusion time for field propagator exp(-tau * L)
    mi_chunk_size   : SNPs per block in chunked MI computation
    verbose         : Print progress messages
    """

    def __init__(self, n_components=20, lambda_reg=1e-3,
                 field_temp=1.0, propagator_tau=0.5,
                 mi_chunk_size=512, verbose=True):
        self.n_components   = n_components
        self.lambda_reg     = lambda_reg
        self.field_temp     = field_temp
        self.propagator_tau = propagator_tau
        self.mi_chunk_size  = mi_chunk_size
        self.verbose        = verbose
        self.scaler         = StandardScaler()
        self.pca            = PCA(n_components=n_components)
        self.weights_       = None
        self.bias_          = None
        self.propagator_    = None
        self.mi_matrix_     = None

    # ── 2a. Vectorised MI with chunking ───────────────────────────────────────

    def _mutual_information(self, G):
        """
        Vectorised pairwise MI for dosage matrices (values in {0,1,2}).
        Uses packed integer trick + NumPy broadcasting. Chunked to bound RAM.
        Returns symmetric MI matrix (n_snps, n_snps).
        """
        n, p  = G.shape
        G_int = G.astype(np.int8)

        if p <= self.mi_chunk_size:
            return self._mi_block(G_int, np.arange(p), np.arange(p))

        MI     = np.zeros((p, p), dtype=np.float64)
        chunks = list(range(0, p, self.mi_chunk_size))
        total  = sum(1 for i in range(len(chunks)) for j in range(i, len(chunks)))
        done   = 0
        for ci, i0 in enumerate(chunks):
            i1   = min(i0 + self.mi_chunk_size, p)
            rows = np.arange(i0, i1)
            for cj in range(ci, len(chunks)):
                j0   = chunks[cj]
                j1   = min(j0 + self.mi_chunk_size, p)
                cols = np.arange(j0, j1)
                block = self._mi_block(G_int, rows, cols)
                MI[np.ix_(rows, cols)] = block
                if i0 != j0:
                    MI[np.ix_(cols, rows)] = block.T
                done += 1
                if self.verbose:
                    print(f"\r[GIFT]   MI chunks: {done}/{total}", end="", flush=True)
        if self.verbose:
            print()
        return MI

    @staticmethod
    def _mi_block(G_int, rows, cols):
        n      = G_int.shape[0]
        nr, nc = len(rows), len(cols)
        Gr     = G_int[:, rows]
        Gc     = G_int[:, cols]
        idx    = Gr[:, :, None].astype(np.int16) * 3 + Gc[:, None, :]  # (n,nr,nc)
        joint_counts = np.zeros((nr, nc, 9), dtype=np.float64)
        for k in range(9):
            joint_counts[:, :, k] = (idx == k).sum(axis=0)
        joint  = np.clip(joint_counts / n, 1e-12, None)
        p_i    = joint.reshape(nr, nc, 3, 3).sum(axis=3)
        p_j    = joint.reshape(nr, nc, 3, 3).sum(axis=2)
        outer  = np.clip((p_i[:,:,:,None] * p_j[:,:,None,:]).reshape(nr,nc,9), 1e-12, None)
        return (joint * np.log(joint / outer)).sum(axis=2)

    # ── 2b. Field propagator ──────────────────────────────────────────────────

    def _build_propagator(self, MI):
        threshold  = np.percentile(MI[MI > 0], 75)
        A          = (MI > threshold).astype(float) * MI
        deg        = A.sum(axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(deg + 1e-12))
        L_norm     = np.eye(len(A)) - D_inv_sqrt @ A @ D_inv_sqrt
        return expm(-self.propagator_tau * L_norm)

    # ── 2c. Field encoding ────────────────────────────────────────────────────

    def _encode_field(self, G):
        G_scaled = self.scaler.transform(G)
        Phi      = np.exp(G_scaled / self.field_temp)
        return Phi / (Phi.sum(axis=1, keepdims=True) + 1e-12)

    def _propagate(self, Phi):
        return Phi @ self.propagator_.T

    # ── 2d. Fit ───────────────────────────────────────────────────────────────

    def fit(self, G, y):
        """Fit GIFT on dosage matrix G and phenotype vector y (NaN rows dropped)."""
        valid = ~np.isnan(y)
        G, y  = G[valid], y[valid]

        if self.verbose:
            print(f"[GIFT] Computing MI matrix ({G.shape[1]} SNPs, chunk={self.mi_chunk_size}) ...")
        self.mi_matrix_  = self._mutual_information(G)

        if self.verbose: print("[GIFT] Building field propagator ...")
        self.propagator_ = self._build_propagator(self.mi_matrix_)

        if self.verbose: print("[GIFT] Encoding and propagating genotype field ...")
        self.scaler.fit(G)
        Phi_prop = self._propagate(self._encode_field(G))

        if self.verbose: print("[GIFT] Compressing field via PCA ...")
        Z = self.pca.fit_transform(Phi_prop)

        if self.verbose: print("[GIFT] Fitting ridge regression ...")
        n, d = Z.shape
        A = Z.T @ Z + self.lambda_reg * np.eye(d)
        self.weights_ = np.linalg.solve(A, Z.T @ y)
        self.bias_    = np.mean(y) - Z.mean(axis=0) @ self.weights_

        if self.verbose: print("[GIFT] Training complete.\n")
        return self

    # ── 2e. Predict ───────────────────────────────────────────────────────────

    def predict(self, G):
        Z = self.pca.transform(self._propagate(self._encode_field(G)))
        return Z @ self.weights_ + self.bias_

    # ── 2f. SNP importance ────────────────────────────────────────────────────

    def snp_importance(self, G):
        V           = self.pca.components_.T            # (n_snps, n_components)
        K_col_norms = np.linalg.norm(self.propagator_, axis=0)
        scores      = K_col_norms * np.abs(V @ self.weights_)
        return scores / scores.max()

    # ── 2g. Field potential energy ────────────────────────────────────────────

    def field_potential(self, G):
        Phi_prop = np.clip(self._propagate(self._encode_field(G)), 1e-10, 1.0)
        return -np.sum(Phi_prop * np.log(Phi_prop), axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. EVALUATION & OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, G_tr, y_tr, G_te, y_te):
    """Compute and print train/test metrics, returning test predictions."""
    def _metrics(y, yp):
        r    = np.corrcoef(y, yp)[0, 1]
        rmse = np.sqrt(np.mean((y - yp)**2))
        ysc  = (yp - yp.min()) / (yp.ptp() + 1e-12)
        auc  = roc_auc_score((y > np.median(y)).astype(int), ysc)
        return r, rmse, auc

    tr = _metrics(y_tr[~np.isnan(y_tr)], model.predict(G_tr[~np.isnan(y_tr)]))
    te = _metrics(y_te[~np.isnan(y_te)], model.predict(G_te[~np.isnan(y_te)]))

    print("=" * 54)
    print(f"{'Metric':<20} {'Train':>10} {'Test':>10}")
    print("-" * 54)
    for name, a, b in zip(["Pearson r", "RMSE", "AUC-ROC"], tr, te):
        print(f"{name:<20} {a:>10.4f} {b:>10.4f}")
    print("=" * 54)
    return model.predict(G_te)


def save_results(out_dir, snps_df, scores, y_pred, samples_df, y_true):
    """Write SNP importance and predictions to TSV files."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # SNP importance
    snp_out = snps_df.copy()
    snp_out["GIFT_SCORE"] = scores
    snp_out = snp_out.sort_values("GIFT_SCORE", ascending=False)
    snp_out.to_csv(out / "snp_importance.tsv", sep="\t", index=False)
    print(f"[OUT] SNP importance -> {out / 'snp_importance.tsv'}")

    # Predictions
    pred_out = samples_df[["FID", "IID"]].copy()
    pred_out["Y_TRUE"] = y_true
    pred_out["Y_PRED"] = y_pred
    pred_out.to_csv(out / "predictions.tsv", sep="\t", index=False)
    print(f"[OUT] Predictions    -> {out / 'predictions.tsv'}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="GIFT: Genomic Informational Field Theory (PLINK input)"
    )
    p.add_argument("--bfile",       required=True,
                   help="PLINK binary file prefix (no extension)")
    p.add_argument("--pheno-file",  default=None,
                   help="External phenotype file (FID IID PHENO ...)")
    p.add_argument("--pheno-col",   type=int, default=6,
                   help="1-based .fam phenotype column (default: 6)")
    p.add_argument("--pheno-name",  default=None,
                   help="Column name in external phenotype file")
    p.add_argument("--missing",     default="mean",
                   choices=["mean","zero","drop"],
                   help="Missing genotype strategy (default: mean)")
    p.add_argument("--out",         default="gift_results/",
                   help="Output directory (default: gift_results/)")
    p.add_argument("--n-components",type=int,   default=20)
    p.add_argument("--lambda-reg",  type=float, default=1e-3)
    p.add_argument("--field-temp",  type=float, default=1.0)
    p.add_argument("--propagator-tau", type=float, default=0.5)
    p.add_argument("--mi-chunk",    type=int,   default=512,
                   help="SNPs per MI chunk (tune to RAM, default: 512)")
    p.add_argument("--test-frac",   type=float, default=0.2,
                   help="Fraction of samples held out for testing")
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    if args is None:
        args = parse_args()

    print("+======================================================+")
    print("|  Genomic Informational Field Theory (GIFT)           |")
    print("|  PLINK .bed/.bim/.fam input                          |")
    print("+======================================================+\n")

    # ── Load PLINK data ───────────────────────────────────────────────────────
    reader = PlinkReader(
        bfile=args.bfile,
        pheno_col=args.pheno_col,
        pheno_name=args.pheno_name,
        missing_geno_strategy=args.missing,
    )
    G, fam, bim, y = reader.load(pheno_file=args.pheno_file)

    # ── Train / test split ────────────────────────────────────────────────────
    rng   = np.random.default_rng(args.seed)
    n     = G.shape[0]
    idx   = rng.permutation(n)
    split = int(n * (1 - args.test_frac))
    tr, te = idx[:split], idx[split:]

    G_tr, G_te = G[tr], G[te]
    y_tr, y_te = y[tr], y[te]
    fam_te     = fam.iloc[te].reset_index(drop=True)

    print(f"Split: {len(tr)} train / {len(te)} test samples\n")

    # ── Fit GIFT ──────────────────────────────────────────────────────────────
    model = GenomicInformationalFieldTheory(
        n_components   = args.n_components,
        lambda_reg     = args.lambda_reg,
        field_temp     = args.field_temp,
        propagator_tau = args.propagator_tau,
        mi_chunk_size  = args.mi_chunk,
    )
    model.fit(G_tr, y_tr)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("Performance Metrics")
    y_pred_te = evaluate(model, G_tr, y_tr, G_te, y_te)

    # ── SNP importance ────────────────────────────────────────────────────────
    scores = model.snp_importance(G_te[~np.isnan(y_te)])
    top10  = bim.iloc[np.argsort(scores)[::-1][:10]][["CHR","SNP","BP","A1","A2"]]
    print(f"\nTop 10 GIFT-ranked SNPs:\n{top10.to_string(index=False)}")

    # ── Field potential ───────────────────────────────────────────────────────
    U = model.field_potential(G_te[~np.isnan(y_te)])
    print(f"\nField Potential Energy - mean: {U.mean():.4f}  std: {U.std():.4f}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    save_results(args.out, bim, scores, y_pred_te, fam_te, y_te)
    print("\n[GIFT] Analysis complete.")


if __name__ == "__main__":
    main()
