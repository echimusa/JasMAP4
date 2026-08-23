"""
GIFT + Multi-way Admixture Mapping + PRS + Full Disease Risk Prediction
=======================================================================
PRS integration modes (all available, selectable via --prs-mode):
  • weighted_sum   – calibrated linear combination: w_prs·PRS + w_gift·GIFT + w_clin·CLINICAL
  • stacked        – ridge meta-learner on [p_PRS | p_GIFT | p_CLINICAL | Q] (default)
  • bayesian       – Bayesian posterior: P(D|PRS,GIFT,CLIN) via log-odds addition

Clinical inputs:  Vitals · Labs · Lifestyle · Symptoms (structured + free-text NLP)
Risk outputs:     Absolute 10-yr risk % · Relative risk vs population · Risk tier

Pipeline
--------
  1. PlinkReader            – .bed/.bim/.fam -> dosage matrix
  2. GlobalAncestryInfer    – NMF admixture, K=3 default, BIC optional
  3. LocalAncestryInfer     – Ensemble HMM + NMF local ancestry
  4. GIFTAdmixture          – Ancestry-stratified GIFT field scores
  5. PRSEngine              – Pre-computed or internally computed PRS
  6. ClinicalEncoder        – Vitals / labs / lifestyle / symptoms -> feature vec
  7. SymptomNLPEncoder      – Free-text symptom -> keyword feature vector
  8. DiseaseRiskPredictor   – All three integration modes, all three output formats
  9. AdmixtureMapping       – Per-window LAI-phenotype association scan
  10. Output                – TSV files + per-individual risk report

Dependencies
------------
  pip install numpy scipy scikit-learn pandas

Usage
-----
  python gift_risk.py \\
      --bfile       data/cohort          \\
      --prs-weights data/prs_weights.tsv \\
      --clinical    data/clinical.tsv    \\
      --disease     T2D                  \\
      --prs-mode    stacked              \\
      --out         results/
"""

import argparse
import re
import warnings
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.special import softmax, expit
from sklearn.decomposition import NMF, PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


###############################################################################
# 1.  PLINK READER
###############################################################################

class PlinkReader:
    GENO_MAP = np.array([0, np.nan, 1, 2], dtype=np.float32)

    def __init__(self, bfile, pheno_col=6, pheno_name=None,
                 missing_geno_strategy="mean"):
        self.bfile    = Path(bfile)
        self.bed      = self.bfile.with_suffix(".bed")
        self.bim_path = self.bfile.with_suffix(".bim")
        self.fam_path = self.bfile.with_suffix(".fam")
        self.pheno_col             = pheno_col
        self.pheno_name            = pheno_name
        self.missing_geno_strategy = missing_geno_strategy
        for p in [self.bed, self.bim_path, self.fam_path]:
            if not p.exists():
                raise FileNotFoundError(f"Not found: {p}")

    def read_fam(self):
        df = pd.read_csv(self.fam_path, sep=r"\s+", header=None,
                         names=["FID","IID","PAT","MAT","SEX","PHENO"],
                         dtype={"FID":str,"IID":str})
        df["PHENO"] = pd.to_numeric(df["PHENO"], errors="coerce")
        df.loc[df["PHENO"].isin([-9,0]),"PHENO"] = np.nan
        return df

    def read_bim(self):
        return pd.read_csv(self.bim_path, sep=r"\s+", header=None,
                           names=["CHR","SNP","CM","BP","A1","A2"],
                           dtype={"CHR":str,"SNP":str,"A1":str,"A2":str})

    def read_bed(self, ns, nsnp):
        raw = np.fromfile(self.bed, dtype=np.uint8)
        if tuple(raw[:2]) != (0x6c,0x1b): raise ValueError("Bad .bed magic.")
        if raw[2] != 0x01: raise ValueError("Only SNP-major .bed supported.")
        bps = (ns+3)//4
        data = raw[3:].reshape(nsnp, bps)
        sh = np.array([0,2,4,6], dtype=np.uint8)
        up = (data[:,:,None] >> sh) & 0x03
        return self.GENO_MAP[up.reshape(nsnp,-1)[:,:ns]].T

    def _impute(self, G):
        if self.missing_geno_strategy == "mean":
            cm = np.nanmean(G,0); nm = np.isnan(G)
            G[nm] = np.take(cm, np.where(nm)[1])
            return G, np.ones(G.shape[1],bool), np.ones(G.shape[0],bool)
        if self.missing_geno_strategy == "zero":
            return np.nan_to_num(G,nan=0.), \
                   np.ones(G.shape[1],bool), np.ones(G.shape[0],bool)
        if self.missing_geno_strategy == "drop":
            ks=np.isnan(G).mean(0)<.05; ss=np.isnan(G).mean(1)<.10
            return G[np.ix_(ss,ks)], ks, ss
        raise ValueError(self.missing_geno_strategy)

    def load(self, pheno_file=None):
        t0 = time.perf_counter()
        print("[PLINK] .fam ...", end=" "); fam=self.read_fam(); print(len(fam))
        print("[PLINK] .bim ...", end=" "); bim=self.read_bim(); print(len(bim))
        print("[PLINK] .bed ...", end=" ")
        G = self.read_bed(len(fam), len(bim))
        print(f"done ({time.perf_counter()-t0:.1f}s)")
        G, ks, ss = self._impute(G)
        bim = bim.iloc[ks].reset_index(drop=True)
        fam = fam.iloc[ss].reset_index(drop=True)
        if pheno_file:
            ext = pd.read_csv(pheno_file,sep=r"\s+",dtype={"FID":str,"IID":str})
            ext.replace(-9,np.nan,inplace=True)
            col = self.pheno_name or ext.columns[2]
            y = fam[["FID","IID"]].merge(ext[["FID","IID",col]],
                    on=["FID","IID"],how="left")[col].values.astype(float)
        else:
            y = fam["PHENO"].values.astype(float)
        print(f"[PLINK] {G.shape[0]} samples x {G.shape[1]} SNPs\n")
        return G, fam, bim, y


###############################################################################
# 2.  GLOBAL ANCESTRY  (NMF + BIC, default K=3)
###############################################################################

class GlobalAncestryInfer:
    def __init__(self, k=3, k_max=5, n_init=3, seed=42):
        self.k=k; self.k_max=k_max; self.n_init=n_init; self.seed=seed
        self.Q_=None; self.F_=None; self.k_=None

    def _nmf(self, G, k):
        best,berr=None,np.inf
        for s in range(self.n_init):
            m=NMF(k,init="nndsvda",max_iter=500,random_state=self.seed+s)
            Q=m.fit_transform(G)
            if m.reconstruction_err_<berr: best=(Q,m.components_); berr=m.reconstruction_err_
        return best,berr

    def fit(self, G):
        if self.k:
            (self.Q_,self.F_),_=self._nmf(G,self.k); self.k_=self.k
        else:
            best_b,bk,bQ,bF=np.inf,2,None,None
            for k in range(2,self.k_max+1):
                (Q,F),err=self._nmf(G,k); n,p=G.shape
                bic=n*p*np.log(err**2/(n*p)+1e-12)+k*(n+p)*np.log(n*p)
                print(f"[GLOBAL]   K={k} BIC={bic:.1f}")
                if bic<best_b: best_b,bk,bQ,bF=bic,k,Q,F
            self.k_,self.Q_,self.F_=bk,bQ,bF
        rs=self.Q_.sum(1,keepdims=True)+1e-12; self.Q_/=rs
        print(f"[GLOBAL] K={self.k_}, Q={self.Q_.shape}\n")
        return self

    @property
    def proportions(self): return self.Q_


###############################################################################
# 3.  LOCAL ANCESTRY  — Ensemble HMM + NMF
###############################################################################

class _HMMEngine:
    def __init__(self, k, ws, ss, tp, seed=42):
        self.k=k; self.ws=ws; self.ss=ss; self.tp=tp; self.seed=seed

    def _windows(self, n):
        return [np.arange(s,min(s+self.ws,n))
                for s in np.arange(0,n-self.ws+1,self.ss)]

    def _emit(self, G, Q, wins):
        n,K=G.shape[0],self.k; Gs=(G-G.mean(0))/(G.std(0)+1e-8)
        nc=min(K*2,G.shape[1]-1,n-1)
        pca=PCA(nc,random_state=self.seed); Zf=pca.fit_transform(Gs)
        cents=(Q.T@Zf)/(Q.sum(0)[:,None]+1e-12)
        em=np.zeros((len(wins),n,K),np.float32)
        for wi,idx in enumerate(wins):
            Gw=Gs[:,idx]; nc2=min(K*2,Gw.shape[1]-1,n-1)
            if nc2<1: em[wi]=Q; continue
            pw=PCA(nc2,random_state=self.seed); Zw=pw.fit_transform(Gw)
            try: cw=pw.transform(pca.inverse_transform(cents)[:,idx])
            except Exception: cw=cents[:,:nc2]
            d=np.sum((Zw[:,None,:]-cw[None,:,:nc2])**2,2)
            em[wi]=softmax(-d/(d.std()+1e-8),1)
        return em

    @staticmethod
    def _viterbi(em, tr, pr):
        T,K=em.shape; ltr=np.log(tr+1e-12)
        v=np.log(pr+1e-12)+np.log(em[0]+1e-12); bk=np.zeros((T,K),np.int16)
        for t in range(1,T):
            sc=v[:,None]+ltr; bp=sc.argmax(0); bk[t]=bp
            v=sc[bp,np.arange(K)]+np.log(em[t]+1e-12)
        st=np.empty(T,np.int16); st[-1]=v.argmax()
        for t in range(T-2,-1,-1): st[t]=bk[t+1,st[t+1]]
        return st

    def infer(self, G, Q):
        K,n=self.k,G.shape[0]; wins=self._windows(G.shape[1])
        em=self._emit(G,Q,wins)
        p=self.tp; tr=np.full((K,K),p/max(K-1,1)); np.fill_diagonal(tr,1-p)
        L=np.zeros((n,len(wins),K),np.float32)
        for i in range(n):
            st=self._viterbi(em[:,i,:],tr,Q[i])
            h=np.zeros((len(wins),K),np.float32)
            h[np.arange(len(wins)),st]=1.
            L[i]=.7*h+.3*em[:,i,:]
        return L,wins


class _NMFEngine:
    def __init__(self, k, ws, ss, seed=42):
        self.k=k; self.ws=ws; self.ss=ss; self.seed=seed

    def _windows(self, n):
        return [np.arange(s,min(s+self.ws,n))
                for s in np.arange(0,n-self.ws+1,self.ss)]

    @staticmethod
    def _align(Qr, Qw):
        K=Qr.shape[1]; used=set(); perm=[]
        for k in range(K):
            bj=max((j for j in range(K) if j not in used),
                   key=lambda j,kk=k: np.corrcoef(Qr[:,kk],Qw[:,j])[0,1])
            perm.append(bj); used.add(bj)
        return Qw[:,perm]

    def infer(self, G, Q):
        K=self.k; wins=self._windows(G.shape[1]); n=G.shape[0]
        L=np.zeros((n,len(wins),K),np.float32)
        for wi,idx in enumerate(wins):
            try:
                m=NMF(K,init="nndsvda",max_iter=300,random_state=self.seed)
                Qw=m.fit_transform(G[:,idx])
                rs=Qw.sum(1,keepdims=True)+1e-12; Qw/=rs
                Qw=self._align(Q,Qw)
            except Exception: Qw=Q.copy()
            L[:,wi,:]=Qw.astype(np.float32)
        return L,wins


class LocalAncestryInfer:
    def __init__(self, k=3, window_size=50, step_size=25,
                 transition_prob=.02, ensemble_alpha=.6, seed=42):
        self.k=k; self.ws=window_size; self.ss=step_size
        self.tp=transition_prob; self.alpha=ensemble_alpha; self.seed=seed
        self.local_ancestry_=None; self.window_snps_=None

    def fit(self, G, Q):
        print("[LOCAL] HMM engine ...")
        Lh,wins=_HMMEngine(self.k,self.ws,self.ss,self.tp,self.seed).infer(G,Q)
        print("[LOCAL] NMF engine ...")
        Ln,_=_NMFEngine(self.k,self.ws,self.ss,self.seed).infer(G,Q)
        L=self.alpha*Lh+(1-self.alpha)*Ln
        rs=L.sum(2,keepdims=True)+1e-12; L/=rs
        self.local_ancestry_=L; self.window_snps_=wins
        print(f"[LOCAL] Tensor {L.shape}  (alpha_HMM={self.alpha})\n")
        return self

    def snp_ancestry(self, n_snps):
        K,n=self.k,self.local_ancestry_.shape[0]
        out=np.zeros((n,n_snps,K),np.float32)
        for wi,idx in enumerate(self.window_snps_):
            out[:,idx,:]=self.local_ancestry_[:,wi:wi+1,:]
        return out


###############################################################################
# 4.  GIFT + ADMIXTURE FIELD
###############################################################################

class GIFTAdmixture:
    """Ancestry-stratified GIFT with all three coupling modes."""
    _MODES = ("stratified","covariate","interaction")

    def __init__(self, k=3, n_components=15, lambda_reg=1e-3,
                 field_temp=1., propagator_tau=.5,
                 mi_chunk=512, ancestry_mode="stratified", verbose=True):
        self.k=k; self.nc=n_components; self.lam=lambda_reg
        self.ft=field_temp; self.tau=propagator_tau
        self.chunk=mi_chunk; self.mode=ancestry_mode; self.verbose=verbose
        self._sc={}; self._pca={}; self._prop={}
        self._w=None; self._b=None; self._ns=None; self._models={}

    # ── MI vectorised ─────────────────────────────────────────────────────────
    @staticmethod
    def _mi_block(Gi, rows, cols):
        n=Gi.shape[0]; nr,nc=len(rows),len(cols)
        idx=(Gi[:,rows][:,:,None].astype(np.int16)*3+
             Gi[:,cols][:,None,:])
        jc=np.zeros((nr,nc,9),np.float64)
        for k in range(9): jc[:,:,k]=(idx==k).sum(0)
        jt=np.clip(jc/n,1e-12,None)
        pi=jt.reshape(nr,nc,3,3).sum(3); pj=jt.reshape(nr,nc,3,3).sum(2)
        outer=np.clip((pi[:,:,:,None]*pj[:,:,None,:]).reshape(nr,nc,9),1e-12,None)
        return (jt*np.log(jt/outer)).sum(2)

    def _mi(self, G, lbl=""):
        n,p=G.shape; Gi=np.clip(np.round(G),0,2).astype(np.int8)
        if p<=self.chunk: return self._mi_block(Gi,np.arange(p),np.arange(p))
        MI=np.zeros((p,p)); cks=list(range(0,p,self.chunk))
        tot=sum(1 for ci in range(len(cks)) for _ in range(ci,len(cks))); done=0
        for ci,i0 in enumerate(cks):
            rows=np.arange(i0,min(i0+self.chunk,p))
            for cj in range(ci,len(cks)):
                j0=cks[cj]; cols=np.arange(j0,min(j0+self.chunk,p))
                bl=self._mi_block(Gi,rows,cols)
                MI[np.ix_(rows,cols)]=bl
                if i0!=j0: MI[np.ix_(cols,rows)]=bl.T
                done+=1
                if self.verbose:
                    print(f"\r[GIFT]   {lbl} MI {done}/{tot}",end="",flush=True)
        if self.verbose: print()
        return MI

    def _build_prop(self, MI):
        thr=np.percentile(MI[MI>0],75) if MI.max()>0 else 0
        A=(MI>thr).astype(float)*MI; deg=A.sum(1)
        D=np.diag(1./np.sqrt(deg+1e-12)); L=np.eye(len(A))-D@A@D
        return expm(-self.tau*L)

    def _enc(self, G, key):
        Gs=self._sc[key].transform(G)
        Ph=np.exp(Gs/self.ft); return Ph/(Ph.sum(1,keepdims=True)+1e-12)

    def _prp(self, Phi, key): return Phi@self._prop[key].T

    def _feats(self, G, Q, SA, mode, fit=False):
        ns=self._ns; parts=[]
        if mode=="stratified":
            for k in range(self.k):
                Lk=SA[:,:ns,k] if SA is not None else np.ones_like(G)
                Gk=G*Lk; key=f"s{k}"
                if fit:
                    if self.verbose: print(f"[GIFT]   propagator k={k} ...")
                    self._sc[key]=StandardScaler().fit(Gk)
                    self._prop[key]=self._build_prop(self._mi(Gk,f"k={k}"))
                    Pp=self._prp(self._enc(Gk,key),key)
                    nc=min(self.nc,Pp.shape[1]-1,G.shape[0]-1)
                    self._pca[key]=PCA(nc).fit(Pp)
                Pp=self._prp(self._enc(Gk,key),key)
                parts.append(self._pca[key].transform(Pp))
        elif mode=="covariate":
            key="cov"
            if fit:
                self._sc[key]=StandardScaler().fit(G)
                self._prop[key]=self._build_prop(self._mi(G,"cov"))
                Pp=self._prp(self._enc(G,key),key)
                nc=min(self.nc,Pp.shape[1]-1,G.shape[0]-1)
                self._pca[key]=PCA(nc).fit(Pp)
            Pp=self._prp(self._enc(G,key),key)
            parts.append(self._pca[key].transform(Pp))
        elif mode=="interaction":
            key="int"
            if fit:
                self._sc[key]=StandardScaler().fit(G)
                self._prop[key]=self._build_prop(self._mi(G,"int"))
                Pp=self._prp(self._enc(G,key),key)
                nc=min(self.nc,Pp.shape[1]-1,G.shape[0]-1)
                self._pca[key]=PCA(nc).fit(Pp)
            Pp=self._prp(self._enc(G,key),key)
            Z=self._pca[key].transform(Pp); parts.append(Z)
            for k in range(self.k): parts.append(Z*Q[:,k:k+1])
        parts.append(Q)
        return np.hstack(parts)

    def _ridge(self, Z, y):
        n,d=Z.shape; A=Z.T@Z+self.lam*np.eye(d)
        w=np.linalg.solve(A,Z.T@y); b=y.mean()-Z.mean(0)@w
        return w,b

    def fit(self, G, y, Q, SA):
        v=~np.isnan(y); Gv,yv,Qv=G[v],y[v],Q[v]
        SAv=SA[v] if SA is not None else None; self._ns=G.shape[1]
        modes=self._MODES if self.mode=="all" else [self.mode]
        for mode in modes:
            if self.verbose: print(f"\n[GIFT] Mode: {mode}")
            sub=GIFTAdmixture(self.k,self.nc,self.lam,self.ft,self.tau,
                              self.chunk,mode,self.verbose)
            sub._ns=self._ns
            Z=sub._feats(Gv,Qv,SAv,mode,fit=True)
            sub._w,sub._b=self._ridge(Z,yv)
            self._models[mode]=sub
        if self.verbose: print("\n[GIFT] Done.\n")
        return self

    def gift_scores(self, G, Q, SA):
        modes=self._MODES if self.mode=="all" else [self.mode]
        return np.mean([self._score_mode(G,Q,SA,m) for m in modes],0)

    def _score_mode(self, G, Q, SA, mode):
        sub=self._models[mode]
        Z=sub._feats(G,Q,SA,mode,fit=False)
        return Z@sub._w+sub._b

    def snp_importance(self, G, Q, SA):
        modes=self._MODES if self.mode=="all" else [self.mode]
        sc=np.zeros(self._ns)
        for mode in modes:
            sub=self._models[mode]
            keys=([f"s{k}" for k in range(self.k)]
                  if mode=="stratified" else
                  ["cov" if mode=="covariate" else "int"])
            idx=0
            for key in keys:
                nc=sub._pca[key].n_components_; wk=sub._w[idx:idx+nc]; idx+=nc
                V=sub._pca[key].components_.T
                Kn=np.linalg.norm(sub._prop[key],axis=0)
                sc+=Kn*np.abs(V@wk)
        return sc/(sc.max()+1e-12)


###############################################################################
# 5.  PRS ENGINE
###############################################################################

class PRSEngine:
    """
    Compute / ingest Polygenic Risk Scores.
    Priority: pre-computed column > internal from weights file.
    """
    def __init__(self, weights_path=None, prs_col=None):
        self.weights_path = Path(weights_path) if weights_path else None
        self.prs_col      = prs_col
        self._mu=0.; self._sigma=1.
        self._betas=None; self._snp_idx=None

    def _load_weights(self, bim):
        w=pd.read_csv(self.weights_path,sep=r"\s+",dtype={"SNP":str,"A1":str})
        if not {"SNP","A1","BETA"}.issubset(w.columns):
            raise ValueError("PRS weights must have columns: SNP A1 BETA")
        w["BETA"]=pd.to_numeric(w["BETA"],errors="coerce")
        m=bim.reset_index().merge(w[["SNP","A1","BETA"]],on="SNP",how="inner")
        m.loc[m["A1_x"]!=m["A1_y"],"BETA"]*=-1
        self._snp_idx=m["index"].values; self._betas=m["BETA"].values.astype(np.float32)
        print(f"[PRS] {len(self._betas)}/{len(bim)} SNPs matched.")

    def get_prs(self, G, bim, clinical_df=None):
        if self.prs_col and clinical_df is not None and self.prs_col in clinical_df.columns:
            print(f"[PRS] Using pre-computed column '{self.prs_col}'.")
            return pd.to_numeric(clinical_df[self.prs_col],errors="coerce").values.astype(float)
        if self.weights_path:
            if self._betas is None: self._load_weights(bim)
            return (G[:,self._snp_idx]@self._betas).astype(float)
        # Fallback: random normal PRS for demo
        print("[PRS] No weights/column provided - using random PRS (demo).")
        return np.random.default_rng(0).normal(0,1,G.shape[0])

    def fit(self, prs):
        self._mu=np.nanmean(prs); self._sigma=np.nanstd(prs)+1e-8; return self

    def transform(self, prs):
        return (prs-self._mu)/self._sigma

    def fit_transform(self, prs):
        return self.fit(prs).transform(prs)


###############################################################################
# 6.  SYMPTOM FREE-TEXT NLP ENCODER
###############################################################################

# Disease-keyed symptom keyword banks
SYMPTOM_KEYWORDS = {
    "T2D": ["thirst","polyuria","fatigue","blurred vision","weight loss",
            "slow healing","tingling","numbness","frequent urination"],
    "CVD": ["chest pain","shortness of breath","palpitations","dizziness",
            "syncope","edema","angina","dyspnea","claudication","sweating"],
    "CKD": ["fatigue","swelling","foam urine","back pain","nausea",
            "itching","decreased urine","hypertension","anemia"],
    "GENERIC": ["fatigue","pain","nausea","weight loss","fever",
                "shortness of breath","swelling","dizziness","loss of appetite"],
}

class SymptomNLPEncoder:
    """
    Lightweight rule-based NLP encoder for free-text symptom descriptions.

    Each symptom field is lowercased and matched against a disease-specific
    keyword bank. Negation detection (no, not, without, denies) flips the flag.
    Output: binary feature vector of length = number of keywords.

    Parameters
    ----------
    disease    : disease key for symptom bank
    extra_keys : additional keywords to include beyond the disease bank
    """
    NEGATION_WORDS = {"no","not","without","denies","denying","absent",
                      "negative","never","none"}

    def __init__(self, disease="GENERIC", extra_keys=None):
        self.disease  = disease.upper()
        bank = SYMPTOM_KEYWORDS.get(self.disease, SYMPTOM_KEYWORDS["GENERIC"])
        self.keywords = bank + (extra_keys or [])

    def _has_negation(self, tokens, kw_tokens, match_pos):
        """Check if any negation word appears within a 4-token window before match."""
        window = tokens[max(0, match_pos-4):match_pos]
        return bool(set(window) & self.NEGATION_WORDS)

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single free-text string -> binary keyword vector."""
        vec    = np.zeros(len(self.keywords), dtype=np.float32)
        tokens = re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
        for ki, kw in enumerate(self.keywords):
            kw_toks = kw.lower().split()
            kw_str  = " ".join(kw_toks)
            txt_str = " ".join(tokens)
            pos = txt_str.find(kw_str)
            if pos == -1: continue
            # Approximate token position
            tok_pos = len(txt_str[:pos].split())
            if not self._has_negation(tokens, kw_toks, tok_pos):
                vec[ki] = 1.
        return vec

    def encode_series(self, texts) -> np.ndarray:
        """Encode an iterable of strings -> (n, n_keywords) array."""
        return np.vstack([self.encode_text(str(t) if pd.notna(t) else "")
                          for t in texts])

    @property
    def feature_names(self):
        return [f"SYM_{kw.replace(' ','_').upper()}" for kw in self.keywords]


###############################################################################
# 7.  CLINICAL ENCODER  (vitals + labs + lifestyle + structured symptoms)
###############################################################################

CLINICAL_SCHEMA = {
    # Vitals
    "AGE":           dict(type="continuous", range=(0,120),   impute="median"),
    "SEX":           dict(type="binary",     values=(0,1),    impute="mode"),
    "BMI":           dict(type="continuous", range=(10,70),   impute="median"),
    "SBP":           dict(type="continuous", range=(60,250),  impute="median"),
    "DBP":           dict(type="continuous", range=(40,150),  impute="median"),
    "HEART_RATE":    dict(type="continuous", range=(30,200),  impute="median"),
    # Labs
    "TOTAL_CHOL":    dict(type="continuous", range=(1,15),    impute="median"),
    "HDL_CHOL":      dict(type="continuous", range=(.2,5),    impute="median"),
    "LDL_CHOL":      dict(type="continuous", range=(.5,10),   impute="median"),
    "TRIGLYCERIDES": dict(type="continuous", range=(.2,30),   impute="median"),
    "GLUCOSE":       dict(type="continuous", range=(1,50),    impute="median"),
    "HBA1C":         dict(type="continuous", range=(2,20),    impute="median"),
    "CREATININE":    dict(type="continuous", range=(20,2000), impute="median"),
    "CRP":           dict(type="continuous", range=(0,500),   impute="median"),
    # Lifestyle
    "SMOKING":       dict(type="ordinal",    levels=3,        impute="mode"),
    "ALCOHOL_UNITS": dict(type="continuous", range=(0,200),   impute="median"),
    "EXERCISE_HRS":  dict(type="continuous", range=(0,40),    impute="median"),
    "DIET_SCORE":    dict(type="continuous", range=(0,10),    impute="median"),
    "SLEEP_HRS":     dict(type="continuous", range=(0,24),    impute="median"),
    # Structured symptoms / comorbidities
    "HYPERTENSION":  dict(type="binary", values=(0,1), impute="mode"),
    "DIABETES":      dict(type="binary", values=(0,1), impute="mode"),
    "CAD":           dict(type="binary", values=(0,1), impute="mode"),
    "FAMILY_HX_CVD": dict(type="binary", values=(0,1), impute="mode"),
    "CHEST_PAIN":    dict(type="binary", values=(0,1), impute="mode"),
    "DYSPNEA":       dict(type="binary", values=(0,1), impute="mode"),
    "FATIGUE":       dict(type="binary", values=(0,1), impute="mode"),
    "POLYURIA":      dict(type="binary", values=(0,1), impute="mode"),
}

DISEASE_PRIORS = {
    "T2D":  {"BMI":2.,"GLUCOSE":3.,"HBA1C":3.,"EXERCISE_HRS":1.5,
             "DIET_SCORE":1.5,"POLYURIA":2.,"DIABETES":2.},
    "CVD":  {"SBP":2.,"DBP":1.5,"TOTAL_CHOL":2.,"LDL_CHOL":2.,
             "SMOKING":2.,"HYPERTENSION":2.,"CAD":3.,
             "FAMILY_HX_CVD":2.,"CHEST_PAIN":2.,"DYSPNEA":1.5},
    "CKD":  {"CREATININE":3.,"DIABETES":2.,"HYPERTENSION":2.,
             "SBP":1.5,"BMI":1.3},
    "GENERIC":{},
}

class ClinicalEncoder:
    """
    Encode vitals, labs, lifestyle, structured symptoms + optional free-text
    symptoms into a standardised numeric matrix.
    """
    def __init__(self, disease="GENERIC", schema=None,
                 symptom_text_col=None):
        self.disease          = disease.upper()
        self.schema           = schema or CLINICAL_SCHEMA
        self.priors           = DISEASE_PRIORS.get(self.disease,{})
        self.symptom_text_col = symptom_text_col
        self._scaler          = StandardScaler()
        self._cols            = None
        self._impute_vals     = {}
        self._nlp             = SymptomNLPEncoder(disease=self.disease) \
                                if symptom_text_col else None

    def _prep(self, df, fit=False):
        X = df[[c for c in self.schema if c in df.columns]].copy()
        if fit: self._cols = list(X.columns)
        else:   X = X[self._cols] if self._cols else X
        for col in X.columns:
            spec=self.schema[col]
            if spec["type"]=="continuous":
                lo,hi=spec["range"]
                X[col]=pd.to_numeric(X[col],errors="coerce").clip(lo,hi)
            else:
                X[col]=pd.to_numeric(X[col],errors="coerce")
            if fit:
                self._impute_vals[col]=(X[col].median()
                    if spec["impute"]=="median"
                    else (X[col].mode().iloc[0] if not X[col].mode().empty else 0))
            X[col].fillna(self._impute_vals.get(col,0), inplace=True)
            if col in self.priors: X[col]*=self.priors[col]
        arr = X.values.astype(np.float32)
        if self._nlp and self.symptom_text_col in df.columns:
            nlp_arr = self._nlp.encode_series(df[self.symptom_text_col])
            arr     = np.hstack([arr, nlp_arr])
        return arr

    def fit(self, df):
        arr=self._prep(df,fit=True); self._scaler.fit(arr)
        n_feats=arr.shape[1]; nlp_n=len(self._nlp.keywords) if self._nlp else 0
        print(f"[CLINICAL] {len(self._cols)} structured + {nlp_n} NLP symptom features "
              f"= {n_feats} total  (disease={self.disease})")
        return self

    def transform(self, df):
        return self._scaler.transform(self._prep(df,fit=False))

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    @property
    def feature_names(self):
        struct = self._cols or []
        nlp    = self._nlp.feature_names if self._nlp else []
        return struct + nlp


###############################################################################
# 8.  DISEASE RISK PREDICTOR  (all three integration modes)
###############################################################################

POPULATION_BASELINE = {"T2D":.08,"CVD":.10,"CKD":.05,"GENERIC":.10}
RISK_TIERS = [(0.05,"Low"),(0.10,"Moderate"),(0.20,"High"),(np.inf,"Very High")]

def _tier(r):
    for th,lb in RISK_TIERS:
        if r<th: return lb
    return "Very High"

def _safe_proba(m, X):
    p=m.predict_proba(X); return p[:,1] if p.shape[1]==2 else p[:,0]

def _binarise(y):
    u=np.unique(y[~np.isnan(y)])
    return y.astype(int) if set(u).issubset({0,1}) else (y>np.nanmedian(y)).astype(int)


class _WeightedSumRisk:
    """
    Calibrated weighted-sum integration.

    Fits three independent logistic regressions, obtains probability
    outputs p_PRS, p_GIFT, p_CLINICAL, then learns optimal weights
    w = (w_prs, w_gift, w_clin) via constrained least-squares so that
    w_prs·p_PRS + w_gift·p_GIFT + w_clin·p_CLINICAL ≈ y_binary.
    Weights are softmax-normalised to sum to 1.
    """
    def __init__(self, seed=42):
        self.seed=seed
        self._m_prs =LogisticRegression(C=1.,max_iter=500,random_state=seed)
        self._m_gift=LogisticRegression(C=1.,max_iter=500,random_state=seed)
        self._m_clin=LogisticRegression(C=1.,max_iter=500,random_state=seed)
        self._w=np.array([1/3,1/3,1/3]); self.name="weighted_sum"

    def fit(self, prs, gift, Xc, Q, yb):
        self._m_prs.fit( prs.reshape(-1,1),  yb)
        self._m_gift.fit(gift.reshape(-1,1), yb)
        self._m_clin.fit(Xc,                 yb)
        pp=_safe_proba(self._m_prs,  prs.reshape(-1,1))
        pg=_safe_proba(self._m_gift, gift.reshape(-1,1))
        pc=_safe_proba(self._m_clin, Xc)
        P=np.column_stack([pp,pg,pc])           # (n,3)
        # Constrained least-squares: min ||Pw-yb||^2  s.t. w≥0, sum=1
        from scipy.optimize import minimize
        def obj(w): return np.mean((P@w-yb)**2)
        res=minimize(obj,self._w,method="SLSQP",
                     constraints={"type":"eq","fun":lambda w:w.sum()-1},
                     bounds=[(0,1)]*3)
        self._w=res.x if res.success else self._w
        print(f"[RISK-WS] Weights: PRS={self._w[0]:.3f}  "
              f"GIFT={self._w[1]:.3f}  CLINICAL={self._w[2]:.3f}")
        return self

    def predict_proba(self, prs, gift, Xc, Q):
        pp=_safe_proba(self._m_prs,  prs.reshape(-1,1))
        pg=_safe_proba(self._m_gift, gift.reshape(-1,1))
        pc=_safe_proba(self._m_clin, Xc)
        raw=self._w[0]*pp+self._w[1]*pg+self._w[2]*pc
        return raw, dict(prs=pp, gift=pg, clinical=pc)

    def component_weights(self):
        return dict(prs=float(self._w[0]),
                    gift=float(self._w[1]),
                    clinical=float(self._w[2]))


class _StackedRisk:
    """
    Ridge meta-learner stacking.
    Layer-1: independent logistic models -> probabilities.
    Layer-2: logistic ridge on [p_prs, p_gift, p_clin, Q].
    """
    def __init__(self, seed=42):
        self.seed=seed
        self._m_prs =LogisticRegression(C=1., max_iter=500,random_state=seed)
        self._m_gift=LogisticRegression(C=1., max_iter=500,random_state=seed)
        self._m_clin=LogisticRegression(C=1., max_iter=500,random_state=seed)
        self._meta  =LogisticRegression(C=.5, max_iter=500,random_state=seed)
        self.name="stacked"

    def fit(self, prs, gift, Xc, Q, yb):
        self._m_prs.fit( prs.reshape(-1,1), yb)
        self._m_gift.fit(gift.reshape(-1,1),yb)
        self._m_clin.fit(Xc,               yb)
        pp=_safe_proba(self._m_prs,  prs.reshape(-1,1)).reshape(-1,1)
        pg=_safe_proba(self._m_gift, gift.reshape(-1,1)).reshape(-1,1)
        pc=_safe_proba(self._m_clin, Xc).reshape(-1,1)
        self._meta.fit(np.hstack([pp,pg,pc,Q]),yb)
        return self

    def predict_proba(self, prs, gift, Xc, Q):
        pp=_safe_proba(self._m_prs,  prs.reshape(-1,1))
        pg=_safe_proba(self._m_gift, gift.reshape(-1,1))
        pc=_safe_proba(self._m_clin, Xc)
        Z=np.hstack([pp.reshape(-1,1),pg.reshape(-1,1),pc.reshape(-1,1),Q])
        raw=_safe_proba(self._meta, Z)
        return raw, dict(prs=pp, gift=pg, clinical=pc)

    def component_weights(self):
        c=self._meta.coef_[0][:3]
        c=np.abs(c); c/=c.sum()+1e-12
        return dict(prs=float(c[0]),gift=float(c[1]),clinical=float(c[2]))


class _BayesianRisk:
    """
    Bayesian log-odds combination.

    P(D | PRS, GIFT, CLIN) ∝ P(D) · LR_PRS · LR_GIFT · LR_CLIN

    where LR_k = P(score_k|D=1) / P(score_k|D=0) is approximated by
    the log-odds from each base logistic model. The prior P(D) is the
    population baseline risk. This yields a principled posterior update.
    """
    def __init__(self, baseline=0.10, seed=42):
        self.baseline=baseline; self.seed=seed
        self._m_prs =LogisticRegression(C=1.,max_iter=500,random_state=seed)
        self._m_gift=LogisticRegression(C=1.,max_iter=500,random_state=seed)
        self._m_clin=LogisticRegression(C=1.,max_iter=500,random_state=seed)
        self.name="bayesian"

    def fit(self, prs, gift, Xc, Q, yb):
        self._m_prs.fit( prs.reshape(-1,1), yb)
        self._m_gift.fit(gift.reshape(-1,1),yb)
        self._m_clin.fit(Xc,               yb)
        return self

    def _log_lr(self, p):
        """Log likelihood ratio from probability: log(p/(1-p)) - log(prior/(1-prior))"""
        logit_p    = np.log(p/(1-p+1e-12)+1e-12)
        logit_base = np.log(self.baseline/(1-self.baseline+1e-12)+1e-12)
        return logit_p - logit_base

    def predict_proba(self, prs, gift, Xc, Q):
        pp=_safe_proba(self._m_prs,  prs.reshape(-1,1))
        pg=_safe_proba(self._m_gift, gift.reshape(-1,1))
        pc=_safe_proba(self._m_clin, Xc)
        logit_prior = np.log(self.baseline/(1-self.baseline+1e-12)+1e-12)
        # Posterior log-odds = prior + Σ log-likelihood-ratios
        logit_post = logit_prior + self._log_lr(pp) + \
                     self._log_lr(pg) + self._log_lr(pc)
        raw = expit(logit_post)
        return raw, dict(prs=pp, gift=pg, clinical=pc)

    def component_weights(self):
        # Report relative magnitude of log-LR contributions (illustrative)
        return dict(prs=1/3, gift=1/3, clinical=1/3)


class DiseaseRiskPredictor:
    """
    Unified disease risk predictor wrapping all three integration strategies.

    prs_mode options
    ----------------
    'weighted_sum'  – learnable weight per signal source
    'stacked'       – ridge logistic meta-learner (default)
    'bayesian'      – Bayesian posterior log-odds update
    'all'           – fit all three and ensemble-average for final output

    Output per individual
    ----------------------
    absolute_risk       : 10-year absolute risk (0–1)
    absolute_risk_pct   : e.g. "14.2%"
    relative_risk       : absolute / population_baseline
    risk_tier           : Low / Moderate / High / Very High
    component_scores    : {prs, gift, clinical} per-source probabilities
    integration_weights : learned signal weights (where applicable)
    top_risk_factors    : top clinical features driving individual risk
    mode_predictions    : per-mode absolute risk when prs_mode='all'
    """
    _MODES = ("weighted_sum","stacked","bayesian")

    def __init__(self, disease="GENERIC", baseline_risk=None,
                 prs_mode="stacked", seed=42):
        self.disease  = disease.upper()
        self.baseline = baseline_risk or \
                        POPULATION_BASELINE.get(self.disease,.10)
        self.prs_mode = prs_mode
        self.seed     = seed
        self._models  = {}
        self._clin_model_ref = None   # for feature importance

    def _make_model(self, mode):
        if mode=="weighted_sum": return _WeightedSumRisk(self.seed)
        if mode=="stacked":      return _StackedRisk(self.seed)
        if mode=="bayesian":     return _BayesianRisk(self.baseline, self.seed)
        raise ValueError(mode)

    def fit(self, prs_z, gift_sc, X_clin, Q, y):
        valid=~np.isnan(y)&~np.isnan(prs_z)
        yb=_binarise(y[valid])
        pv,gv,cv,Qv=prs_z[valid],gift_sc[valid],X_clin[valid],Q[valid]
        modes=self._MODES if self.prs_mode=="all" else [self.prs_mode]
        for mode in modes:
            m=self._make_model(mode); m.fit(pv,gv,cv,Qv,yb)
            self._models[mode]=m
        # Keep a separate clinical logistic for feature importance
        self._clin_model_ref=LogisticRegression(C=1.,max_iter=500,
                                                random_state=self.seed)
        self._clin_model_ref.fit(cv,yb)
        print(f"[RISK] Fitted mode(s): {modes}, baseline={self.baseline:.1%}\n")
        return self

    def predict_risk(self, prs_z, gift_sc, X_clin, Q,
                     feature_names=None):
        modes=self._MODES if self.prs_mode=="all" else [self.prs_mode]
        all_raw=[]; all_comp=[]
        for mode in modes:
            raw,comp=self._models[mode].predict_proba(prs_z,gift_sc,X_clin,Q)
            all_raw.append(raw); all_comp.append(comp)

        abs_risk = np.mean(all_raw,0)   # ensemble mean if 'all'
        clin_coef= self._clin_model_ref.coef_[0]
        feat_names=feature_names or [f"f{i}" for i in range(X_clin.shape[1])]

        results=[]
        for i in range(len(abs_risk)):
            ar=float(abs_risk[i]); rr=ar/self.baseline
            contribs=np.abs(clin_coef*X_clin[i])
            ti=np.argsort(contribs)[::-1][:5]
            top=[(feat_names[j],float(contribs[j]))
                 for j in ti if j<len(feat_names)]
            # per-mode breakdown when mode=='all'
            mode_preds={m:float(r[i])
                        for m,r in zip(modes,all_raw)} if len(modes)>1 else {}
            # component scores averaged across modes
            avg_comp={k:float(np.mean([c[k][i] for c in all_comp]))
                      for k in ["prs","gift","clinical"]}
            # integration weights from first (or only) model
            iw=self._models[modes[0]].component_weights()
            results.append(dict(
                absolute_risk      = ar,
                absolute_risk_pct  = f"{ar*100:.1f}%",
                relative_risk      = rr,
                risk_tier          = _tier(ar),
                component_scores   = avg_comp,
                integration_weights= iw,
                top_risk_factors   = top,
                mode_predictions   = mode_preds,
            ))
        return results

    def evaluate(self, prs_z, gift_sc, X_clin, Q, y, feature_names=None):
        valid=~np.isnan(y)&~np.isnan(prs_z); yb=_binarise(y[valid])
        res=self.predict_risk(prs_z[valid],gift_sc[valid],
                              X_clin[valid],Q[valid],feature_names)
        scores=np.array([r["absolute_risk"] for r in res])
        auc=roc_auc_score(yb,scores); ap=average_precision_score(yb,scores)
        tiers=[r["risk_tier"] for r in res]
        tc={t:tiers.count(t) for t in ["Low","Moderate","High","Very High"]}
        print(f"\n{'-'*56}")
        print(f"Disease Risk Evaluation  [{self.disease}]  mode={self.prs_mode}")
        print(f"{'-'*56}")
        print(f"  AUC-ROC              : {auc:.4f}")
        print(f"  Avg Precision        : {ap:.4f}")
        print(f"  Mean absolute risk   : {scores.mean()*100:.1f}%")
        print(f"  Population baseline  : {self.baseline*100:.1f}%")
        if self.prs_mode=="all" and res[0]["mode_predictions"]:
            print("  Per-mode mean risks:")
            for m,v in res[0]["mode_predictions"].items():
                mode_sc=np.array([r["mode_predictions"][m] for r in res])
                print(f"    {m:<15} {mode_sc.mean()*100:.1f}%")
        print("  Risk tier distribution:")
        for t,c in tc.items(): print(f"    {t:<12}{c:>5} ({c/len(res)*100:.1f}%)")
        print(f"{'-'*56}\n")
        return scores, res


###############################################################################
# 9.  ADMIXTURE MAPPING
###############################################################################

def _ncdf(x):
    return .5*(1+np.sign(x)*np.sqrt(1-np.exp(-2/np.pi*x**2)))

class AdmixtureMapping:
    @staticmethod
    def scan(LA, y, bim, wins):
        v=~np.isnan(y); yv=y[v]; yc=yv-yv.mean(); LAv=LA[v]
        n,W,K=LAv.shape; rows=[]
        for w,idx in enumerate(wins):
            chrom=bim.iloc[idx[0]]["CHR"]
            sbp=int(bim.iloc[idx[0]]["BP"]); ebp=int(bim.iloc[idx[-1]]["BP"])
            for k in range(K):
                x=LAv[:,w,k]; xc=x-x.mean(); d=xc@xc
                if d<1e-10: continue
                beta=(xc@yc)/d; r=yc-beta*xc
                se=np.sqrt((r@r)/max(n-2,1)/d); z=beta/(se+1e-12)
                p=2*(1-_ncdf(abs(z)))
                rows.append(dict(WINDOW=w,CHR=chrom,START_BP=sbp,END_BP=ebp,
                                 ANCESTRY_K=k,BETA=beta,SE=se,Z=z,P=p))
        return pd.DataFrame(rows).sort_values("P").reset_index(drop=True)


###############################################################################
# 10.  OUTPUT
###############################################################################

def save_results(out_dir, bim, snp_scores, risk_results,
                 fam_te, y_te, admix_df, Q, k):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    # SNP importance
    si=bim.copy(); si["GIFT_SCORE"]=snp_scores
    si.sort_values("GIFT_SCORE",ascending=False).to_csv(
        out/"snp_importance.tsv",sep="\t",index=False)
    # Risk report
    rows=[]
    for i,((_,row),res) in enumerate(zip(fam_te.iterrows(),risk_results)):
        r=res; mp_str="|".join(f"{m}:{v:.3f}" for m,v in
                               r["mode_predictions"].items()) if r["mode_predictions"] else ""
        rows.append(dict(
            FID=row["FID"],IID=row["IID"],
            Y_TRUE=y_te[i] if i<len(y_te) else np.nan,
            ABS_RISK_PCT=r["absolute_risk_pct"],
            RELATIVE_RISK=f"{r['relative_risk']:.2f}",
            RISK_TIER=r["risk_tier"],
            SCORE_PRS=f"{r['component_scores']['prs']:.4f}",
            SCORE_GIFT=f"{r['component_scores']['gift']:.4f}",
            SCORE_CLINICAL=f"{r['component_scores']['clinical']:.4f}",
            WEIGHT_PRS=f"{r['integration_weights'].get('prs',0):.3f}",
            WEIGHT_GIFT=f"{r['integration_weights'].get('gift',0):.3f}",
            WEIGHT_CLINICAL=f"{r['integration_weights'].get('clinical',0):.3f}",
            MODE_PREDICTIONS=mp_str,
            TOP_RISK_FACTORS="; ".join(f"{f}({v:.3f})"
                                       for f,v in r["top_risk_factors"]),
        ))
    pd.DataFrame(rows).to_csv(out/"risk_report.tsv",sep="\t",index=False)
    admix_df.to_csv(out/"admixture_mapping.tsv",sep="\t",index=False)
    pd.DataFrame(Q,columns=[f"Q{k_}" for k_ in range(k)]).to_csv(
        out/"global_ancestry.tsv",sep="\t",index=False)
    for f in ["snp_importance.tsv","risk_report.tsv",
              "admixture_mapping.tsv","global_ancestry.tsv"]:
        print(f"[OUT] {f:<30} -> {out/f}")


###############################################################################
# 11.  CLI
###############################################################################

def parse_args():
    p=argparse.ArgumentParser(
        description="GIFT + Admixture + PRS + Disease Risk (all modes)")
    p.add_argument("--bfile",           required=True)
    p.add_argument("--pheno-file",      default=None)
    p.add_argument("--pheno-col",       type=int,   default=6)
    p.add_argument("--clinical",        default=None,
                   help="TSV: clinical/lifestyle/symptom columns")
    p.add_argument("--symptom-text-col",default=None,
                   help="Column name containing free-text symptom descriptions")
    p.add_argument("--prs-weights",     default=None,
                   help="TSV: SNP A1 BETA (GWAS summary stats)")
    p.add_argument("--prs-col",         default=None,
                   help="Pre-computed PRS column in --clinical file")
    p.add_argument("--prs-mode",        default="stacked",
                   choices=["weighted_sum","stacked","bayesian","all"],
                   help="PRS+GIFT+Clinical integration strategy (default: stacked)")
    p.add_argument("--disease",         default="GENERIC",
                   choices=["T2D","CVD","CKD","GENERIC"])
    p.add_argument("--baseline-risk",   type=float, default=None)
    p.add_argument("--out",             default="gift_risk_results/")
    # Ancestry
    p.add_argument("--k",               type=int,   default=3)
    p.add_argument("--k-max",           type=int,   default=5)
    p.add_argument("--window-size",     type=int,   default=50)
    p.add_argument("--window-step",     type=int,   default=25)
    p.add_argument("--trans-prob",      type=float, default=.02)
    p.add_argument("--ensemble-alpha",  type=float, default=.6)
    # GIFT
    p.add_argument("--ancestry-mode",   default="stratified",
                   choices=["stratified","covariate","interaction","all"])
    p.add_argument("--n-components",    type=int,   default=15)
    p.add_argument("--lambda-reg",      type=float, default=1e-3)
    p.add_argument("--field-temp",      type=float, default=1.)
    p.add_argument("--propagator-tau",  type=float, default=.5)
    p.add_argument("--mi-chunk",        type=int,   default=512)
    p.add_argument("--test-frac",       type=float, default=.2)
    p.add_argument("--seed",            type=int,   default=42)
    return p.parse_args()


###############################################################################
# 12.  MAIN
###############################################################################

def main(args=None):
    if args is None: args=parse_args()

    print("+==========================================================+")
    print("|  GIFT + Admixture + PRS + Full Disease Risk Predictor    |")
    print("|  3 PRS modes . Ensemble LAI . All clinical domains       |")
    print("|  Output: Absolute risk . Relative risk . Tier            |")
    print("+==========================================================+\n")

    # ── 1. PLINK ──────────────────────────────────────────────────────────────
    reader=PlinkReader(args.bfile,pheno_col=args.pheno_col)
    G,fam,bim,y=reader.load(pheno_file=args.pheno_file)

    # ── 2. Clinical data ──────────────────────────────────────────────────────
    if args.clinical:
        cdf=pd.read_csv(args.clinical,sep=r"\s+",dtype={"FID":str,"IID":str})
        cdf=fam[["FID","IID"]].merge(cdf,on=["FID","IID"],how="left")
        print(f"[CLINICAL] Loaded {cdf.shape[0]}x{cdf.shape[1]}")
    else:
        print("[CLINICAL] No file - generating synthetic clinical data.")
        rng=np.random.default_rng(args.seed); n=len(fam)
        cdf=fam[["FID","IID"]].copy()
        cdf["AGE"]          =rng.integers(30,75,n).astype(float)
        cdf["BMI"]          =rng.normal(27,5,n).clip(15,50)
        cdf["SBP"]          =rng.normal(125,18,n).clip(80,220)
        cdf["DBP"]          =rng.normal(80,12,n).clip(50,130)
        cdf["HEART_RATE"]   =rng.normal(72,12,n).clip(40,160)
        cdf["TOTAL_CHOL"]   =rng.normal(5,1.2,n).clip(2,12)
        cdf["HDL_CHOL"]     =rng.normal(1.4,.4,n).clip(.4,4)
        cdf["LDL_CHOL"]     =rng.normal(3.2,.9,n).clip(.5,8)
        cdf["GLUCOSE"]      =rng.normal(5.5,1.1,n).clip(2,25)
        cdf["HBA1C"]        =rng.normal(5.8,.8,n).clip(3,15)
        cdf["CREATININE"]   =rng.normal(85,20,n).clip(40,400)
        cdf["CRP"]          =rng.exponential(2,n).clip(0,100)
        cdf["SMOKING"]      =rng.choice([0,1,2],n,p=[.6,.15,.25]).astype(float)
        cdf["ALCOHOL_UNITS"]=rng.exponential(7,n).clip(0,100)
        cdf["EXERCISE_HRS"] =rng.exponential(3,n).clip(0,30)
        cdf["DIET_SCORE"]   =rng.uniform(2,9,n)
        cdf["SLEEP_HRS"]    =rng.normal(7,1.2,n).clip(3,12)
        cdf["HYPERTENSION"] =(cdf["SBP"]>140).astype(float)
        cdf["DIABETES"]     =(cdf["HBA1C"]>6.5).astype(float)
        cdf["CAD"]          =rng.binomial(1,.06,n).astype(float)
        cdf["FAMILY_HX_CVD"]=rng.binomial(1,.25,n).astype(float)
        cdf["CHEST_PAIN"]   =rng.binomial(1,.05,n).astype(float)
        cdf["DYSPNEA"]      =rng.binomial(1,.07,n).astype(float)
        cdf["FATIGUE"]      =rng.binomial(1,.15,n).astype(float)
        cdf["POLYURIA"]     =rng.binomial(1,.04,n).astype(float)
        # Synthetic free-text symptoms
        SYM_POOL=["fatigue and frequent urination","no symptoms",
                  "chest pain shortness of breath","thirst polyuria blurred vision",
                  "denies chest pain","palpitations dizziness","no complaints",
                  "mild fatigue","chest pain and dyspnea","polyuria and thirst"]
        cdf["SYMPTOMS"]     =rng.choice(SYM_POOL,n)
        args.symptom_text_col="SYMPTOMS"

    # ── 3. Global ancestry ────────────────────────────────────────────────────
    print("-- Global Ancestry (NMF) ----------------------------------")
    kf=None if args.k==0 else args.k
    gai=GlobalAncestryInfer(k=kf,k_max=args.k_max,seed=args.seed)
    gai.fit(G); Q_all=gai.proportions; K=gai.k_

    # ── 4. Local ancestry ─────────────────────────────────────────────────────
    print("-- Local Ancestry (Ensemble HMM + NMF) -------------------")
    lai=LocalAncestryInfer(k=K,window_size=args.window_size,
                           step_size=args.window_step,
                           transition_prob=args.trans_prob,
                           ensemble_alpha=args.ensemble_alpha,seed=args.seed)
    lai.fit(G,Q_all); SA_all=lai.snp_ancestry(G.shape[1])

    # ── 5. Admixture mapping ──────────────────────────────────────────────────
    print("-- Admixture Mapping -------------------------------------")
    admix_df=AdmixtureMapping.scan(SA_all,y,bim,lai.window_snps_)
    print(f"Top 5:\n{admix_df.head(5)[['CHR','START_BP','ANCESTRY_K','Z','P']].to_string(index=False)}\n")

    # ── 6. Split ──────────────────────────────────────────────────────────────
    rng2=np.random.default_rng(args.seed); n=G.shape[0]
    idx=rng2.permutation(n); split=int(n*(1-args.test_frac))
    tr,te=idx[:split],idx[split:]
    G_tr,G_te=G[tr],G[te]; y_tr,y_te=y[tr],y[te]
    Q_tr,Q_te=Q_all[tr],Q_all[te]; SA_tr,SA_te=SA_all[tr],SA_all[te]
    cdf_tr=cdf.iloc[tr].reset_index(drop=True)
    cdf_te=cdf.iloc[te].reset_index(drop=True)
    fam_te=fam.iloc[te].reset_index(drop=True)
    print(f"Split: {len(tr)} train / {len(te)} test\n")

    # ── 7. Clinical encoding ──────────────────────────────────────────────────
    print("-- Clinical + Symptom Encoding ---------------------------")
    enc=ClinicalEncoder(disease=args.disease,
                        symptom_text_col=args.symptom_text_col)
    Xc_tr=enc.fit_transform(cdf_tr); Xc_te=enc.transform(cdf_te)

    # ── 8. PRS ────────────────────────────────────────────────────────────────
    print("\n-- PRS Engine --------------------------------------------")
    prs_eng=PRSEngine(weights_path=args.prs_weights,prs_col=args.prs_col)
    prs_z_tr=prs_eng.fit_transform(prs_eng.get_prs(G_tr,bim,cdf_tr))
    prs_z_te=prs_eng.transform(    prs_eng.get_prs(G_te,bim,cdf_te))

    # ── 9. GIFT ───────────────────────────────────────────────────────────────
    print(f"\n-- GIFT (mode={args.ancestry_mode}, K={K}) ----------------------")
    gift=GIFTAdmixture(k=K,n_components=args.n_components,
                       lambda_reg=args.lambda_reg,field_temp=args.field_temp,
                       propagator_tau=args.propagator_tau,mi_chunk=args.mi_chunk,
                       ancestry_mode=args.ancestry_mode)
    gift.fit(G_tr,y_tr,Q_tr,SA_tr)
    gs_tr=gift.gift_scores(G_tr,Q_tr,SA_tr)
    gs_te=gift.gift_scores(G_te,Q_te,SA_te)

    # ── 10. Disease risk ───────────────────────────────────────────────────────
    print(f"\n-- Disease Risk Predictor [mode={args.prs_mode}] ----------------")
    risk=DiseaseRiskPredictor(disease=args.disease,
                              baseline_risk=args.baseline_risk,
                              prs_mode=args.prs_mode,seed=args.seed)
    risk.fit(prs_z_tr,gs_tr,Xc_tr,Q_tr,y_tr)
    risk_scores,risk_results=risk.evaluate(
        prs_z_te,gs_te,Xc_te,Q_te,y_te,
        feature_names=enc.feature_names)

    # ── 11. Sample individual report ──────────────────────────────────────────
    r=risk_results[0]
    print("-- Sample Individual Risk Report -------------------------")
    print(f"  Absolute 10-yr risk   : {r['absolute_risk_pct']}")
    print(f"  Relative risk         : {r['relative_risk']:.2f}x population")
    print(f"  Risk tier             : {r['risk_tier']}")
    print("  Component scores      : "
          f"PRS={r['component_scores']['prs']:.3f}  "
          f"GIFT={r['component_scores']['gift']:.3f}  "
          f"Clinical={r['component_scores']['clinical']:.3f}")
    print("  Integration weights   : "
          + "  ".join(f"{k}={v:.3f}" for k,v in r['integration_weights'].items()))
    if r["mode_predictions"]:
        print("  Per-mode predictions  : "
              + "  ".join(f"{m}={v*100:.1f}%" for m,v in r['mode_predictions'].items()))
    print("  Top risk factors      : "
          + ", ".join(f"{f}({v:.2f})" for f,v in r['top_risk_factors']))

    # ── 12. SNP importance + save ──────────────────────────────────────────────
    v_te=~np.isnan(y_te)
    snp_sc=gift.snp_importance(G_te[v_te],Q_te[v_te],SA_te[v_te])
    print("\nTop 5 GIFT SNPs:\n"
          f"{bim.iloc[np.argsort(snp_sc)[::-1][:5]][['CHR','SNP','BP']].to_string(index=False)}")
    save_results(args.out,bim,snp_sc,risk_results,
                 fam_te,y_te,admix_df,Q_all,K)
    print("\n[GIFT] Done.")

if __name__ == "__main__":
    main()
