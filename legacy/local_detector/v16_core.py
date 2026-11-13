"""Shared components for the v16 supplementary experiments.

Reuses the audited classes of run_uploaded_experiments.py (Data, ReferenceHIN, Local,
grouped_split, scaled, apply_scale, candidates, classifier) so that partitions, graph
descriptors and local features are identical to the submitted manuscript.
"""
import pickle,time,json,warnings,csv
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score,confusion_matrix,matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import run_uploaded_experiments as base

warnings.filterwarnings('ignore')
HERE=Path(__file__).resolve().parent

def load_all(data_root,cache=HERE/'work'/'data_cache.pkl'):
    if cache.exists():return pickle.loads(cache.read_bytes())
    P,T,A=[base.Data(data_root,n) for n in ['2020','2024','lt2020']]
    for d in (P,T,A):attach_meta(d,data_root)
    cache.write_bytes(pickle.dumps((P,T,A),protocol=5));return P,T,A

def attach_meta(d,root):
    """Audit-only metadata (never predictors): linking, stripping, packing, family, opcode n-grams."""
    import gzip
    folder=Path(root)/('du_lieu_'+d.name)
    with gzip.open(folder/'bang_v5_dense.csv.gz','rt',encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    assert [r['file_hash'].lower() for r in rows]==list(d.hash)
    d.is_static=np.array([float(r['LNK_is_static'] or 0)>0 for r in rows])
    d.stripped=np.array([float(r['SEC_is_stripped'] or 0)>0 for r in rows])
    d.packing=np.array([r['dong_goi'] for r in rows])
    d.family=np.array([(r['family'] or 'missing').lower() for r in rows])
    d.raw_source=np.array([r['nguon'] for r in rows])
    fam=Path(root)/'dataset-f1_ho_chuan_hoa.csv'
    if d.name=='2024' and fam.exists():
        with fam.open(encoding='utf-8') as f:lk={r['file_hash'].lower():r['ho_chuan_hoa'] for r in csv.DictReader(f)}
        d.family=np.array([lk.get(h,'missing') if y==1 else 'benign' for h,y in zip(d.hash,d.y)])
    voc=json.loads((folder/'sp_tu_vung.json').read_text(encoding='utf-8'))
    d.vocab2=voc['opc2'];d.vocab2_index={t:i for i,t in enumerate(d.vocab2)}
    d.opc2=sp.load_npz(folder/'sp_opc2.npz').tocsr()
    d.sem=sp.hstack([sp.load_npz(folder/'sp_sem1.npz'),sp.load_npz(folder/'sp_sem2.npz')],format='csr')
    d.sem_vocab=list(voc['sem1'])+list(voc['sem2'])
    assert d.opc2.shape[0]==len(d.y)==d.sem.shape[0] and d.sem.shape[1]==len(d.sem_vocab)
    d.has_sc=d.calls.any(1)

def external_indices(P,T,A):
    hashes=set(P.hash);fp=set(P.fingerprints)
    ti=np.array([i for i,h in enumerate(T.hash) if h not in hashes and T.fingerprints[i] not in fp])
    ai=np.array([i for i,h in enumerate(A.hash) if h not in hashes and A.fingerprints[i] not in fp])
    return ti,ai

def partition(P,seed,setting='holdout',purge=True):
    """purge=False (LOAO only): keep source files whose similarity groups also occur in the held-out architecture
    (e.g. the same Debian package built for other architectures). Only held-out-architecture files are removed."""
    purged=0
    if setting=='holdout':pool,test=base.grouped_split(P,np.arange(len(P.y)),seed)
    else:
        pool=np.flatnonzero(P.arch!=setting);test=np.flatnonzero(P.arch==setting)
        if purge:
            held=set(P.groups[test]);keep=np.array([P.groups[i] not in held for i in pool]);purged=int((~keep).sum());pool=pool[keep]
    train,ref=base.grouped_split(P,pool,seed+1701)
    assert not(set(P.groups[train])&set(P.groups[ref]))
    if purge or setting=='holdout':assert not(set(P.groups[train])&set(P.groups[test]) or set(P.groups[ref])&set(P.groups[test]))
    assert not(set(P.arch[train])|set(P.arch[ref]))&({setting} if setting!='holdout' else set())
    return train,ref,test,purged

# ---------------------------------------------------------------- metrics
def metrics(y,p,s):
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    out=dict(n=int(len(y)),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),fpr=float(fp/(fp+tn)) if fp+tn else None,
             recall=float(tp/(tp+fn)) if tp+fn else None,accuracy=float((tp+tn)/len(y)))
    if len(set(y))==2:
        out.update(macro_f1=float(f1_score(y,p,average='macro')),binary_f1=float(f1_score(y,p,zero_division=0)),
                   precision=float(tp/(tp+fp)) if tp+fp else 0.,mcc=float(matthews_corrcoef(y,p)),
                   roc_auc=float(roc_auc_score(y,s)),average_precision=float(average_precision_score(y,s)))
    else:out.update(macro_f1=None,binary_f1=None,precision=None,mcc=None,roc_auc=None,average_precision=None)
    return out

def best_threshold(y,s):
    """Threshold maximising macro F1 on source out-of-fold scores (no target labels)."""
    qs=np.unique(np.quantile(s,np.linspace(0.005,0.995,199)))
    f=[f1_score(y,(s>=q).astype(int),average='macro') for q in qs]
    return float(qs[int(np.argmax(f))])

# ---------------------------------------------------------------- models
NEW_MODELS=['rf','lgbm','lr','knn','mlp']
ORIG_MODELS=['svm','dt','nb']
RAW_MODELS=['hindroid','jacknn']   # consume unscaled [52 binary calls | arch one-hot]

def scale_fit(X,y,k):
    if k=='raw':return X,None
    return base.scaled(X,y,k)

def scale_apply(X,prep):
    return X if prep is None else base.apply_scale(X,prep)

class NgramTfidf:
    """Top-n tokens by summed training counts, TF-IDF fitted on training rows, aligned by token name."""
    def __init__(self,matrix_attr,vocab_attr,topn):self.m,self.v,self.topn=matrix_attr,vocab_attr,topn
    def fit(self,d,ix):
        from sklearn.feature_extraction.text import TfidfTransformer
        M=getattr(d,self.m);V=getattr(d,self.v);freq=np.asarray(M[ix].sum(0)).ravel()
        order=sorted(np.flatnonzero(freq>0),key=lambda j:(-freq[j],V[j]))[:self.topn];self.tokens_=[V[j] for j in order]
        self.tfidf_=TfidfTransformer().fit(M[ix][:,order]);return self
    def transform(self,d,ix):
        M=getattr(d,self.m);V=getattr(d,self.v);pos={t:i for i,t in enumerate(V)}
        cols=[M[ix,pos[t]] if t in pos else sp.csr_matrix((len(ix),1)) for t in self.tokens_]
        return self.tfidf_.transform(sp.hstack(cols,format='csr')).toarray()

class HinDroidSVM:
    """HinDroid-style multi-kernel SVM (Hou et al., KDD 2017) adapted to file-call-arch relations.
    Meta-path kernels between files: A A^T (file-call-file), A B A^T (call co-occurrence, B from
    training files), D D^T (same architecture). Each kernel is cosine-normalised; the combination
    and C are selected by inner CV. Identical training rows are collapsed with sample weights,
    which is exactly equivalent to duplicating them in the SVM dual."""
    def __init__(self,paths=('AA',),C=1.0,n_calls=52):self.paths,self.C,self.n_calls=paths,C,n_calls
    def _feats(self,X):
        A=X[:,:self.n_calls];D=X[:,self.n_calls:];return A,D
    def _kernel(self,X1,X2,n1=None,n2=None):
        A1,D1=self._feats(X1);A2,D2=self._feats(X2);K=np.zeros((len(X1),len(X2)))
        for p in self.paths:
            if p=='AA':k=A1@A2.T;d1=(A1*A1).sum(1);d2=(A2*A2).sum(1)
            elif p=='ABA':k=A1@self.B_@A2.T;d1=np.einsum('ij,jk,ik->i',A1,self.B_,A1);d2=np.einsum('ij,jk,ik->i',A2,self.B_,A2)
            elif p=='DD':k=D1@D2.T;d1=(D1*D1).sum(1);d2=(D2*D2).sum(1)
            K+=k/np.sqrt(np.outer(np.maximum(d1,1e-12),np.maximum(d2,1e-12)))
        return K/len(self.paths)
    def fit(self,X,y):
        from sklearn.svm import SVC
        A,_=self._feats(X);self.B_=A.T@A;np.fill_diagonal(self.B_,0)
        Xy=np.column_stack([X,y]);U,inv,cnt=np.unique(Xy,axis=0,return_inverse=True,return_counts=True)
        self.U_=U[:,:-1];uy=U[:,-1].astype(int)
        w=np.bincount(y,minlength=2);cw=len(y)/(2*w)  # balanced class weights
        self.svc_=SVC(kernel='precomputed',C=self.C).fit(self._kernel(self.U_,self.U_),uy,sample_weight=cnt*cw[uy])
        self.classes_=self.svc_.classes_;return self
    def decision_function(self,X):
        out=np.empty(len(X))
        for i in range(0,len(X),4000):out[i:i+4000]=self.svc_.decision_function(self._kernel(X[i:i+4000],self.U_))
        return out
    def predict(self,X):return (self.decision_function(X)>0).astype(int)

def grid(model,mode):
    if model in ORIG_MODELS:return base.candidates(model,'presence' if mode=='presence' else mode)
    if model=='svm_all':return [c for c in base.candidates('svm',mode) if c['k']=='all']
    if model=='hindroid':return [dict(k='raw',paths=p,C=c) for p in [('AA',),('AA','ABA'),('AA','ABA','DD')] for c in [0.1,1.0]]
    if model=='jacknn':return [dict(k='raw',n_neighbors=n) for n in [5,25]]
    if model=='identity':return [dict(k='raw')]
    g={'rf':[dict(max_features='sqrt',min_samples_leaf=1),dict(max_features=0.2,min_samples_leaf=1)],
       'lgbm':[dict(num_leaves=31,learning_rate=0.05),dict(num_leaves=63,learning_rate=0.05)],
       'lr':[dict(C=0.1),dict(C=1.0)],
       'knn':[dict(n_neighbors=5),dict(n_neighbors=25)],
       'mlp':[dict(alpha=1e-4),dict(alpha=1e-3)]}[model]
    return [dict(k='all',**c) for c in g]

def make(model,params,seed):
    p={k:v for k,v in params.items() if k!='k'}
    if model in ORIG_MODELS:return base.classifier(model,params,seed)
    if model=='svm_all':return base.classifier('svm',params,seed)
    if model=='hindroid':return HinDroidSVM(**p)
    if model=='jacknn':return JaccardKNN(**p)
    if model=='identity':return IdentityScore()
    if model=='rf':return RandomForestClassifier(n_estimators=200,class_weight='balanced',n_jobs=1,random_state=seed,**p)
    if model=='lgbm':
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=300,class_weight='balanced',subsample=0.8,subsample_freq=1,colsample_bytree=0.8,
                                  random_state=seed,n_jobs=1,verbose=-1,**p)
    if model=='lr':return LogisticRegression(class_weight='balanced',max_iter=3000,**p)
    if model=='knn':return KNeighborsClassifier(weights='distance',n_jobs=1,**p)
    if model=='mlp':return MLPClassifier(hidden_layer_sizes=(256,64),early_stopping=True,max_iter=300,random_state=seed,**p)
    raise ValueError(model)

class IdentityScore:
    """Training-free scorer: the single input column is already a malicious probability (e.g. reference-kNN
    fraction); predicts malicious when it is >= 0.5."""
    def fit(self,X,y):self.classes_=np.array([0,1]);return self
    def predict_proba(self,X):s=np.clip(X[:,0],0,1);return np.column_stack([1-s,s])
    def predict(self,X):return (X[:,0]>=0.5).astype(int)

class JaccardKNN:
    """kNN on Jaccard distance between static syscall-indicator sets (architecture ignored)."""
    def __init__(self,n_neighbors=5,n_calls=52):self.k,self.n_calls=n_neighbors,n_calls
    def fit(self,X,y):
        self.knn_=KNeighborsClassifier(n_neighbors=self.k,metric='jaccard',algorithm='brute',n_jobs=1).fit(X[:,:self.n_calls].astype(bool),y)
        self.classes_=self.knn_.classes_;return self
    def predict_proba(self,X):return self.knn_.predict_proba(X[:,:self.n_calls].astype(bool))
    def predict(self,X):return (self.predict_proba(X)[:,1]>=0.5).astype(int)

def score_of(clf,X):
    if hasattr(clf,'decision_function'):return clf.decision_function(X)
    return clf.predict_proba(X)[:,1]

def select_and_fit(model,mode,X,y,folds,fold_mats,seed,want_oof=False):
    """Inner group-CV selection by macro F1, then refit on all classifier-training data.
    fold_mats[f][k]=(xa,xb) pre-scaled matrices for inner fold f and feature budget k."""
    t0=time.perf_counter();G=grid(model,mode);sc=np.zeros((len(G),len(folds)));oof={}
    for j,prm in enumerate(G):
        o=np.zeros(len(y))
        for f,(a,b) in enumerate(folds):
            xa,xb=fold_mats[f][prm['k']];c=make(model,prm,seed).fit(xa,y[a])
            sc[j,f]=f1_score(y[b],c.predict(xb),average='macro')
            if want_oof:o[b]=score_of(c,xb)
        oof[j]=o
    best=int(sc.mean(1).argmax());prm=G[best]
    XX,prep=scale_fit(X,y,prm['k']);clf=make(model,prm,seed).fit(XX,y)
    return clf,prep,prm,float(sc[best].mean()),time.perf_counter()-t0,(oof[best] if want_oof else None)

# ======================================================================= HIN-v2
class HIN2:
    """Label-aware, identity-preserving meta-path descriptors over a disjoint reference pool R.

    Node types: file, static syscall (52), semantic opcode n-gram (sem1+sem2, architecture-independent),
    architecture. Meta-paths: F-S-F (IDF-weighted syscall overlap), F-N-F (TF-IDF semantic n-gram
    cosine), and their mean (multi-path, HinDroid-style). For each meta-path similarity s(q, r) to the
    reference files we summarise the *labelled* neighbourhood of q in R:
      knn_mal  : similarity-weighted malicious fraction among the top-k references
      max_mal, max_ben, margin = max_mal - max_ben, mean_topk
    The F-N-F path is additionally summarised over references of a *different* architecture only
    (cross-architecture evidence). Queries never belong to R, so there is no self-match; R labels are
    used, but R never trains the downstream classifier.
    """
    def __init__(self,k=10):self.k=k
    def fit(self,d,ref):
        self.sem_=NgramTfidf('sem','sem_vocab',len(d.sem_vocab)).fit(d,ref)
        S=d.calls[ref].astype(float);df=S.sum(0);self.idf_=np.log((1+len(ref))/(1+df))+1
        self.RS_=self._norm(S*self.idf_);self.RN_=self._norm(self.sem_.transform(d,ref))
        self.y_=d.y[ref].astype(float);self.arch_=d.arch[ref];return self
    @staticmethod
    def _norm(X):
        n=np.linalg.norm(X,axis=1,keepdims=True);return X/np.maximum(n,1e-12)
    def _summ(self,Ssim,mask=None):
        S=Ssim.copy()
        if mask is not None:S[~mask]=-1.0
        k=self.k;idx=np.argpartition(-S,k-1,axis=1)[:,:k];top=np.take_along_axis(S,idx,1);ty=self.y_[idx]
        top=np.clip(top,0,None);w=top.sum(1)
        knn=np.where(w>0,(top*ty).sum(1)/np.maximum(w,1e-12),0.5)
        Smal=np.where(self.y_[None,:]==1,S,-1);Sben=np.where(self.y_[None,:]==0,S,-1)
        mm=np.clip(Smal.max(1),0,None);mb=np.clip(Sben.max(1),0,None)
        return np.column_stack([knn,mm,mb,mm-mb,top.mean(1)])
    def transform(self,d,ix,chunk=3000):
        out=[]
        QS_all=self._norm(d.calls[ix].astype(float)*self.idf_);QN_all=self._norm(self.sem_.transform(d,ix))
        for i in range(0,len(ix),chunk):
            QS,QN=QS_all[i:i+chunk],QN_all[i:i+chunk];a=d.arch[ix][i:i+chunk]
            sS=QS@self.RS_.T;sN=QN@self.RN_.T;sM=(sS+sN)/2
            other=a[:,None]!=self.arch_[None,:]
            out.append(np.column_stack([self._summ(sS),self._summ(sN),self._summ(sM),self._summ(sN,other)]))
        return np.vstack(out)
HIN2_NAMES=[f'{p}:{s}' for p in ['S','N','SN','N_xarch'] for s in ['knn_mal','max_mal','max_ben','margin','mean_topk']]
