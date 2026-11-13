"""Independent HIN/ML evaluation of user-supplied feature tables.
Does not execute supplied scripts or binaries. Metadata never enters predictors.
"""
import argparse,csv,gzip,json,hashlib,collections,time,platform,warnings
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import StratifiedGroupKFold,ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import f_classif
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score,matthews_corrcoef,average_precision_score,confusion_matrix
import sklearn,scipy

PREFIXES={'BH','BEH','ENT','CFG','STR','SC'}
EXCLUDE_CALLS={'SC_sc_sites','SC_sc_unique'}

class Data:
    def __init__(self,root,name):
        folder=Path(root)/('du_lieu_'+name);self.name=name
        with gzip.open(folder/'bang_v5_dense.csv.gz','rt',encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f);self.columns=[k for k in reader.fieldnames if k.split('_')[0] in PREFIXES]
            self.meta=[];values=[]
            for r in reader:
                self.meta.append({k:r[k] for k in reader.fieldnames[:18]})
                values.append([float(r[k]) if r[k] else np.nan for k in self.columns])
        self.x=np.asarray(values,dtype=np.float64)
        if not np.isfinite(self.x).all():raise ValueError('Non-finite predictor')
        self.hash=np.array([r['file_hash'].lower() for r in self.meta]);self.y=np.array([int(r['label']) for r in self.meta])
        self.arch=np.array([r['arch'] for r in self.meta]);self.source=np.array([source_group(r['nguon']) for r in self.meta])
        self.groups=np.array([r['nhom_chia'] or 'sha:'+r['file_hash'] for r in self.meta],dtype=object)
        if len(set(self.hash))!=len(self.hash):raise ValueError('Duplicate file hash')
        if any(len(h)!=64 or any(c not in '0123456789abcdef' for c in h) for h in self.hash):raise ValueError('Invalid SHA256')
        self.call_columns=[i for i,k in enumerate(self.columns) if k.startswith('SC_sc_') and k not in EXCLUDE_CALLS]
        self.calls=(self.x[:,self.call_columns]>0).astype(np.uint8)
        voc=json.loads((folder/'sp_tu_vung.json').read_text(encoding='utf-8'))
        self.vocab=voc['opc1'];self.vocab_index={t:i for i,t in enumerate(self.vocab)}
        self.opc=sp.load_npz(folder/'sp_opc1.npz').tocsr()
        if self.opc.shape!=(len(self.y),len(self.vocab)):raise ValueError('Sparse/dense/vocabulary alignment mismatch')
        if self.opc.nnz and (not np.isfinite(self.opc.data).all() or (self.opc.data<0).any()):raise ValueError('Invalid opcode count')
        # Identical raw predictor vectors are grouped, independently of labels.
        fingerprints=[]
        for i in range(len(self.y)):
            h=hashlib.sha256(np.round(self.x[i],6).tobytes());row=self.opc[i]
            tokens=sorted((self.vocab[j],float(v)) for j,v in zip(row.indices,row.data))
            h.update(json.dumps(tokens,separators=(',',':')).encode());fingerprints.append(h.hexdigest())
        self.fingerprints=np.array(fingerprints)
        self.groups=merge_groups(self.groups,self.fingerprints)

def source_group(s):
    s=s.lower()
    if 'debian' in s:return 'Debian'
    if 'open' in s:return 'OpenWrt'
    if 'ddwrt' in s or 'dd-wrt' in s:return 'DD-WRT'
    if 'alpine' in s:return 'Alpine'
    return s

def merge_groups(groups,fingerprints):
    parent={}
    def find(v):
        parent.setdefault(v,v)
        if parent[v]!=v:parent[v]=find(parent[v])
        return parent[v]
    for g,h in zip(groups,fingerprints):
        a,b=find('g:'+g),find('v:'+h)
        if a!=b:parent[b]=a
    return np.array([find('g:'+g) for g in groups],dtype=object)

class ReferenceHIN:
    """Exact weighted compression of a disjoint file--call--architecture graph."""
    def __init__(self,use_arch=True):self.use_arch=use_arch
    def fit(self,calls,arch):
        self.active_=np.flatnonzero(calls.any(0));X=calls[:,self.active_].astype(float)
        self.patterns_,self.weights_=np.unique(X,axis=0,return_counts=True)
        self.frequency_=X.sum(0);self.B_=X.T@X;np.fill_diagonal(self.B_,0)
        self.arches_=sorted(set(arch));D=np.array([[a==b for b in self.arches_] for a in arch],dtype=float)
        self.arch_count_=D.sum(0);self.call_arch_=X.T@D
        return self
    def transform(self,calls,arch):
        Q=calls[:,self.active_].astype(float);A,inv=np.unique(Q,axis=0,return_inverse=True)
        k1=A@self.patterns_.T;k2=A@self.B_@self.patterns_.T
        z=np.column_stack([A.sum(1),A@self.frequency_,(k1>0)@self.weights_,
            k1.max(1),A@self.B_@self.frequency_,(k2>0)@self.weights_])[inv]
        d=np.array([self.arch_count_[self.arches_.index(a)] if a in self.arches_ else 0 for a in arch])
        mix=Q@self.call_arch_@self.arch_count_
        if not self.use_arch:d=np.zeros(len(Q));mix=np.zeros(len(Q))
        return np.log1p(np.column_stack([z,d,mix]))

class Local:
    def fit(self,data,ix):
        freq=np.asarray(data.opc[ix].sum(0)).ravel()
        order=sorted(np.flatnonzero(freq>0),key=lambda j:(-freq[j],data.vocab[j]))[:100]
        self.tokens_=[data.vocab[j] for j in order]
        self.tfidf_=TfidfTransformer().fit(data.opc[ix][:,order])
        return self
    def transform(self,data,ix):
        numeric=data.x[ix].copy()
        for j,k in enumerate(data.columns):
            if not k.startswith(('BH_','BEH_')):numeric[:,j]=np.sign(numeric[:,j])*np.log1p(abs(numeric[:,j]))
        cols=[]
        for t in self.tokens_:
            j=data.vocab_index.get(t);cols.append(data.opc[ix,j] if j is not None else sp.csr_matrix((len(ix),1)))
        matrix=sp.hstack(cols,format='csr')
        return np.column_stack([numeric,self.tfidf_.transform(matrix).toarray()])

def metric(y,p,s):
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    base=dict(n=len(y),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),fpr=float(fp/(fp+tn)) if fp+tn else None,
              accuracy=float(accuracy_score(y,p)))
    if len(set(y))<2:
        base.update(macro_f1=None,binary_f1=None,precision=None,recall=float(tp/(tp+fn)) if tp+fn else None,mcc=None,average_precision=None)
    else:base.update(macro_f1=float(f1_score(y,p,average='macro')),binary_f1=float(f1_score(y,p)),
        precision=float(precision_score(y,p,zero_division=0)),recall=float(recall_score(y,p)),
        mcc=float(matthews_corrcoef(y,p)),average_precision=float(average_precision_score(y,s)))
    return base

def grouped_split(data,ix,seed,n=5):
    strata=np.array([str(data.y[i])+':'+data.arch[i] for i in ix])
    a,b=next(StratifiedGroupKFold(n,shuffle=True,random_state=seed).split(ix,strata,data.groups[ix]))
    return ix[a],ix[b]

def candidates(model,mode):
    ks=['all'] if mode=='presence' else ['all',4 if mode=='relations' else 32]
    params={'svm':dict(C=[.1,1],k=ks),'dt':dict(max_depth=[8,None],min_samples_leaf=[5,20],k=ks),
        'nb':dict(var_smoothing=[1e-9,1e-7,1e-5],k=ks)}
    return list(ParameterGrid(params[model]))

def scaled(X,y,k,valid=None):
    if valid is None:valid=np.flatnonzero(np.var(X,axis=0)>0)
    Z=X[:,valid];scale=StandardScaler().fit(Z);Z=scale.transform(Z)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');scores=f_classif(Z,y)[0]
    selected=np.arange(Z.shape[1]) if k=='all' else np.argsort(-np.nan_to_num(scores,nan=-np.inf),kind='stable')[:min(k,len(valid))]
    return Z[:,selected],(valid,scale,selected)

def apply_scale(X,prep):
    valid,scale,selected=prep;return scale.transform(X[:,valid])[:,selected]

def classifier(name,params,seed):
    params={k:v for k,v in params.items() if k!='k'}
    if name=='svm':return SVC(kernel='rbf',gamma='scale',class_weight='balanced',**params)
    if name=='dt':return DecisionTreeClassifier(class_weight='balanced',random_state=seed,**params)
    return GaussianNB(**params)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--protocol',choices=['main','loao','ablation'],default='main');ap.add_argument('--seeds',nargs='+',type=int,default=[10,42,100,314,555,777,999,1234,2024,2026]);ap.add_argument('--architectures',nargs='+')
    ap.add_argument('--no-arch',action='store_true');ap.add_argument('--modes',nargs='+',choices=['relations','local','fusion','presence']);args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    P,T,A=[Data(args.data,n) for n in ['2020','2024','lt2020']]
    if P.columns!=T.columns or P.columns!=A.columns:raise ValueError('Dense schemas differ')
    hashes=set(P.hash);fp=set(P.fingerprints)
    ti=np.array([i for i,h in enumerate(T.hash) if h not in hashes and T.fingerprints[i] not in fp])
    ai=np.array([i for i,h in enumerate(A.hash) if h not in hashes and A.fingerprints[i] not in fp])
    audit=dict(primary_rows=len(P.y),external_rows=len(T.y),auxiliary_rows=len(A.y),external_retained=len(ti),auxiliary_retained=len(ai),
        exact_hash_overlap=len(set(P.hash)&set(T.hash)),external_identical_predictor_rows_removed=len(T.y)-len(ti),
        feature_prefixes=sorted(PREFIXES),dense_predictor_count=len(P.columns),call_indicator_count=len(P.call_columns),
        source_provenance='Author confirms homd positives derive from YNU Dataset C and SC_sc_* are static syscall indicators; external positives are supplied as f1full.',
        chronology='Mixed or missing years: this is cross-collection external evaluation, not a clean chronological deployment test.')
    (out/'data_audit.json').write_text(json.dumps(audit,indent=2))
    results=[];predictions=[];partitions=[];completed=[]
    for seed in args.seeds:
        settings=['holdout'] if args.protocol!='loao' else (args.architectures or sorted(set(P.arch)))
        for setting in settings:
            purged_source=0
            if setting=='holdout':pool,test=grouped_split(P,np.arange(len(P.y)),seed)
            else:
                pool=np.flatnonzero(P.arch!=setting);test=np.flatnonzero(P.arch==setting)
                held_groups=set(P.groups[test]);keep=np.array([P.groups[i] not in held_groups for i in pool])
                purged_source=int((~keep).sum());pool=pool[keep]
            train,ref=grouped_split(P,pool,seed+1701)
            assert not(set(P.groups[train])&set(P.groups[ref]) or set(P.groups[train])&set(P.groups[test]) or set(P.groups[ref])&set(P.groups[test]))
            for role,ix in [('classifier_train',train),('reference',ref),('primary_test',test)]:
                partitions.extend(dict(seed=seed,setting=setting,role=role,sha256=P.hash[i]) for i in ix)
            external=ti if setting=='holdout' else ti[T.arch[ti]==setting]
            auxiliary=ai if setting=='holdout' else ai[A.arch[ai]==setting]
            evals=[('primary',P,test),('external',T,external)]
            if len(auxiliary):evals.append(('auxiliary_benign',A,auxiliary))
            no_arch=args.protocol=='ablation' or args.no_arch
            hin=ReferenceHIN(not no_arch).fit(P.calls[ref],P.arch[ref]);Z=hin.transform(P.calls[train],P.arch[train])
            evalZ=[hin.transform(d.calls[ix],d.arch[ix]) for _,d,ix in evals]
            folds=list(StratifiedGroupKFold(3,shuffle=True,random_state=seed).split(train,P.y[train],P.groups[train]))
            local_folds=[]
            for a,b in folds:
                f=Local().fit(P,train[a]);local_folds.append((f.transform(P,train[a]),f.transform(P,train[b])))
            local=Local().fit(P,train);L=local.transform(P,train);evalL=[local.transform(d,ix) for _,d,ix in evals]
            modes=['relations','local','fusion'] if args.protocol=='main' else (['local','fusion'] if args.protocol=='loao' else ['fusion','presence'])
            if args.modes:modes=args.modes
            for mode in modes:
                X=Z if mode=='relations' else (L if mode=='local' else np.column_stack([L,Z]))
                EX=evalZ if mode=='relations' else (evalL if mode=='local' else [np.column_stack([l,z]) for l,z in zip(evalL,evalZ)])
                if mode=='presence':X=P.calls[train].any(1).astype(float)[:,None];EX=[d.calls[ix].any(1).astype(float)[:,None] for _,d,ix in evals]
                matrices=[]
                for fold,(a,b) in enumerate(folds):
                    la,lb=local_folds[fold]
                    xa,xb=(Z[a],Z[b]) if mode=='relations' else ((la,lb) if mode=='local' else (np.column_stack([la,Z[a]]),np.column_stack([lb,Z[b]])))
                    if mode=='presence':xa,xb=X[a],X[b]
                    transformed={}
                    for k in set(c['k'] for m in ['svm','dt','nb'] for c in candidates(m,mode)):
                        xx,prep=scaled(xa,P.y[train[a]],k);transformed[k]=(xx,apply_scale(xb,prep))
                    matrices.append(transformed)
                for model in ['svm','dt','nb']:
                    started=time.perf_counter();grid=candidates(model,mode);scores=np.zeros((len(grid),len(folds)))
                    for j,params in enumerate(grid):
                        for f,(a,b) in enumerate(folds):
                            xa,xb=matrices[f][params['k']];clf=classifier(model,params,seed);clf.fit(xa,P.y[train[a]])
                            scores[j,f]=f1_score(P.y[train[b]],clf.predict(xb),average='macro')
                    best=int(scores.mean(1).argmax());params=grid[best];XX,prep=scaled(X,P.y[train],params['k'])
                    clf=classifier(model,params,seed);clf.fit(XX,P.y[train]);fit_seconds=time.perf_counter()-started
                    for (corpus,d,ix),xx in zip(evals,EX):
                        xx=apply_scale(xx,prep);p=clf.predict(xx);s=clf.decision_function(xx) if hasattr(clf,'decision_function') else clf.predict_proba(xx)[:,1]
                        groups=[('all',np.ones(len(ix),dtype=bool))]
                        if corpus=='external':
                            groups += [('arch:'+a,d.arch[ix]==a) for a in sorted(set(d.arch[ix]))]
                            groups += [('benign_source:'+v,(d.source[ix]==v)&(d.y[ix]==0)) for v in sorted(set(d.source[ix][d.y[ix]==0]))]
                        for group,mask in groups:
                            if not mask.any():continue
                            results.append(dict(seed=seed,setting=setting,protocol=args.protocol,mode=mode,model=model,corpus=corpus,group=group,
                                n_classifier_train=len(train),n_reference=len(ref),n_purged_source_groups_samples=purged_source,best_params=params,inner_macro_f1=float(scores[best].mean()),
                                selected_dimension=XX.shape[1],search_refit_seconds=fit_seconds,**metric(d.y[ix][mask],p[mask],s[mask])))
                        predictions.extend(dict(seed=seed,setting=setting,protocol=args.protocol,mode=mode,model=model,corpus=corpus,sha256=d.hash[i],arch=d.arch[i],
                            source=d.source[i],label=int(d.y[i]),prediction=int(pp),score=float(ss)) for i,pp,ss in zip(ix,p,s))
                    completed.append(dict(seed=seed,setting=setting,mode=mode,model=model));print(json.dumps(completed[-1]),flush=True)
                    (out/'results.json').write_text(json.dumps(results,indent=2));(out/'completed.json').write_text(json.dumps(completed))
    for name,rows in [('results',results),('predictions',predictions),('partitions',partitions)]:
        with (out/(name+'.csv')).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'run_info.json').write_text(json.dumps(dict(args=vars(args),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),metadata_excluded=True,disjoint_reference_graph=True),indent=2))

if __name__=='__main__':main()
