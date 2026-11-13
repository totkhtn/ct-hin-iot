"""v16 supplementary runs: shortcut controls, more classifier families, multi-seed LOAO.

Example:
  python v16_run.py --protocol holdout --seeds 42 --modes local fusion --models svm dt nb --tag repro
Outputs (per tag): results/<tag>/rows.jsonl and results/<tag>/pred_<seed>_<setting>.npz
"""
import argparse,json,time,os
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import run_uploaded_experiments as base
import v16_core as C

ARCHES=['EM_386','EM_ARM','EM_MIPS','EM_PPC','EM_X86_64']
NGRAM={'opc2_tfidf':('opc2','vocab2',1000),'sem_tfidf':('sem','sem_vocab',495)}

def fit_extra(d,ix,modes):
    need=set(m for m in modes if m in NGRAM)|({'sem_tfidf'} if 'local_sem' in modes else set())
    return {m:C.NgramTfidf(*NGRAM[m]).fit(d,ix) for m in need}

HIN2_CACHE={}
def blocks_for(P,d,ix,hin,local,extra):
    calls=d.calls[ix]
    b=dict(Z=hin.transform(calls,d.arch[ix]),L=local.transform(d,ix) if local is not None else None,
           pres=calls.any(1).astype(float)[:,None],deg=np.log1p(calls.sum(1)).astype(float)[:,None],raw52=calls.astype(float),
           arch1h=np.array([[a==b for b in ARCHES] for a in d.arch[ix]],dtype=float))
    for m,t in extra.items():b[m]=t.transform(d,ix)
    if 'hin2' in HIN2_CACHE:b['Z2']=HIN2_CACHE['hin2'].transform(d,ix)
    if b['L'] is not None:
        keep=[j for j,c in enumerate(P.columns) if not c.startswith('SC_')]+list(range(len(P.columns),b['L'].shape[1]))
        b['L_nosc']=b['L'][:,keep]
    return b

def assemble(mode,b):
    Z=b['Z']
    if mode=='relations':return Z
    if mode in NGRAM:return b[mode]
    if mode=='numeric_only':return b['L'][:,:633]
    if mode=='opc1_only':return b['L'][:,633:]
    if mode=='calls_arch':return np.column_stack([b['raw52'],b['arch1h']])
    if mode=='presence':return b['pres']
    if mode in('local','local_full'):return b['L']
    if mode=='fusion':return np.column_stack([b['L'],Z])
    if mode=='local_pres':return np.column_stack([b['L'],b['pres']])
    if mode=='hin2':return b['Z2']
    if mode=='hin2_noS':Z=b['Z2'];return Z[:,list(range(5,10))+list(range(15,20))]
    if mode=='hin2_noX':return b['Z2'][:,:15]
    if mode=='hin2_nolabel':
        Z=b['Z2'];return np.column_stack([np.maximum(Z[:,5*p+1],Z[:,5*p+2]) for p in range(4)]+[Z[:,5*p+4] for p in range(4)])
    if mode=='semknn':return b['Z2'][:,[5]]   # N-path similarity-weighted malicious fraction (top-10 reference neighbours)
    if mode=='local_hin2':return np.column_stack([b['L'],b['Z2']])
    if mode=='local_sem':return np.column_stack([b['L'],b['sem_tfidf']])
    if mode=='local_nosc':return b['L_nosc']
    if mode=='local_pres8':return np.column_stack([b['L']]+[b['pres']]*8)
    if mode=='local_deg':return np.column_stack([b['L'],b['deg']])
    if mode=='local_raw52':return np.column_stack([b['L'],b['raw52']])
    if mode=='fusion_nodeg':return np.column_stack([b['L'],Z[:,1:]])
    if mode.startswith('fusion_drop'):
        c=int(mode[len('fusion_drop'):]);return np.column_stack([b['L'],np.delete(Z,c,axis=1)])
    if mode.startswith('fusion_only'):  # local + a single relational coordinate
        c=int(mode[len('fusion_only'):]);return np.column_stack([b['L'],Z[:,[c]]])
    raise ValueError(mode)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='work/input');ap.add_argument('--out',default='results')
    ap.add_argument('--protocol',choices=['holdout','loao'],default='holdout');ap.add_argument('--seeds',nargs='+',type=int,required=True)
    ap.add_argument('--architectures',nargs='+');ap.add_argument('--modes',nargs='+',required=True);ap.add_argument('--models',nargs='+',required=True)
    ap.add_argument('--tag',required=True);ap.add_argument('--ref-fraction-folds',type=int,default=5,
        help='number of folds in the second split; 1/k of the pool becomes the reference graph (default 5 -> ~16%% of corpus)')
    ap.add_argument('--threshold',action='store_true',help='also report source-OOF calibrated threshold')
    ap.add_argument('--no-purge',action='store_true',help='LOAO: do not purge source files sharing similarity groups with the held-out architecture')
    ap.add_argument('--eval-all-external',action='store_true',help='LOAO: evaluate on the full external corpus (seen + held-out architectures)')
    args=ap.parse_args()
    out=Path(args.out)/args.tag;out.mkdir(parents=True,exist_ok=True)
    P,T,A=C.load_all(args.data);ti,ai=C.external_indices(P,T,A)
    def unit_path(seed,setting,mode,model):return out/'units'/f'{seed}_{setting}'/f'{mode}__{model}.npz'
    def wanted(mode):return [m for m in args.models if not(mode=='presence' and m not in C.ORIG_MODELS+['lgbm','rf'])]
    for seed in args.seeds:
        settings=['holdout'] if args.protocol=='holdout' else (args.architectures or sorted(set(P.arch)))
        for setting in settings:
            key=f'{seed}_{setting}'
            if all(unit_path(seed,setting,m,mm).exists() for m in args.modes for mm in wanted(m)):print('skip',key,flush=True);continue
            (out/'units'/key).mkdir(parents=True,exist_ok=True)
            t0=time.perf_counter()
            if args.ref_fraction_folds==5:train,ref,test,purged=C.partition(P,seed,setting,purge=not args.no_purge)
            else:
                pool,test=(base.grouped_split(P,np.arange(len(P.y)),seed) if setting=='holdout' else (None,None));purged=0
                strata=np.array([str(P.y[i])+':'+P.arch[i] for i in pool])
                a,b=next(StratifiedGroupKFold(args.ref_fraction_folds,shuffle=True,random_state=seed+1701).split(pool,strata,P.groups[pool]))
                train,ref=pool[a],pool[b]
            ext=ti if (setting=='holdout' or args.eval_all_external) else ti[T.arch[ti]==setting]
            aux=ai if setting=='holdout' else ai[A.arch[ai]==setting]
            evals=[('primary',P,test),('external',T,ext)]+([('auxiliary_benign',A,aux)] if len(aux) else [])
            hin=base.ReferenceHIN(True).fit(P.calls[ref],P.arch[ref])
            HIN2_CACHE.clear()
            if any(m.startswith('hin2') or m in ('local_hin2','semknn') for m in args.modes):HIN2_CACHE['hin2']=C.HIN2().fit(P,ref)
            for full in ([False,True] if 'local_full' in args.modes else [False]):
                modes=[m for m in args.modes if (m=='local_full')==full and not all(unit_path(seed,setting,m,mm).exists() for mm in wanted(m))]
                if not modes:continue
                tr=np.concatenate([train,ref]) if full else train
                ytr=P.y[tr]
                folds=list(StratifiedGroupKFold(3,shuffle=True,random_state=seed).split(tr,ytr,P.groups[tr]))
                fold_blocks=[]
                for a,b in folds:
                    loc=base.Local().fit(P,tr[a]);ex=fit_extra(P,tr[a],modes)
                    fold_blocks.append((blocks_for(P,P,tr[a],hin,loc,ex),blocks_for(P,P,tr[b],hin,loc,ex)))
                loc=base.Local().fit(P,tr);ex=fit_extra(P,tr,modes)
                trb=blocks_for(P,P,tr,hin,loc,ex);evb=[blocks_for(P,d,ix,hin,loc,ex) for _,d,ix in evals]
                for mode in modes:
                    X=assemble(mode,trb);EX=[assemble(mode,b) for b in evb]
                    ks=sorted({g['k'] for m in args.models for g in C.grid(m,mode)},key=str)
                    fold_mats=[]
                    for (ba,bb),(a,b) in zip(fold_blocks,folds):
                        xa,xb=assemble(mode,ba),assemble(mode,bb);d={}
                        for k in ks:
                            za,prep=C.scale_fit(xa,ytr[a],k);d[k]=(za,C.scale_apply(xb,prep))
                        fold_mats.append(d)
                    for model in wanted(mode):
                        if unit_path(seed,setting,mode,model).exists():continue
                        preds={};rows=[]
                        clf,prep,prm,inner,secs,oof=C.select_and_fit(model,mode,X,ytr,folds,fold_mats,seed,want_oof=args.threshold)
                        thr=C.best_threshold(ytr,oof) if args.threshold else None
                        if args.threshold:preds[f'{mode}|{model}|oof']=oof.astype(np.float32)
                        for (corpus,d,ix),xx in zip(evals,EX):
                            t1=time.perf_counter();xs=C.scale_apply(xx,prep);p=clf.predict(xs).astype(int);s=C.score_of(clf,xs);pred_secs=time.perf_counter()-t1
                            preds[f'{mode}|{model}|{corpus}|pred']=p.astype(np.int8);preds[f'{mode}|{model}|{corpus}|score']=s.astype(np.float32)
                            groups=[('all',np.ones(len(ix),bool))]
                            if corpus=='external':
                                groups+=[('arch:'+a,d.arch[ix]==a) for a in sorted(set(d.arch[ix]))]
                                groups+=[('benign_source:'+v,(d.source[ix]==v)&(d.y[ix]==0)) for v in sorted(set(d.source[ix][d.y[ix]==0]))]
                            for g,mask in groups:
                                if not mask.any():continue
                                r=dict(seed=seed,setting=setting,mode=mode,model=model,corpus=corpus,group=g,params=str(prm),inner_macro_f1=inner,
                                       dim=int(X.shape[1]),n_train=int(len(tr)),n_ref=int(len(ref)),purged=purged,fit_seconds=secs,predict_seconds=pred_secs,
                                       **C.metrics(d.y[ix][mask],p[mask],s[mask]))
                                if thr is not None:
                                    pt=(s[mask]>=thr).astype(int);m2=C.metrics(d.y[ix][mask],pt,s[mask])
                                    r.update({'cal_'+k:m2[k] for k in ['macro_f1','fpr','recall']},threshold=thr)
                                rows.append(r)
                        up=unit_path(seed,setting,mode,model);tmp=up.with_suffix('.tmp.npz')
                        idx={f'{c}|index':ix for c,_,ix in evals}
                        np.savez_compressed(tmp,rows=np.array(json.dumps(rows)),train=tr,ref=ref,**idx,**preds);os.replace(tmp,up)
                        print(json.dumps(dict(seed=seed,setting=setting,mode=mode,model=model,secs=round(secs,1))),flush=True)
            print('DONE',key,round(time.perf_counter()-t0,1),flush=True)

if __name__=='__main__':main()
