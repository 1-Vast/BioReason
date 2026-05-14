
import argparse, json, re
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
from scipy.sparse import issparse

PERT_KEYS=['perturbation','condition','gene','guide','target','perturbation_name','treatment','group']
CONTROL=['control','non-targeting','non_targeting','nt','safe','aavs1','no_target','untreated','ctrl','vehicle']

def norm_name(x):
    s=str(x).strip().replace(' ','_').replace('/','_')
    return re.sub(r'[^A-Za-z0-9_.+-]+','_',s) or 'unknown'

def pick_col(obs, requested, candidates):
    if requested and requested!='auto': return requested
    cols=list(obs.columns); low={c.lower():c for c in cols}
    for k in candidates:
        if k in low: return low[k]
    for c in cols:
        cl=c.lower()
        if any(k in cl for k in candidates): return c
    raise ValueError(f'No matching column for {candidates}; available={cols[:30]}')

def pick_control(vals, requested):
    if requested and requested!='auto': return str(requested)
    counts=pd.Series(vals).astype(str).value_counts()
    # NaN / null = non-targeting control (common in CRISPR screens)
    nan_vals = [v for v in counts.index if v.lower() in ('nan', 'none', 'null', '')]
    if nan_vals:
        return str(nan_vals[0])
    for v in counts.index:
        vl=v.lower()
        if any(c in vl for c in CONTROL): return str(v)
    return str(counts.index[0])

def split_stratified(a, test_frac, val_frac, seed):
    rng=np.random.default_rng(seed); pert=a.obs['perturbation'].astype(str).values
    ct=a.obs['cell_type'].astype(str).values if 'cell_type' in a.obs else np.array(['unknown']*a.n_obs)
    split=np.array(['train']*a.n_obs, dtype=object)
    for key in sorted(set(zip(pert,ct))):
        idx=np.where((pert==key[0]) & (ct==key[1]))[0]
        if len(idx)<5: continue
        rng.shuffle(idx); n_test=max(1,int(len(idx)*test_frac)); n_val=max(1,int(len(idx)*val_frac)) if len(idx)>=10 else 0
        split[idx[:n_test]]='test'; split[idx[n_test:n_test+n_val]]='val'
    a.obs['split']=split
    return a

def select_hvg_train(a,n_hvg):
    if n_hvg<=0 or a.n_vars<=n_hvg: return a, list(a.var_names)
    train=np.asarray(a.obs['split']=='train')
    X=a.X[train]
    if issparse(X):
        mean=np.asarray(X.mean(axis=0)).ravel(); mean2=np.asarray(X.power(2).mean(axis=0)).ravel(); var=mean2-mean**2
    else:
        var=np.var(np.asarray(X),axis=0)
    idx=np.argsort(var)[-n_hvg:]
    genes=list(a.var_names[idx])
    return a[:,idx].copy(), genes

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--h5ad',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--pert_key',default='auto'); ap.add_argument('--condition_key',default='auto'); ap.add_argument('--control_label',default='auto')
    ap.add_argument('--top_perts',type=int,default=20); ap.add_argument('--min_cells_per_pert',type=int,default=80); ap.add_argument('--n_hvg',type=int,default=2000)
    ap.add_argument('--test_frac',type=float,default=0.2); ap.add_argument('--val_frac',type=float,default=0.1); ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--max_cells',type=int,default=0)
    ap.add_argument('--external_condition_mode',action='store_true'); ap.add_argument('--report',required=True)
    args=ap.parse_args(); a=ad.read_h5ad(args.h5ad)
    key=pick_col(a.obs, args.condition_key if args.external_condition_mode else args.pert_key, PERT_KEYS)
    raw=a.obs[key].astype(str).values; ctrl=pick_control(raw,args.control_label)
    perts=np.array([('control' if str(x)==ctrl or str(x).lower()==ctrl.lower() else norm_name(x)) for x in raw], dtype=object)
    a.obs['perturbation']=perts
    if 'cell_type' not in a.obs: a.obs['cell_type']='unknown'
    if 'batch' not in a.obs: a.obs['batch']='batch0'
    if not args.external_condition_mode:
        counts=pd.Series(perts).value_counts(); keep=['control']+[p for p,c in counts.items() if p!='control' and c>=args.min_cells_per_pert][:args.top_perts]
        a=a[np.isin(perts,keep)].copy(); a.obs['perturbation']=a.obs['perturbation'].astype(str)
    if args.max_cells>0 and a.n_obs>args.max_cells:
        rng=np.random.default_rng(args.seed)
        pert_vals=a.obs['perturbation'].astype(str).values
        keep_idx=[]
        for p in sorted(set(pert_vals)):
            p_idx=np.where(pert_vals==p)[0]; n_keep=max(1,int(len(p_idx)*args.max_cells/a.n_obs))
            keep_idx.extend(rng.choice(p_idx,size=min(n_keep,len(p_idx)),replace=False).tolist())
        a=a[keep_idx].copy()
    a=split_stratified(a,args.test_frac,args.val_frac,args.seed)
    a, genes=select_hvg_train(a,args.n_hvg if not args.external_condition_mode else min(1000,a.n_vars))
    counts=a.obs['split'].value_counts().to_dict(); pc=a.obs['perturbation'].value_counts().to_dict()
    a.uns['selected_var_names']=list(genes); a.uns['hvg_source']='train_only'; a.uns['split_info']={'mode':'stratified_cell','seed':args.seed,'test_frac':args.test_frac,'val_frac':args.val_frac,'hvg_from_train':True,'hvg_source':'train','group_means_source':'train','pert_key':key,'control_label':ctrl,'selected_genes':len(genes)}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); a.write_h5ad(args.out)
    summary={'input':args.h5ad,'output':args.out,'shape':[int(a.n_obs),int(a.n_vars)],'pert_key':key,'control_label':ctrl,'split_counts':counts,'perturbation_counts':pc,'hvg_from_train':True,'n_selected_genes':len(genes)}
    Path(args.report).parent.mkdir(parents=True,exist_ok=True); Path(args.report).write_text('# Preprocess Report\n\n```json\n'+json.dumps(summary,indent=2,ensure_ascii=False)+'\n```\n',encoding='utf-8')
    Path(str(args.report).replace('.md','.json')).write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__': main()
