"""Batch inference for BioReason model."""
import argparse, json
from pathlib import Path
import numpy as np, torch, anndata as ad

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5ad', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--config')
    ap.add_argument('--split', default='all')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch_size', type=int, default=512)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.infer import load_model
    from models.data import PertDataset, build_loader, align_adata_to_genes
    from utils.device import move_to_device

    model, cfg = load_model(args.checkpoint, device=args.device)
    meta = cfg.get('data_meta', {})
    a = ad.read_h5ad(args.h5ad)
    
    # Gene alignment
    genes = meta.get('selected_var_names')
    if genes:
        a = align_adata_to_genes(a, genes)
    
    # Split filter
    if args.split != 'all' and 'split' in a.obs:
        a = a[a.obs['split'].astype(str).values == args.split].copy()
    
    ds = PertDataset(a, use_hvg=False)
    if meta.get('pert_to_id'):
        ds.set_vocab(meta['pert_to_id'], meta.get('id_to_pert'))
    
    # Safe cov mapping: clip OOV covariate IDs to 0
    train_cov_dims = meta.get('cov_dims', {})
    if train_cov_dims:
        for key, max_dim in train_cov_dims.items():
            if key in ds.cov_maps:
                for k in ds.cov_maps[key]:
                    ds.cov_maps[key][k] = min(ds.cov_maps[key][k], max_dim - 1)
    
    loader = build_loader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    model.to(device).eval()
    
    preds, deltas, lat = [], [], []
    idx, pids, pstr = [], [], []
    
    with torch.inference_mode():
        for b in loader:
            bd = move_to_device(b, device)
            out = model(bd['x'], bd['pert'], cov=bd.get('cov'), evidence=bd.get('evidence'), return_latent=True)
            preds.append(out['pred'].float().cpu().numpy())
            deltas.append(out['delta'].float().cpu().numpy())
            lat.append(out['latent'].float().cpu().numpy())
            idx += [m.get('idx', len(idx)) for m in b.get('meta', [])]
            pids += bd['pert'].cpu().tolist()
            pstr += b.get('pert_str', [])
    
    Path(args.out).mkdir(parents=True, exist_ok=True)
    np.savez(
        Path(args.out) / 'pred.npz',
        preds=np.vstack(preds), deltas=np.vstack(deltas),
        latents=np.vstack(lat), indices=np.array(idx),
        pert=np.array(pids), pert_str=np.array(pstr),
    )
    print(json.dumps({'cells': len(idx), 'out': str(Path(args.out) / 'pred.npz')}))

if __name__ == '__main__':
    import sys
    main()
