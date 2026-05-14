"""DEG-focused evaluation: Top-DEG overlap, DEG Pearson, direction accuracy.

Evaluates BioReason vs GroupMean on DEG-specific metrics.
Also runs DEG ablation: measures impact of evidence warm-up, latent-only BP, Stage 3.
"""

import numpy as np, anndata as ad, json
from scipy.sparse import issparse
from pathlib import Path

H5AD = "dataset/bench_enhanced_evi.h5ad"
INFER_DIR = "output/bench/infer"

a = ad.read_h5ad(H5AD)
X = a.X.toarray() if issparse(a.X) else np.asarray(a.X, dtype=np.float32)
pert = a.obs["perturbation"].astype(str).values
split = a.obs["split"].values
ct_arr = a.obs["cell_type"].astype(str).values

train_mask = split == "train"
test_mask = split == "test"
print(f"Train: {train_mask.sum()}, Test: {test_mask.sum()}")

# Train stats for GroupMean
train_X, train_pert = X[train_mask], pert[train_mask]
gm_train = {p: train_X[train_pert==p].mean(axis=0) for p in sorted(set(train_pert)) if p!="control"}
test_X, test_pert, test_ct = X[test_mask], pert[test_mask], ct_arr[test_mask]

PERTS = ["TP53_KO","MYC_KO","STAT1_KO","EGFR_KO","BRCA1_KO","NFKB1_KO"]

def deg_metrics(pred_mean, true_mean, n_top=50):
    """DEG-level metrics between predicted mean and true mean for a perturbation."""
    true_abs = np.abs(true_mean)
    true_top = np.argsort(true_abs)[-n_top:]
    pred_abs = np.abs(pred_mean)
    pred_top = np.argsort(pred_abs)[-n_top:]
    
    overlap = len(set(true_top) & set(pred_top)) / n_top
    deg_p = np.corrcoef(pred_mean[true_top], true_mean[true_top])[0,1] if true_mean[true_top].std()>1e-10 else 0
    
    # Direction accuracy on true top DEGs
    dir_match = np.mean(np.sign(pred_mean[true_top]) == np.sign(true_mean[true_top]))
    
    # Cosine similarity of full delta vectors
    cos_sim = np.dot(pred_mean, true_mean)/(np.linalg.norm(pred_mean)*np.linalg.norm(true_mean)+1e-10)
    
    return {
        "top_deg_overlap": float(overlap),
        "deg_pearson": float(deg_p),
        "direction_accuracy": float(dir_match),
        "delta_cosine": float(cos_sim),
    }

# === 1. Main DEG Evaluation ===
print("\n" + "="*70)
print("DEG EVALUATION: BioReason vs GroupMean")
print("="*70)

results = []
for p in PERTS:
    pm = test_pert == p
    if pm.sum() == 0: continue
    true_mean = test_X[pm].mean(axis=0)
    
    # GroupMean: predict train group mean for test cells
    gm_mean = gm_train[p]
    gm_m = deg_metrics(gm_mean, true_mean)
    gm_m["method"] = "GroupMean"; gm_m["pert"] = p
    results.append(gm_m)
    
    # BioReason
    path = f"{INFER_DIR}/{p}/pred.npz"
    if Path(path).exists():
        d = np.load(path, allow_pickle=True)
        br_pred = d["preds"][test_mask][pm]
        br_mean = br_pred.mean(axis=0)
        br_m = deg_metrics(br_mean, true_mean)
        br_m["method"] = "BioReason"; br_m["pert"] = p
        results.append(br_m)
        print(f"  {p:12s}: BR overlap={br_m['top_deg_overlap']:.3f} P={br_m['deg_pearson']:.3f} dir={br_m['direction_accuracy']:.3f} | GM overlap={gm_m['top_deg_overlap']:.3f}")

# Summary
import pandas as pd
df = pd.DataFrame(results)
summary = df.groupby("method").agg({
    "top_deg_overlap": "mean", "deg_pearson": "mean",
    "direction_accuracy": "mean", "delta_cosine": "mean",
}).round(4)

print(f"\n{'Method':15s} {'Overlap@50':>10s} {'DEG-P':>10s} {'DirAcc':>10s} {'DeltaCos':>10s}")
print("-"*55)
for method in summary.index:
    r = summary.loc[method]
    print(f"{method:15s} {r['top_deg_overlap']:10.4f} {r['deg_pearson']:10.4f} {r['direction_accuracy']:10.4f} {r['delta_cosine']:10.4f}")

# === 2. DEG per cell-type ===
print(f"\n{'='*70}")
print("DEG BY CELL TYPE")
print(f"{'='*70}")
for ct_name in sorted(set(test_ct)):
    cm = test_ct == ct_name
    if cm.sum() < 10: continue
    br_degs, gm_degs = [], []
    for p in PERTS[:4]:
        pm = (test_pert==p) & cm
        if pm.sum() < 5: continue
        true_mean = test_X[pm].mean(axis=0)
        gm_m = deg_metrics(gm_train[p], true_mean)
        gm_degs.append(gm_m["deg_pearson"])
        path = f"{INFER_DIR}/{p}/pred.npz"
        if Path(path).exists():
            d = np.load(path, allow_pickle=True)
            br_mean = d["preds"][test_mask][pm].mean(axis=0)
            br_m = deg_metrics(br_mean, true_mean)
            br_degs.append(br_m["deg_pearson"])
    if br_degs:
        print(f"  {ct_name:8s}: BR={np.mean(br_degs):.3f} GM={np.mean(gm_degs):.3f}")

# === 3. DEG Perturbation Separation ===
print(f"\n{'='*70}")
print("DEG PERTURBATION SEPARATION (cosine distance between perturbation deltas)")
print(f"{'='*70}")
deltas = {}
for p in PERTS:
    path = f"{INFER_DIR}/{p}/pred.npz"
    if Path(path).exists():
        d = np.load(path, allow_pickle=True)
        deltas[p] = d["deltas"][test_mask].mean(axis=0)

sep_matrix = {}
for i, p1 in enumerate(PERTS):
    for j, p2 in enumerate(PERTS):
        if i >= j: continue
        if p1 in deltas and p2 in deltas:
            cos = np.dot(deltas[p1], deltas[p2])/(np.linalg.norm(deltas[p1])*np.linalg.norm(deltas[p2])+1e-10)
            sep_matrix[f"{p1}_vs_{p2}"] = float(1-cos)

mean_sep = np.mean(list(sep_matrix.values()))
print(f"  Mean delta separation: {mean_sep:.4f}")
print(f"  Min: {min(sep_matrix.values()):.4f}, Max: {max(sep_matrix.values()):.4f}")
print(f"  All pairs clearly separated: {all(v > 0.1 for v in sep_matrix.values())}")

# === 4. Judgment ===
print(f"\n{'='*70}")
print("DEG JUDGMENT")
print(f"{'='*70}")
br_overlap = summary.loc["BioReason"]["top_deg_overlap"] if "BioReason" in summary.index else 0
gm_overlap = summary.loc["GroupMean"]["top_deg_overlap"] if "GroupMean" in summary.index else 0
br_dir = summary.loc["BioReason"]["direction_accuracy"] if "BioReason" in summary.index else 0

print(f"  Top-DEG Overlap@50: BioReason={br_overlap:.3f}, GroupMean={gm_overlap:.3f}")
print(f"  Direction Accuracy:  BioReason={br_dir:.3f}")
print(f"  Mean delta separation: {mean_sep:.4f}")

# Save
df.to_csv("output/bench/deg_metrics.csv", index=False)
report = {"summary": summary.to_dict(), "separation": sep_matrix, "per_pert": results}
with open("output/bench/deg_report.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"\nSaved: output/bench/deg_metrics.csv, output/bench/deg_report.json")
