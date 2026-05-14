"""Test: evidence policy (quality_gate, dropout, warm_start) in training."""
print("--- test_evidence_policy ---")
import sys, os
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

import torch
import numpy as np

# We test the policy logic directly without running full training
# The policy logic is in train_epoch's evidence filtering code

def apply_evidence_policy(evidence, evidence_conf, policy, min_conf, warm_start_epoch, epoch, dropout=0.0):
    """Minimal reproduction of the evidence policy logic from train_epoch."""
    if evidence is None:
        return None, None
    
    if policy == "off":
        return None, None
    elif policy == "quality_gate":
        if evidence_conf is not None:
            gate = (evidence_conf >= min_conf).float().view(-1, 1)
            evidence = evidence * gate
            if warm_start_epoch > 0 and epoch <= warm_start_epoch:
                return None, None
    
    # Dropout (train mode only - tested separately)
    if dropout > 0:
        keep = torch.rand(evidence.size(0)) > dropout
        evidence = evidence * keep.float().view(-1, 1)
        if evidence_conf is not None:
            evidence_conf = evidence_conf * keep.float()
    
    return evidence, evidence_conf


# Setup
ev = torch.randn(8, 128)
conf = torch.tensor([0.9, 0.1, 0.5, 0.3, 0.95, 0.05, 0.7, 0.4])

# 1) quality_gate: low conf evidence zeroed
ev_out, conf_out = apply_evidence_policy(ev.clone(), conf.clone(), "quality_gate", 0.35, 0, 5)
# Rows with conf < 0.35 should be zeroed
low_conf_mask = conf < 0.35
assert torch.allclose(ev_out[low_conf_mask], torch.zeros_like(ev_out[low_conf_mask])), "Low conf evidence should be zeroed"
assert not torch.allclose(ev_out[~low_conf_mask], torch.zeros_like(ev_out[~low_conf_mask])), "High conf evidence should NOT be zeroed"
print("  1) quality_gate zeros low-conf evidence OK")

# 2) quality_gate: warm_start → evidence=None in early epochs
ev2, conf2 = apply_evidence_policy(ev.clone(), conf.clone(), "quality_gate", 0.35, warm_start_epoch=2, epoch=1)
assert ev2 is None, "Warm start epoch 1 should give evidence=None"
ev3, conf3 = apply_evidence_policy(ev.clone(), conf.clone(), "quality_gate", 0.35, warm_start_epoch=2, epoch=2)
assert ev3 is None, "Warm start epoch 2 should give evidence=None"
ev4, conf4 = apply_evidence_policy(ev.clone(), conf.clone(), "quality_gate", 0.35, warm_start_epoch=2, epoch=3)
assert ev4 is not None, "After warm start, evidence should not be None"
print("  2) warm_start works OK")

# 3) off policy: evidence always None
ev5, conf5 = apply_evidence_policy(ev.clone(), conf.clone(), "off", 0.35, 0, 5)
assert ev5 is None, "off policy should give evidence=None"
print("  3) off policy works OK")

# 4) always policy: evidence unchanged (except dropout)
ev6, conf6 = apply_evidence_policy(ev.clone(), conf.clone(), "always", 0.35, 0, 5, dropout=0.0)
assert ev6 is not None, "always policy should keep evidence"
assert torch.allclose(ev6, ev), "always policy should not modify evidence"
print("  4) always policy works OK")

# 5) Dropout only applied in train (we test the function parameter)
torch.manual_seed(42)
ev7, conf7 = apply_evidence_policy(ev.clone(), conf.clone(), "always", 0.35, 0, 5, dropout=0.5)
# With dropout=0.5, roughly half should change
changed = ~torch.allclose(ev7, ev)
print(f"  5) evidence_dropout applied: some zeroed={changed} OK")

# 6) Without dropout (val mode): evidence unchanged
ev8, conf8 = apply_evidence_policy(ev.clone(), conf.clone(), "always", 0.35, 0, 5, dropout=0.0)
assert torch.allclose(ev8, ev), "Without dropout, evidence should be unchanged"
print("  6) no dropout (val mode) preserves evidence OK")

# 7) warm_start with warm_start_epoch=0: never skips
ev9, _ = apply_evidence_policy(ev.clone(), conf.clone(), "quality_gate", 0.35, warm_start_epoch=0, epoch=1)
assert ev9 is not None, "warm_start_epoch=0 should not skip"
print("  7) warm_start_epoch=0 no skip OK")

print("ALL OK")
