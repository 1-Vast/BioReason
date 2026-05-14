"""BioReason: run all verification checks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

tests = [
    "test_forward", "test_loss", "test_data", "test_infer",
    "test_llm_config", "test_control_input", "test_vocab_checkpoint",
    "test_gene_align", "test_target_latent_mask", "test_cov_dims",
    "test_device_flow", "test_loader_perf_flags",
    "test_progress_smoke", "test_infer_progress",
    "test_sparse_memory", "test_mmd_amp_stability",
    "test_atomic_checkpoint", "test_batchnorm_tail_batch",
    "test_infer_prealloc",
    "test_split_no_leak", "test_groupmean_no_leak", "test_target_latent_no_leak",
    "test_eval_split_only", "test_leak_audit",
    "test_clean_api", "test_reason_mode",
    "test_strict_no_leak",
    "test_llm_budget", "test_llm_cache",
    "test_prep_kb", "test_prep_text", "test_prep_evi", "test_prep_cli",
    "test_llm_json", "test_stage2_warm", "test_latent_only_bp",
    "test_stage3_init", "test_trust_gate",
    "test_structured_evidence", "test_gate_add_evidence",
    "test_evidence_policy", "test_evidence_conf_loss",
    "test_evidence_pert_embedding",
    "test_evi_contrast", "test_adaptive_evidence_gate",
]
passed, failed = 0, 0

for name in tests:
    sys.stdout.write(f"[....] {name}\r"); sys.stdout.flush()
    try:
        mod = __import__(f"tools.checks.{name}", fromlist=[""])
        passed += 1; print(f"[PASS] {name}")
    except SystemExit as e:
        if e.code == 0: passed += 1; print(f"[PASS] {name}")
        else: failed += 1; print(f"[FAIL] {name} (exit {e.code})")
    except Exception as e:
        failed += 1; print(f"[FAIL] {name}: {e}")

print(f"\n{'='*40}")
print(f"PASS: {passed}  FAIL: {failed}")
if failed: sys.exit(1)

