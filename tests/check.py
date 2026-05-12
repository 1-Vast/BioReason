"""BioReason: run all verification tests."""
import sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

tests = [
    "test_forward", "test_loss", "test_data", "test_infer",
    "test_llm_config", "test_control_input", "test_vocab_checkpoint",
    "test_gene_align", "test_target_latent_mask", "test_cov_dims",
    "test_device_flow", "test_loader_perf_flags",
    "test_progress_smoke", "test_infer_progress",
    "test_sparse_memory", "test_mmd_amp_stability",
    "test_atomic_checkpoint", "test_batchnorm_tail_batch",
    "test_infer_prealloc",
    "test_prior_kb", "test_prior_encode", "test_prior_pipeline",
    "test_llm_json", "test_trust_gate",
]
passed, failed = 0, 0

for name in tests:
    sys.stdout.write(f"[....] {name}\r"); sys.stdout.flush()
    try:
        mod = __import__(f"tests.{name}", fromlist=[""])
        passed += 1; print(f"[PASS] {name}")
    except SystemExit as e:
        if e.code == 0: passed += 1; print(f"[PASS] {name}")
        else: failed += 1; print(f"[FAIL] {name} (exit {e.code})")
    except Exception as e:
        failed += 1; print(f"[FAIL] {name}: {e}")

print(f"\n{'='*40}")
print(f"PASS: {passed}  FAIL: {failed}")
if failed: sys.exit(1)
