"""BioReason: run all verification tests."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

tests = [
    "test_forward", "test_loss", "test_data", "test_infer",
    "test_llm_config", "test_control_input", "test_vocab_checkpoint",
    "test_gene_align", "test_target_latent_mask", "test_cov_dims",
]
failed = []
for name in tests:
    try:
        mod = __import__(f"tests.{name}", fromlist=[""])
        print(f"\n{'='*40}")
    except SystemExit as e:
        if e.code != 0:
            failed.append(name)
    except Exception as e:
        print(f"  ERROR: {e}")
        failed.append(name)

if failed:
    print(f"\nFAILED: {failed}")
    sys.exit(1)
else:
    print(f"\nALL {len(tests)} TESTS PASSED")
