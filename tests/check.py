"""BioReason: run all verification tests."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

tests = [
    ("test_forward", "tests.test_forward"),
    ("test_loss", "tests.test_loss"),
    ("test_data", "tests.test_data"),
    ("test_infer", "tests.test_infer"),
    ("test_llm_config", "tests.test_llm_config"),
]

failed = []
for name, mod in tests:
    try:
        __import__(mod)
        print(f"\n{'='*40}")
    except SystemExit as e:
        if e.code != 0:
            failed.append(name)

if failed:
    print(f"\nFAILED: {failed}")
    sys.exit(1)
else:
    print(f"\nALL {len(tests)} TESTS PASSED")
