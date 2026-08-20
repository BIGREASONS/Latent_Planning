import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

"""Script to audit all internal project imports."""

import sys
import os
import importlib
import pkgutil


def main():
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

    modules_to_test = [
        "data_processing.action_parser",
        "data_processing.trajectory_dataset",
        "evaluation.coherence",
        "evaluation.probes",
        "models.diagnostic_decoder",
        "models.model_loader",
        "models.transition_model",
        "training.train_decoder",
        "training.train_transition",
    ]

    failed = 0
    for mod in modules_to_test:
        try:
            importlib.import_module(mod)
            print(f"[OK] Successfully imported: {mod}")
        except Exception as e:
            print(f"[FAIL] Failed to import: {mod}")
            print(f"       Error: {type(e).__name__}: {str(e)}")
            failed += 1

    if failed > 0:
        print(f"\nAudit failed. {failed} module(s) could not be imported.")
        sys.exit(1)
    else:
        print("\nAudit passed! All internal modules imported successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
