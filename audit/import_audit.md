# Phase 3 — Import Audit

**Method:** read every module's imports; ran the project's own `scripts/import_audit.py` mentally against the full file set; ran the full `pytest` suite (imports all core modules — **29 passed, 1 skipped**, verified).

## 3.1 The project's own import audit is incomplete — MEDIUM

`scripts/import_audit.py:17-27` tests only 9 modules. It **omits**:
- `evaluation/oracle_coherence.py`
- `evaluation/intrinsic_noise.py`
- `evaluation/plotter.py`
- **every `scripts/run_phase_*.py`**

So a green "Audit passed!" gives false assurance — the modules most likely to break (the phase scripts, with their `.cuda()` and hardcoded dims) are never imported. **Fix:** discover modules with `pkgutil.walk_packages` (already imported but unused) instead of a hand list.

## 3.2 Latent missing import, masked by `from __future__ import annotations` — LOW

`evaluation/intrinsic_noise.py`:
```
8   from __future__ import annotations
12  from typing import Dict, List, Tuple        # <-- Optional NOT imported
23  def get_symbolic_states(traj) -> List[Optional[Tuple[...]]]:   # uses Optional
```
`Optional` is referenced in the return annotation but never imported. It does **not** crash today because `from __future__ import annotations` turns annotations into un-evaluated strings. But:
- It will raise `NameError` the moment anyone calls `typing.get_type_hints()` on this module, or removes the `__future__` import.
- `get_symbolic_states` is imported by 5 phase scripts, so the latent bug is widely fanned out.

**Fix:** add `Optional` to the import.

## 3.3 `sys.path.insert` hacking — LOW (works, but fragile)

Every `scripts/*.py` begins with a duplicated
```python
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
```
injected by `inject_sys_path.py`. In `run_phase_a.py` this produced **duplicate `import os`/`import sys`** (lines 19-26) and a doubled insert. It works, but the package is `pip install -e .`-able (`pyproject.toml` declares the packages), which would make all of this unnecessary. `scripts/` is **not** in the installed packages list (`pyproject.toml` `[tool.setuptools.packages.find]` includes only `data_processing, models, training, evaluation`), so `from scripts.run_phase_b import ...` (used in `run_phase_a.py:406`) relies on the path hack and on `scripts/__init__.py` existing (it does).

## 3.4 No circular imports — OK

Dependency direction is clean: `scripts → training/evaluation → models/data_processing → (torch, sklearn)`. `run_phase_a.py:406` imports from `scripts.run_phase_b` but `run_phase_b` does not import `run_phase_a`, so no cycle. Verified by reading.

## 3.5 Heavy optional deps imported unconditionally — LOW

- `models/model_loader.py:11` imports `peft` and `:9 bitsandbytes` (`BitsAndBytesConfig`) at module top. `bitsandbytes` frequently fails to import on CPU/Windows. Since `run_phase_a` imports `load_model` from this module, a broken `bitsandbytes`/`peft` install **blocks Phase A even though Phase A never uses quantization.** Risk realized on any machine without a working bnb. **Fix:** lazy-import bnb/peft inside `load_qlora_model`/`load_checkpoint`.
- `seaborn` imported in `intrinsic_noise.py:267` (inside function — good) and `run_phase_c1_advanced.py:13` (top-level — adds a hard dep not in some envs).

## 3.6 Verdict

No active import crash in the core path (pytest proves it). But: the self-audit tool covers the wrong subset, a missing `Optional` is one refactor away from breaking 5 scripts, and top-level `bitsandbytes` import gates the entire pipeline on a notoriously fragile dependency the experiment doesn't even use.
