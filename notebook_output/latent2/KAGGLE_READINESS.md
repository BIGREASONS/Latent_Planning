# KAGGLE_READINESS.md

Execution readiness for a large-scale **Phase C.4A** run on Kaggle.

> Nothing in this document was executed during the build/verify pass. These are
> the steps to run **on Kaggle**, in order. Steps 1–2 invoke TinyLlama and were
> intentionally NOT run here.

## Run path / dependency chain

```
generate_countdown_dataset.py                 (CPU, no model)
        │  data/{train,val,test,test_ood}.jsonl
        ▼
run_phase_a.py  (TinyLlama forward passes)     ← produces trajectories
        │  <reports_dir>/trajectories/{train,val,test}.pt
        ▼
run_phase_c4a_pca_ablation.py                  (CPU/sklearn, no model)
        │  reads <reports_dir>/trajectories/train.pt   (splits it 80/20 internally)
        ▼
<reports_dir>/phase_c4a_*.csv, phase_c4a_pca_ablation_report.md,
              phase_c4a_controls.csv, coverage_report.md
```

C.4A consumes **only `train.pt`** and splits it into a retrieval database (train)
and query set (test) by `--train_frac`. It does not read `test.pt`. All paths are
repo-relative and rooted at `--reports_dir`.

## Step 1 — Generate data (CPU, no model)

```bash
python scripts/generate_countdown_dataset.py \
  --num_train 50000 --num_val 2000 --num_test 2000 --num_test_ood 2000 \
  --seed 0 --min_target 100 --max_target 999 \
  --output_dir data
```
Reproducible (seeded), targets bounded to 100–999, includes integer division.

## Step 2 — Extract trajectories (TinyLlama; GPU recommended)

```bash
python scripts/run_phase_a.py \
  --model TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
  --data_dir data --out_dir reports --layer -1 --batch_size 64
```
Writes `reports/trajectories/{train,val,test}.pt`. Device is auto-selected
(`cuda` if available, else CPU). Do NOT pass `--smoke` for a real run; leave
`--cap` unset to use the full dataset.

> If you only need trajectories (not the full Phase A diagnostics), this is still
> the script that produces them — there is no standalone extraction entry point
> wired to the same `trajectories/` layout that C.4A reads.

## Step 3 — Run Phase C.4A (CPU/sklearn, no model)

```bash
python scripts/run_phase_c4a_pca_ablation.py \
  --reports_dir reports \
  --max_trajectories 0 \
  --train_frac 0.8 \
  --seed 0
```

### CLI arguments

| Arg | Default | Purpose |
|---|---|---|
| `--reports_dir` | `reports` | Holds `trajectories/` and receives outputs. On Kaggle use e.g. `/kaggle/working/reports`. |
| `--max_trajectories` | `10000` | Cap on trajectories loaded from `train.pt`; **set `0` to use all** for a large-scale run. |
| `--train_frac` | `0.8` | Fraction used as retrieval DB (train); remainder is the query set. |
| `--seed` | `0` | Seed for the permutation-null control. |
| `--smoke` | off | Debug only — limits to 500 trajectories. |

### Outputs

- `phase_c4a_pca_ablation_metrics.csv` — Depth/Probe A/Probe C across PC ablations
- `phase_c4a_retrieval.csv` — Top-1/5/10 + gain across ablations
- `phase_c4a_geometry.csv` — within/between ratio
- `phase_c4a_controls.csv` — permutation null + raw-operand baseline
- `coverage_report.md` — train/test state coverage + **prefix-collision count**
- `phase_c4a_pca_ablation_report.md` — full report incl. Controls table

## Path / portability notes

- No absolute or Windows-specific paths in code; everything uses `os.path.join`
  rooted at `--reports_dir` / `--output_dir`.
- On Kaggle, the writable dir is `/kaggle/working`. Point `--out_dir` (Step 2)
  and `--reports_dir` (Step 3) at the same location, e.g.
  `/kaggle/working/reports`.
- `configs/kaggle.yaml` uses repo-relative paths (Phase A oriented); C.4A does
  not read it — pass C.4A settings via CLI.

## Build verification completed (this pass)

- `compileall` over all packages → OK
- `import scripts.run_phase_c4a_pca_ablation`, `import scripts.generate_countdown_dataset` → OK
- `run_phase_c4a_pca_ablation.py --help` → new args present
- Generator model-free smoke → deterministic, bounded, division present, all steps parse
- `pytest tests/test_dataset.py tests/test_action_parser.py` → 13 passed

## Pre-run checklist for Kaggle

- [ ] HuggingFace access to TinyLlama available (Step 2 downloads it)
- [ ] `--out_dir` (Step 2) == `--reports_dir` (Step 3), both writable
- [ ] `--max_trajectories 0` for the full-scale run
- [ ] Inspect `coverage_report.md`: non-trivial **test coverage** (smoke data gave
      0%) and the **prefix-collision count** before trusting retrieval numbers
- [ ] Read C.4A retrieval/probe numbers against the **controls** row (permutation
      null = noise floor; raw-operand baseline = trivial-decodability bar)

## Known issues that do NOT block C.4A

See `BUILD_STATUS.md` §4 — Phase D `.cuda()` and Phase C.3 PCA leakage are off the
C.4A path and were left unchanged.
