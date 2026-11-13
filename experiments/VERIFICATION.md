# Verification report

## Structure
- 41 experiment folders (32 Main + 7 Ablation + 2 Robustness), IDs unique.
- Each folder has: `<ID>.py`, `<ID>.ipynb`, `requirements.txt`, `README.md`.
- No `fedtgen_core.py`; no shared config/partition/checkpoint; no cross-experiment imports.
- No aggregate result files (`all_experiments.csv`, `main_results.csv`,
  `summary_all_experiments.csv`, `paper_results.xlsx`, `summary_all_seeds.csv`).
- Each experiment writes its own single-row `final_results.csv` (206-column schema).

## Static checks (run in the build sandbox)
- `python -m compileall FedTGen_Standalone_Experiments` -> **passed** (all 41 `.py` compile).
- Self-consistency audit -> **0 errors, 0 warnings**:
  - 41 notebooks, 41 `.py`, 41 `requirements.txt`, 41 `README.md`, 41 unique IDs.
  - No `TODO` / `FIXME` / `NotImplementedError` / bare `pass`.
  - Ablations change exactly one variable each (A01 model, A02/A03 `lora_r`,
    A04/A05 `local_epochs`, A06 `participation_rate`, A07 `num_clients`);
    R01/R02 change `dirichlet_alpha` only. A01 uses `google/flan-t5-base`.
  - `final_results.csv` schema present with 206 unique columns; every notebook valid nbformat 4.
- Model-independent logic test -> **LOGIC_OK**:
  - Partition checksum is deterministic for identical (data, K, alpha, seed) — so
    experiments sharing dataset+K+alpha reproduce the same partition checksum.
  - alpha 0.1 / 0.5 / 1.0 yield distinct partitions (M29 vs R01 vs R02).
  - FedTGen quality-aware weights are finite, sum to 1, increase with generation
    quality, and fall back to sample-weighted FedAvg on NaN/Inf.
  - Communication-reduction math validated (example LoRA vs full = 98.85%).

## Fixed issues
- v2: `DataCollatorForSeq2Seq` crash (`too many dimensions 'str'`) — tokenized items
  now contain only numeric fields (`input_ids`, `attention_mask`, `labels`); reference/
  input texts are read from the examples list. Tokenizer cache bumped to `tok_v2_*`.
- v3: `datasets>=4.0` removed script-based datasets (`RuntimeError: Dataset scripts are
  no longer supported, but found qg_squad.py`). The install now pins `datasets>=2.16,<4.0`,
  `load_dataset` retries with `trust_remote_code=True`, and SQuAD-QG falls back to the
  script-free `rajpurkar/squad` (validation used as test; train split 90/10 with seed 42)
  if `lmqg/qg_squad` is unavailable.
- v4: FULL-DATA mode — dataset caps removed (`max_train_samples`/`max_validation_samples`/
  `max_test_samples = None`). Every experiment now trains and evaluates on the ENTIRE
  dataset. Expect much longer runtimes (QQP/SQuAD are large); rely on checkpoint/resume.

## What could NOT be run in the build sandbox
The build sandbox has **no GPU and no network access to Hugging Face**, so the
train/download path (model + dataset downloads, BERTScore) cannot execute here.
Run the end-to-end smoke test on Colab or any GPU machine with HF access:

```bash
python run_smoke_test.py        # one experiment per family, SMOKE mode, ~minutes
```

It runs each of the 8 method families end-to-end on a tiny subset (2 clients, 1
round, tiny BERTScore model), writing to a throwaway `_smoke_out/` folder, and
asserts a valid one-row `final_results.csv` with positive inference latency for
each. A single full experiment is just `python <ID>.py` or `Runtime -> Run all`
in its notebook.
