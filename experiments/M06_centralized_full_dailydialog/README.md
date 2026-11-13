# M06_centralized_full_dailydialog

Standalone FedTGen experiment. **Only seed 42.** No cross-experiment imports, no
`fedtgen_core.py`, no shared config/partition/checkpoint, no aggregate result files.

| Field | Value |
|-------|-------|
| Experiment ID | `M06_centralized_full_dailydialog` |
| Research question(s) | RQ1,RQ3 |
| Method family | `centralized_full` |
| Training setting | `centralized` |
| Tuning mode | `full` |
| Aggregation | `none` |
| Proposed method | False |
| Model | `google/flan-t5-small` |
| Dataset key | `dailydialog` |
| Changed variable vs baseline | n/a (baseline) |
| Overrides applied | `none (BASE_CONFIG)` |

## How to run (Google Colab)
1. Upload `M06_centralized_full_dailydialog.ipynb` to Colab, select a GPU runtime (T4 is enough).
2. `Runtime -> Run all`. The first cell removes `torchao` and installs dependencies;
   the second mounts Google Drive; the third defines all code; the fourth runs it.

## How to run (plain Python)
```bash
pip install -r requirements.txt
python M06_centralized_full_dailydialog.py
```
Outputs default to `/content/drive/MyDrive/fedtgen_experiments/M06_centralized_full_dailydialog/seed_42/` on
Colab, or `./fedtgen_experiments/M06_centralized_full_dailydialog/seed_42/` locally. Override with the
`FEDTGEN_OUTPUT_ROOT` environment variable.

## Resume
Checkpoints are written to `.../seed_42/checkpoints/` after every round (federated),
epoch (centralized) or client-epoch (local). Only the two newest are kept. Re-running
the script or notebook auto-resumes from the latest valid checkpoint (falling back to
the previous one if the newest is corrupt); completed rounds are never recomputed.

## Output CSVs
`history.csv`, `client_history.csv`, `communication_history.csv`, `resource_history.csv`,
`latency_details.csv`, `flops_results.csv`, `test_predictions.csv`, and the single-row
`final_results.csv` (plus JSON siblings, `best_model/`, `final_model/`, `checkpoints/`,
`logs/`).

## Reading inference latency
`final_results.csv` reports `inference_latency_mean_ms_per_sample`,
`inference_latency_p95_ms_per_sample`, and throughput. For LoRA methods latency is
measured both unmerged and after `merge_and_unload()`; the **merged** value is the
primary result (`primary_latency_source = merged_lora`). If merge fails it falls back to
`unmerged_lora_fallback`. Local uses `weighted_local_models`; full FT uses `full_model`.

## Notes
- `secure_aggregation_implemented = False` (no cryptographic secure aggregation).
- BLEU is on the 0-100 scale; ROUGE and BERTScore are on 0-1.
- Dataset caps (`max_train_samples`, etc.) are recorded transparently in `config.json`
  so runs finish on Colab; set them to `null` in `CONFIG_OVERRIDES` for the full data.
