# A01_centralized_base_full_samsum - final_results.csv (reconstructed 2026-09-28)

The original final_results.csv was lost (partial Google-Drive zip; Drive data deleted). Rebuilt with the same formulas as assemble_final_results() from the surviving artefacts below. Columns blank by design for centralized_full (LoRA / federated / per-client) are blank as in M05.

## Sources
- Test metrics, start/end time: run_summary.json (first, correct run: early stop at epoch 10, best epoch 7).
- Validation-at-best, training/generation time, VRAM: history_partial.csv (epochs 1-10).
- Latency (200 samples): latency_details.csv; latency_total_time = sum of per-sample times (excludes <0.1% loop overhead).
- Environment: environment.json; params 247,577,856 (notebook log); FLOPs: analytical 2ND formula (equals flops_results.csv).
- Dataset/partition/hash/client-stat columns: identical in every SAMSum run (verified M05, M09, M13).
- total_experiment_time_seconds = 7503.6 from the notebook log (code measures the final session only).
- resume_count: Colab sessions detected from elapsed-time resets in resource_history.csv.
- APPROXIMATE: cpu_peak_memory_MB = max cpu_memory_MB in resource_history.csv (code records RSS at run end); checkpoint_size_MB = 2 x epoch_0009 state.pt.

## Columns left blank (value not recoverable)
- (none)
