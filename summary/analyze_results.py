import csv, os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

base = os.path.join(os.environ['USERPROFILE'], 'Downloads', 'paper2', 'paper2-Thiện', 'FedTGen_Standalone_Experiments', 'FedTGen_Standalone_Experiments')

key_cols = [
    'experiment_id', 'method', 'model_name', 'dataset_name', 'training_setting', 'tuning_mode',
    'test_bleu', 'test_rouge1', 'test_rouge2', 'test_rougeL',
    'test_bertscore_f1', 'test_bertscore_precision', 'test_bertscore_recall',
    'trainable_params', 'total_params', 'trainable_ratio', 'adapter_params', 'adapter_size_MB',
    'communication_MB_per_round_average', 'upload_MB_per_round_average', 'download_MB_per_round_average',
    'total_communication_MB', 'total_communication_GB',
    'communication_reduction_per_round_vs_full_percent',
    'communication_reduction_total_vs_full_same_rounds_percent',
    'training_time_seconds', 'total_experiment_time_seconds', 'average_time_per_round_seconds',
    'inference_latency_mean_ms_per_sample', 'inference_latency_median_ms_per_sample',
    'inference_latency_p95_ms_per_sample', 'inference_throughput_samples_per_second',
    'inference_throughput_tokens_per_second',
    'max_vram_allocated_MB', 'cpu_peak_memory_MB',
    'estimated_total_training_flops', 'forward_flops_per_sample',
    'lora_r', 'lora_alpha', 'lora_dropout', 'lora_target_modules',
    'local_epochs', 'participation_rate', 'num_clients', 'dirichlet_alpha',
    'num_train_samples', 'completed_rounds', 'best_round', 'early_stopped',
    'client_test_bleu_mean', 'client_test_bleu_std',
    'client_test_rougeL_mean', 'client_test_rougeL_std',
    'client_test_rougeL_min', 'client_test_rougeL_max',
    'client_test_bertscore_f1_mean', 'client_test_bertscore_f1_std',
    'client_update_divergence_mean', 'client_update_divergence_std',
    'model_size_MB', 'checkpoint_size_MB',
    'reference_full_MB_per_round',
    'unmerged_inference_latency_mean_ms', 'merged_inference_latency_mean_ms',
    'average_input_tokens', 'average_generated_tokens',
    'fedprox_mu', 'lambda_n', 'beta_loss', 'beta_divergence', 'beta_quality',
    'mean_pairwise_client_js_divergence',
    'aggregation_fallback_count', 'aggregation_fallback_rate',
    'average_aggregation_weight_entropy',
]

rows = []
all_headers = None
dirs = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])
for d in dirs:
    csv_path = os.path.join(base, d, 'results', 'final_results.csv')
    if not os.path.exists(csv_path):
        print(f'MISSING: {d}')
        continue
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if all_headers is None:
                all_headers = list(row.keys())
            r = {k: row.get(k, '') for k in key_cols}
            rows.append(r)

print(f'Total experiments: {len(rows)}')
print()
print('Summary:')
for r in rows:
    eid = r['experiment_id']
    method = r['method']
    ds = r['dataset_name'].split('/')[-1] if '/' in r['dataset_name'] else r['dataset_name']
    bleu = r['test_bleu'][:7] if r['test_bleu'] else 'N/A'
    rougeL = r['test_rougeL'][:7] if r['test_rougeL'] else 'N/A'
    bs = r['test_bertscore_f1'][:7] if r['test_bertscore_f1'] else 'N/A'
    trainable = r['trainable_params']
    comm = r['communication_MB_per_round_average'][:7] if r['communication_MB_per_round_average'] else 'N/A'
    comm_red = r['communication_reduction_per_round_vs_full_percent'][:6] if r['communication_reduction_per_round_vs_full_percent'] else 'N/A'
    print(f"{eid:42s} {method:20s} {ds:15s} BLEU={bleu} ROUGE-L={rougeL} BS-F1={bs} trainable={trainable:12s} comm_MB/round={comm} comm_red={comm_red}%")

# Save the focused CSV
out_path = os.path.join(os.environ['USERPROFILE'], 'Downloads', 'paper2', 'paper2-Thiện',
                        'FedTGen_Standalone_Experiments', 'all_results_paper_focused.csv')
with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=key_cols)
    writer.writeheader()
    writer.writerows(rows)
print()
print(f'Saved paper-focused CSV ({len(key_cols)} columns) to: all_results_paper_focused.csv')
