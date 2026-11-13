"""
Build paper-ready CSVs from experiment results.
Based on outline analysis of 1-Outline-FedTGen-Aug10.pdf:

Tables needed:
1. Main Results Table (5.2) - BLEU, ROUGE-1/2/L, BERTScore, Trainable Params, Comm Cost, Training Time
   → per method, per dataset
2. Communication Efficiency Table (5.2)
   → MB/round, total_MB, comm_reduction%, total params, adapter params
3. Computational Efficiency Table (5.2)
   → trainable_params, VRAM, training_time, FLOPs, inference_latency
4. Ablation: SLM vs Larger LM (A01 vs M05/M09 vs M29)
5. Ablation: Full vs PEFT (M13 vs M17 vs M29, across datasets)
6. Ablation: LoRA Rank (A02 vs M29/default vs A03)
7. Ablation: Local Epochs (A04 vs M29/default vs A05)
8. Ablation: Client Participation (A06 vs M29/default)
9. Ablation: Num Clients (A07 vs M29/default)
10. Robustness under non-IID (R01 vs M29 vs R02 - different alpha)
"""

import csv, os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

base = os.path.join(os.environ['USERPROFILE'], 'Downloads', 'paper2', 'paper2-Thiện', 'FedTGen_Standalone_Experiments', 'FedTGen_Standalone_Experiments')
out_dir = os.path.join(os.environ['USERPROFILE'], 'Downloads', 'paper2', 'paper2-Thiện', 'FedTGen_Standalone_Experiments')

# ---- Load all experiments ----
all_rows = {}
dirs = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])
for d in dirs:
    csv_path = os.path.join(base, d, 'results', 'final_results.csv')
    if not os.path.exists(csv_path):
        continue
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_rows[row['experiment_id']] = row

print(f"Loaded {len(all_rows)} experiments")

def get(row, key, default=''):
    v = row.get(key, default)
    return v if v not in (None, 'None', '') else default

def fmt_float(val, decimals=4):
    try:
        return round(float(val), decimals)
    except:
        return val

# ========================================================
# TABLE 1: Main Results - Quality Metrics per Method+Dataset
# ========================================================
main_cols = [
    'experiment_id', 'group', 'method_label', 'model_label', 'training_setting', 'tuning_mode',
    'dataset', 'task',
    'test_bleu', 'test_rouge1', 'test_rouge2', 'test_rougeL',
    'test_bertscore_precision', 'test_bertscore_recall', 'test_bertscore_f1',
    'trainable_params', 'total_params', 'trainable_ratio_pct',
    'communication_MB_per_round', 'total_communication_MB',
    'communication_reduction_pct',
    'training_time_seconds', 'inference_latency_mean_ms',
    'completed_rounds', 'best_round',
    'lora_r', 'local_epochs', 'participation_rate', 'num_clients', 'dirichlet_alpha',
]

# Method label mapping
method_labels = {
    'local_lora':       'Local-LoRA',
    'centralized_full': 'Centralized-Full',
    'centralized_lora': 'Centralized-LoRA',
    'fedavg_full':      'FedAvg-Full',
    'fedavg_lora':      'FedAvg-LoRA',
    'fedprox_lora':     'FedProx-LoRA',
    'scaffold_lora':    'SCAFFOLD-LoRA',
    'fedtgen':          'FedTGen (Ours)',
}

# Dataset label mapping
dataset_labels = {
    'knkarthick/samsum': 'SAMSum (Summarization)',
    'roskoN/dailydialog': 'DailyDialog (Dialogue Gen.)',
    'lmqg/qg_squad':     'SQuAD-QG (Question Gen.)',
    'nyu-mll/glue':      'QQP (Paraphrase Gen.)',
}

# Model label
model_labels = {
    'google/flan-t5-small': 'Flan-T5-Small (77M)',
    'google/flan-t5-base':  'Flan-T5-Base (248M)',
}

# Group
group_map = {
    'A': 'Ablation',
    'M': 'Main',
    'R': 'Robustness',
}

main_rows = []
for eid, row in all_rows.items():
    prefix = eid[0]
    method = get(row, 'method')
    ds = get(row, 'dataset_name')
    model = get(row, 'model_name')

    r = {
        'experiment_id': eid,
        'group': group_map.get(prefix, prefix),
        'method_label': method_labels.get(method, method),
        'model_label': model_labels.get(model, model),
        'training_setting': get(row, 'training_setting'),
        'tuning_mode': get(row, 'tuning_mode'),
        'dataset': dataset_labels.get(ds, ds.split('/')[-1]),
        'task': get(row, 'task_name'),
        'test_bleu': fmt_float(get(row, 'test_bleu'), 4),
        'test_rouge1': fmt_float(get(row, 'test_rouge1'), 4),
        'test_rouge2': fmt_float(get(row, 'test_rouge2'), 4),
        'test_rougeL': fmt_float(get(row, 'test_rougeL'), 4),
        'test_bertscore_precision': fmt_float(get(row, 'test_bertscore_precision'), 4),
        'test_bertscore_recall': fmt_float(get(row, 'test_bertscore_recall'), 4),
        'test_bertscore_f1': fmt_float(get(row, 'test_bertscore_f1'), 4),
        'trainable_params': get(row, 'trainable_params'),
        'total_params': get(row, 'total_params'),
        'trainable_ratio_pct': fmt_float(float(get(row, 'trainable_ratio', '0')) * 100, 2) if get(row, 'trainable_ratio') else '',
        'communication_MB_per_round': fmt_float(get(row, 'communication_MB_per_round_average'), 3),
        'total_communication_MB': fmt_float(get(row, 'total_communication_MB'), 2),
        'communication_reduction_pct': fmt_float(get(row, 'communication_reduction_per_round_vs_full_percent'), 2),
        'training_time_seconds': fmt_float(get(row, 'training_time_seconds'), 1),
        'inference_latency_mean_ms': fmt_float(get(row, 'inference_latency_mean_ms_per_sample'), 2),
        'completed_rounds': get(row, 'completed_rounds'),
        'best_round': get(row, 'best_round'),
        'lora_r': get(row, 'lora_r'),
        'local_epochs': get(row, 'local_epochs'),
        'participation_rate': get(row, 'participation_rate'),
        'num_clients': get(row, 'num_clients'),
        'dirichlet_alpha': get(row, 'dirichlet_alpha'),
    }
    main_rows.append(r)

out_main = os.path.join(out_dir, 'paper_table1_main_results.csv')
with open(out_main, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=main_cols)
    writer.writeheader()
    writer.writerows(main_rows)
print(f"Saved Table 1 (Main Results): {len(main_rows)} rows → paper_table1_main_results.csv")


# ========================================================
# TABLE 2: Communication & Computational Efficiency
# ========================================================
eff_cols = [
    'experiment_id', 'method_label', 'dataset', 'model_label',
    'trainable_params', 'total_params', 'trainable_ratio_pct', 'adapter_size_MB',
    'communication_MB_per_round', 'total_communication_MB',
    'communication_reduction_pct',
    'reference_full_MB_per_round',
    'max_vram_allocated_MB', 'cpu_peak_memory_MB',
    'training_time_seconds', 'average_time_per_round_seconds',
    'estimated_total_training_flops', 'forward_flops_per_sample',
    'inference_latency_mean_ms', 'inference_latency_p95_ms',
    'inference_throughput_samples_per_second', 'inference_throughput_tokens_per_second',
    'average_input_tokens', 'average_generated_tokens',
    'completed_rounds',
]

eff_rows = []
for eid, row in all_rows.items():
    method = get(row, 'method')
    ds = get(row, 'dataset_name')
    model = get(row, 'model_name')
    r = {
        'experiment_id': eid,
        'method_label': method_labels.get(method, method),
        'dataset': dataset_labels.get(ds, ds.split('/')[-1]),
        'model_label': model_labels.get(model, model),
        'trainable_params': get(row, 'trainable_params'),
        'total_params': get(row, 'total_params'),
        'trainable_ratio_pct': fmt_float(float(get(row, 'trainable_ratio', '0')) * 100, 2) if get(row, 'trainable_ratio') else '',
        'adapter_size_MB': fmt_float(get(row, 'adapter_size_MB'), 3),
        'communication_MB_per_round': fmt_float(get(row, 'communication_MB_per_round_average'), 3),
        'total_communication_MB': fmt_float(get(row, 'total_communication_MB'), 2),
        'communication_reduction_pct': fmt_float(get(row, 'communication_reduction_per_round_vs_full_percent'), 2),
        'reference_full_MB_per_round': fmt_float(get(row, 'reference_full_MB_per_round'), 2),
        'max_vram_allocated_MB': fmt_float(get(row, 'max_vram_allocated_MB'), 1),
        'cpu_peak_memory_MB': fmt_float(get(row, 'cpu_peak_memory_MB'), 1),
        'training_time_seconds': fmt_float(get(row, 'training_time_seconds'), 1),
        'average_time_per_round_seconds': fmt_float(get(row, 'average_time_per_round_seconds'), 1),
        'estimated_total_training_flops': get(row, 'estimated_total_training_flops'),
        'forward_flops_per_sample': get(row, 'forward_flops_per_sample'),
        'inference_latency_mean_ms': fmt_float(get(row, 'inference_latency_mean_ms_per_sample'), 2),
        'inference_latency_p95_ms': fmt_float(get(row, 'inference_latency_p95_ms_per_sample'), 2),
        'inference_throughput_samples_per_second': fmt_float(get(row, 'inference_throughput_samples_per_second'), 4),
        'inference_throughput_tokens_per_second': fmt_float(get(row, 'inference_throughput_tokens_per_second'), 2),
        'average_input_tokens': fmt_float(get(row, 'average_input_tokens'), 1),
        'average_generated_tokens': fmt_float(get(row, 'average_generated_tokens'), 1),
        'completed_rounds': get(row, 'completed_rounds'),
    }
    eff_rows.append(r)

out_eff = os.path.join(out_dir, 'paper_table2_efficiency.csv')
with open(out_eff, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=eff_cols)
    writer.writeheader()
    writer.writerows(eff_rows)
print(f"Saved Table 2 (Efficiency): {len(eff_rows)} rows → paper_table2_efficiency.csv")


# ========================================================
# TABLE 3: Ablation Studies
# ========================================================
ablation_map = {
    # SLM vs Larger LM (RQ3)
    'A01_centralized_base_full_samsum': ('SLM-vs-LargerLM', 'Centralized-Full (Flan-T5-Base 248M)'),
    'M05_centralized_full_samsum':      ('SLM-vs-LargerLM', 'Centralized-Full (Flan-T5-Small 77M)'),
    'M09_centralized_lora_samsum':      ('SLM-vs-LargerLM', 'Centralized-LoRA (Flan-T5-Small 77M)'),
    'M29_fedtgen_samsum':               ('SLM-vs-LargerLM', 'FedTGen (Flan-T5-Small 77M)'),
    # LoRA Rank
    'A02_fedtgen_rank4_samsum':         ('LoRA-Rank', 'FedTGen rank=4'),
    # A04 is default rank=8
    'A04_fedtgen_epoch3_samsum':        ('LoRA-Rank', 'FedTGen rank=8 (default, epoch=3)'),
    'A03_fedtgen_rank16_samsum':        ('LoRA-Rank', 'FedTGen rank=16'),
    # Local Epochs
    'A04_fedtgen_epoch3_samsum':        ('Local-Epochs', 'FedTGen epochs=3'),
    # A29 default is epoch=1
    'M29_fedtgen_samsum':               ('Local-Epochs', 'FedTGen epochs=1 (default)'),
    'A05_fedtgen_epoch5_samsum':        ('Local-Epochs', 'FedTGen epochs=5'),
    # Client Participation
    'A06_fedtgen_participation05_samsum': ('Participation', 'FedTGen participation=0.5'),
    # M29 is participation=1.0 (default)
    # Num Clients
    'A07_fedtgen_clients10_samsum':     ('Num-Clients', 'FedTGen clients=10'),
    # M29 is clients=5 (default)
    # Non-IID Robustness (RQ4)
    'R01_fedtgen_alpha01_samsum':       ('NonIID-Robustness', 'FedTGen alpha=0.1 (highly heterogeneous)'),
    'M29_fedtgen_samsum':               ('NonIID-Robustness', 'FedTGen alpha=0.5 (default)'),
    'R02_fedtgen_alpha10_samsum':       ('NonIID-Robustness', 'FedTGen alpha=1.0 (near IID)'),
}

abl_cols = [
    'experiment_id', 'ablation_type', 'variant_label',
    'method_label', 'model_label', 'dataset',
    'lora_r', 'local_epochs', 'participation_rate', 'num_clients', 'dirichlet_alpha',
    'trainable_params', 'trainable_ratio_pct', 'communication_MB_per_round',
    'communication_reduction_pct', 'total_communication_MB',
    'test_bleu', 'test_rouge1', 'test_rouge2', 'test_rougeL', 'test_bertscore_f1',
    'client_test_rougeL_mean', 'client_test_rougeL_std', 'client_test_rougeL_min', 'client_test_rougeL_max',
    'client_test_bertscore_f1_mean', 'client_test_bertscore_f1_std',
    'client_update_divergence_mean', 'client_update_divergence_std',
    'mean_pairwise_client_js_divergence',
    'training_time_seconds', 'inference_latency_mean_ms', 'completed_rounds', 'best_round',
]

abl_rows = []
# For ablation, we include all M* experiments + A* + R* with ablation type annotation
ablation_experiment_groups = {
    'SLM-vs-LargerLM': ['A01_centralized_base_full_samsum', 'M05_centralized_full_samsum', 'M09_centralized_lora_samsum', 'M29_fedtgen_samsum'],
    'LoRA-Rank': ['A02_fedtgen_rank4_samsum', 'M29_fedtgen_samsum', 'A03_fedtgen_rank16_samsum'],
    'Local-Epochs': ['M29_fedtgen_samsum', 'A04_fedtgen_epoch3_samsum', 'A05_fedtgen_epoch5_samsum'],
    'Participation-Rate': ['A06_fedtgen_participation05_samsum', 'M29_fedtgen_samsum'],
    'Num-Clients': ['M29_fedtgen_samsum', 'A07_fedtgen_clients10_samsum'],
    'NonIID-Robustness': ['R01_fedtgen_alpha01_samsum', 'M29_fedtgen_samsum', 'R02_fedtgen_alpha10_samsum'],
}

seen = set()
for abl_type, eids in ablation_experiment_groups.items():
    for eid in eids:
        if eid not in all_rows:
            continue
        row = all_rows[eid]
        method = get(row, 'method')
        ds = get(row, 'dataset_name')
        model = get(row, 'model_name')
        key = (abl_type, eid)
        r = {
            'experiment_id': eid,
            'ablation_type': abl_type,
            'variant_label': ablation_map.get(eid, ('', eid))[1] if eid in ablation_map else eid,
            'method_label': method_labels.get(method, method),
            'model_label': model_labels.get(model, model),
            'dataset': dataset_labels.get(ds, ds.split('/')[-1]),
            'lora_r': get(row, 'lora_r'),
            'local_epochs': get(row, 'local_epochs'),
            'participation_rate': get(row, 'participation_rate'),
            'num_clients': get(row, 'num_clients'),
            'dirichlet_alpha': get(row, 'dirichlet_alpha'),
            'trainable_params': get(row, 'trainable_params'),
            'trainable_ratio_pct': fmt_float(float(get(row, 'trainable_ratio', '0')) * 100, 2) if get(row, 'trainable_ratio') else '',
            'communication_MB_per_round': fmt_float(get(row, 'communication_MB_per_round_average'), 3),
            'communication_reduction_pct': fmt_float(get(row, 'communication_reduction_per_round_vs_full_percent'), 2),
            'total_communication_MB': fmt_float(get(row, 'total_communication_MB'), 2),
            'test_bleu': fmt_float(get(row, 'test_bleu'), 4),
            'test_rouge1': fmt_float(get(row, 'test_rouge1'), 4),
            'test_rouge2': fmt_float(get(row, 'test_rouge2'), 4),
            'test_rougeL': fmt_float(get(row, 'test_rougeL'), 4),
            'test_bertscore_f1': fmt_float(get(row, 'test_bertscore_f1'), 4),
            'client_test_rougeL_mean': fmt_float(get(row, 'client_test_rougeL_mean'), 4),
            'client_test_rougeL_std': fmt_float(get(row, 'client_test_rougeL_std'), 4),
            'client_test_rougeL_min': fmt_float(get(row, 'client_test_rougeL_min'), 4),
            'client_test_rougeL_max': fmt_float(get(row, 'client_test_rougeL_max'), 4),
            'client_test_bertscore_f1_mean': fmt_float(get(row, 'client_test_bertscore_f1_mean'), 4),
            'client_test_bertscore_f1_std': fmt_float(get(row, 'client_test_bertscore_f1_std'), 4),
            'client_update_divergence_mean': fmt_float(get(row, 'client_update_divergence_mean'), 6),
            'client_update_divergence_std': fmt_float(get(row, 'client_update_divergence_std'), 6),
            'mean_pairwise_client_js_divergence': fmt_float(get(row, 'mean_pairwise_client_js_divergence'), 4),
            'training_time_seconds': fmt_float(get(row, 'training_time_seconds'), 1),
            'inference_latency_mean_ms': fmt_float(get(row, 'inference_latency_mean_ms_per_sample'), 2),
            'completed_rounds': get(row, 'completed_rounds'),
            'best_round': get(row, 'best_round'),
        }
        abl_rows.append(r)

out_abl = os.path.join(out_dir, 'paper_table3_ablation.csv')
with open(out_abl, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=abl_cols)
    writer.writeheader()
    writer.writerows(abl_rows)
print(f"Saved Table 3 (Ablation): {len(abl_rows)} rows → paper_table3_ablation.csv")


print("\n=== SUMMARY OF PAPER CSV FILES ===")
print("1. all_results_combined.csv      - All 40 experiments, all 206 columns (raw)")
print("2. all_results_paper_focused.csv - All 40 experiments, 75 paper-relevant columns")
print("3. paper_table1_main_results.csv - Main results table (BLEU/ROUGE/BERTScore/Comm/Time)")
print("4. paper_table2_efficiency.csv   - Communication & Computational Efficiency")
print("5. paper_table3_ablation.csv     - Ablation study results")
print()
print("NOTE: M08_centralized_full_qqp is MISSING (no final_results.csv)")
print("      This is the Centralized-Full baseline for QQP dataset.")
print("      You may want to re-run that experiment or note it in the paper.")
