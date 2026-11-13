import pandas as pd, numpy as np, glob, json
from rouge_score import rouge_scorer
df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('*/results/final_results.csv'))], ignore_index=True).set_index('experiment_id')
# --- correct 4 runs that trained past early stopping (notebook ran main() twice) ---
corr = {}
for e in ['M07_centralized_full_squad_qg','M12_centralized_lora_qqp','M25_scaffold_lora_samsum','M26_scaffold_lora_dailydialog']:
    h = pd.read_csv(f'{e}/results/history.csv'); r = df.loc[e]
    n = int(r.best_round) + 3
    corr[e] = dict(old_rounds=int(r.completed_rounds), rounds=n,
                   train_time=float(h.training_time_seconds.iloc[:n].sum()),
                   total_comm_MB=float(r.communication_MB_per_round_average) * n if r.total_communication_MB > 0 else 0.0,
                   flops=float(r.training_flops_per_step) * n)
    df.loc[e, 'completed_rounds'] = n; df.loc[e, 'training_time_seconds'] = corr[e]['train_time']
    df.loc[e, 'total_communication_MB'] = corr[e]['total_comm_MB']; df.loc[e,'estimated_total_training_flops']=corr[e]['flops']
    df.loc[e, 'rounds_without_improvement'] = 3
json.dump(corr, open('/home/claude/corrections.json','w'), indent=1)
df.to_csv('/home/claude/all41_corrected.csv')
print(json.dumps(corr, indent=1))
