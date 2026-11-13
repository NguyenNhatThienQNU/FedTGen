# FedTGen Standalone Experiments (41)

Communication-Efficient Federated Text Generation via Parameter-Efficient
Fine-Tuning of Small Language Models.

**41 fully independent experiments** (32 Main + 7 Ablation + 2 Robustness), each in
its own folder with a standalone `.py`, a standalone Colab `.ipynb`, `requirements.txt`
and `README.md`. Single seed = **42**. No shared core module, no shared config/
partition/checkpoint, and no aggregate result files: each experiment writes its own
`final_results.csv`.

## Experiments

| ID | Family | Setting | Tuning | Dataset | Model | RQ |
|----|--------|---------|--------|---------|-------|----|
| M01_local_lora_samsum | local_lora | local | lora | samsum | flan-t5-small | RQ1,RQ3 |
| M02_local_lora_dailydialog | local_lora | local | lora | dailydialog | flan-t5-small | RQ1,RQ3 |
| M03_local_lora_squad_qg | local_lora | local | lora | squad_qg | flan-t5-small | RQ1,RQ3 |
| M04_local_lora_qqp | local_lora | local | lora | qqp | flan-t5-small | RQ1,RQ3 |
| M05_centralized_full_samsum | centralized_full | centralized | full | samsum | flan-t5-small | RQ1,RQ3 |
| M06_centralized_full_dailydialog | centralized_full | centralized | full | dailydialog | flan-t5-small | RQ1,RQ3 |
| M07_centralized_full_squad_qg | centralized_full | centralized | full | squad_qg | flan-t5-small | RQ1,RQ3 |
| M08_centralized_full_qqp | centralized_full | centralized | full | qqp | flan-t5-small | RQ1,RQ3 |
| M09_centralized_lora_samsum | centralized_lora | centralized | lora | samsum | flan-t5-small | RQ1,RQ2,RQ3 |
| M10_centralized_lora_dailydialog | centralized_lora | centralized | lora | dailydialog | flan-t5-small | RQ1,RQ2,RQ3 |
| M11_centralized_lora_squad_qg | centralized_lora | centralized | lora | squad_qg | flan-t5-small | RQ1,RQ2,RQ3 |
| M12_centralized_lora_qqp | centralized_lora | centralized | lora | qqp | flan-t5-small | RQ1,RQ2,RQ3 |
| M13_fedavg_full_samsum | fedavg_full | federated | full | samsum | flan-t5-small | RQ1,RQ2 |
| M14_fedavg_full_dailydialog | fedavg_full | federated | full | dailydialog | flan-t5-small | RQ1,RQ2 |
| M15_fedavg_full_squad_qg | fedavg_full | federated | full | squad_qg | flan-t5-small | RQ1,RQ2 |
| M16_fedavg_full_qqp | fedavg_full | federated | full | qqp | flan-t5-small | RQ1,RQ2 |
| M17_fedavg_lora_samsum | fedavg_lora | federated | lora | samsum | flan-t5-small | RQ1,RQ2 |
| M18_fedavg_lora_dailydialog | fedavg_lora | federated | lora | dailydialog | flan-t5-small | RQ1,RQ2 |
| M19_fedavg_lora_squad_qg | fedavg_lora | federated | lora | squad_qg | flan-t5-small | RQ1,RQ2 |
| M20_fedavg_lora_qqp | fedavg_lora | federated | lora | qqp | flan-t5-small | RQ1,RQ2 |
| M21_fedprox_lora_samsum | fedprox_lora | federated | lora | samsum | flan-t5-small | RQ1,RQ4 |
| M22_fedprox_lora_dailydialog | fedprox_lora | federated | lora | dailydialog | flan-t5-small | RQ1,RQ4 |
| M23_fedprox_lora_squad_qg | fedprox_lora | federated | lora | squad_qg | flan-t5-small | RQ1,RQ4 |
| M24_fedprox_lora_qqp | fedprox_lora | federated | lora | qqp | flan-t5-small | RQ1,RQ4 |
| M25_scaffold_lora_samsum | scaffold_lora | federated | lora | samsum | flan-t5-small | RQ1,RQ4 |
| M26_scaffold_lora_dailydialog | scaffold_lora | federated | lora | dailydialog | flan-t5-small | RQ1,RQ4 |
| M27_scaffold_lora_squad_qg | scaffold_lora | federated | lora | squad_qg | flan-t5-small | RQ1,RQ4 |
| M28_scaffold_lora_qqp | scaffold_lora | federated | lora | qqp | flan-t5-small | RQ1,RQ4 |
| M29_fedtgen_samsum | fedtgen | federated | lora | samsum | flan-t5-small | RQ1,RQ2,RQ4,RQ5 |
| M30_fedtgen_dailydialog | fedtgen | federated | lora | dailydialog | flan-t5-small | RQ1,RQ2,RQ4,RQ5 |
| M31_fedtgen_squad_qg | fedtgen | federated | lora | squad_qg | flan-t5-small | RQ1,RQ2,RQ4,RQ5 |
| M32_fedtgen_qqp | fedtgen | federated | lora | qqp | flan-t5-small | RQ1,RQ2,RQ4,RQ5 |
| A01_centralized_base_full_samsum | centralized_full | centralized | full | samsum | flan-t5-base | RQ3 |
| A02_fedtgen_rank4_samsum | fedtgen | federated | lora | samsum | flan-t5-small | RQ5 |
| A03_fedtgen_rank16_samsum | fedtgen | federated | lora | samsum | flan-t5-small | RQ5 |
| A04_fedtgen_epoch3_samsum | fedtgen | federated | lora | samsum | flan-t5-small | RQ5 |
| A05_fedtgen_epoch5_samsum | fedtgen | federated | lora | samsum | flan-t5-small | RQ5 |
| A06_fedtgen_participation05_samsum | fedtgen | federated | lora | samsum | flan-t5-small | RQ5 |
| A07_fedtgen_clients10_samsum | fedtgen | federated | lora | samsum | flan-t5-small | RQ5 |
| R01_fedtgen_alpha01_samsum | fedtgen | federated | lora | samsum | flan-t5-small | RQ4 |
| R02_fedtgen_alpha10_samsum | fedtgen | federated | lora | samsum | flan-t5-small | RQ4 |

## Run one experiment
Open its `.ipynb` in Google Colab (GPU runtime) and `Runtime -> Run all`, or run
`python <ID>.py` after `pip install -r requirements.txt`. Results are saved to
`/content/drive/MyDrive/fedtgen_experiments/<ID>/seed_42/`.

## Models & datasets
SLM = `google/flan-t5-small`; larger LM = `google/flan-t5-base` (only A01).
Datasets: SAMSum, DailyDialog, SQuAD-QG, QQP (label==1 paraphrases).

## Guarantees
- Each experiment is independent and self-contained.
- Checkpoint / resume / early stopping / progress bars.
- Quality, communication, time, VRAM, FLOPs and inference-latency metrics.
- `secure_aggregation_implemented = False` everywhere.
