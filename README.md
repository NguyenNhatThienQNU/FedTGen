# FedTGen: Communication-Efficient Federated Text Generation via Parameter-Efficient Fine-Tuning of Small Language Models

Code and results for the paper by Nhat-Thien Nguyen and Quang-Hung Le
(Department of Information Technology, Quy Nhon University).

FedTGen lets each client freeze a small language model (FLAN-T5-small) and
fine-tune and exchange only a LoRA adapter (0.45% of the parameters), while
the server weights each adapter update by data size, its loss and ROUGE-L on a
small shared probe set, and its divergence from the consensus update.

## Repository layout

| Path | Content |
|------|---------|
| `experiments/` | 41 standalone experiments (32 main, 7 ablation, 2 robustness). Each folder has a self-contained `.py` script, a Colab notebook (`.ipynb`), `requirements.txt`, a `README.md`, and its `results/` (final metrics, per-round history, communication, resource and latency logs). |
| `experiments/README.md` | Table of all experiments and how to run them. |

Model checkpoints, tokenizer caches, and test-set predictions are not included
because of their size (about 44 GB); every experiment regenerates them.

## Running an experiment

```bash
cd experiments/M29_fedtgen_samsum
pip install -r requirements.txt
python M29_fedtgen_samsum.py
```

or open the notebook in Google Colab (GPU runtime) and run all cells.
All experiments use seed 42, `google/flan-t5-small` (FLAN-T5-base for A01),
and the full SAMSum, DailyDialog, SQuAD-QG, and QQP datasets.

## Experiment IDs

- `M01`–`M04`: Local-LoRA; `M05`–`M08`: Centralized-Full; `M09`–`M12`: Centralized-LoRA
- `M13`–`M16`: FedAvg-Full; `M17`–`M20`: FedAvg-LoRA; `M21`–`M24`: FedProx-LoRA; `M25`–`M28`: SCAFFOLD-LoRA
- `M29`–`M32`: FedTGen (SAMSum, DailyDialog, SQuAD-QG, QQP)
- `A01`: FLAN-T5-base (centralized); `A02`–`A07`: ablations (rank, local epochs, participation, clients)
- `R01`–`R02`: non-IID skew (Dirichlet α = 0.1 and 1.0)
