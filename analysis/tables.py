import pandas as pd, numpy as np, json
fr = pd.read_csv('/home/claude/all41_corrected.csv', index_col=0)
boot = json.load(open('/home/claude/boot.json'))['boot']
DS = ['samsum','dailydialog','squad_qg','qqp']; DSN = ['SAMSum','DailyDialog','SQuAD-QG','QQP']
METH = [('Local-LoRA','local_lora',1,'Local'),('Centralized-Full','centralized_full',5,'Centralized'),
        ('Centralized-LoRA','centralized_lora',9,'Centralized'),('FedAvg-Full','fedavg_full',13,'Federated'),
        ('FedAvg-LoRA','fedavg_lora',17,'Federated'),('FedProx-LoRA','fedprox_lora',21,'Federated'),
        ('SCAFFOLD-LoRA','scaffold_lora',25,'Federated'),('FedTGen (ours)','fedtgen',29,'Federated')]
def eid(b,f,i): return f"M{b+i:02d}_{f}_{DS[i]}"
def g(e,c): return fr.loc[e,c]
T = {}
# ---------- main results ----------
L=[]
L.append(r'\begin{table}[t]\centering\caption{Test-set generation quality (single seed 42). R-L and BS are ROUGE-L F1 and BERTScore F1 ($\times$100). \textbf{Bold}: best overall; \underline{underline}: best among PEFT-based federated methods. Local-LoRA scores are sample-weighted averages over the five clients'"'"' own test splits.}\label{tab:main}')
L.append(r'\setlength{\tabcolsep}{2.6pt}\resizebox{\textwidth}{!}{\begin{tabular}{l ccc ccc ccc ccc}\toprule')
L.append(' & '+' & '.join([r'\multicolumn{3}{c}{%s}'%n for n in DSN])+r'\\')
L.append(''.join([r'\cmidrule(lr){%d-%d}'%(2+3*i,4+3*i) for i in range(4)]))
L.append('Method & '+' & '.join(['BLEU & R-L & BS']*4)+r'\\\midrule')
vals={}
for lab,f,b,grp in METH:
    for i in range(4):
        e=eid(b,f,i); vals[(lab,i)]=(g(e,'test_bleu'),100*g(e,'test_rougeL'),100*g(e,'test_bertscore_f1'))
peft_fed=['FedAvg-LoRA','FedProx-LoRA','SCAFFOLD-LoRA','FedTGen (ours)']
prev=None
for lab,f,b,grp in METH:
    if prev and grp!=prev: L.append(r'\midrule')
    prev=grp
    cells=[]
    for i in range(4):
        for k in range(3):
            v=vals[(lab,i)][k]; s=f'{v:.2f}'
            best=max(vals[(m[0],i)][k] for m in METH)
            bpf=max(vals[(m,i)][k] for m in peft_fed)
            if abs(v-best)<1e-9: s=r'\textbf{'+s+'}'
            if lab in peft_fed and abs(v-bpf)<1e-9: s=r'\underline{'+s+'}'
            cells.append(s)
    L.append(lab+' & '+' & '.join(cells)+r'\\')
L.append(r'\bottomrule\end{tabular}}\end{table}')
T['main']='\n'.join(L)
# ---------- communication ----------
L=[r'\begin{table}[t]\centering\caption{Communication cost of the federated methods (float32 payloads, $K{=}5$ clients, full participation). Params/round counts the parameters sent in each direction per client; totals use the number of rounds actually run before early stopping (in parentheses).}\label{tab:comm}',
   r'\setlength{\tabcolsep}{3pt}\resizebox{\textwidth}{!}{\begin{tabular}{l r r r cccc}\toprule',
   r'Method & Params/round & MB/round & Reduction & \multicolumn{4}{c}{Total communication in MB (rounds)}\\\cmidrule(lr){5-8}',
   r' & (per client) & (all clients) & vs.\ Full & '+' & '.join(DSN)+r'\\\midrule']
for lab,f,b,grp in METH[3:]:
    e0=eid(b,f,0)
    p=int(g(e0,'communication_parameters_per_round')); mb=g(e0,'communication_MB_per_round_average')
    red=(1-mb/g('M13_fedavg_full_samsum','communication_MB_per_round_average'))*100
    tot=[f"{g(eid(b,f,i),'total_communication_MB'):,.1f} ({int(g(eid(b,f,i),'completed_rounds'))})" for i in range(4)]
    rs = '--' if red<1e-6 else ('%.2f' % red) + r'\%'
    L.append(f"{lab} & {p:,} & {mb:,.2f} & {rs} & "+' & '.join(tot)+r'\\')
L.append(r'\bottomrule\end{tabular}}\end{table}')
T['comm']='\n'.join(L)
# ---------- compute ----------
L=[r'\begin{table}[t]\centering\caption{Computational cost on a single NVIDIA T4 (bf16 autocast). Training time is summed over the four datasets (wall-clock of all training rounds/epochs, including per-round validation). FLOPs follow the analytical $2ND$ estimate over total parameters and therefore do not credit LoRA for skipped weight gradients. Latency: mean per-sample beam-search generation time on 200 test inputs (SAMSum / QQP).}\label{tab:compute}',
   r'\setlength{\tabcolsep}{3pt}\resizebox{\textwidth}{!}{\begin{tabular}{l r r cc r r cc}\toprule',
   r'Method & Trainable & Ratio & \multicolumn{2}{c}{Peak VRAM (MB)} & Train time & Est.\ FLOPs & \multicolumn{2}{c}{Latency (ms)}\\\cmidrule(lr){4-5}\cmidrule(lr){8-9}',
   r' & params & (\%) & SAMSum & QQP & (h, 4 tasks) & (4 tasks) & SAMSum & QQP\\\midrule']
for lab,f,b,grp in METH:
    es=[eid(b,f,i) for i in range(4)]
    tp=int(g(es[0],'trainable_params')); ratio=100*g(es[0],'trainable_ratio')
    tt=sum(g(e,'training_time_seconds') for e in es)/3600; fl=sum(g(e,'estimated_total_training_flops') for e in es)
    ex=f"{fl:.2e}".replace('e+',r'\!\times\!10^{')+'}'; ex=ex.replace('10^{0','10^{')
    L.append(f"{lab} & {tp:,} & {ratio:.2f} & {g(es[0],'max_vram_allocated_MB'):,.0f} & {g(es[3],'max_vram_allocated_MB'):,.0f} & {tt:.1f} & ${ex}$ & {g(es[0],'inference_latency_mean_ms_per_sample'):.0f} & {g(es[3],'inference_latency_mean_ms_per_sample'):.0f}\\\\")
L.append(r'\bottomrule\end{tabular}}\end{table}')
T['compute']='\n'.join(L)
# ---------- significance ----------
L=[r'\begin{table}[t]\centering\caption{Paired bootstrap comparison of FedTGen against each baseline on per-example test ROUGE-L (2{,}000 resamples). Entries: mean difference FedTGen$-$baseline ($\times$100) with 95\% CI; $^\ast$ marks CIs that exclude zero. For Local-LoRA, examples are matched by input--reference pair.}\label{tab:boot}',
   r'\setlength{\tabcolsep}{3pt}\resizebox{\textwidth}{!}{\begin{tabular}{l cccc}\toprule',
   r'Baseline & '+' & '.join(DSN)+r'\\\midrule']
for m,name in [('FedAvg-LoRA','FedAvg-LoRA'),('FedProx-LoRA','FedProx-LoRA'),('SCAFFOLD-LoRA','SCAFFOLD-LoRA'),('Local-LoRA','Local-LoRA'),('Cent-LoRA','Centralized-LoRA'),('FedAvg-Full','FedAvg-Full')]:
    cells=[]
    for ds in DS:
        v=boot[f'{ds}|{m}']; sig = v['lo']>0 or v['hi']<0
        cells.append(f"${100*v['diff']:+.2f}$ [{100*v['lo']:+.2f}, {100*v['hi']:+.2f}]"+('$^\\ast$' if sig else ''))
    L.append(name+' & '+' & '.join(cells)+r'\\')
L.append(r'\bottomrule\end{tabular}}\end{table}')
T['boot']='\n'.join(L)
# ---------- RQ3 ----------
rows=[('FLAN-T5-small, full','M05_centralized_full_samsum'),('FLAN-T5-small, LoRA','M09_centralized_lora_samsum'),('FLAN-T5-base, full','A01_centralized_base_full_samsum')]
L=[r'\begin{table}[t]\centering\caption{Quality--efficiency trade-off between the SLM (FLAN-T5-small) and a larger LM (FLAN-T5-base), centralized training on SAMSum.}\label{tab:rq3}',
   r'\setlength{\tabcolsep}{2.5pt}\resizebox{\textwidth}{!}{\begin{tabular}{l cccc r r r r r}\toprule',
   r'Model & BLEU & R-1 & R-L & BS & Trainable & Peak VRAM & Train time & Latency & Ckpt/model\\',
   r' & & & & & params & (MB) & (h) & (ms) & size (MB)\\\midrule']
for n,e in rows:
    L.append(f"{n} & {g(e,'test_bleu'):.2f} & {100*g(e,'test_rouge1'):.2f} & {100*g(e,'test_rougeL'):.2f} & {100*g(e,'test_bertscore_f1'):.2f} & {int(g(e,'trainable_params')):,} & {g(e,'max_vram_allocated_MB'):,.0f} & {g(e,'training_time_seconds')/3600:.2f} & {g(e,'inference_latency_mean_ms_per_sample'):.0f} & {g(e,'model_size_MB'):.0f}\\\\")
L.append(r'\bottomrule\end{tabular}}\end{table}')
T['rq3']='\n'.join(L)
# ---------- RQ5 ablations ----------
abl=[('\\multicolumn{9}{l}{\\emph{LoRA rank} $r$ ($E{=}1$, $K{=}5$, $p{=}1.0$)}',None),('$r=4$','A02_fedtgen_rank4_samsum'),('$r=8$ (default)','M29_fedtgen_samsum'),('$r=16$','A03_fedtgen_rank16_samsum'),
     ('\\multicolumn{9}{l}{\\emph{Local epochs} $E$ ($r{=}8$)}',None),('$E=1$ (default)','M29_fedtgen_samsum'),('$E=3$','A04_fedtgen_epoch3_samsum'),('$E=5$','A05_fedtgen_epoch5_samsum'),
     ('\\multicolumn{9}{l}{\\emph{Client participation} $p$}',None),('$p=1.0$ (default)','M29_fedtgen_samsum'),('$p=0.5$','A06_fedtgen_participation05_samsum'),
     ('\\multicolumn{9}{l}{\\emph{Number of clients} $K$}',None),('$K=5$ (default)','M29_fedtgen_samsum'),('$K=10$','A07_fedtgen_clients10_samsum')]
L=[r'\begin{table}[t]\centering\caption{FedTGen ablations on SAMSum ($\alpha{=}0.5$). ``Rounds (best)'' gives the number of rounds run before early stopping and the round whose adapter was selected.}\label{tab:abl}',
   r'\setlength{\tabcolsep}{3pt}\resizebox{\textwidth}{!}{\begin{tabular}{l cccc c r r r}\toprule',
   r'Setting & BLEU & R-1 & R-L & BS & Rounds (best) & Params/client & MB/round & Total MB\\\midrule']
for n,e in abl:
    if e is None: L.append(n+r'\\'); continue
    L.append(f"\\quad {n} & {g(e,'test_bleu'):.2f} & {100*g(e,'test_rouge1'):.2f} & {100*g(e,'test_rougeL'):.2f} & {100*g(e,'test_bertscore_f1'):.2f} & {int(g(e,'completed_rounds'))} ({int(g(e,'best_round'))}) & {int(g(e,'communication_parameters_per_round')):,} & {g(e,'communication_MB_per_round_average'):.2f} & {g(e,'total_communication_MB'):.1f}\\\\")
L.append(r'\bottomrule\end{tabular}}\end{table}')
T['abl']='\n'.join(L)
# ---------- RQ4 ----------
L=[r'\begin{table}[t]\centering\caption{FedTGen under different Dirichlet concentrations $\alpha$ on SAMSum (smaller $\alpha$ = more skewed pseudo-topic distribution). JS: mean pairwise Jensen--Shannon divergence between client topic distributions. $H(w)$ and $\max w$: average entropy and maximum of the FedTGen aggregation weights ($\ln 5\approx1.61$ for uniform weights).}\label{tab:noniid}',
   r'\setlength{\tabcolsep}{3pt}\resizebox{\textwidth}{!}{\begin{tabular}{l c r r cccc c cc}\toprule',
   r'$\alpha$ & JS & Min $n_k$ & Max $n_k$ & BLEU & R-1 & R-L & BS & Rounds (best) & $H(w)$ & $\max w$\\\midrule']
for a,e in [('0.1','R01_fedtgen_alpha01_samsum'),('0.5','M29_fedtgen_samsum'),('1.0','R02_fedtgen_alpha10_samsum')]:
    L.append(f"{a} & {g(e,'mean_pairwise_client_js_divergence'):.3f} & {int(g(e,'client_sample_count_min')):,} & {int(g(e,'client_sample_count_max')):,} & {g(e,'test_bleu'):.2f} & {100*g(e,'test_rouge1'):.2f} & {100*g(e,'test_rougeL'):.2f} & {100*g(e,'test_bertscore_f1'):.2f} & {int(g(e,'completed_rounds'))} ({int(g(e,'best_round'))}) & {g(e,'average_aggregation_weight_entropy'):.2f} & {g(e,'average_max_client_weight'):.2f}\\\\")
L.append(r'\bottomrule\end{tabular}}\end{table}')
T['noniid']='\n'.join(L)
# ---------- datasets ----------
L=[r'\begin{table}[t]\centering\caption{Datasets, sequence lengths, and client partitions ($K{=}5$, $\alpha{=}0.5$).}\label{tab:data}',
   r'\setlength{\tabcolsep}{3pt}\resizebox{\textwidth}{!}{\begin{tabular}{l l r r r c r r c}\toprule',
   r'Dataset & Task & Train & Val & Test & Src/Tgt len & Min $n_k$ & Max $n_k$ & JS\\\midrule']
task=['Dialogue summarization','Dialogue response generation','Question generation','Paraphrase generation']
for i,d in enumerate(DS):
    e=eid(29,'fedtgen',i)
    L.append(f"{DSN[i]} & {task[i]} & {int(g(e,'num_train_samples')):,} & {int(g(e,'num_validation_samples')):,} & {int(g(e,'num_test_samples')):,} & {int(g(e,'max_source_length'))}/{int(g(e,'max_target_length'))} & {int(g(e,'client_sample_count_min')):,} & {int(g(e,'client_sample_count_max')):,} & {g(e,'mean_pairwise_client_js_divergence'):.3f}\\\\")
L.append(r'\bottomrule\end{tabular}}\end{table}')
T['data']='\n'.join(L)
# ---------- appendix R1/R2 ----------
L=[r'\begin{table}[h]\centering\caption{ROUGE-1 / ROUGE-2 F1 ($\times$100) on the test sets. $^\dagger$Pending: the per-example predictions of this run were lost and will be regenerated from the saved best checkpoint.}\label{tab:r12}',
   r'\setlength{\tabcolsep}{3pt}\resizebox{\textwidth}{!}{\begin{tabular}{l cc cc cc cc}\toprule',
   ' & '+' & '.join([r'\multicolumn{2}{c}{%s}'%n for n in DSN])+r'\\'+''.join([r'\cmidrule(lr){%d-%d}'%(2+2*i,3+2*i) for i in range(4)]),
   'Method & '+' & '.join(['R-1 & R-2']*4)+r'\\\midrule']
for lab,f,b,grp in METH:
    cells=[]
    for i in range(4):
        e=eid(b,f,i); r1=g(e,'test_rouge1'); r2=g(e,'test_rouge2')
        cells += ([f'{100*r1:.2f}',f'{100*r2:.2f}'] if r1==r1 else [r'--$^\dagger$',r'--$^\dagger$'])
    L.append(lab+' & '+' & '.join(cells)+r'\\')
L.append(r'\bottomrule\end{tabular}}\end{table}')
T['r12']='\n'.join(L)
import os; os.makedirs('/home/claude/paper/tables',exist_ok=True)
for k,v in T.items(): open(f'/home/claude/paper/tables/{k}.tex','w').write(v+'\n')
print(T['main']); print(T['comm']); print(T['compute'])
