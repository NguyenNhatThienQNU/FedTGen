import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.size':7.5,'mathtext.fontset':'dejavuserif','axes.spines.top':False,'axes.spines.right':False,
                     'axes.edgecolor':'#555','axes.linewidth':0.6,'xtick.color':'#333','ytick.color':'#333','pdf.fonttype':42})
fr = pd.read_csv('/home/claude/all41_corrected.csv', index_col=0)
DS = [('samsum','SAMSum'),('dailydialog','DailyDialog'),('squad_qg','SQuAD-QG'),('qqp','QQP')]
METH = [('FedAvg-Full','fedavg_full',13,'#2a78d6','s'),('FedAvg-LoRA','fedavg_lora',17,'#eb6834','o'),
        ('FedProx-LoRA','fedprox_lora',21,'#1baf7a','^'),('SCAFFOLD-LoRA','scaffold_lora',25,'#eda100','D'),
        ('FedTGen','fedtgen',29,'#4a3aa7','*')]
def eid(base,fam,i,ds): return f"M{base+i:02d}_{fam}_{ds}"
# ---- Fig: quality vs total communication ----
fig, axs = plt.subplots(2,2, figsize=(4.8, 3.9)); axs=axs.ravel()
for i,(ds,name) in enumerate(DS):
    ax = axs[i]
    for lab,fam,b,col,mk in METH:
        r = fr.loc[eid(b,fam,i,ds)]
        ax.scatter(r.total_communication_MB/1024, r.test_rougeL, s=60 if mk=='*' else 30, marker=mk, color=col,
                   edgecolor='white', linewidth=0.6, zorder=3, label=lab)
    ax.set_xscale('log'); ax.set_title(name, fontsize=8.5)
    ax.set_xlabel('Total communication (GB, log)' if i>=2 else '')
    ax.grid(True, which='major', color='#e6e6e6', linewidth=0.5); ax.set_axisbelow(True)
    ax.set_xlim(0.03, 200)
axs[0].set_ylabel('Test ROUGE-L'); axs[2].set_ylabel('Test ROUGE-L')
h,l = axs[0].get_legend_handles_labels()
fig.legend(h,l, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5,1.09), fontsize=7)
fig.tight_layout(); fig.savefig('/home/claude/paper/figs/quality_vs_comm.pdf', bbox_inches='tight'); fig.savefig('/home/claude/paper/figs/quality_vs_comm.png', dpi=160, bbox_inches='tight')
# ---- Fig: validation convergence ----
fig, axs = plt.subplots(2,2, figsize=(4.8, 3.9)); axs=axs.ravel()
for i,(ds,name) in enumerate(DS):
    ax = axs[i]
    for lab,fam,b,col,mk in METH:
        e = eid(b,fam,i,ds); h = pd.read_csv(f'{e}/results/history.csv')
        n = int(fr.loc[e,'completed_rounds']); h = h[h['round']<=n]
        ax.plot(h['round'], h.validation_rougeL, color=col, lw=1.4, marker=mk, ms=3 if mk!='*' else 5, label=lab)
    if ds=='samsum':
        h = pd.read_csv('A05_fedtgen_epoch5_samsum/results/history.csv')
        ax.plot(h['round'], h.validation_rougeL, color='#4a3aa7', lw=1.2, ls='--', label='FedTGen ($E$=5)')
    ax.set_title(name, fontsize=8.5); ax.set_xlabel('Communication round' if i>=2 else '')
    ax.grid(True, color='#e6e6e6', linewidth=0.5); ax.set_axisbelow(True); ax.set_xlim(0.5,20.5)
axs[0].set_ylabel('Validation ROUGE-L'); axs[2].set_ylabel('Validation ROUGE-L')
h1,l1 = axs[0].get_legend_handles_labels()
fig.legend(h1,l1, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5,1.09), fontsize=7)
fig.tight_layout(); fig.savefig('/home/claude/paper/figs/convergence.pdf', bbox_inches='tight'); fig.savefig('/home/claude/paper/figs/convergence.png', dpi=160, bbox_inches='tight')
print('ok')
