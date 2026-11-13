import pandas as pd, matplotlib, sys
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R=sys.argv[1]; FR=sys.argv[2]
plt.rcParams.update({'font.family':'serif','font.size':7.5,'mathtext.fontset':'dejavuserif','axes.spines.top':False,'axes.spines.right':False,
 'axes.edgecolor':'#555','axes.linewidth':0.6,'xtick.color':'#333','ytick.color':'#333','pdf.fonttype':42})
fr=pd.read_csv(FR,index_col=0)
DS=[('samsum','SAMSum'),('dailydialog','DailyDialog'),('squad_qg','SQuAD-QG'),('qqp','QQP')]
METH=[('FedAvg-Full','fedavg_full',13,'#2a78d6','s'),('FedProx-LoRA','fedprox_lora',21,'#1baf7a','^'),
 ('SCAFFOLD-LoRA','scaffold_lora',25,'#eda100','D'),(r'FedTGen ($\beta$=0)','fedavg_lora',17,'#eb6834','o'),('FedTGen','fedtgen',29,'#4a3aa7','*')]
def eid(b,fam,i,ds): return f"M{b+i:02d}_{fam}_{ds}"
fig,axs=plt.subplots(2,2,figsize=(4.8,3.9)); axs=axs.ravel()
for i,(ds,name) in enumerate(DS):
    ax=axs[i]
    for lab,fam,b,col,mk in METH:
        e=eid(b,fam,i,ds); h=pd.read_csv(f'{R}/{e}/results/history.csv')
        n=int(fr.loc[e,'completed_rounds']); h=h[h['round']<=n]
        ax.plot(h['round'],h.validation_rougeL*100,color=col,lw=1.4,marker=mk,ms=3 if mk!='*' else 5,label=lab)
    if ds=='samsum':
        h=pd.read_csv(f'{R}/A05_fedtgen_epoch5_samsum/results/history.csv')
        ax.plot(h['round'],h.validation_rougeL*100,color='#4a3aa7',lw=1.2,ls='--',label='FedTGen ($E$=5)')
    ax.set_title(name,fontsize=8.5); ax.set_xlabel('Communication round' if i>=2 else '')
    ax.grid(True,color='#e6e6e6',linewidth=0.5); ax.set_axisbelow(True); ax.set_xlim(0.5,20.5)
axs[0].set_ylabel('Validation ROUGE-L'); axs[2].set_ylabel('Validation ROUGE-L')
h1,l1=axs[0].get_legend_handles_labels()
fig.legend(h1,l1,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(0.5,1.09),fontsize=7)
fig.tight_layout(); fig.savefig('figs/convergence_v2.pdf',bbox_inches='tight'); fig.savefig('figs/convergence_v2.png',dpi=110,bbox_inches='tight')
