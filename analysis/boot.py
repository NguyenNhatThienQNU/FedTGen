import pandas as pd, numpy as np, json, os
from rouge_score import rouge_scorer
from concurrent.futures import ProcessPoolExecutor
sc = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
DS = ['samsum','dailydialog','squad_qg','qqp']
M = {'Local-LoRA':1,'Cent-Full':5,'Cent-LoRA':9,'FedAvg-Full':13,'FedAvg-LoRA':17,'FedProx-LoRA':21,'SCAFFOLD-LoRA':25,'FedTGen':29}
fam = {'Local-LoRA':'local_lora','Cent-Full':'centralized_full','Cent-LoRA':'centralized_lora','FedAvg-Full':'fedavg_full','FedAvg-LoRA':'fedavg_lora','FedProx-LoRA':'fedprox_lora','SCAFFOLD-LoRA':'scaffold_lora','FedTGen':'fedtgen'}
def eid(m,i): return f"M{M[m]+i:02d}_{fam[m]}_{DS[i]}"
def score(path):
    cache = '/home/claude/cache_'+os.path.basename(os.path.dirname(os.path.dirname(path)))+'.pkl'
    if os.path.exists(cache): return pd.read_pickle(cache)
    p = pd.read_csv(path, keep_default_na=False)
    s = [sc.score(r, q) for r, q in zip(p.reference_text, p.prediction_text)]
    out = pd.DataFrame({'key': p.input_text + '||' + p.reference_text,
                        'r1':[x['rouge1'].fmeasure for x in s],'r2':[x['rouge2'].fmeasure for x in s],'rl':[x['rougeL'].fmeasure for x in s]})
    out.to_pickle(cache); return out
jobs = [(m,i) for m in M for i in range(4) if not (m=='Cent-Full' and i==3)]
paths = [f"{eid(m,i)}/results/test_predictions.csv" for m,i in jobs]
with ProcessPoolExecutor(8) as ex: res = dict(zip(jobs, ex.map(score, paths)))
fr = pd.read_csv('/home/claude/all41_corrected.csv', index_col=0)
check = {}
for (m,i),s in res.items():
    check[eid(m,i)] = (round(s.rl.mean(),5), round(fr.loc[eid(m,i),'test_rougeL'],5), len(s))
rng = np.random.default_rng(42); B = 2000
def boot(a, b):
    d = a - b; n = len(d); idx = rng.integers(0, n, (B, n)); bs = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float(min(1, 2*min((bs<=0).mean(), (bs>=0).mean())))
out = {}
for i, ds in enumerate(DS):
    t = res[('FedTGen', i)]
    for m in ['FedAvg-LoRA','FedProx-LoRA','SCAFFOLD-LoRA','Local-LoRA','Cent-LoRA','FedAvg-Full']:
        o = res[(m, i)]
        if m == 'Local-LoRA':  # local test set = union of client test splits; align by text
            j = t.merge(o, on='key', suffixes=('_t','_o')).drop_duplicates('key')
            a, b, n = j.rl_t.values, j.rl_o.values, len(j)
        else:
            assert (t.key.values == o.key.values).all(); a, b, n = t.rl.values, o.rl.values, len(t)
        out[f'{ds}|{m}'] = dict(zip(['diff','lo','hi','p'], boot(a, b)), n=n)
json.dump({'check': check, 'boot': out}, open('/home/claude/boot.json','w'), indent=1)
for k,v in check.items(): print(k, v)
for k,v in out.items(): print(f"{k:28s} dRL={v['diff']:+.4f} CI[{v['lo']:+.4f},{v['hi']:+.4f}] p={v['p']:.3f} n={v['n']}")
