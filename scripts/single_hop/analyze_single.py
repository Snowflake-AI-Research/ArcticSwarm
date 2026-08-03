#!/usr/bin/env python3
import json, glob, os, itertools, math
import numpy as np
from collections import Counter, defaultdict

DATA=os.path.join(os.path.dirname(__file__),"data")
files=sorted(glob.glob(f"{DATA}/run_*.json"), key=lambda p:int(p.split("_")[-1].split(".")[0]))
runs=[json.load(open(f)) for f in files]; R=len(runs)
common=set(runs[0])
for r in runs[1:]: common&=set(r)
common=sorted(common); Q=len(common)
print(f"R={R} runs  common judged qids={Q}")
per_run_acc=[np.mean([runs[i][q]['correct'] for q in common]) for i in range(R)]
print(f"mean single-run acc={np.mean(per_run_acc):.4f} (min {min(per_run_acc):.4f} max {max(per_run_acc):.4f})")
C=np.array([[runs[i][q]['correct'] for i in range(R)] for q in common])   # QxR
k=C.sum(1)                                                                # per-q #correct
tok=np.array([[runs[i][q]['total_tok'] for i in range(R)] for q in common])
mean_run_tok=tok.mean()/1e6
print(f"mean single-run tokens/case={mean_run_tok:.3f}M")

# pass-rate hist + union ceiling
hist=[int((k==kk).sum()) for kk in range(R+1)]
print("\npass-rate hist (k solved of R):")
for kk in range(R+1): print(f"  k={kk:2d}: {hist[kk]:4d} ({100*hist[kk]/Q:.1f}%)")
print(f"never solved k=0: {hist[0]}/{Q}={hist[0]/Q:.4f}; union(best-of-{R})={1-hist[0]/Q:.4f}; all-{R}={hist[R]/Q:.4f}")

# answer correctness map
ans=[[runs[i][q]['answer_norm'] for i in range(R)] for q in common]
ans_corr=[]
for qi in range(Q):
    d=defaultdict(list)
    for i in range(R): d[ans[qi][i]].append(runs[i][common[qi]]['correct'])
    ans_corr.append({a:(1 if np.mean(v)>=.5 else 0) for a,v in d.items()})

def bestN_exact(N):
    # per q: 1 - C(R-k,N)/C(R,N)
    denom=math.comb(R,N)
    return float(np.mean([1 - (math.comb(R-int(kk),N)/denom if R-int(kk)>=N else 0.0) for kk in k]))
def weighted_exact(N):
    return float(np.mean(k/R))  # hypergeometric mean, independent of N
def majority(N, mc=30000, seed=0):
    nC=math.comb(R,N)
    if nC<=30000:
        subsets=list(itertools.combinations(range(R),N)); exact=True
    else:
        rng=np.random.default_rng(seed)
        subsets=[tuple(rng.choice(R,size=N,replace=False)) for _ in range(mc)]; exact=False
    vals=[]
    for cb in subsets:
        cols=list(cb); acc=0.0
        for qi in range(Q):
            cnt=Counter(ans[qi][i] for i in cols); mx=max(cnt.values())
            top=[a for a,c in cnt.items() if c==mx]
            acc+=np.mean([ans_corr[qi][a] for a in top])
        vals.append(acc/Q)
    return float(np.mean(vals)), float(np.std(vals)), exact, len(subsets)

Ns_full=list(range(1,R+1))
best={N:bestN_exact(N) for N in Ns_full}
wt={N:weighted_exact(N) for N in Ns_full}
report_Ns=[1,2,3,5,10,15,16,18,20]
maj={}
print(f"\n{'N':>2} {'bestN':>8} {'majority':>9} {'wt':>7} {'tok/case':>10} {'maj_mode':>10}")
for N in report_Ns:
    m,s,ex,ns=majority(N)
    maj[N]=(m,s,ex,ns)
    print(f"{N:>2} {best[N]:>8.4f} {m:>9.4f} {wt[N]:>7.4f} {N*mean_run_tok:>9.2f}M {('exact' if ex else f'MC{ns}'):>10}")

# overlap: exact global mean pairwise title Jaccard (all C(R,2) pairs)
title_sets={q:[set(runs[i][q]['titles']) for i in range(R)] for q in common}
def jac(a,b):
    u=len(a|b); return (len(a&b)/u) if u else None
perq=[]
for q in common:
    ts=title_sets[q]; ps=[]
    for i in range(R):
        for j in range(i+1,R):
            v=jac(ts[i],ts[j])
            if v is not None: ps.append(v)
    if ps: perq.append(np.mean(ps))
Jbar=float(np.mean(perq))
print(f"\nglobal mean pairwise title Jaccard = {Jbar:.4f}")
for N in [2,5,10,15,20]:
    print(f"  N={N:2d} N_eff=N/(1+(N-1)J)={N/(1+(N-1)*Jbar):.3f}")

json.dump({"R":R,"Q":Q,"mean_single_acc":float(np.mean(per_run_acc)),"per_run_acc":per_run_acc,
 "mean_run_tok":mean_run_tok,"hist":hist,
 "best":best,"weighted":wt,"majority":{str(n):maj[n] for n in maj},
 "best_full":{str(n):best[n] for n in Ns_full},
 "Jbar":Jbar}, open(f"{DATA}/../results_single20.json","w"), indent=1)
print("\nsaved results_single20.json")
