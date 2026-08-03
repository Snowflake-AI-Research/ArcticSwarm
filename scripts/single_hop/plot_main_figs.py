import json, glob, os, itertools, math
import numpy as np
from collections import Counter, defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
DATA="data"; IMG=os.path.abspath(os.path.join("..","..","..","images"))
files=sorted(glob.glob(f"{DATA}/run_*.json"),key=lambda p:int(p.split('_')[-1].split('.')[0]))
runs=[json.load(open(f)) for f in files]; R=len(runs)
common=set(runs[0])
for r in runs[1:]: common&=set(r)
common=sorted(common); Q=len(common)
C=np.array([[runs[i][q]['correct'] for i in range(R)] for q in common]); k=C.sum(1)
tok=np.array([[runs[i][q]['total_tok'] for i in range(R)] for q in common]); mtok=tok.mean()/1e6
ans=[[runs[i][q]['answer_norm'] for i in range(R)] for q in common]
ac=[]
for qi in range(Q):
    d=defaultdict(list)
    for i in range(R): d[ans[qi][i]].append(runs[i][common[qi]]['correct'])
    ac.append({a:(1 if np.mean(v)>=.5 else 0) for a,v in d.items()})
Ns=list(range(1,R+1))
def bestN(N):
    dn=math.comb(R,N); return float(np.mean([1-(math.comb(R-int(x),N)/dn if R-int(x)>=N else 0.0) for x in k]))
def maj(N,mc=6000,seed=1):
    nC=math.comb(R,N)
    subs=list(itertools.combinations(range(R),N)) if nC<=15000 else [tuple(np.random.default_rng(seed+t).choice(R,N,replace=False)) for t in range(mc)]
    vals=[]
    for cb in subs:
        cols=list(cb); acc=0.0
        for qi in range(Q):
            cnt=Counter(ans[qi][i] for i in cols); mx=max(cnt.values())
            top=[a for a,c in cnt.items() if c==mx]; acc+=np.mean([ac[qi][a] for a in top])
        vals.append(acc/Q)
    return np.mean(vals)
bestv=np.array([bestN(N) for N in Ns]); majv=np.array([maj(N) for N in Ns]); wt=float(np.mean(k/R))
AS_ACC=0.8257; ceil=bestv[-1]  # 3-run mean (was single-run 0.8313)

# ---------------- FIG A: best-of-N main (single panel) ----------------
fig,ax=plt.subplots(figsize=(3.5,3.0))
ax.plot(Ns,bestv*100,"-o",ms=3.5,color="#1b6ca8",label="Best-of-$N$ (oracle select)")
ax.plot(Ns,majv*100,"-s",ms=3.5,color="#c1440e",label="Majority vote")
ax.plot(Ns,[wt*100]*len(Ns),":",color="#5a5a5a",lw=1.4,label="Weighted (partial credit)")
ax.axhline(AS_ACC*100,ls="--",color="#2a9d8f",lw=1.8,label="ArcticSwarm (82.6%)")
ax.axhline(ceil*100,ls="-",color="#bbb",lw=1)
ax.annotate(f"best-of-{R} ceiling {ceil*100:.1f}%",(1,ceil*100),xytext=(4.6,ceil*100+1.0),fontsize=6.3,color="#777")
ax.set_xlabel("$N$ independent single-agent runs"); ax.set_ylabel("Accuracy (%)")
ax.set_xticks([1,5,10,15,20]); ax.set_ylim(44,88); ax.grid(alpha=.25); ax.legend(fontsize=6.6,loc="lower right")
plt.tight_layout(); fa=f"{IMG}/bestofn_main.png"; plt.savefig(fa,dpi=220,bbox_inches="tight"); plt.close()

# ---------------- FIG B: overlap main (empirical single-pool line + band) ----------------
# precompute per-q 20x20 pairwise jaccard
title_sets=[[set(runs[i][q]['titles']) for i in range(R)] for q in common]
PJ=np.full((Q,R,R),np.nan)
for qi in range(Q):
    S=title_sets[qi]
    for i in range(R):
        for j in range(i+1,R):
            u=len(S[i]|S[j])
            if u: PJ[qi,i,j]=len(S[i]&S[j])/u
def single_pool_stats(N,S=120,seed=7):
    rng=np.random.default_rng(seed+N)
    means=[]
    for _ in range(S):
        idx=rng.choice(R,N,replace=False); pv=[]
        for a in range(N):
            for b in range(a+1,N):
                i,j=sorted((idx[a],idx[b])); col=PJ[:,i,j]; pv.append(col)
        M=np.nanmean(np.vstack(pv),axis=0)      # per-q mean pairwise over chosen pairs
        means.append(np.nanmean(M))
    return np.mean(means),np.std(means)
res=json.load(open(f"{DATA}/../results_overlap_main.json")); AS=[(N,J,nc) for N,J,nc in res["AS"] if nc>=40]
axs_x=[a[0] for a in AS]; axs_y=[a[1] for a in AS]
Ns_sp=list(range(2,17))
spm=[]; sps=[]
for N in Ns_sp:
    m,s=single_pool_stats(N); spm.append(m); sps.append(s)
spm=np.array(spm); sps=np.array(sps)
fig,ax=plt.subplots(figsize=(3.6,3.0))
ax.plot(Ns_sp,spm,"--s",ms=3,color="#c1440e",label="Independent single-agent\nruns (pool)")
ax.fill_between(Ns_sp,spm-sps,spm+sps,color="#c1440e",alpha=.15)
ax.plot(axs_x,axs_y,"-o",ms=4,color="#1b6ca8",label="ArcticSwarm (16-agent\nswarm, 82.6%)")
ax.set_xlabel("$N$ search paths (agents / runs)"); ax.set_ylabel("Mean pairwise Jaccard\n(retrieved corpus docs)")
ax.set_ylim(0,0.24); ax.set_xticks(range(2,17,2)); ax.grid(alpha=.25); ax.legend(fontsize=6.6,loc="lower left")
ax.annotate("lower = more\ndecorrelated",(12,axs_y[-1]),xytext=(7.3,0.20),fontsize=6.3,color="#555",
            arrowprops=dict(arrowstyle="->",color="#aaa",lw=.8))
plt.tight_layout(); fb=f"{IMG}/overlap_bcp_main.png"; plt.savefig(fb,dpi=220,bbox_inches="tight"); plt.close()
print("wrote",fa,fb,sep="\n  ")
print(f"single-pool empirical mean pairwise by N: "+", ".join(f'{n}:{m:.3f}' for n,m in zip(Ns_sp,spm)))
print(f"best20={bestv[-1]:.4f} maj20={majv[-1]:.4f} wt={wt:.4f}")
