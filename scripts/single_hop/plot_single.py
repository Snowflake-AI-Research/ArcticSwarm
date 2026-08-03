#!/usr/bin/env python3
import json, glob, os, itertools, math
import numpy as np
from collections import Counter, defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=os.path.dirname(__file__); DATA=os.path.join(HERE,"data")
IMG=os.path.abspath(os.path.join(HERE,"..","..","..","images")); os.makedirs(IMG,exist_ok=True)
files=sorted(glob.glob(f"{DATA}/run_*.json"), key=lambda p:int(p.split("_")[-1].split(".")[0]))
runs=[json.load(open(f)) for f in files]; R=len(runs)
common=set(runs[0])
for r in runs[1:]: common&=set(r)
common=sorted(common); Q=len(common)
C=np.array([[runs[i][q]['correct'] for i in range(R)] for q in common]); k=C.sum(1)
tok=np.array([[runs[i][q]['total_tok'] for i in range(R)] for q in common]); mean_run_tok=tok.mean()/1e6
ans=[[runs[i][q]['answer_norm'] for i in range(R)] for q in common]
ans_corr=[]
for qi in range(Q):
    d=defaultdict(list)
    for i in range(R): d[ans[qi][i]].append(runs[i][common[qi]]['correct'])
    ans_corr.append({a:(1 if np.mean(v)>=.5 else 0) for a,v in d.items()})
Ns=list(range(1,R+1))
def bestN(N):
    dn=math.comb(R,N); return float(np.mean([1-(math.comb(R-int(x),N)/dn if R-int(x)>=N else 0.0) for x in k]))
def maj(N,mc=6000,seed=0):
    nC=math.comb(R,N)
    subs=list(itertools.combinations(range(R),N)) if nC<=20000 else \
         [tuple(np.random.default_rng(seed+t).choice(R,N,replace=False)) for t in range(mc)]
    vals=[]
    for cb in subs:
        cols=list(cb); acc=0.0
        for qi in range(Q):
            cnt=Counter(ans[qi][i] for i in cols); mx=max(cnt.values())
            top=[a for a,c in cnt.items() if c==mx]; acc+=np.mean([ans_corr[qi][a] for a in top])
        vals.append(acc/Q)
    return np.mean(vals),np.std(vals)
bestv=np.array([bestN(N) for N in Ns])
mv=[maj(N) for N in Ns]; majm=np.array([m[0] for m in mv]); majs=np.array([m[1] for m in mv])
wt=float(np.mean(k/R))
xt=np.array(Ns)*mean_run_tok
# end-to-end tokens/case (3-run means); forced-isolation e2e unavailable, so it is
# omitted from the token panel to avoid mixing orchestrator- and end-to-end accounting.
AS=(24.86,0.8257,"ArcticSwarm"); UB=(24.38,0.8000,"Unrestricted BBS")
ceil=bestv[-1]

# FIG1
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,3.9))
ax1.plot(Ns,bestv*100,"-o",ms=4,color="#1b6ca8",label="Best-of-$N$ (oracle select)")
ax1.plot(Ns,majm*100,"-s",ms=4,color="#c1440e",label="Majority vote")
ax1.fill_between(Ns,(majm-majs)*100,(majm+majs)*100,color="#c1440e",alpha=.15)
ax1.plot(Ns,[wt*100]*len(Ns),"--",color="#5a5a5a",label="Weighted (partial credit)")
ax1.axhline(AS[1]*100,ls=":",color="#2a9d8f",lw=1.6,label=f"{AS[2]} ({AS[1]*100:.1f}%)")
ax1.axhline(ceil*100,ls="-",color="#999",lw=1,alpha=.7)
ax1.annotate(f"best-of-{R} union ceiling {ceil*100:.1f}%",(1,ceil*100),xytext=(1.2,ceil*100+1.1),fontsize=7.3,color="#666")
ax1.set_xlabel("$N$ independent single-agent runs"); ax1.set_ylabel("Accuracy (%)")
ax1.set_xticks([1,5,10,15,20]); ax1.set_ylim(44,88); ax1.legend(fontsize=7.3,loc="center right"); ax1.grid(alpha=.25)
ax1.set_title("(a) Aggregating $N$ single-agent runs",fontsize=9.5)
ax2.plot(xt,bestv*100,"-o",ms=4,color="#1b6ca8",label="Best-of-$N$ (oracle)")
ax2.plot(xt,majm*100,"-s",ms=4,color="#c1440e",label="Majority vote")
for (x,y,lab),mk,col in [(AS,"*","#2a9d8f"),(UB,"X","#e9a319")]:
    ax2.scatter([x],[y*100],marker=mk,s=95,color=col,zorder=5,edgecolor="k",lw=.4,label=lab)
ax2.set_xscale("log"); ax2.set_xlabel("Tokens per case (millions, log scale)"); ax2.set_ylabel("Accuracy (%)")
ax2.set_ylim(44,88); ax2.legend(fontsize=7.1,loc="lower right"); ax2.grid(alpha=.25,which="both")
ax2.set_title("(b) Accuracy vs. compute",fontsize=9.5)
plt.tight_layout(); plt.savefig(f"{IMG}/single_bestofn_bcp.png",dpi=200,bbox_inches="tight"); plt.close()

# FIG2 hist
fig,ax=plt.subplots(figsize=(5.6,3.2))
hist=[int((k==kk).sum()) for kk in range(R+1)]
ax.bar(range(R+1),hist,color="#1b6ca8",alpha=.85,edgecolor="k",lw=.3)
ax.set_xlabel(f"# of {R} single-agent runs that solve the question"); ax.set_ylabel("# questions")
ax.set_xticks([0,5,10,15,20]); ax.set_title(f"Per-question solve rate (BrowseComp-Plus, $n$={Q})",fontsize=9.5); ax.grid(alpha=.25,axis="y")
plt.tight_layout(); plt.savefig(f"{IMG}/single_passrate_bcp.png",dpi=200,bbox_inches="tight"); plt.close()

# FIG3 overlap
title_sets={q:[set(runs[i][q]['titles']) for i in range(R)] for q in common}
def jac(a,b):
    u=len(a|b); return (len(a&b)/u) if u else None
perq=[]
for q in common:
    ts=title_sets[q]; ps=[jac(ts[i],ts[j]) for i in range(R) for j in range(i+1,R)]
    ps=[p for p in ps if p is not None]
    if ps: perq.append(np.mean(ps))
Jbar=float(np.mean(perq))
Ns2=list(range(2,R+1)); Neff=[N/(1+(N-1)*Jbar) for N in Ns2]
fig,(axa,axb)=plt.subplots(1,2,figsize=(9,3.6))
axa.plot(Ns2,[Jbar]*len(Ns2),"-o",ms=3,color="#1b6ca8")
axa.set_xlabel("$N$ independent single-agent runs"); axa.set_ylabel("Mean pairwise Jaccard\n(retrieved corpus docs)")
axa.set_ylim(0,0.32); axa.set_xticks([2,5,10,15,20]); axa.grid(alpha=.25); axa.set_title("(a) Search-path overlap",fontsize=9.5)
axb.plot(Ns2,Neff,"-o",ms=3,color="#1b6ca8",label="Overlap-adjusted $N/(1+(N-1)\\bar J)$")
axb.plot(Ns2,Ns2,"--",color="#999",label="Perfect independence ($N$)")
axb.set_xlabel("$N$ independent single-agent runs"); axb.set_ylabel("Effective independent rollouts")
axb.set_xticks([2,5,10,15,20]); axb.legend(fontsize=7.4); axb.grid(alpha=.25); axb.set_title("(b) Effective rollout count",fontsize=9.5)
plt.tight_layout(); plt.savefig(f"{IMG}/single_overlap_bcp.png",dpi=200,bbox_inches="tight"); plt.close()
print(f"R={R} Q={Q} single={C.mean():.4f} best20={bestv[-1]:.4f} maj20={majm[-1]:.4f} wt={wt:.4f} Jbar={Jbar:.4f} Neff20={Neff[-1]:.3f} tok={mean_run_tok:.3f}M")
print("figures written.")
