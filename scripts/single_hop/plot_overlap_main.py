import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
DATA="data"; IMG=os.path.abspath(os.path.join("..","..","..","images"))
r=json.load(open(f"{DATA}/../results_overlap_main.json"))
AS=[(N,J,nc) for (N,J,nc) in r["AS"] if nc>=40]   # keep N with >=40 supporting cases
sJ=r["single_J"]
xs=[a[0] for a in AS]; ys=[a[1] for a in AS]
fig,ax=plt.subplots(figsize=(3.5,3.0))
xr=list(range(2, max(xs)+1))
ax.plot(xr,[sJ]*len(xr),"--s",ms=3,color="#c1440e",label="Independent single-agent\nruns (pool, $N{=}20$)")
ax.plot(xs,ys,"-o",ms=4,color="#1b6ca8",label="\\textbf{ArcticSwarm} (16-agent\nswarm, 82.6\\%)")
ax.set_xlabel("$N$ search paths (agents / runs)")
ax.set_ylabel("Mean pairwise Jaccard\n(retrieved corpus docs)")
ax.set_ylim(0,0.24); ax.set_xticks(range(2,max(xs)+1,2))
ax.grid(alpha=.25); ax.legend(fontsize=6.6,loc="lower left")
ax.annotate("lower = more\ndecorrelated search", (max(xs),min(ys)), xytext=(max(xs)-4.5,0.045),
            fontsize=6.3,color="#555")
plt.tight_layout(); f=f"{IMG}/overlap_bcp_main.png"
plt.savefig(f,dpi=220,bbox_inches="tight"); plt.close()
print("wrote",f); print("AS pts:",list(zip(xs,[round(y,4) for y in ys]))); print("single flat:",round(sJ,4))
