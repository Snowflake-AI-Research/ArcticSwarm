import json, glob, itertools
import numpy as np
DATA="data"
# ---- ArcticSwarm multi-agent (0710) ----
multi=json.load(open(f"{DATA}/multi_0710.json"))
def case_pair_jaccs(sets):
    S=[set(x) for x in sets]; out=[]
    for i in range(len(S)):
        for j in range(i+1,len(S)):
            u=len(S[i]|S[j])
            if u: out.append(len(S[i]&S[j])/u)
    return out
# per case: list of agents' sets; case_mean_pairwise + n_agents
case_stats=[]  # (n_agents, mean_pairwise or None, pair_list)
for qid,sets in multi.items():
    m=len(sets)
    if m>=2:
        pj=case_pair_jaccs(sets)
        case_stats.append((m, np.mean(pj) if pj else None))
    else:
        case_stats.append((m, None))
maxN=15
AS=[]  # (N, meanJ, ncases)
for N in range(2, maxN+1):
    vals=[cm for (m,cm) in case_stats if m>=N and cm is not None]
    if len(vals)>=20:
        AS.append((N, float(np.mean(vals)), len(vals)))
print("ArcticSwarm multi-agent within-swarm overlap vs N:")
for N,J,nc in AS: print(f"  N={N:2d} meanJ={J:.4f} ncases={nc}")

# ---- single-agent pool (20 runs) ----
files=sorted(glob.glob(f"{DATA}/run_*.json"), key=lambda p:int(p.split('_')[-1].split('.')[0]))
runs=[json.load(open(f)) for f in files]; R=len(runs)
common=set(runs[0])
for r in runs[1:]: common&=set(r)
common=sorted(common)
# per-question all-pairs mean pairwise Jaccard over the 20 runs (== expectation for any N-subset)
perq=[]
for q in common:
    S=[set(runs[i][q]['titles']) for i in range(R)]; pj=[]
    for i in range(R):
        for j in range(i+1,R):
            u=len(S[i]|S[j])
            if u: pj.append(len(S[i]&S[j])/u)
    if pj: perq.append(np.mean(pj))
single_J=float(np.mean(perq))
print(f"\nSingle-agent pool: flat mean pairwise Jaccard = {single_J:.4f} (independent of N)")
json.dump({"AS":AS,"single_J":single_J}, open(f"{DATA}/../results_overlap_main.json","w"), indent=1)
