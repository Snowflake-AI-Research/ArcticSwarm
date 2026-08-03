#!/usr/bin/env python3
"""Distinct-passage COVERAGE (accumulation) curves on the corpus.
Unit = distinct retrieved passage (normalized web_search chunk body, hashed).
Single: expected #distinct passages covered by k of 20 independent runs.
ArcticSwarm(0710): expected #distinct passages covered by k of a case's active browsing subagents."""
import json, glob, re, hashlib, math, os, time, statistics as st
DS="/data/soyoung/important/arcticswarm/0725_bcp_single"
DM="/data/soyoung/important/arcticswarm/0710_bcp_again_v3"
OUT="/tmp/coverage_results.json"
SPLIT=re.compile(r'(?:^|\n)\s*\d+\.\s*\(cosine=[\d.]+\)\s*')
HDR=re.compile(r'^\s*---\s*\n.*?\n---\s*\n', re.S)
SQ=re.compile(r'\[Source Quality:.*?\]', re.S)
def passages(txt):
    out=[]
    for p in SPLIT.split(txt)[1:]:
        b=SQ.sub('',HDR.sub('',p)); b=re.sub(r'\s+',' ',b).strip().lower()
        if len(b)>=20: out.append(hashlib.sha1(b[:160].encode()).hexdigest()[:12])
    return out
def agent_pass(msgs):
    id2n={}
    for m in msgs:
        c=m.get("content")
        if isinstance(c,list):
            for b in c:
                if isinstance(b,dict) and b.get("type")=="tool_use": id2n[b.get("id")]=b.get("name")
    ps=set()
    for m in msgs:
        c=m.get("content")
        if not isinstance(c,list): continue
        for b in c:
            if isinstance(b,dict) and b.get("type")=="tool_result" and id2n.get(b.get("tool_use_id"))=="web_search":
                inner=b.get("content"); txt="\n".join(x.get("text","") for x in inner if isinstance(x,dict) and x.get("type")=="text") if isinstance(inner,list) else (inner if isinstance(inner,str) else "")
                ps.update(passages(txt))
    return ps
def exp_union_curve(agent_sets):
    """Return dict k-> expected #distinct passages by random k-subset (k=1..m)."""
    m=len(agent_sets)
    if m==0: return {}
    cnt={}
    for s in agent_sets:
        for p in s: cnt[p]=cnt.get(p,0)+1
    out={}
    for k in range(1,m+1):
        denom=math.comb(m,k); tot=0.0
        for p,c in cnt.items():
            tot += 1 - (math.comb(m-c,k)/denom if m-c>=k else 0.0)
        out[k]=tot
    return out

# ---- SINGLE (20 runs) ----
t0=time.time()
qids=None
runsets={}   # qid -> list of 20 sets
for i in range(20):
    for f in glob.glob(f"{DS}/run_{i}/trajectories/*.json"):
        q=os.path.basename(f)[:-5]
        try: conv=json.load(open(f))[0]
        except: continue
        runsets.setdefault(q,[None]*20)[i]=agent_pass(conv.get("messages",[]))
# keep qids present in all 20
single_curves={}
for q,lst in runsets.items():
    if any(s is None for s in lst): continue
    single_curves[q]=exp_union_curve(lst)
Qs=len(single_curves)
single_k={}
for k in range(1,21):
    vals=[c[k] for c in single_curves.values() if k in c]
    single_k[k]=(st.mean(vals), st.pstdev(vals), len(vals))
print(f"single: Q={Qs} k1={single_k[1][0]:.1f} k20={single_k[20][0]:.1f} ({time.time()-t0:.0f}s)")

# ---- ArcticSwarm (0710) ----
t1=time.time()
AS_curves={}; Ndist=[]
for f in glob.glob(f"{DM}/trajectories/*.json"):
    q=os.path.basename(f)[:-5]
    try: node=json.load(open(f))["trajectory"][0]
    except: continue
    sets=[agent_pass(s.get("messages",[])) for s in node.get("subagents",[])]
    sets=[s for s in sets if s]   # active browsing agents (>=1 search passage)
    Ndist.append(len(sets))
    if sets: AS_curves[q]=exp_union_curve(sets)
AS_k={}
maxN=max(len(c) for c in AS_curves.values())
for k in range(1,maxN+1):
    vals=[c[k] for c in AS_curves.values() if k in c]
    if len(vals)>=40: AS_k[k]=(st.mean(vals), st.pstdev(vals), len(vals))
print(f"AS: cases={len(AS_curves)} meanN={st.mean(Ndist):.2f} maxN={max(Ndist)} k1={AS_k[1][0]:.1f} ({time.time()-t1:.0f}s)")
import collections
hist=collections.Counter(Ndist)
json.dump({"single_k":single_k,"AS_k":AS_k,"Ndist_hist":dict(sorted(hist.items())),
           "single_Q":Qs,"AS_meanN":st.mean(Ndist)}, open(OUT,"w"))
print("saved",OUT)
