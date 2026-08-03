#!/usr/bin/env python3
"""Extract per-case browsing-subagent retrieved-title SETS from the ArcticSwarm
multi-agent run (0710_bcp_again_v3), for within-swarm search-path overlap."""
import json, glob, re, os, sys, time
D="/data/soyoung/important/arcticswarm/0710_bcp_again_v3"
OUT="/tmp/multi_extract.json"
TITLE_RE=re.compile(r'title:\s*(.+?)\s*\nauthor:', re.S)
def agent_titles(msgs):
    titles=set()
    for m in msgs or []:
        c=m.get("content")
        if not isinstance(c,list): continue
        for b in c:
            if isinstance(b,dict) and b.get("type")=="tool_result":
                inner=b.get("content"); txt=""
                if isinstance(inner,list): txt="\n".join(x.get("text","") for x in inner if isinstance(x,dict) and x.get("type")=="text")
                elif isinstance(inner,str): txt=inner
                for ti in TITLE_RE.findall(txt): titles.add(ti.strip())
    return sorted(titles)
out={}; t0=time.time()
for f in sorted(glob.glob(f"{D}/trajectories/*.json")):
    qid=os.path.basename(f)[:-5]
    try: t=json.load(open(f))
    except Exception: continue
    node=t.get("trajectory",[{}])
    node=node[0] if isinstance(node,list) and node else {}
    sa=node.get("subagents",[]) or []
    sets=[]
    for s in sa:
        ts=agent_titles(s.get("messages",[]))
        if ts: sets.append(ts)      # only agents that actually retrieved (browsing search paths)
    out[qid]=sets
json.dump(out, open(OUT,"w"))
nag=[len(v) for v in out.values()]
import statistics as st
print(f"cases={len(out)} mean_browsing_agents/case={st.mean(nag):.2f} max={max(nag)} "
      f"cases>=2agents={sum(1 for x in nag if x>=2)} ({time.time()-t0:.1f}s)")
