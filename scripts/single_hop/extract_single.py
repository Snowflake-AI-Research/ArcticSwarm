#!/usr/bin/env python3
"""Extract per-(run,qid) records from 0725_bcp_single for best-of-N + overlap analysis.
Outputs one compact JSON per run: {qid: {correct,answer_raw,answer_norm,in_tok,out_tok,total_tok,titles:[...]}}.
Titles are retrieved-corpus-doc identifiers parsed from tool_result texts (analog of URLs)."""
import json, glob, re, os, sys, string

D = "/data/soyoung/important/arcticswarm/0725_bcp_single"
OUT = "/tmp/single_extract"
os.makedirs(OUT, exist_ok=True)

TITLE_RE = re.compile(r'title:\s*(.+?)\s*\nauthor:', re.S)
EFA_RE = re.compile(r'extracted_final_answer:\s*(.*)')
_punct = str.maketrans('', '', string.punctuation)

def norm(a):
    if a is None: return ""
    a = a.strip().lower().translate(_punct)
    return re.sub(r'\s+', ' ', a).strip()

def titles_from_traj(path):
    try:
        t = json.load(open(path))
    except Exception:
        return []
    conv = t[0] if isinstance(t, list) and t else t
    msgs = conv.get("messages") if isinstance(conv, dict) else (conv if isinstance(conv, list) else [])
    titles = set()
    for m in msgs or []:
        c = m.get("content")
        if not isinstance(c, list): continue
        for b in c:
            if isinstance(b, dict) and b.get("type") == "tool_result":
                inner = b.get("content")
                text = ""
                if isinstance(inner, list):
                    text = "\n".join(x.get("text","") for x in inner if isinstance(x, dict) and x.get("type")=="text")
                elif isinstance(inner, str):
                    text = inner
                for ti in TITLE_RE.findall(text):
                    titles.add(ti.strip())
    return sorted(titles)

def run_one(i):
    rd = f"{D}/run_{i}"
    rep = json.load(open(f"{rd}/report.json"))
    out = {}
    for c in rep.get("per_case", []):
        qid = c.get("conv_id")
        jc = c.get("judge_correct")
        if jc is None:  # unjudged -> skip
            continue
        efa = ""
        m = EFA_RE.search(c.get("judge_raw_output","") or "")
        if m: efa = m.group(1).strip()
        tu = c.get("token_usage",{}) or {}
        intok = tu.get("input_tokens", 0) or 0
        outtok = tu.get("output_tokens", 0) or 0
        tot = c.get("total_tokens", intok+outtok) or (intok+outtok)
        titles = titles_from_traj(f"{rd}/trajectories/{qid}.json")
        out[qid] = {
            "correct": 1 if jc else 0,
            "answer_raw": efa,
            "answer_norm": norm(efa),
            "in_tok": intok, "out_tok": outtok, "total_tok": tot,
            "n_titles": len(titles), "titles": titles,
        }
    json.dump(out, open(f"{OUT}/run_{i}.json","w"))
    accs = sum(v["correct"] for v in out.values())/len(out)
    mt = sum(v["total_tok"] for v in out.values())/len(out)
    print(f"run_{i}: n={len(out)} acc={accs:.4f} mean_total_tok={mt:.0f} mean_titles={sum(v['n_titles'] for v in out.values())/len(out):.1f}")

if __name__ == "__main__":
    args = sys.argv[1:]
    runs = [int(x) for x in args] if args else list(range(10))
    import time
    for i in runs:
        t0=time.time(); run_one(i); print(f"  ({time.time()-t0:.1f}s)")
