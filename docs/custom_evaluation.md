# Custom evaluation

Evaluate an ArcticSwarm agent on **your own dataset** with **your own judge
rubric** — entirely through config, without editing framework code.

There are three pieces:

1. A CSV of your questions + gold answers in the *unified eval format*.
2. A YAML config that points the eval at that CSV.
3. (Optional) A custom judge rubric `.txt` that grades every case your way.

A ready-to-copy template lives at
[`conf/bench/custom_example.yaml`](../conf/bench/custom_example.yaml).

---

## 1. Prepare your dataset CSV

ArcticSwarm reads one "unified eval" CSV: one row per question, with the
question text, the reference (gold) answer, and JSON-encoded metadata. The
column header is:

```
TURN_INDEX, TURN, PAST_TURNS, TOOL_CHOICE, TOOLS, REFERENCE_MESSAGE,
CONV_ID, TOOL_RESOURCES, ATTRIBUTES, TURNS, DATE_OVERRIDE, REFERENCE_TOOLS
```

You don't have to hand-write that. The helper
[`write_unified_eval_csv`](../arcticswarm/eval/data/external/utils.py) builds a
correct CSV from a simple list of dicts. Each dict needs `id`, `question`, and
`expected_answer` (plus an optional `attributes` dict for extra per-row
metadata):

```python
from pathlib import Path
from arcticswarm.eval.data.external.utils import write_unified_eval_csv

examples = [
    {
        "id": "myq-001",
        "question": "What year was the Eiffel Tower completed?",
        "expected_answer": "1889",
    },
    {
        "id": "myq-002",
        "question": "Who wrote the novel 'Beloved'?",
        "expected_answer": "Toni Morrison",
    },
]

write_unified_eval_csv(
    examples,
    output_path=Path("path/to/my_dataset.csv"),
    dataset_name="MY_DATASET",   # must match eval.datasets in your config
    eval_mode="QA",
)
```

This writes every row with `ATTRIBUTES = {"dataset": "MY_DATASET",
"eval_mode": "QA", "is_vip": true}`, so the default `eval.vip_only: true`
already includes your cases. The `dataset_name` you pass here is what you list
in `eval.datasets` (matched case-insensitively).

If you prefer to write the CSV yourself, match the header above exactly: a
1-row example (after the header) looks like:

```csv
0,"What year was the Eiffel Tower completed?",[],"{""type"": ""auto""}",[],"{""text"": ""1889""}",myq-001,{},"{""dataset"": ""MY_DATASET"", ""eval_mode"": ""QA"", ""is_vip"": true}","[{""Role"": ""user"", ""Content"": [{""Type"": ""text"", ""Text"": ""What year was the Eiffel Tower completed?""}]}]",,"[""web_search""]"
```

> The bundled benchmark CSVs (BrowseComp etc.) are **not** shipped in the repo —
> they're regenerated from public sources with `bash scripts/fetch_datasets.sh`.
> If you point `eval.csv_path` at a missing file, the loader raises a clear
> error telling you to check the path or run that script.

---

## 2. Point a config at your CSV

Copy the template and edit the `eval` block:

```yaml
eval:
  csv_path: path/to/my_dataset.csv   # the CSV you just wrote (repo-root-relative)
  datasets: [MY_DATASET]             # matches dataset_name from step 1
  judge_model: openai-gpt-4.1        # any judge-capable model id
  output: results/custom_example     # where results land
  parallel: 3
  timeout: 300
```

The rest of the YAML (`llm`, `swarm`, `web`, `tools`) controls *how the agent
runs*. The template defaults to a small web-research swarm on
`claude-sonnet-4-5`; change `llm.model`, set `swarm.enabled: false` for a
single agent, or `web.enabled: false` for an offline/closed-book eval.

---

## 3. (Optional) Custom judge rubric

By default the agent's answer is graded by the built-in QA / BrowseComp judge.
To grade with **your own rubric**, set `eval.custom_judge_prompt` to a `.txt`
template:

```yaml
eval:
  custom_judge_prompt: arcticswarm/eval/prompts/custom_example_eval.txt
```

When set, this rubric grades **every** case, overriding the built-in
per-dataset prompts. The template may reference three placeholders:

| Placeholder        | Filled with                       |
| ------------------ | --------------------------------- |
| `{question}`       | the question text                 |
| `{response}`       | the agent's answer                |
| `{correct_answer}` | the reference (gold) answer       |

**Verdict format contract.** Your rubric must make the judge end with a verdict
the parser understands. Any one of these (case-insensitive) works:

* a line `correct: true` / `correct: false` (also accepts `yes` / `no`);
* a line `GRADE: CORRECT` / `GRADE: INCORRECT`;
* a JSON object `{"correct": true, "reasoning": "..."}` (an optional
  `confidence` / `judge_confidence` number is mapped into the result).

If no verdict can be parsed, the case is scored **incorrect** (conservative).

A ready example is
[`arcticswarm/eval/prompts/custom_example_eval.txt`](../arcticswarm/eval/prompts/custom_example_eval.txt).

---

## 4. Run it

```bash
arcticswarm-eval --config conf/bench/custom_example.yaml
```

Override any knob inline, e.g. a quick 5-case smoke test on 8 workers:

```bash
arcticswarm-eval --config conf/bench/custom_example.yaml eval.limit=5 eval.parallel=8
```

---

## 5. Where results land

Everything is written under `results/<output>/` (i.e. the directory named by
`eval.output`):

| File / dir            | Contents                                                  |
| --------------------- | --------------------------------------------------------- |
| `report.json`         | full per-case + aggregate metrics (accuracy, latency, …)  |
| `breakdown.md`        | human-readable per-dataset breakdown                      |
| `resolved_config.yaml`| the fully-resolved config used for the run                |
| `trajectories/`       | per-case agent trajectories (one JSON per `CONV_ID`)      |

Per-case correctness lives in `report.json` under `per_case[].judge_correct`;
the overall accuracy is in the aggregate metrics. Re-grade a finished run
without re-running the agent via `arcticswarm-eval --config ... --rejudge
results/custom_example` (handy when iterating on a custom rubric).
