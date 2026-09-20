#!/usr/bin/env bash
# Re-derive per_instance.csv (with outcomes + pytest_summary) + aggregate
# for any run dir. Reads model name from each chunk's preds.json so the
# eval-inconsistency detector knows the deterministic test_output path.
set -u
RUN="${1:-}"
if [ -z "$RUN" ] || [ ! -d "$RUN" ]; then echo "usage: $0 <run-dir>"; exit 2; fi
cd /data/users/krishna/sky/skylight
source env.sh
RUN="$RUN" .venv/bin/python - <<'PY'
import os, json
from pathlib import Path
from collections import Counter
from skylight.bench.agentic import _read_exit_statuses, _preds_patches, _write_per_instance_csv
from skylight.bench.benchmarks.mini_swe_agent import MiniSweAgent
run = Path(os.environ["RUN"])
agg = Counter(); chunks = 0
total_in_chunk = total_attempted = total_resolved = 0
for d in sorted(run.glob("gpu-*")):
    if not d.is_dir(): continue
    chunk_file = run / d.name.replace("gpu-", "chunk-")
    if not chunk_file.exists(): print(f"{d.name}: no chunk file"); continue
    iids = [l.strip() for l in chunk_file.read_text().splitlines() if l.strip()]
    parsed = MiniSweAgent().parse_results(d)
    patches = _preds_patches(d / "preds.json")
    exits = _read_exit_statuses(d)
    # discover model from preds.json (model_name_or_path), e.g. "openai/Qwen/Qwen3.5-27B"
    model = "unknown"
    pfile = d / "preds.json"
    if pfile.exists():
        try:
            pj = json.load(open(pfile))
            for v in pj.values():
                if isinstance(v, dict) and v.get("model_name_or_path"):
                    model = v["model_name_or_path"]
                    if model.startswith("openai/"): model = model[len("openai/"):]
                    break
        except Exception: pass
    try:
        csvp, outcomes = _write_per_instance_csv(
            out_dir=d, instance_ids=iids, patches=patches,
            resolved=set(parsed.get("resolved_ids") or []),
            empty_swebench=set(parsed.get("empty_patch_ids") or []),
            error_swebench=set(parsed.get("error_ids") or []),
            completed=set(parsed.get("completed_ids") or []),
            exit_statuses=exits,
            model=model,
            submitted_swebench=set(parsed.get("submitted_ids") or []),
        )
        chunks += 1
        n_attempted = sum(1 for iid in iids if iid in patches)
        n_resolved = len(set(iids) & set(parsed.get("resolved_ids") or []))
        total_in_chunk += len(iids); total_attempted += n_attempted; total_resolved += n_resolved
        print(f"{d.name}: {dict(outcomes)}")
        agg.update(outcomes)
    except Exception as e:
        print(f"{d.name}: SKIP {e}")
print()
print(f"=== aggregate over {chunks} chunks in {run} ===")
total = sum(agg.values())
ORDER = ["resolved","swebench_eval_inconsistent","submitted_but_failed",
         "submitted_not_resolved","agent_gave_up_empty","lost_to_timeout",
         "lost_to_walltime","lost_to_context_overflow","error_swebench",
         "dropped_by_agent","unknown"]
for k in ORDER:
    v = agg.get(k, 0)
    if v == 0: continue
    bar = "#" * int(40 * v / max(1,total))
    print(f"  {k:<28s} {v:>4d}  {v/max(1,total):6.1%}  {bar}")
print(f"  TOTAL:                       {total}")
print()
print(f"  instances_in_chunk:   {total_in_chunk}")
print(f"  instances_attempted:  {total_attempted}  (mini-extra ran agent)")
print(f"  instances_resolved:   {total_resolved}")
print(f"  pass@1 over chunk:      {total_resolved/max(1,total_in_chunk):.1%}")
print(f"  pass@1 over attempted:  {total_resolved/max(1,total_attempted):.1%}  <-- honest agent-quality metric")
out_agg = {
    "outcomes": dict(agg),
    "instances_in_chunk": total_in_chunk,
    "instances_attempted": total_attempted,
    "instances_resolved": total_resolved,
    "pass_at_1_over_chunk": total_resolved/max(1,total_in_chunk),
    "pass_at_1_over_attempted": total_resolved/max(1,total_attempted),
}
(run / "aggregate_outcomes.json").write_text(json.dumps(out_agg, indent=2) + "\n")
print(f"\nwritten: {run}/aggregate_outcomes.json")
PY
