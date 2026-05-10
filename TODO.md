# Pending TODOs

## 1. Top up OpenRouter credits
- Visit https://openrouter.ai/settings/credits
- Unblocks all LLM rerun jobs below

## 2. Rerun HICO-DET Milvus LLM s42 + s200 (blocked on #1)
Old results (s42=4273, s200=3854) were produced by buggy `while/else` dedup code — effectively
pure random search. Fixed code confirmed working (s99=7327, beats VDTuner 7273).
```bash
cd /work/hdd/bdjd/vdms_workflow/semantic_vdms/milvus/hico_det
sbatch run_milvus_llm_s42.sh
sbatch run_milvus_llm_s200.sh
```

## 3. Fix `all_verb_ids` in constraint filter (code change)
Currently both Milvus and VDMS map each image to a single bucket using `primary_object_id` /
`primary_verb_id` only. 56% of images have multiple verbs — the verb filter misses secondary
verb associations. Fix: use `all_object_ids` and `all_verb_ids` from `hico_metadata.json`.

- **Milvus**: trivial — update `obj_img_map` / `verb_img_map` building in
  `src/milvus/milvus_hico_optimizer.py` to iterate over `all_object_ids` / `all_verb_ids`
- **VDMS**: requires rebuilding the graph database (see #5)

Near-miss that motivates this fix: VDMS LLM seed99 iter41 got QPS=394 with `cs=verb` but
mAP=0.1293 (threshold τ=0.15). Fixing the verb map may push this above threshold, raising
LLM's best VDMS HICO-DET score from 300 → 394 QPS.

## 4. Add `constraint_strategy` to Grid search space (code change)
Grid is currently fixed to `constraint_strategy=none`. This is methodologically unfair at VLDB:
LLM has access to a dimension Grid cannot explore. Per NeurIPS 2021 "HPO Is Deceiving Us",
fair comparison requires all methods to search the same configuration space.

Fix: in `src/hico_det/hico_agent_optimizer.py`, remove the `cs=none` lock on Grid and add
`constraint_strategy ∈ {none, object, verb, object_and_verb}` to its sweep.

## 5. Rebuild VDMS HICO-DET database (after #3)
Wipe the VDMS graph DB and re-insert all 47K descriptors using corrected `all_verb_ids`
mapping so the PMGD constraint filter uses full metadata.

## 6. Rerun all VDMS HICO-DET experiments (after #4 and #5)
6 methods × 3 seeds = 18 runs: LLM, Random, Optuna, GP-BO, VDTuner, Grid (now with
`constraint_strategy` in sweep).

## 7. Update paper (after #2 and #6)
- Report corrected Milvus HICO-DET LLM results (s42, s200)
- Report corrected VDMS HICO-DET results with fair Grid baseline and `all_verb_ids` fix
- Add one sentence acknowledging `primary_object_id` limitation as a design choice
