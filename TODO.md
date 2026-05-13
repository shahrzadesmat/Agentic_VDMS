# Pending TODOs

## COMPLETED

### ~~1. Top up OpenRouter credits~~  ✓
Credits restored — LLM jobs unblocked.

### ~~2. Rerun HICO-DET Milvus LLM s42 + s200~~  ✓
Buggy `while/else` dedup code fixed. Corrected results:
- s42: 7327.7 QPS, mAP=0.1596
- s99: 7327.4 QPS, mAP=0.1525 (prior reference run)
- s200: 7162.3 QPS, mAP=0.1614

### ~~4. Add `constraint_strategy` to Grid search space~~  ✓
Grid extended to 50 systematic configs with full `constraint_strategy` sweep
(`none`, `object`, `verb`, `object_and_verb`). See commit d5cb52d.

---

## IN PROGRESS

### 3 / 5 / 6. ConditionB — VDMS HICO-DET with `all_verb_ids`  (running)
All 16 SLURM scripts submitted under `supplementary/experiments/hico_det/conditionB/`.
Results land in `supplementary/results/hico_det_condB/`.

**Completed so far:**
| Method | Seed | QPS | mAP |
|--------|------|-----|-----|
| random | 42 | 401.9 | 0.1589 |
| random | 99 | 182.7 | 0.2220 |

**Still running / pending:**
- random/seed200
- grid × 3 seeds
- optuna × 3 seeds
- gpbo × 3 seeds
- vdtuner × 3 seeds
- llm × 3 seeds (needs OPENROUTER_API_KEY exported before sbatch)

Submission order: random → grid → optuna → gpbo → vdtuner → llm (last).
Max 4 concurrent GPU jobs (gpuA40x4 partition).

### Fix `all_verb_ids` in Milvus constraint filter  (pending)
VDMS side fixed via ConditionB (`--use-all-verb-ids` flag). Milvus side still maps each
image to a single bucket using `primary_verb_id`. Fix: update `obj_img_map` / `verb_img_map`
building in `src/milvus/milvus_hico_optimizer.py` to iterate over `all_verb_ids`.

---

## 7. Update paper  (after ConditionB completes)
- Report corrected Milvus HICO-DET LLM results (s42=7327.7, s200=7162.3)
- Decide ConditionA vs ConditionB as primary results (methodologically B is correct;
  whichever is higher goes in main table, other in ablation)
- Add one sentence acknowledging `primary_verb_id` as a design choice limitation
