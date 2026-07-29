#!/usr/bin/env python3
"""Smoke test for the g(H) / hints ablation flags -- no VDMS, no API calls.

Intercepts requests.post so the prompt is captured and a canned response is
returned. Verifies each flag is SURGICAL: it must remove its own component and
nothing else. A leaky ablation would silently confound the experiment (e.g. if
--ablation-no-guidance also dropped history, any measured effect could not be
attributed to g(H)).

  G1  baseline prompt contains a real diagnosis, hints, history and a phase label
  G2  --ablation-no-guidance removes g(H) ONLY (history, phases, hints intact)
  G3  --ablation-no-hints removes UNTRIED hints ONLY (history, phases, g(H) intact)
  G4  both flags together remove both, still keeping history and phases
  G5  the flags reach LLMProvider through run_optimization's signature

Exit code 0 = all passed.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import sys
import time

import importlib.util
import os

import requests

# Load the STAGED optimizer by explicit path. Two reasons: (1) it is the exact
# file the campaign executes, so the test cannot pass against a stale sibling
# copy; (2) only from src/hico_det does the module's own sys.path fix-up
# (__file__.parent.parent -> src/) resolve vdtuner_ehvi.
_STAGED = "/path/to/Agentic_VDMS/src/hico_det/hico_agent_optimizer_ghint.py"
if not os.path.exists(_STAGED):
    sys.exit(f"staged optimizer not found: {_STAGED}")
_spec = importlib.util.spec_from_file_location("hico_agent_optimizer_ghint", _STAGED)
M = importlib.util.module_from_spec(_spec)
sys.modules["hico_agent_optimizer_ghint"] = M
_spec.loader.exec_module(M)
print(f"  (testing staged module: {_STAGED})")

CAPTURED: list[str] = []
_results: list[tuple[str, bool, str]] = []

CANNED = json.dumps({
    "engine": "FaissHNSWFlat",
    "params": {"M": 64, "efConstruction": 200, "efSearch": 64},
    "k_neighbors": 50, "alpha": 0.80, "n_refs": 5,
    "ref_strategy": "centroid", "constraint_strategy": "object",
    "reasoning": "canned",
})


class _Resp:
    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": CANNED}}]}


def fake_post(url, **kw):
    CAPTURED.append(kw["json"]["messages"][0]["content"])
    return _Resp()


def bench(qps, mp):
    return M.BenchmarkResult(
        qps=qps, map_score=mp, latency_ms=10.0, t_clip_avg_ms=5.0,
        t_rerank_avg_ms=2.0, index_build_s=1.0, precision_at_10=0.4,
        ndcg_at_10=0.4, recall_at_10=0.1, engine="FaissHNSWFlat",
        params={"M": 64, "efConstruction": 200, "efSearch": 64},
        k_neighbors=100, alpha=0.65, timestamp=time.time(),
    )


def history(n=12):
    out = []
    for i in range(1, n + 1):
        cfg = {"engine": "FaissHNSWFlat",
               "params": {"M": 64, "efConstruction": 200, "efSearch": 64},
               "k_neighbors": 100, "alpha": round(0.30 + 0.05 * (i % 6), 2),
               "n_refs": 5, "ref_strategy": "centroid",
               "constraint_strategy": "object", "_iter": i}
        out.append(M.IterationResult(i, cfg, bench(140.0 + i, 0.16 + 0.001 * i),
                                     "r", "hyperparameter_only"))
    return out


def prompt_for(**flags) -> str:
    CAPTURED.clear()
    p = M.LLMProvider("dummy-key", seed=42, **flags)
    asyncio.run(p.query_llm(iteration=30, history=history(), total_iterations=50))
    assert CAPTURED, "no prompt captured"
    return CAPTURED[-1]


def rec(name, ok, detail=""):
    _results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


HAS_GUID_OFF = "ABLATION: diagnostic guidance disabled"
HINT_MARK = "UNTRIED"


def main() -> int:
    requests.post = fake_post
    M.requests.post = fake_post

    print("=" * 72)
    print("g(H) / hints ablation smoke test — no VDMS, no API calls")
    print("=" * 72)

    base = prompt_for()
    g_off = prompt_for(ablation_no_guidance=True)
    h_off = prompt_for(ablation_no_hints=True)
    both = prompt_for(ablation_no_guidance=True, ablation_no_hints=True)

    # Signals that must survive every ablation.
    def has_history(p):  # the prompt lists previously tried configs
        return "ALL CONFIGS TRIED SO FAR" in p and "alpha=" in p

    def has_phase(p):
        return "Phase: EXPLOITATION" in p

    rec("G1 baseline has diagnosis + hints + history + phase",
        HAS_GUID_OFF not in base and HINT_MARK in base
        and has_history(base) and has_phase(base),
        f"len={len(base)}, hints={'yes' if HINT_MARK in base else 'no'}")

    rec("G2 --ablation-no-guidance removes g(H) only",
        HAS_GUID_OFF in g_off and HINT_MARK in g_off
        and has_history(g_off) and has_phase(g_off),
        "guidance off; hints/history/phase retained")

    rec("G3 --ablation-no-hints removes hints only",
        HINT_MARK not in h_off and HAS_GUID_OFF not in h_off
        and has_history(h_off) and has_phase(h_off),
        "hints off; guidance/history/phase retained")

    rec("G4 both flags remove both, keep history+phase",
        HAS_GUID_OFF in both and HINT_MARK not in both
        and has_history(both) and has_phase(both))

    sig = inspect.signature(M.run_optimization).parameters
    rec("G5 flags plumbed through run_optimization",
        "ablation_no_guidance" in sig and "ablation_no_hints" in sig,
        f"params present: {sorted(k for k in sig if k.startswith('ablation'))}")

    # Report how much text each ablation actually removes -- a near-zero delta
    # would mean the flag is a no-op in this configuration.
    print()
    print(f"  prompt sizes: base={len(base)}  no-guidance={len(g_off)} "
          f"({len(g_off)-len(base):+d})  no-hints={len(h_off)} ({len(h_off)-len(base):+d})")

    n_ok = sum(1 for _, ok, _ in _results if ok)
    print("=" * 72)
    print(f"{n_ok}/{len(_results)} passed")
    print("=" * 72)
    return 0 if n_ok == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
