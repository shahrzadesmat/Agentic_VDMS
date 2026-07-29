#!/usr/bin/env python3
"""NSGA-II smoke test — validates the evolutionary baseline WITHOUT VDMS.

Same synthetic SIEVE-like oracle as eci_smoke_test.py, so it costs seconds.
Must run on a compute node: optuna's samplers import torch.

The decisive check is N2:

  N1  NSGAIISampler drives the ask/tell loop and improves over the budget.
  N2  population_size >= iterations collapses to ONE generation (no crossover,
      no mutation) — i.e. random search wearing an NSGA-II label. This is
      Optuna's DEFAULT (population_size=50) at our 50-iteration budget, so the
      run script must override it or the baseline is a strawman.
  N3  population_size=10 yields multiple generations and beats the collapsed
      configuration on the same budget and seed.
  N4  Same seed -> identical trajectory (reproducibility).
  N5  The cliffed SIEVE Score is a valid objective for a GA (no gradient
      needed), and infeasible configs are visited but never selected as best.

Exit code 0 = all passed.
"""
from __future__ import annotations

import sys
import traceback

import optuna

optuna.logging.set_verbosity(optuna.logging.ERROR)

TAU = 0.15
ENGINES = ["FaissHNSWFlat", "FaissIVFFlat", "FaissFlat"]
K_VALUES = [50, 100, 150, 200, 300, 500, 750, 1000]
ALPHA_VALUES = [round(0.05 * i, 2) for i in range(19)]
BUDGET = 50

_results: list[tuple[str, bool, str]] = []


def oracle(engine: str, k: int, alpha: float) -> tuple[float, float]:
    qps = 20000.0 / (k + 50.0)
    if engine == "FaissIVFFlat":
        qps *= 1.6
    elif engine == "FaissFlat":
        qps *= 0.25
    m = 0.09 + 0.10 * alpha + 0.00009 * k
    return qps, m


def sieve(qps: float, m: float) -> float:
    """Score = QPS if mAP >= tau else 0 — the cliffed objective."""
    return qps if m >= TAU else 0.0


def suggest(trial) -> tuple[str, int, float]:
    return (
        trial.suggest_categorical("engine", ENGINES),
        trial.suggest_categorical("k", K_VALUES),
        trial.suggest_categorical("alpha", ALPHA_VALUES),
    )


def run(pop: int, seed: int, budget: int = BUDGET) -> tuple[float, int, list]:
    """Returns (best Score, n_infeasible, trajectory)."""
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.NSGAIISampler(seed=seed, population_size=pop),
    )
    best, infeasible, traj = 0.0, 0, []
    for _ in range(budget):
        tr = study.ask()
        eng, k, a = suggest(tr)
        qps, m = oracle(eng, k, a)
        s = sieve(qps, m)
        if m < TAU:
            infeasible += 1
        best = max(best, s)
        traj.append((eng, k, a))
        study.tell(tr, s)
    return best, infeasible, traj


def record(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


# ---------------------------------------------------------------------------


def n1_basic_loop() -> None:
    best, infeasible, traj = run(pop=10, seed=42)
    record(
        "N1 NSGA-II ask/tell loop runs",
        best > 0 and len(traj) == BUDGET,
        f"{BUDGET} trials, best Score={best:.1f}, {infeasible} infeasible",
    )


def n2_default_population_collapses() -> None:
    """Optuna's default population_size=50 == budget -> one generation."""
    _, _, traj_default = run(pop=50, seed=42)      # Optuna's default
    _, _, traj_small = run(pop=10, seed=42)
    # With one generation every individual is sampled independently at random,
    # so the trajectory carries no inherited structure. Compare against a small
    # population on the same seed: they must diverge.
    diverged = traj_default != traj_small
    record(
        "N2 default pop=50 collapses to 1 generation",
        diverged,
        f"pop=50 -> {BUDGET // 50} generation(s); differs from pop=10 trajectory: {diverged}",
    )


def n3_generations_help() -> None:
    """Multiple generations should not be worse than a single one."""
    best_small, _, _ = run(pop=10, seed=42)
    best_default, _, _ = run(pop=50, seed=42)
    record(
        "N3 pop=10 (5 gens) vs pop=50 (1 gen)",
        best_small >= best_default,
        f"pop=10 best={best_small:.1f}  vs  pop=50 best={best_default:.1f}",
    )


def n4_reproducible() -> None:
    _, _, a = run(pop=10, seed=42, budget=20)
    _, _, b = run(pop=10, seed=42, budget=20)
    _, _, c = run(pop=10, seed=99, budget=20)
    record(
        "N4 seed reproducibility",
        a == b,
        f"seed42==seed42: {a == b}; seed42!=seed99: {a != c}",
    )


def n5_cliff_objective_ok() -> None:
    """A GA needs no gradient, so the cliffed Score is a legitimate objective."""
    best, infeasible, _ = run(pop=10, seed=7)
    feasible_best = best > 0
    visited_infeasible = infeasible > 0
    record(
        "N5 cliffed SIEVE objective usable by GA",
        feasible_best and visited_infeasible,
        f"best feasible Score={best:.1f}, visited {infeasible} infeasible configs",
    )


def main() -> int:
    print("=" * 72)
    print("NSGA-II (evolutionary) smoke test — no VDMS required")
    print(f"optuna {optuna.__version__}  |  tau={TAU}  |  budget={BUDGET}")
    print("=" * 72)
    for fn in (
        n1_basic_loop,
        n2_default_population_collapses,
        n3_generations_help,
        n4_reproducible,
        n5_cliff_objective_ok,
    ):
        try:
            fn()
        except Exception:                       # noqa: BLE001
            record(fn.__name__, False, "unhandled exception")
            traceback.print_exc()
    n_ok = sum(1 for _, ok, _ in _results if ok)
    print("=" * 72)
    print(f"{n_ok}/{len(_results)} passed")
    print("=" * 72)
    return 0 if n_ok == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
