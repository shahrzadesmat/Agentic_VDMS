#!/usr/bin/env python3
"""ECI smoke test — validates constrained-BO mechanics WITHOUT VDMS.

Runs against a synthetic SIEVE-like oracle so it costs seconds, not GPU-hours.
Must run on a compute node: optuna's GPSampler imports torch.

Each test maps to a finding from the code audit:

  T1  constraints_func fires on every completed tell under the ask/tell API,
      and ECI reaches a feasible high-QPS config.
  T2  BLOCKER #1 (bug): a dedup-style tell that does not set the constraint
      crashes the run. Proves the un-patched path is fatal.
  T3  BLOCKER #1 (fix): the same dedup tell WITH the constraint supplied from
      the earlier observation completes normally.
  T4  A FAIL-state tell (crashed benchmark) is excluded from the model and does
      not break the "same number of constraints for all trials" invariant.
  T5  An all-infeasible warm-up does not crash the sampler.
  T6  Same seed -> identical trajectory (reproducibility).
  T7  Objective semantics: telling raw QPS is not the same as telling SIEVE
      Score. If these matched, ECI would be gp_bo with extra steps.

Exit code 0 = all passed.
"""
from __future__ import annotations

import sys
import traceback

import optuna
from optuna.trial import TrialState

optuna.logging.set_verbosity(optuna.logging.ERROR)

TAU = 0.15
ENGINES = ["FaissHNSWFlat", "FaissIVFFlat", "FaissFlat"]
K_VALUES = [50, 100, 150, 200, 300, 500, 750, 1000]
ALPHA_VALUES = [round(0.05 * i, 2) for i in range(19)]   # 0.00 .. 0.90
CONSTRAINT_KEY = "eci_c"

_results: list[tuple[str, bool, str]] = []


def oracle(engine: str, k: int, alpha: float) -> tuple[float, float]:
    """Synthetic oracle with the same shape as the real one.

    QPS falls with k; mAP rises with BOTH k and alpha, so a small-k (high-QPS)
    config is only feasible when alpha compensates — the (k, alpha) coupling the
    paper is about.
    """
    qps = 20000.0 / (k + 50.0)
    if engine == "FaissIVFFlat":
        qps *= 1.6
    elif engine == "FaissFlat":
        qps *= 0.25
    m = 0.09 + 0.10 * alpha + 0.00009 * k
    return qps, m


def constraint_of(m: float) -> float:
    """Optuna convention: feasible iff c <= 0."""
    return TAU - m


def suggest(trial) -> tuple[str, int, float]:
    engine = trial.suggest_categorical("engine", ENGINES)
    k = trial.suggest_categorical("k", K_VALUES)
    alpha = trial.suggest_categorical("alpha", ALPHA_VALUES)
    return engine, k, alpha


def constraints_func(trial):
    """Mirrors the patched _eci_constraints_func: fail loud, never default."""
    if CONSTRAINT_KEY not in trial.user_attrs:
        raise RuntimeError(
            f"trial {trial.number} completed without a constraint value"
        )
    return (float(trial.user_attrs[CONSTRAINT_KEY]),)


def make_study(seed: int, startup: int = 5):
    return optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.GPSampler(
            seed=seed, constraints_func=constraints_func, n_startup_trials=startup
        ),
    )


def record(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


# ---------------------------------------------------------------------------


def t1_basic_loop() -> None:
    """Full ECI loop: constraints honoured, feasible optimum reached."""
    study = make_study(seed=42)
    best_feasible_qps, n_feasible = 0.0, 0
    for _ in range(25):
        tr = study.ask()
        eng, k, a = suggest(tr)
        qps, m = oracle(eng, k, a)
        tr.set_user_attr(CONSTRAINT_KEY, constraint_of(m))
        study.tell(tr, qps)
        if m >= TAU:
            n_feasible += 1
            best_feasible_qps = max(best_feasible_qps, qps)
    stored = [
        study._storage.get_trial_system_attrs(t._trial_id).get("constraints")
        for t in study.get_trials(deepcopy=False)
    ]
    all_stored = all(c is not None and len(c) == 1 for c in stored)
    record(
        "T1 basic ECI loop (25 trials)",
        all_stored and n_feasible > 0 and best_feasible_qps > 0,
        f"{n_feasible}/25 feasible, best feasible QPS={best_feasible_qps:.1f}, "
        f"constraints stored for all trials={all_stored}",
    )


def t2_dedup_without_constraint_crashes() -> None:
    """BLOCKER #1 — the un-patched dedup path is fatal."""
    study = make_study(seed=42)
    for _ in range(6):
        tr = study.ask()
        eng, k, a = suggest(tr)
        qps, m = oracle(eng, k, a)
        tr.set_user_attr(CONSTRAINT_KEY, constraint_of(m))
        study.tell(tr, qps)
    # Simulate the dedup branch: tell WITHOUT running the oracle / setting attr.
    crashed, err = False, ""
    try:
        tr = study.ask()
        suggest(tr)
        study.tell(tr, 123.0)          # <- what the un-patched code does
        for _ in range(3):             # next asks trigger the GP path
            tr2 = study.ask()
            suggest(tr2)
            qps, m = oracle("FaissFlat", 100, 0.5)
            tr2.set_user_attr(CONSTRAINT_KEY, constraint_of(m))
            study.tell(tr2, qps)
    except Exception as e:             # noqa: BLE001 - we are asserting it raises
        crashed, err = True, f"{type(e).__name__}: {str(e)[:90]}"
    record("T2 dedup WITHOUT constraint crashes (expected)", crashed, err or "did NOT crash")


def t3_dedup_with_constraint_ok() -> None:
    """BLOCKER #1 fix — supplying the prior observation's constraint works."""
    study = make_study(seed=42)
    seen: dict[tuple, tuple[float, float]] = {}
    ok, err = True, ""
    try:
        for i in range(20):
            tr = study.ask()
            eng, k, a = suggest(tr)
            key = (eng, k, a)
            if key in seen:                       # dedup branch, patched
                qps, m = seen[key]
                tr.set_user_attr(CONSTRAINT_KEY, constraint_of(m))
                study.tell(tr, qps)
                continue
            qps, m = oracle(eng, k, a)
            seen[key] = (qps, m)
            tr.set_user_attr(CONSTRAINT_KEY, constraint_of(m))
            study.tell(tr, qps)
    except Exception as e:                        # noqa: BLE001
        ok, err = False, f"{type(e).__name__}: {str(e)[:90]}"
    record("T3 dedup WITH constraint completes", ok, err or f"20 trials, {len(seen)} unique")


def t4_failed_trial_excluded() -> None:
    """FAIL-state tells need no constraint and don't break the invariant."""
    study = make_study(seed=7)
    ok, err = True, ""
    try:
        for i in range(10):
            tr = study.ask()
            eng, k, a = suggest(tr)
            if i == 6:                            # simulate a crashed benchmark
                study.tell(tr, state=TrialState.FAIL)
                continue
            qps, m = oracle(eng, k, a)
            tr.set_user_attr(CONSTRAINT_KEY, constraint_of(m))
            study.tell(tr, qps)
        n_fail = len(study.get_trials(deepcopy=False, states=(TrialState.FAIL,)))
        n_done = len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
        ok = n_fail == 1 and n_done == 9
        err = f"{n_done} complete / {n_fail} failed"
    except Exception as e:                        # noqa: BLE001
        ok, err = False, f"{type(e).__name__}: {str(e)[:90]}"
    record("T4 FAIL-state tell tolerated", ok, err)


def t5_all_infeasible_warmup() -> None:
    """Every early trial infeasible — sampler must not crash."""
    study = make_study(seed=13)
    ok, err = True, ""
    try:
        for i in range(15):
            tr = study.ask()
            suggest(tr)
            if i < 10:
                tr.set_user_attr(CONSTRAINT_KEY, 1.0)     # hard infeasible
                study.tell(tr, 5.0)
            else:
                qps, m = oracle("FaissHNSWFlat", 1000, 0.9)
                tr.set_user_attr(CONSTRAINT_KEY, constraint_of(m))
                study.tell(tr, qps)
    except Exception as e:                                # noqa: BLE001
        ok, err = False, f"{type(e).__name__}: {str(e)[:90]}"
    record("T5 all-infeasible warm-up survives", ok, err or "10 infeasible then 5 feasible")


def t6_reproducible() -> None:
    """Same seed -> same suggestion trajectory."""
    def run(seed: int) -> list[tuple]:
        study = make_study(seed=seed)
        traj = []
        for _ in range(12):
            tr = study.ask()
            eng, k, a = suggest(tr)
            traj.append((eng, k, a))
            qps, m = oracle(eng, k, a)
            tr.set_user_attr(CONSTRAINT_KEY, constraint_of(m))
            study.tell(tr, qps)
        return traj
    a, b, c = run(42), run(42), run(99)
    record(
        "T6 seed reproducibility",
        a == b,
        f"seed42==seed42: {a == b}; seed42!=seed99: {a != c}",
    )


def t7_objective_semantics() -> None:
    """Telling raw QPS must differ from telling SIEVE Score.

    If identical, the ECI run would be gp_bo with a redundant constraint model
    and the experiment would be a null result caused by the harness.
    """
    diffs = 0
    for eng in ENGINES:
        for k in K_VALUES:
            for a in ALPHA_VALUES:
                qps, m = oracle(eng, k, a)
                sieve = qps if m >= TAU else 0.0
                if abs(sieve - qps) > 1e-9:
                    diffs += 1
    total = len(ENGINES) * len(K_VALUES) * len(ALPHA_VALUES)
    record(
        "T7 raw-QPS objective != SIEVE Score",
        diffs > 0,
        f"{diffs}/{total} configs differ (infeasible region the constraint GP must model)",
    )


def main() -> int:
    print("=" * 72)
    print("ECI (constrained BO) smoke test — no VDMS required")
    print(f"optuna {optuna.__version__}  |  tau={TAU}")
    print("=" * 72)
    for fn in (
        t1_basic_loop,
        t2_dedup_without_constraint_crashes,
        t3_dedup_with_constraint_ok,
        t4_failed_trial_excluded,
        t5_all_infeasible_warmup,
        t6_reproducible,
        t7_objective_semantics,
    ):
        try:
            fn()
        except Exception:                                  # noqa: BLE001
            record(fn.__name__, False, "unhandled exception")
            traceback.print_exc()
    n_ok = sum(1 for _, ok, _ in _results if ok)
    print("=" * 72)
    print(f"{n_ok}/{len(_results)} passed")
    print("=" * 72)
    return 0 if n_ok == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
