"""
VDTuner EHVI optimizer adapted for VDMS discrete parameter spaces.

Core EHVIBO class adapted from:
  Yang et al., "VDTuner: Automated Performance Tuning for Vector Data
  Management Systems," ICDE 2024. https://github.com/tiannuo-yang/VDTuner

Adaptations vs. original:
  - Generic Matern-2.5 kernel (original had Milvus-specific active_dims split)
  - KnobEncoder handles VDMS enum + integer params (continuous [0,1] <-> discrete)
  - VDTunerOptimizer replaces PollingBayesianOptimization (single engine, no SA)
  - Warm-up via LHS before EHVI; ε-constraint post-hoc selection from Pareto front
"""

import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
)
from botorch.optim import optimize_acqf
from gpytorch.mlls import SumMarginalLogLikelihood
from scipy.stats import qmc


# ---------------------------------------------------------------------------
# Utilities (adapted from VDTuner utils)
# ---------------------------------------------------------------------------

def fast_non_dominated_sort(P):
    """Non-dominated sorting — adapted from VDTuner (yang2024vdtuner)."""
    P_size = len(P)
    n = np.zeros(P_size, dtype=int)
    S = [[] for _ in range(P_size)]
    rank = np.full(P_size, -1)
    f = [[]]

    for p in range(P_size):
        for q in range(P_size):
            if p == q:
                continue
            p_dom = all(P[p][i] >= P[q][i] for i in range(len(P[p]))) and \
                    any(P[p][i] >  P[q][i] for i in range(len(P[p])))
            q_dom = all(P[q][i] >= P[p][i] for i in range(len(P[p]))) and \
                    any(P[q][i] >  P[p][i] for i in range(len(P[p])))
            if p_dom:
                S[p].append(q)
            elif q_dom:
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            f[0].append(p)

    i = 0
    while f[i]:
        Q = []
        for p in f[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        f.append(Q)
    return rank, f


# ---------------------------------------------------------------------------
# Parameter encoding  (adapted from VDTuner KnobStand)
# ---------------------------------------------------------------------------

class KnobEncoder:
    """
    Encodes VDMS discrete parameters to/from continuous [0,1] for the GP.

    param_defs: list of dicts, one per tunable parameter:
      {"name": str, "type": "enum",    "values": [v0, v1, ...]}
      {"name": str, "type": "integer", "min": int, "max": int}
    """

    def __init__(self, param_defs):
        self.param_defs = param_defs
        self.n_dims = len(param_defs)

    def encode(self, config: dict) -> np.ndarray:
        """Dict config → float64 array in [0,1]^n_dims."""
        vec = []
        for p in self.param_defs:
            val = config[p["name"]]
            if p["type"] == "enum":
                idx = p["values"].index(val)
                # map index → [0,1]; for a single-value enum this is always 0
                vec.append(idx / max(len(p["values"]) - 1, 1))
            else:
                vec.append((val - p["min"]) / max(p["max"] - p["min"], 1))
        return np.array(vec, dtype=np.float64)

    def decode(self, vec: np.ndarray) -> dict:
        """Float64 [0,1] array → dict, snapping to nearest valid value."""
        config = {}
        for i, p in enumerate(self.param_defs):
            v = float(np.clip(vec[i], 0.0, 1.0))
            if p["type"] == "enum":
                idx = int(round(v * (len(p["values"]) - 1)))
                idx = max(0, min(len(p["values"]) - 1, idx))
                config[p["name"]] = p["values"][idx]
            else:
                val = int(round(v * (p["max"] - p["min"]) + p["min"]))
                config[p["name"]] = max(p["min"], min(p["max"], val))
        return config


# ---------------------------------------------------------------------------
# EHVIBO  (adapted from VDTuner optimizer_pobo_sa.py)
# ---------------------------------------------------------------------------

class EHVIBO:
    """
    Expected Hypervolume Improvement Bayesian Optimization.
    Adapted from VDTuner (Yang et al., ICDE 2024).
    Fits one SingleTaskGP per objective; uses qEHVI + Sobol sampler.
    """

    def __init__(self, knob_num: int, seed: int):
        self.knob_num = knob_num
        self.bounds = torch.tensor(
            [[0.0] * knob_num, [1.0] * knob_num], dtype=torch.float64
        )
        self.seed = seed
        self.model = None

    def update_samples(self, X: np.ndarray, Y: np.ndarray):
        """Fit ModelListGP (one GP per objective) on normalized data."""
        X_t = torch.tensor(X, dtype=torch.float64)
        Y_t = torch.tensor(Y, dtype=torch.float64)

        models = []
        for i in range(Y_t.shape[-1]):
            models.append(SingleTaskGP(
                X_t, Y_t[..., i: i + 1],
                outcome_transform=Standardize(m=1),
            ))
        self.model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        self._X_t = X_t  # keep for partitioning

    def recommend(self, seed: int) -> np.ndarray:
        """Return next candidate in [0,1]^d using qEHVI."""
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))

        with torch.no_grad():
            pred = self.model.posterior(self._X_t).mean

        ref_point = torch.tensor([0.5, 0.5], dtype=torch.float64)
        partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=pred)

        acq = qLogExpectedHypervolumeImprovement(
            model=self.model,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
        )

        candidate, _ = optimize_acqf(
            acq,
            bounds=self.bounds,
            q=1,
            num_restarts=10,
            raw_samples=100,
            options={"seed": seed},
        )
        return candidate.detach().numpy()[0]


# ---------------------------------------------------------------------------
# VDTunerOptimizer  — drop-in replacement for the gp_bo/optuna branch
# ---------------------------------------------------------------------------

class VDTunerOptimizer:
    """
    VDTuner-style multi-objective optimizer for VDMS.

    Optimizes [quality, QPS] jointly via EHVI (no hard threshold during search).
    After all iterations, applies the ε-constraint (SIEVE quality ≥ τ) post-hoc
    to select the Pareto-optimal config with highest QPS — placing VDTuner on
    the same evaluation footing as all other methods.

    Usage (per iteration):
        config = vdtuner.ask()          # returns dict
        # ... run benchmark ...
        vdtuner.tell(quality, qps)      # record both raw objectives

    After loop:
        best = vdtuner.select(quality_threshold)
    """

    N_WARMUP = 5  # LHS warm-up iterations before EHVI

    def __init__(self, encoder: KnobEncoder, seed: int):
        self.encoder = encoder
        self.seed = seed
        self.ehvi = EHVIBO(encoder.n_dims, seed)
        self._lhs = qmc.LatinHypercube(d=encoder.n_dims, seed=seed)
        self._lhs_samples = self._lhs.random(n=self.N_WARMUP)

        self._iter = 0
        self._last_vec = None

        # Raw observation history
        self.X_vecs: list = []      # [0,1] encoded vectors
        self.Y_quality: list = []   # raw quality metric (mAP or Recall@10)
        self.Y_qps: list = []       # raw QPS
        self.configs: list = []     # decoded config dicts

    def ask(self) -> dict:
        """Return next config dict to evaluate."""
        if self._iter < self.N_WARMUP:
            vec = self._lhs_samples[self._iter]
        else:
            Y_norm = self._normalize_Y()
            self.ehvi.update_samples(np.array(self.X_vecs), Y_norm)
            vec = self.ehvi.recommend(seed=self.seed + self._iter)

        self._last_vec = vec
        config = self.encoder.decode(vec)
        return config

    def tell(self, config: dict, quality: float, qps: float):
        """Record benchmark result.

        Stores _last_vec (the continuous vector the GP recommended) rather than
        re-encoding the actual config.  Re-encoding would crash on conditional
        overrides whose values sit outside the encoder's enum lists
        (e.g. aqe_weight=0.0 when n_aqe==1, but AQE_WEIGHT_VALUES=[0.1…0.5]).
        This matches VDTuner's own approach of training the GP on the recommended
        continuous point.
        """
        self.X_vecs.append(self._last_vec)
        self.Y_quality.append(float(quality))
        self.Y_qps.append(float(qps))
        self.configs.append(config)
        self._iter += 1

    def select(self, quality_threshold: float):
        """
        ε-constraint selection from Pareto front (SIEVE post-hoc):
        Returns (config, quality, qps) with highest QPS where quality ≥ threshold.
        Falls back to best-quality config if none are feasible.
        """
        feasible = [
            (cfg, q, qps)
            for cfg, q, qps in zip(self.configs, self.Y_quality, self.Y_qps)
            if q >= quality_threshold
        ]
        if feasible:
            return max(feasible, key=lambda x: x[2])
        # No feasible config — return best quality as fallback
        best_idx = int(np.argmax(self.Y_quality))
        return (self.configs[best_idx],
                self.Y_quality[best_idx],
                self.Y_qps[best_idx])

    def _normalize_Y(self) -> np.ndarray:
        """Normalize both objectives to ~[0,1] by running max (VDTuner approach)."""
        q_arr = np.array(self.Y_quality)
        qps_arr = np.array(self.Y_qps)
        max_q   = max(float(np.max(q_arr)),   1e-9)
        max_qps = max(float(np.max(qps_arr)), 1e-9)
        return np.column_stack([q_arr / max_q, qps_arr / max_qps])
