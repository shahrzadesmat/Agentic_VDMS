"""
Milvus-backend GLDv2 optimizer (pure FAISS, no VDMS).
Methods: hyperparameter_only (LLM), random, grid, optuna, gp_bo, vdtuner.
Stage 1: FAISS IP search on CLIP features.
Stage 1b: AQE (numpy).
Stage 2: DINOv2 reranking (numpy).
Metric: mAP, threshold τ=0.15, Score = QPS if mAP ≥ τ else 0.
"""

import sys, os, time, json, random, asyncio, re, argparse
import numpy as np
import faiss
import optuna
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

# Reuse GLDv2 evaluation functions from benchmark_gldv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../gldv2"))
from benchmark_gldv2 import (
    load_gldv2_data,
    build_prf_query, rerank_with_dinov2,
    compute_ap, compute_precision_at_k, compute_recall_at_k, compute_ndcg_at_k,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vdtuner_ehvi import VDTunerOptimizer, KnobEncoder

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
_MAP_THRESHOLD: float = 0.15
_GLDV2_CACHE: Dict = {}
_FAISS_INDEX  = None
_FAISS_LAST_CFG: Optional[str] = None


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _load_data(dataset_dir: str) -> Dict:
    global _GLDV2_CACHE
    if not _GLDV2_CACHE:
        _GLDV2_CACHE = load_gldv2_data(dataset_dir)
    return _GLDV2_CACHE


# ---------------------------------------------------------------------------
# FAISS index (IP metric for CLIP)
# ---------------------------------------------------------------------------

def _build_index(engine: str, params: Dict, clip_db: np.ndarray) -> faiss.Index:
    d = clip_db.shape[1]
    if engine == "FaissFlat":
        idx = faiss.IndexFlatIP(d)
    elif engine == "FaissHNSWFlat":
        M   = int(params.get("M", 32))
        efC = int(params.get("efConstruction", 200))
        idx = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction = efC
    else:
        raise ValueError(f"Unknown engine: {engine}")
    # Normalize for IP = cosine similarity
    vecs = clip_db / (np.linalg.norm(clip_db, axis=1, keepdims=True) + 1e-10)
    idx.add(vecs.astype(np.float32))
    return idx


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    map_score:     float
    qps:           float
    latency_ms:    float
    index_build_s: float
    precision_at_10: float = 0.0
    recall_at_10:    float = 0.0

    def score(self) -> float:
        return self.qps if self.map_score >= _MAP_THRESHOLD else 0.0


def run_benchmark(cfg: Dict, dataset_dir: str, iteration: int = 0) -> BenchmarkResult:
    global _FAISS_INDEX, _FAISS_LAST_CFG

    data = _load_data(dataset_dir)
    clip_db   = data["clip_db"]
    clip_q    = data["clip_q"]
    dinov2_db = data["dinov2_db"]
    dinov2_q  = data["dinov2_q"]
    rel_sets  = data["rel_sets"]
    junk_sets = data["junk_sets"]

    engine    = cfg.get("engine", "FaissHNSWFlat")
    params    = cfg.get("params", {})
    k         = int(cfg.get("k_neighbors", 100))
    alpha     = float(cfg.get("alpha", 0.30))
    n_refs    = int(cfg.get("n_refs", 1))
    ref_strat = str(cfg.get("ref_strategy", "first"))
    n_aqe     = int(cfg.get("n_aqe", 1))
    aqe_w     = float(cfg.get("aqe_weight", 0.0))

    print(f"\n[Iter {iteration}] engine={engine} params={params} k={k} "
          f"alpha={alpha:.2f} n_refs={n_refs}/{ref_strat} n_aqe={n_aqe} aqe_w={aqe_w:.2f}")

    cfg_key = json.dumps({"engine": engine, "params": {k2: v for k2, v in params.items()
                                                        if k2 != "efSearch"}}, sort_keys=True)
    if cfg_key != _FAISS_LAST_CFG:
        t0 = time.time()
        print(f"[FAISS] Building index: {engine} {params}…")
        _FAISS_INDEX    = _build_index(engine, params, clip_db)
        _FAISS_LAST_CFG = cfg_key
        build_s = time.time() - t0
        print(f"[FAISS] Index built in {build_s:.1f}s")
    else:
        build_s = 0.0

    efS = int(params.get("efSearch", 500))
    if engine == "FaissHNSWFlat":
        _FAISS_INDEX.hnsw.efSearch = efS

    # Normalize query vectors
    q_norm = clip_q / (np.linalg.norm(clip_q, axis=1, keepdims=True) + 1e-10)
    db_norm = clip_db / (np.linalg.norm(clip_db, axis=1, keepdims=True) + 1e-10)

    n_queries = len(q_norm)
    search_start = time.time()

    # Stage 1: batch FAISS search
    t_faiss = time.perf_counter()
    D, I = _FAISS_INDEX.search(q_norm.astype(np.float32), k)
    t_faiss_ms = (time.perf_counter() - t_faiss) * 1000

    aps, precs, recs, ndcgs = [], [], [], []
    t_rerank_total = 0.0

    for qi in range(n_queries):
        cand_ids   = [int(x) for x in I[qi] if x >= 0]
        cand_dists = [float(D[qi][j]) for j in range(len(cand_ids))]
        clip_results = [{"id": cand_ids[j], "distance": -cand_dists[j]}
                        for j in range(len(cand_ids))]  # negate: higher IP = closer

        # Stage 1b: AQE
        cq = q_norm[qi]
        if n_aqe > 1 and aqe_w > 0.0 and len(cand_ids) >= n_aqe:
            n_refs_aqe = min(n_aqe - 1, len(cand_ids))
            q_exp = ((1.0 - aqe_w) * cq
                     + aqe_w * db_norm[np.array(cand_ids[:n_refs_aqe])].mean(axis=0))
            q_exp /= (np.linalg.norm(q_exp) + 1e-10)
            new_sims = db_norm[np.array(cand_ids)] @ q_exp
            clip_results = sorted(
                [{"id": cand_ids[j], "distance": -float(new_sims[j])}
                 for j in range(len(cand_ids))],
                key=lambda r: r["distance"],
            )

        # Stage 2: DINOv2 rerank
        t0 = time.perf_counter()
        dq = dinov2_q[qi]
        aug_dq   = build_prf_query(dq, clip_results, dinov2_db, n_refs, ref_strat)
        reranked = rerank_with_dinov2(clip_results, aug_dq, dinov2_db, alpha)
        t_rerank_total += (time.perf_counter() - t0) * 1000

        ranked_ids = [r["id"] for r in reranked]
        aps.append(compute_ap(ranked_ids, rel_sets[qi], junk_sets[qi]))
        precs.append(compute_precision_at_k(ranked_ids, rel_sets[qi], junk_sets[qi], k=10))
        recs.append(compute_recall_at_k(ranked_ids, rel_sets[qi], junk_sets[qi], k=10))
        ndcgs.append(compute_ndcg_at_k(ranked_ids, rel_sets[qi], junk_sets[qi], k=10))

    total_time = time.time() - search_start
    map_score  = float(np.mean(aps))
    qps        = float(n_queries / total_time)
    lat_ms     = float(total_time / n_queries * 1000)

    print(f"  Score={qps if map_score >= _MAP_THRESHOLD else 0:.1f}  "
          f"mAP={map_score:.4f}  QPS={qps:.1f}  build={build_s:.1f}s")

    return BenchmarkResult(
        map_score=map_score, qps=qps, latency_ms=lat_ms,
        index_build_s=build_s,
        precision_at_10=float(np.mean(precs)),
        recall_at_10=float(np.mean(recs)),
    )


# ---------------------------------------------------------------------------
# Config provider
# ---------------------------------------------------------------------------

class ConfigProvider:
    ENGINE_PARAMS = {
        "FaissFlat":    {},
        "FaissHNSWFlat": {
            "M":              [8, 16, 32, 48, 64],
            "efConstruction": [100, 200, 400],
            "efSearch":       [32, 64, 100, 150, 200, 300, 500],
        },
    }
    K_NEIGHBORS_VALUES  = [50, 100, 150, 200, 300, 500, 750, 1000]
    ALPHA_VALUES        = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    N_REFS_VALUES       = [1, 3, 5, 10]
    REF_STRATEGY_VALUES = ["first", "centroid", "diverse"]
    N_AQE_VALUES        = [1, 3, 5, 10]
    AQE_WEIGHT_VALUES   = [0.1, 0.2, 0.3, 0.5]

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        random.seed(seed)

    def random_config(self) -> Dict:
        engine = "FaissHNSWFlat"
        params = {k: int(self.rng.choice(v)) for k, v in self.ENGINE_PARAMS[engine].items()}
        n_aqe  = int(self.rng.choice(self.N_AQE_VALUES))
        ref_strat = str(self.rng.choice(self.REF_STRATEGY_VALUES))
        n_refs = 1 if ref_strat == "centroid" else int(self.rng.choice(self.N_REFS_VALUES))
        return {
            "engine": engine, "params": params,
            "k_neighbors":  int(self.rng.choice(self.K_NEIGHBORS_VALUES)),
            "alpha":        float(self.rng.choice(self.ALPHA_VALUES)),
            "n_refs":       n_refs,
            "ref_strategy": ref_strat,
            "n_aqe":        n_aqe,
            "aqe_weight":   float(self.rng.choice(self.AQE_WEIGHT_VALUES)) if n_aqe > 1 else 0.0,
        }

    def optuna_suggest(self, trial) -> Dict:
        M   = trial.suggest_categorical("M",   self.ENGINE_PARAMS["FaissHNSWFlat"]["M"])
        efC = trial.suggest_categorical("efC", self.ENGINE_PARAMS["FaissHNSWFlat"]["efConstruction"])
        efS = trial.suggest_categorical("efS", self.ENGINE_PARAMS["FaissHNSWFlat"]["efSearch"])
        k   = trial.suggest_categorical("k",   self.K_NEIGHBORS_VALUES)
        alpha = trial.suggest_categorical("alpha", self.ALPHA_VALUES)
        ref_strat = trial.suggest_categorical("ref_strat", self.REF_STRATEGY_VALUES)
        n_refs = 1 if ref_strat == "centroid" else trial.suggest_categorical("n_refs", self.N_REFS_VALUES)
        n_aqe  = trial.suggest_categorical("n_aqe", self.N_AQE_VALUES)
        aqe_w  = trial.suggest_categorical("aqe_w", self.AQE_WEIGHT_VALUES) if n_aqe > 1 else 0.0
        return {
            "engine": "FaissHNSWFlat",
            "params": {"M": M, "efConstruction": efC, "efSearch": efS},
            "k_neighbors": k, "alpha": alpha,
            "n_refs": n_refs, "ref_strategy": ref_strat,
            "n_aqe": n_aqe, "aqe_weight": aqe_w,
        }

    def grid_configs(self, n: int) -> List[Dict]:
        cfgs = []
        for M in [32, 64]:
            for efS in [64, 200, 500]:
                for k in [100, 200]:
                    for alpha in [0.30, 0.70]:
                        cfgs.append({
                            "engine": "FaissHNSWFlat",
                            "params": {"M": M, "efConstruction": 200, "efSearch": efS},
                            "k_neighbors": k, "alpha": alpha,
                            "n_refs": 1, "ref_strategy": "centroid",
                            "n_aqe": 1, "aqe_weight": 0.0,
                        })
        return cfgs[:n]


# ---------------------------------------------------------------------------
# LLM agent (simplified)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are tuning a two-stage FAISS retrieval pipeline for GLDv2 (762K × 768-d CLIP IP vectors).
Stage 1: FAISS HNSW IP search. Stage 1b: AQE (alpha query expansion). Stage 2: DINOv2 reranking.

Goal: maximize QPS subject to mAP ≥ 0.15.
Score = QPS if mAP ≥ 0.15 else 0.

Parameters:
  M: [8, 16, 32, 48, 64]         — HNSW graph connectivity
  efConstruction: [100, 200, 400] — build quality
  efSearch: [32, 64, 100, 150, 200, 300, 500] — search beam width (main QPS lever)
  k_neighbors: [50, 100, 150, 200, 300, 500, 750, 1000]
  alpha: [0.0..0.90]              — DINOv2 weight (0=CLIP only, 0.9=mostly DINOv2)
  n_refs: [1, 3, 5, 10]          — PRF references for DINOv2 query augmentation
  ref_strategy: [first, centroid, diverse]
  n_aqe: [1, 3, 5, 10]          — AQE candidates (1=disabled)
  aqe_weight: [0.1, 0.2, 0.3, 0.5] — AQE blend (only when n_aqe>1)

YOU MUST respond with ONLY a valid JSON object. No explanation, no markdown, no code blocks.
Example: {"engine": "FaissHNSWFlat", "params": {"M": 32, "efConstruction": 200, "efSearch": 64}, "k_neighbors": 100, "alpha": 0.3, "n_refs": 1, "ref_strategy": "first", "n_aqe": 1, "aqe_weight": 0.0, "reasoning": "low efSearch for high QPS"}"""


class LLMAgent:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key  = api_key
        self.base_url = base_url
        self.model    = model

    async def suggest(self, context: str, provider: ConfigProvider) -> Dict:
        import aiohttp
        prompt = f"{_SYSTEM_PROMPT}\n\n{context}"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        max_tokens = 4096
        for attempt in range(3):
            try:
                payload = {"model": self.model,
                           "messages": [{"role": "user", "content": prompt}],
                           "temperature": 0.3, "max_tokens": max_tokens}
                async with aiohttp.ClientSession() as sess:
                    async with sess.post(f"{self.base_url}/chat/completions",
                                         headers=headers, json=payload,
                                         timeout=aiohttp.ClientTimeout(total=120)) as r:
                        data = await r.json()
                if "choices" not in data:
                    raise RuntimeError(f"API error: {data}")
                choice = data["choices"][0]
                if choice.get("finish_reason") == "length":
                    max_tokens = min(max_tokens * 2, 16384)
                    print(f"  [LLM] finish_reason=length; retrying with max_tokens={max_tokens}")
                    await asyncio.sleep(2 ** attempt)
                    continue
                msg = choice["message"]
                content = msg.get("content") or msg.get("reasoning_content") or ""
                if not content:
                    raise RuntimeError(f"Empty content: {data}")
                content = content.strip()
                content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.MULTILINE).strip()
                cfg = json.loads(content)
                cfg.pop("reasoning", None)
                cfg.setdefault("engine", "FaissHNSWFlat")
                cfg.setdefault("params", {})
                for fld, default in [("k_neighbors", 100), ("alpha", 0.30),
                                      ("n_refs", 1), ("ref_strategy", "first"),
                                      ("n_aqe", 1), ("aqe_weight", 0.0)]:
                    cfg.setdefault(fld, default)
                if cfg.get("n_aqe", 1) == 1:
                    cfg["aqe_weight"] = 0.0
                if cfg.get("ref_strategy") == "centroid":
                    cfg["n_refs"] = 1
                return cfg
            except Exception as e:
                if attempt == 2:
                    print(f"  [LLM error after 3 attempts: {e}] → random fallback")
                    return provider.random_config()
                await asyncio.sleep(2 ** attempt)


# ---------------------------------------------------------------------------
# Optimizer loop
# ---------------------------------------------------------------------------

@dataclass
class IterationResult:
    iteration:    int
    config:       Dict
    benchmark:    BenchmarkResult
    search_method: str


async def run_optimizer(args) -> List[IterationResult]:
    global _MAP_THRESHOLD
    _MAP_THRESHOLD = args.map_threshold

    provider    = ConfigProvider(seed=args.seed)
    results:    List[IterationResult] = []
    dataset_dir = args.dataset_dir

    if args.method == "hyperparameter_only":
        api_key  = os.getenv("OPENROUTER_API_KEY", "")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        model    = os.getenv("LLM_MODEL", "minimax/minimax-m2.1")
        agent    = LLMAgent(api_key, base_url, model)
        seen: set = set()

        for i in range(args.iterations):
            history_str = "\n".join(
                f"  iter {r.iteration}: {json.dumps(r.config)}  mAP={r.benchmark.map_score:.4f}  qps={r.benchmark.qps:.1f}"
                for r in results[-8:]
            )
            context = (f"Iteration {i+1}/{args.iterations}. τ={_MAP_THRESHOLD:.2f}.\n"
                       f"Recent:\n{history_str}\nSuggest next config.")
            cfg = await agent.suggest(context, provider)
            cfg_key = json.dumps(cfg, sort_keys=True)
            if cfg_key in seen:
                cfg = provider.random_config()
            seen.add(json.dumps(cfg, sort_keys=True))
            cfg["_iter"] = i + 1
            bm = run_benchmark(cfg, dataset_dir, iteration=i + 1)
            results.append(IterationResult(i+1, cfg, bm, "llm"))

    elif args.method == "random":
        for i in range(args.iterations):
            cfg = provider.random_config()
            cfg["_iter"] = i + 1
            bm = run_benchmark(cfg, dataset_dir, iteration=i + 1)
            results.append(IterationResult(i+1, cfg, bm, "random"))

    elif args.method == "grid":
        for i, cfg in enumerate(provider.grid_configs(args.iterations)):
            cfg["_iter"] = i + 1
            bm = run_benchmark(cfg, dataset_dir, iteration=i + 1)
            results.append(IterationResult(i+1, cfg, bm, "grid"))

    elif args.method in ("optuna", "gp_bo"):
        sampler = (optuna.samplers.TPESampler(seed=args.seed)
                   if args.method == "optuna"
                   else optuna.samplers.GPSampler(seed=args.seed))
        _res = results

        def _objective(trial):
            cfg = provider.optuna_suggest(trial)
            cfg["_iter"] = trial.number + 1
            bm = run_benchmark(cfg, dataset_dir, iteration=trial.number + 1)
            _res.append(IterationResult(trial.number+1, cfg, bm, args.method))
            return bm.score()

        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(_objective, n_trials=args.iterations)

    elif args.method == "vdtuner":
        _encoder = KnobEncoder([
            {"name": "M",        "type": "enum",
             "values": ConfigProvider.ENGINE_PARAMS["FaissHNSWFlat"]["M"]},
            {"name": "efSearch", "type": "enum",
             "values": ConfigProvider.ENGINE_PARAMS["FaissHNSWFlat"]["efSearch"]},
            {"name": "k_neighbors", "type": "enum",
             "values": ConfigProvider.K_NEIGHBORS_VALUES},
            {"name": "alpha",    "type": "enum",
             "values": ConfigProvider.ALPHA_VALUES},
            {"name": "n_refs",   "type": "enum",
             "values": ConfigProvider.N_REFS_VALUES},
            {"name": "ref_strategy", "type": "enum",
             "values": ConfigProvider.REF_STRATEGY_VALUES},
            {"name": "n_aqe",    "type": "enum",
             "values": ConfigProvider.N_AQE_VALUES},
            {"name": "aqe_weight", "type": "enum",
             "values": ConfigProvider.AQE_WEIGHT_VALUES},
        ])
        vdtuner_opt = VDTunerOptimizer(_encoder, seed=args.seed)
        for i in range(args.iterations):
            partial = vdtuner_opt.ask()
            n_aqe = partial["n_aqe"]
            aqe_w = partial["aqe_weight"] if n_aqe > 1 else 0.0
            cfg = {
                "engine": "FaissHNSWFlat",
                "params": {"M": partial["M"], "efConstruction": 200,
                           "efSearch": partial["efSearch"]},
                "k_neighbors":  partial["k_neighbors"],
                "alpha":        partial["alpha"],
                "n_refs":       partial["n_refs"],
                "ref_strategy": partial["ref_strategy"],
                "n_aqe":        n_aqe,
                "aqe_weight":   aqe_w,
                "_iter": i + 1,
            }
            bm = run_benchmark(cfg, dataset_dir, iteration=i + 1)
            vdtuner_opt.tell(partial, bm.map_score, bm.qps)
            results.append(IterationResult(i+1, cfg, bm, "vdtuner"))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir",    required=True)
    parser.add_argument("--method",         default="hyperparameter_only",
                        choices=["hyperparameter_only", "random", "grid", "optuna", "gp_bo", "vdtuner"])
    parser.add_argument("--iterations",     type=int, default=50)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--map-threshold",  type=float, default=0.15)
    parser.add_argument("--output",         default="milvus_gldv2_results.json")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Milvus GLDv2 Optimizer  method={args.method}  seed={args.seed}  τ={args.map_threshold}")
    print("=" * 60)

    results = asyncio.run(run_optimizer(args))

    best = max(results, key=lambda r: r.benchmark.score(), default=None)
    feasible = [r for r in results if r.benchmark.map_score >= args.map_threshold]

    summary = {
        "method": args.method, "seed": args.seed, "iterations": args.iterations,
        "map_threshold": args.map_threshold,
        "best_score":  best.benchmark.score()      if best else 0.0,
        "best_map":    best.benchmark.map_score    if best else 0.0,
        "best_qps":    best.benchmark.qps          if best else 0.0,
        "best_config": best.config                 if best else {},
        "n_feasible":  len(feasible),
        "all_results": [
            {"iteration": r.iteration, "config": r.config, "search_method": r.search_method,
             "score": r.benchmark.score(), "map": r.benchmark.map_score,
             "qps": r.benchmark.qps, "latency_ms": r.benchmark.latency_ms}
            for r in results
        ],
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved → {args.output}")
    if best:
        print(f"Best: mAP={best.benchmark.map_score:.4f}  QPS={best.benchmark.qps:.1f}  "
              f"Score={best.benchmark.score():.1f}")


if __name__ == "__main__":
    main()
