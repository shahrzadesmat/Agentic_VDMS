"""
Milvus-backend SIFT1M optimizer (pure FAISS, no VDMS).
Methods: hyperparameter_only (LLM), random, grid, optuna, gp_bo, vdtuner.
Metric: Recall@10, threshold τ=0.90, Score = QPS if Recall@10 ≥ τ else 0.
"""

import sys, os, time, json, random, asyncio, re, argparse
import numpy as np
import faiss
import optuna
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vdtuner_ehvi import VDTunerOptimizer, KnobEncoder

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
_RECALL_THRESHOLD: float = 0.90
_SIFT1M_CACHE: Dict = {}
_FAISS_INDEX  = None
_FAISS_LAST_CFG: Optional[str] = None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _read_fvecs(path: str) -> np.ndarray:
    a = np.fromfile(path, dtype="int32")
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy().view("float32")

def _read_ivecs(path: str) -> np.ndarray:
    a = np.fromfile(path, dtype="int32")
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def _load_data(dataset_dir: str) -> Dict:
    global _SIFT1M_CACHE
    if _SIFT1M_CACHE:
        return _SIFT1M_CACHE
    d = Path(dataset_dir) / "sift"
    print("[SIFT1M] Loading base vectors (1M × 128)…")
    t0 = time.time()
    base_vecs  = _read_fvecs(str(d / "sift_base.fvecs"))
    query_vecs = _read_fvecs(str(d / "sift_query.fvecs"))
    gt_array   = _read_ivecs(str(d / "sift_groundtruth.ivecs"))
    gt_sets    = [set(row[:10].tolist()) for row in gt_array]
    print(f"[SIFT1M] Loaded in {time.time()-t0:.1f}s  base={base_vecs.shape}  queries={query_vecs.shape}")
    _SIFT1M_CACHE = {"base": base_vecs, "queries": query_vecs, "gt_sets": gt_sets}
    return _SIFT1M_CACHE


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------

def _build_index(engine: str, params: Dict, base: np.ndarray) -> faiss.Index:
    d = base.shape[1]
    if engine == "FaissFlat":
        idx = faiss.IndexFlatL2(d)
    elif engine == "FaissHNSWFlat":
        M   = int(params.get("M", 32))
        efC = int(params.get("efConstruction", 200))
        idx = faiss.IndexHNSWFlat(d, M, faiss.METRIC_L2)
        idx.hnsw.efConstruction = efC
    else:
        raise ValueError(f"Unknown engine: {engine}")
    idx.add(base)
    return idx


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    recall_at_10: float
    qps: float
    latency_ms: float
    index_build_s: float

    def score(self) -> float:
        return self.qps if self.recall_at_10 >= _RECALL_THRESHOLD else 0.0


def run_benchmark(cfg: Dict, dataset_dir: str, iteration: int = 0) -> BenchmarkResult:
    global _FAISS_INDEX, _FAISS_LAST_CFG

    data   = _load_data(dataset_dir)
    base   = data["base"]
    qvecs  = data["queries"]
    gt_sets = data["gt_sets"]

    engine = cfg.get("engine", "FaissHNSWFlat")
    params = cfg.get("params", {})
    k      = int(cfg.get("k_neighbors", 50))

    cfg_key = json.dumps({"engine": engine, "params": params}, sort_keys=True)
    if cfg_key != _FAISS_LAST_CFG:
        t0 = time.time()
        print(f"[FAISS] Building index: {engine} params={params}…")
        _FAISS_INDEX    = _build_index(engine, params, base)
        _FAISS_LAST_CFG = cfg_key
        build_s = time.time() - t0
        print(f"[FAISS] Index built in {build_s:.1f}s")
    else:
        build_s = 0.0

    efS = int(params.get("efSearch", 32))
    if engine == "FaissHNSWFlat":
        _FAISS_INDEX.hnsw.efSearch = efS

    n_q = len(qvecs)
    t0  = time.time()
    _, I = _FAISS_INDEX.search(qvecs, k)
    elapsed = time.time() - t0

    recalls = []
    for qi in range(n_q):
        retrieved = set(int(x) for x in I[qi] if x >= 0)
        gt = gt_sets[qi]
        recalls.append(len(retrieved & gt) / max(len(gt), 1))
    recall = float(np.mean(recalls))
    qps    = float(n_q / elapsed)
    lat_ms = float(elapsed / n_q * 1000)

    print(f"  Score={qps if recall >= _RECALL_THRESHOLD else 0:.1f}  "
          f"Recall@10={recall:.4f}  QPS={qps:.1f}  build={build_s:.1f}s")

    return BenchmarkResult(recall_at_10=recall, qps=qps,
                           latency_ms=lat_ms, index_build_s=build_s)


# ---------------------------------------------------------------------------
# Config provider
# ---------------------------------------------------------------------------

class ConfigProvider:
    ENGINE_PARAMS = {
        "FaissFlat":    {},
        "FaissHNSWFlat": {
            "M":              [8, 16, 32, 48, 64],
            "efConstruction": [100, 200, 400],
            "efSearch":       [16, 32, 64, 100, 150, 200, 300, 500],
        },
    }
    K_NEIGHBORS = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500]

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        random.seed(seed)

    def random_config(self) -> Dict:
        engine = str(self.rng.choice(["FaissHNSWFlat", "FaissFlat"]))
        params = {k: int(self.rng.choice(v)) for k, v in self.ENGINE_PARAMS[engine].items()}
        return {"engine": engine, "params": params,
                "k_neighbors": int(self.rng.choice(self.K_NEIGHBORS))}

    def optuna_suggest(self, trial) -> Dict:
        engine = trial.suggest_categorical("engine", ["FaissHNSWFlat"])
        cfg = {"engine": engine, "params": {}, "k_neighbors": trial.suggest_categorical("k", self.K_NEIGHBORS)}
        if engine == "FaissHNSWFlat":
            cfg["params"]["M"]              = trial.suggest_categorical("M", self.ENGINE_PARAMS[engine]["M"])
            cfg["params"]["efConstruction"] = trial.suggest_categorical("efC", self.ENGINE_PARAMS[engine]["efConstruction"])
            cfg["params"]["efSearch"]       = trial.suggest_categorical("efS", self.ENGINE_PARAMS[engine]["efSearch"])
        return cfg

    def grid_configs(self, n: int) -> List[Dict]:
        cfgs = []
        for M in [16, 32, 64]:
            for efS in [32, 64, 100, 200, 500]:
                for k in [10, 50, 100, 200]:
                    cfgs.append({"engine": "FaissHNSWFlat",
                                 "params": {"M": M, "efConstruction": 200, "efSearch": efS},
                                 "k_neighbors": k})
        cfgs.append({"engine": "FaissFlat", "params": {}, "k_neighbors": 10})
        return cfgs[:n]


# ---------------------------------------------------------------------------
# LLM agent (simplified)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are tuning a FAISS HNSW index for SIFT1M (1M × 128-d L2 vectors).

Goal: maximize QPS subject to Recall@10 ≥ 0.90.
Score = QPS if Recall@10 ≥ 0.90 else 0.

Parameters (FaissHNSWFlat only — ignore FaissFlat/FaissIVFFlat):
  M:              [8, 16, 32, 48, 64]      — graph connectivity, higher=better recall
  efConstruction: [100, 200, 400]          — build quality, not tunable at search time
  efSearch:       [16, 32, 64, 100, 150, 200, 300, 500] — search beam width, main QPS/recall lever
  k_neighbors:    [10, 20, 30, 50, 75, 100, 150, 200, 300, 500]

Key tradeoffs:
  - Lower efSearch → higher QPS, lower Recall@10
  - Higher M → better Recall@10 at same efSearch (but slower build)
  - k_neighbors: return exactly this many results; lower k = faster
  - efConstruction does NOT affect search speed

Good starting point: M=16 efC=200 efS=32 k=10 → Recall≈0.91 ≥ τ, QPS~1000+

YOU MUST respond with ONLY a valid JSON object. No explanation, no markdown, no code blocks.
Example: {"engine": "FaissHNSWFlat", "params": {"M": 16, "efConstruction": 200, "efSearch": 32}, "k_neighbors": 10, "reasoning": "low efSearch for high QPS"}"""


class LLMAgent:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key  = api_key
        self.base_url = base_url
        self.model    = model
        self.history: List[Dict] = []

    async def suggest(self, context: str, provider: ConfigProvider) -> Dict:
        import aiohttp
        prompt = f"{_SYSTEM_PROMPT}\n\n{context}"
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
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
                cfg.setdefault("k_neighbors", 50)
                if cfg["engine"] not in ("FaissHNSWFlat", "FaissFlat"):
                    cfg["engine"] = "FaissHNSWFlat"
                for param, vals in provider.ENGINE_PARAMS.get(cfg["engine"], {}).items():
                    if param in cfg["params"] and cfg["params"][param] not in vals:
                        cfg["params"][param] = min(vals, key=lambda x: abs(x - cfg["params"][param]))
                if cfg["k_neighbors"] not in provider.K_NEIGHBORS:
                    cfg["k_neighbors"] = min(provider.K_NEIGHBORS, key=lambda x: abs(x - cfg["k_neighbors"]))
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
    global _RECALL_THRESHOLD
    _RECALL_THRESHOLD = args.recall_threshold

    provider = ConfigProvider(seed=args.seed)
    results: List[IterationResult] = []
    dataset_dir = args.dataset_dir

    if args.method == "hyperparameter_only":
        api_key  = os.getenv("OPENROUTER_API_KEY", "")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        model    = os.getenv("LLM_MODEL", "minimax/minimax-m2.1")
        agent    = LLMAgent(api_key, base_url, model)
        seen: set = set()

        for i in range(args.iterations):
            history_str = "\n".join(
                f"  iter {r.iteration}: {json.dumps(r.config)}  recall={r.benchmark.recall_at_10:.4f}  qps={r.benchmark.qps:.1f}"
                for r in results[-10:]
            )
            context = (f"Iteration {i+1}/{args.iterations}. τ={_RECALL_THRESHOLD:.2f}.\n"
                       f"Recent results:\n{history_str}\n"
                       f"Suggest next config.")
            cfg = await agent.suggest(context, provider)
            cfg_key = json.dumps(cfg, sort_keys=True)
            if cfg_key in seen:
                cfg = provider.random_config()
                cfg_key = json.dumps(cfg, sort_keys=True)
            seen.add(cfg_key)
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
            {"name": "efConstruction", "type": "enum",
             "values": ConfigProvider.ENGINE_PARAMS["FaissHNSWFlat"]["efConstruction"]},
            {"name": "efSearch", "type": "enum",
             "values": ConfigProvider.ENGINE_PARAMS["FaissHNSWFlat"]["efSearch"]},
            {"name": "k_neighbors", "type": "enum",
             "values": ConfigProvider.K_NEIGHBORS},
        ])
        vdtuner_opt = VDTunerOptimizer(_encoder, seed=args.seed)
        for i in range(args.iterations):
            partial = vdtuner_opt.ask()
            cfg = {
                "engine": "FaissHNSWFlat",
                "params": {"M": partial["M"], "efConstruction": partial["efConstruction"],
                           "efSearch": partial["efSearch"]},
                "k_neighbors": partial["k_neighbors"],
                "_iter": i + 1,
            }
            bm = run_benchmark(cfg, dataset_dir, iteration=i + 1)
            vdtuner_opt.tell(partial, bm.recall_at_10, bm.qps)
            results.append(IterationResult(i+1, cfg, bm, "vdtuner"))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir",       required=True)
    parser.add_argument("--method",            default="hyperparameter_only",
                        choices=["hyperparameter_only", "random", "grid", "optuna", "gp_bo", "vdtuner"])
    parser.add_argument("--iterations",        type=int, default=50)
    parser.add_argument("--seed",              type=int, default=42)
    parser.add_argument("--recall-threshold",  type=float, default=0.90)
    parser.add_argument("--output",            default="milvus_sift1m_results.json")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Milvus SIFT1M Optimizer  method={args.method}  seed={args.seed}  τ={args.recall_threshold}")
    print("=" * 60)

    results = asyncio.run(run_optimizer(args))

    best = max(results, key=lambda r: r.benchmark.score(), default=None)
    feasible = [r for r in results if r.benchmark.recall_at_10 >= args.recall_threshold]

    summary = {
        "method": args.method, "seed": args.seed, "iterations": args.iterations,
        "recall_threshold": args.recall_threshold,
        "best_score":   best.benchmark.score()        if best else 0.0,
        "best_recall":  best.benchmark.recall_at_10   if best else 0.0,
        "best_qps":     best.benchmark.qps            if best else 0.0,
        "best_config":  best.config                   if best else {},
        "n_feasible":   len(feasible),
        "all_results": [
            {"iteration": r.iteration, "config": r.config, "search_method": r.search_method,
             "score": r.benchmark.score(), "recall_at_10": r.benchmark.recall_at_10,
             "qps": r.benchmark.qps, "latency_ms": r.benchmark.latency_ms}
            for r in results
        ],
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved → {args.output}")
    if best:
        print(f"Best: Recall@10={best.benchmark.recall_at_10:.4f}  QPS={best.benchmark.qps:.1f}  "
              f"Score={best.benchmark.score():.1f}")


if __name__ == "__main__":
    main()
