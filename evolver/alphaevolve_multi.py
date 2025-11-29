# evolver/alphaevolve_multi.py
"""
Data-Aware AlphaEvolve (HF dataset, ICL-only baseline, OpenRouter).

- 结构化 Prompt 进化：PromptGenome + GA
- Few-shot ICL 示例随基因一起进化（icl_strategy, k_shots）
- 当前阶段 RAG=none, summary=false, teacher=false（baseline）
- 多指标评估：QWK / Pearson / RMSE / Accuracy
  * 优先级：QWK > Pearson > RMSE
- ✅ 新增：每代统计 LLM text/ICL 变异触发率、成功率
"""

from __future__ import annotations

import json
import math
import os
import random
import time
import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from evolver.prompt_evolver import (
    Individual,
    build_initial_population,
    tournament_selection,
    crossover_genome,
    mutate_genome,
    set_llm_feedback,
    load_template_pool,
    update_template_pool,
    reset_llm_stats,   # ✅ 新增
    get_llm_stats,     # ✅ 新增
)
from evolver.data_aware_prompt import PromptGenome, build_full_prompt, INSTRUCTION_TEMPLATES
from evolver.icl_sampler import select_icl_examples
from evolver.checkpoint import (  # 🔥 新增：断点续传
    save_checkpoint,
    load_checkpoint,
    restore_population,
    restore_best_individual,
    clean_old_checkpoints,
)
from llm_api.openrouter_api import call_scoring_llm

# ================== 路径 & 常量配置 ================== #

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_HF_TRAIN = BASE_DIR / "data" / "raw" / "hf_dataset" / "train.csv"
PROC_DIR = BASE_DIR / "data" / "processed" / "hf_dataset"
TRAIN_CLEAN = PROC_DIR / "train_clean.csv"
EVAL_CLEAN = PROC_DIR / "eval_clean.csv"

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

BEST_JSON = LOG_DIR / "best_scoring_prompt_hf.json"
BEST_TXT = LOG_DIR / "best_scoring_prompt_hf.txt"
BEST_PRED_CSV = LOG_DIR / "best_prompt_predictions_hf.csv"
METRIC_FIG = LOG_DIR / "metrics_curve_hf.png"

# ✅ 模板池持久化路径
TEMPLATE_POOL_JSON = LOG_DIR / "template_pool.json"

# 🔥 断点续传配置
CHECKPOINT_DIR = LOG_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
ENABLE_CHECKPOINT = os.getenv("ENABLE_CHECKPOINT", "1") == "1"
CHECKPOINT_EVERY_GEN = int(os.getenv("CHECKPOINT_EVERY_GEN", "1"))  # 每 N 代保存一次

# -------- GA 超参数 --------
POP_SIZE = 10
N_GENERATIONS = 6
TOURNAMENT_K = 4
CROSSOVER_RATE = 0.85
MUTATION_RATE = 0.35

# 🔥 分阶段评估配置
USE_STAGED_EVAL = os.getenv("USE_STAGED_EVAL", "0") == "1"
N_EVAL_SAMPLES = int(os.getenv("N_EVAL_SAMPLES", "64"))
N_EVAL_SAMPLES_EARLY = int(os.getenv("N_EVAL_SAMPLES_EARLY", "32"))
N_EVAL_SAMPLES_LATE = int(os.getenv("N_EVAL_SAMPLES_LATE", "64"))
EARLY_PHASE_RATIO = float(os.getenv("EARLY_PHASE_RATIO", "0.67"))

MAX_CONTEXT_CHARS = 12000
EARLYSTOP_CONSEC_FAIL = 3
EARLYSTOP_FAIL_RATE = 0.6
MIN_SAMPLES_BEFORE_EARLYSTOP = 8

SINGLE_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")

# 可配置的最小调用间隔（秒）
MIN_INTERVAL = float(os.getenv("OPENROUTER_MIN_INTERVAL", "1.0"))
_last_call_ts = 0.0


def throttle():
    global _last_call_ts
    now = time.time()
    wait = MIN_INTERVAL - (now - _last_call_ts)
    if wait > 0:
        time.sleep(wait)
    _last_call_ts = time.time()


_LLM_CACHE: Dict[str, Optional[str]] = {}


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_band_from_text(text: str, default: float = 5.0) -> float:
    import re
    if not text:
        return default
    clean = text.replace(",", " ").replace("\n", " ")
    nums = re.findall(r"(?<!\d)([0-9](?:\.5)?)(?!\d)", clean)
    if not nums:
        nums = re.findall(r"\d+(?:\.\d+)?", clean)
        if not nums:
            return default
    val = float(nums[0])
    val = max(0.0, min(9.0, val))
    return round(val * 2) / 2.0


def quadratic_weighted_kappa(y_true: List[float], y_pred: List[float]) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    y_true_i = np.round(y_true * 2).astype(int)
    y_pred_i = np.round(y_pred * 2).astype(int)

    min_rating, max_rating = 0, 18
    n_ratings = max_rating - min_rating + 1

    O = np.zeros((n_ratings, n_ratings), dtype=float)
    for a, b in zip(y_true_i, y_pred_i):
        if 0 <= a <= max_rating and 0 <= b <= max_rating:
            O[a, b] += 1.0

    hist_true = np.zeros(n_ratings, dtype=float)
    hist_pred = np.zeros(n_ratings, dtype=float)
    for a in y_true_i:
        if 0 <= a <= max_rating:
            hist_true[a] += 1.0
    for b in y_pred_i:
        if 0 <= b <= max_rating:
            hist_pred[b] += 1.0

    E = np.outer(hist_true, hist_pred)
    if E.sum() == 0:
        return 0.0
    E = E / E.sum() * O.sum()

    W = np.zeros((n_ratings, n_ratings), dtype=float)
    for i in range(n_ratings):
        for j in range(n_ratings):
            W[i, j] = ((i - j) ** 2) / ((max_rating - min_rating) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 0.0
    return 1.0 - num / den


def compute_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    y_true_arr = np.array(y_true, dtype=float)
    y_pred_arr = np.array(y_pred, dtype=float)

    qwk = quadratic_weighted_kappa(y_true_arr.tolist(), y_pred_arr.tolist())

    try:
        if np.std(y_pred_arr) == 0 or np.std(y_true_arr) == 0:
            pearson = 0.0
        else:
            pearson = float(pearsonr(y_true_arr, y_pred_arr)[0])
    except Exception:
        pearson = 0.0

    rmse = float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))

    exact_acc = float(np.mean(y_true_arr == y_pred_arr))
    adj_acc = float(np.mean(np.abs(y_true_arr - y_pred_arr) <= 0.5))

    return {
        "qwk": qwk,
        "pearson": pearson,
        "rmse": rmse,
        "exact_acc": exact_acc,
        "adj_acc": adj_acc,
    }


def fitness_from_metrics(m: Dict[str, float]) -> float:
    qwk = m["qwk"]
    pearson = max(m["pearson"], 0.0)
    rmse = m["rmse"]
    return qwk + 0.3 * pearson - 0.2 * rmse


# ================== 偏差统计（给 LLM 反馈用）================== #

def compute_bias_stats(labels: List[float], preds: List[float]) -> Dict[str, Any]:
    """统计 pred-true 的偏差：整体 + 分 band。"""
    if not labels:
        return {}

    labels_arr = np.array(labels, dtype=float)
    preds_arr = np.array(preds, dtype=float)
    err = preds_arr - labels_arr

    mean_err = float(np.mean(err))
    mae = float(np.mean(np.abs(err)))
    over_rate = float(np.mean(err > 0))
    under_rate = float(np.mean(err < 0))

    by_band: Dict[int, Dict[str, Any]] = {}
    for b in sorted(set(np.round(labels_arr * 2).astype(int))):
        mask = (np.round(labels_arr * 2).astype(int) == b)
        e = err[mask]
        if e.size == 0:
            continue
        by_band[int(b)] = {
            "n": int(e.size),
            "mean_err": float(np.mean(e)),
            "mae": float(np.mean(np.abs(e))),
        }

    return {
        "mean_err": mean_err,
        "mae": mae,
        "over_rate": over_rate,
        "under_rate": under_rate,
        "by_band": by_band,
    }


# ================== HF 数据加载 ================== #

Sample = Tuple[int, str, float, str]

def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(str(x).strip())
    except Exception:
        return None


def load_hf_dataset() -> Tuple[List[Sample], List[Sample]]:
    if TRAIN_CLEAN.exists() and EVAL_CLEAN.exists():
        train_df = pd.read_csv(TRAIN_CLEAN)
        eval_df = pd.read_csv(EVAL_CLEAN)
        log(f"Loaded HF train_clean: {len(train_df)} samples.")
        log(f"Loaded HF eval_clean : {len(eval_df)} samples.")
    else:
        df = pd.read_csv(RAW_HF_TRAIN)
        df = df.dropna(subset=["essay", "band"]).reset_index(drop=True)
        train_df = df
        eval_df = df.sample(frac=0.05, random_state=42)
        train_df = df.drop(eval_df.index).reset_index(drop=True)
        eval_df = eval_df.reset_index(drop=True)
        log(f"Loaded RAW HF train: {len(train_df)} | eval: {len(eval_df)}")

    def to_samples(d: pd.DataFrame) -> List[Sample]:
        out: List[Sample] = []
        for i, row in d.iterrows():
            essay = str(row.get("essay", "")).strip()
            prompt = str(row.get("prompt", "")).strip()
            band = _safe_float(row.get("band"))
            if band is None:
                continue
            out.append((i, essay, float(band), prompt))
        return out

    return to_samples(train_df), to_samples(eval_df)


# ================== RAG (stub) ================== #

class DummyRAG:
    def retrieve(self, essay: str, strategy: str = "none", k: int = 3):
        return []


# ================== 个体评估 ================== #

def evaluate_individual(
    ind: Individual,
    eval_pool: List[Sample],
    train_pool: List[Sample],
    rag: DummyRAG,
) -> Dict[str, float]:
    preds: List[float] = []
    labels: List[float] = []

    consecutive_fail = 0
    fail_cnt = 0
    valid_cnt = 0

    MIN_VALID_RATIO = 0.6
    MAX_FAIL_STREAK = EARLYSTOP_CONSEC_FAIL

    for i, (sid, essay, true_band, prompt_text) in enumerate(eval_pool, start=1):

        # 🔥 根据模式选择 ICL 示例
        if ind.genome.use_icl_indices and ind.genome.icl_indices:
            # 新模式：使用索引列表
            icl_examples = select_icl_examples(
                train_pool,
                strategy="random",  # 不使用
                k=ind.genome.k_shots,
                seed=sid,
                indices=ind.genome.icl_indices,
            )
        else:
            # 旧模式：使用策略
            icl_examples = select_icl_examples(
                train_pool,
                strategy=ind.genome.icl_strategy,
                k=ind.genome.k_shots,
                seed=sid,
            )

        rag_examples = rag.retrieve(
            essay,
            strategy=ind.genome.rag_strategy,
            k=3,
        )

        full_prompt = build_full_prompt(
            genome=ind.genome,
            essay=essay,
            icl_examples=icl_examples,
            rag_examples=rag_examples,
            summary_text=None,
        )

        if len(full_prompt) > MAX_CONTEXT_CHARS:
            head = full_prompt[: int(MAX_CONTEXT_CHARS * 0.75)]
            tail = full_prompt[-int(MAX_CONTEXT_CHARS * 0.20):]
            full_prompt = head + "\n\n[Truncated]\n\n" + tail

        cache_key = full_prompt
        cached = _LLM_CACHE.get(cache_key)

        if cached is None:
            throttle()
            reply = call_scoring_llm(
                full_prompt,
                temperature=0.0,
                model=SINGLE_MODEL,
                max_tokens=8,
                max_retries=3,
                timeout=60,
            )
            _LLM_CACHE[cache_key] = reply
        else:
            reply = cached

        if not reply:
            consecutive_fail += 1
            fail_cnt += 1
            raw_band = 5.0
        else:
            consecutive_fail = 0
            raw_band = parse_band_from_text(reply, default=5.0)
            valid_cnt += 1

        calibrated = round(float(raw_band) * 2) / 2.0
        preds.append(calibrated)
        labels.append(true_band)

        log(
            f"  [{i}/{len(eval_pool)}] "
            f"True={true_band:.1f} | Raw={raw_band:.1f} | Pred={calibrated:.1f}"
        )

        cur_fail_rate = fail_cnt / i
        if consecutive_fail >= MAX_FAIL_STREAK:
            log(f"  [EarlyStop] consecutive_fail={consecutive_fail} → stop this individual.")
            break

        if i >= MIN_SAMPLES_BEFORE_EARLYSTOP and cur_fail_rate >= EARLYSTOP_FAIL_RATE:
            log(f"  [EarlyStop] fail_rate={cur_fail_rate:.2%} ≥ {EARLYSTOP_FAIL_RATE:.0%} → stop.")
            break

        if i >= MIN_SAMPLES_BEFORE_EARLYSTOP:
            remain = len(eval_pool) - i
            if (valid_cnt + remain) / len(eval_pool) < MIN_VALID_RATIO:
                log(f"  [EarlyStop] valid_cnt too low → stop.")
                break

    m = compute_metrics(labels, preds)

    if valid_cnt < MIN_VALID_RATIO * len(labels):
        log(f"  [LowValid] valid={valid_cnt}/{len(labels)} → force low fitness.")
        ind.fitness = -1e9
    else:
        ind.fitness = fitness_from_metrics(m)

    ind.metrics = m
    ind.preds = preds
    ind.labels = labels

    log(
        f"  → QWK={m['qwk']:.4f}, Pearson={m['pearson']:.4f}, "
        f"RMSE={m['rmse']:.4f}, Exact={m['exact_acc']:.3f}, "
        f"Adj={m['adj_acc']:.3f}, Fitness={ind.fitness:.4f}, "
        f"FailRate={fail_cnt}/{len(labels)}"
    )
    return m


def get_eval_pool_size(gen: int, total_gens: int) -> int:
    """
    根据代数返回评估样本数。
    
    如果启用分阶段评估：
    - 前期（前 67% 代数）：使用较少样本快速筛选
    - 后期（后 33% 代数）：使用较多样本精确评估
    
    Args:
        gen: 当前代数
        total_gens: 总代数
    
    Returns:
        评估样本数
    """
    if not USE_STAGED_EVAL:
        return N_EVAL_SAMPLES
    
    if gen <= total_gens * EARLY_PHASE_RATIO:
        return N_EVAL_SAMPLES_EARLY
    else:
        return N_EVAL_SAMPLES_LATE


def stratified_sample(eval_pool_full, n=32, seed=42):
    rng = random.Random(seed)
    buckets = {}
    for s in eval_pool_full:
        key = int(round(s[2] * 2))
        buckets.setdefault(key, []).append(s)

    all_keys = sorted(buckets.keys())
    total = len(eval_pool_full)
    out = []

    for k in all_keys:
        frac = len(buckets[k]) / total
        take = max(1, round(frac * n))
        out.extend(rng.sample(buckets[k], min(take, len(buckets[k]))))

    if len(out) < n:
        rest = [s for s in eval_pool_full if s not in out]
        out.extend(rng.sample(rest, n - len(out)))

    if len(out) > n:
        out = rng.sample(out, n)

    rng.shuffle(out)
    return out


# ================== GA 主流程 ================== #

def run_evolution_hf_icl_only():
    print("==== Data-Aware AlphaEvolve (ICL-only baseline, OpenRouter) ====")
    print(f"Single model: {SINGLE_MODEL}")
    print(f"Checkpoint enabled: {ENABLE_CHECKPOINT}")

    # ✅ 启动时加载模板池
    load_template_pool(TEMPLATE_POOL_JSON)

    train_pool, eval_pool_full = load_hf_dataset()
    
    print(f"Train pool: {len(train_pool)} | Eval pool (full): {len(eval_pool_full)}")
    
    if USE_STAGED_EVAL:
        print(f"🎯 Staged evaluation enabled:")
        print(f"   Early phase (gen 1-{int(N_GENERATIONS * EARLY_PHASE_RATIO)}): {N_EVAL_SAMPLES_EARLY} samples")
        print(f"   Late phase (gen {int(N_GENERATIONS * EARLY_PHASE_RATIO)+1}-{N_GENERATIONS}): {N_EVAL_SAMPLES_LATE} samples")
    else:
        print(f"📊 Fixed evaluation: {N_EVAL_SAMPLES} samples per generation")

    # 🔥 尝试从检查点恢复
    checkpoint = None
    if ENABLE_CHECKPOINT:
        latest_checkpoint = CHECKPOINT_DIR / "checkpoint_latest.json"
        checkpoint = load_checkpoint(latest_checkpoint)
    
    # 初始化或恢复状态
    if checkpoint is not None:
        print("\n🔄 Resuming from checkpoint...")
        start_gen = checkpoint["generation"] + 1
        population = restore_population(checkpoint["population"])
        best_overall_ind = restore_best_individual(checkpoint["best_overall"])
        best_overall_fitness = checkpoint["best_overall_fitness"]
        history_qwk = checkpoint["history"]["qwk"]
        history_pearson = checkpoint["history"]["pearson"]
        history_rmse = checkpoint["history"]["rmse"]
        history_llm_stats = checkpoint["history"]["llm_stats"]
        
        # 恢复 LLM 缓存
        global _LLM_CACHE
        _LLM_CACHE = checkpoint.get("llm_cache", {})
        print(f"   Restored {len(_LLM_CACHE)} cached LLM calls")
        print(f"   Starting from generation {start_gen}")
    else:
        print("\n🆕 Starting fresh evolution...")
        start_gen = 1
        population = build_initial_population(
            pop_size=POP_SIZE,
            train_pool_size=len(train_pool)
        )
        history_qwk, history_pearson, history_rmse = [], [], []
        history_llm_stats: List[Dict[str, Any]] = []
        best_overall_ind: Optional[Individual] = None
        best_overall_fitness = -math.inf

    rag = DummyRAG()

    for gen in range(start_gen, N_GENERATIONS + 1):
        print(f"\n=== Generation {gen}/{N_GENERATIONS} ===")

        # ✅ 每代开始清空 LLM 统计：统计“本代产生下一代时”的 LLM 贡献
        reset_llm_stats()

        gen_best_ind: Optional[Individual] = None
        gen_best_metrics: Optional[Dict[str, float]] = None
        gen_best_fitness = -math.inf

        # ====== 评估本代所有个体 ======
        for i, ind in enumerate(population, start=1):
            print(f"\n[Gen {gen}] Individual {i}/{len(population)}")
            print(f"Genome: {ind.genome}")

            metrics = evaluate_individual(ind, eval_pool, train_pool, rag)
            fit = ind.fitness if ind.fitness is not None else -math.inf

            if fit > gen_best_fitness:
                gen_best_fitness = fit
                gen_best_ind = copy.deepcopy(ind)  # ✅ 保留 labels/preds/metrics
                gen_best_metrics = metrics

        assert gen_best_ind is not None and gen_best_metrics is not None

        history_qwk.append(gen_best_metrics["qwk"])
        history_pearson.append(gen_best_metrics["pearson"])
        history_rmse.append(gen_best_metrics["rmse"])

        print(
            f"\n[Gen {gen}] Best in generation: "
            f"QWK={gen_best_metrics['qwk']:.4f}, "
            f"Pearson={gen_best_metrics['pearson']:.4f}, "
            f"RMSE={gen_best_metrics['rmse']:.4f}, "
            f"Fitness={gen_best_fitness:.4f}"
        )

        # ✅ 更新 overall best
        if gen_best_fitness > best_overall_fitness:
            best_overall_fitness = gen_best_fitness
            best_overall_ind = copy.deepcopy(gen_best_ind)

        # ✅ 计算偏差统计 + 喂给 LLM
        best_text = (
            gen_best_ind.genome.instruction_text
            or INSTRUCTION_TEMPLATES.get(
                gen_best_ind.genome.instruction_id, INSTRUCTION_TEMPLATES[0]
            )
        )
        bias_stats = compute_bias_stats(gen_best_ind.labels or [], gen_best_ind.preds or [])
        set_llm_feedback(best_text, bias_stats, gen_best_metrics, gen)

        # ✅ 把好模板写进模板池并持久化
        update_template_pool(best_text, gen_best_fitness, gen_best_metrics, gen)

        # ====== 产生下一代（最后一代不需要生成） ======
        if gen < N_GENERATIONS:
            parents = tournament_selection(
                population, k=TOURNAMENT_K, num_winners=POP_SIZE
            )
            new_population: List[Individual] = []
            # elitism: 保留本代最优
            new_population.append(
                Individual(genome=copy.deepcopy(gen_best_ind.genome))
            )

            rng = random.Random(gen * 999)
            while len(new_population) < POP_SIZE:
                p1, p2 = rng.sample(parents, 2)
                child_genome = copy.deepcopy(p1.genome)

                if rng.random() < CROSSOVER_RATE:
                    child_genome = crossover_genome(p1.genome, p2.genome, rng)

                child_genome = mutate_genome(
                    child_genome, 
                    mutation_rate=MUTATION_RATE, 
                    rng=rng,
                    train_pool_size=len(train_pool)
                )
                new_population.append(Individual(genome=child_genome))

            population = new_population

        # ✅ 变异都发生完了（或最后一代无变异），再统计/打印
        stats = get_llm_stats()
        history_llm_stats.append({"gen": gen, **stats})
        print(f"\n[Gen {gen}] LLM mutation stats:")
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        
        # 🔥 保存检查点
        if ENABLE_CHECKPOINT and gen % CHECKPOINT_EVERY_GEN == 0:
            try:
                # 保存到 latest（用于恢复）
                latest_checkpoint = CHECKPOINT_DIR / "checkpoint_latest.json"
                save_checkpoint(
                    latest_checkpoint,
                    gen,
                    population,
                    best_overall_ind,
                    best_overall_fitness,
                    history_qwk,
                    history_pearson,
                    history_rmse,
                    history_llm_stats,
                    _LLM_CACHE,
                )
                
                # 同时保存带代数的备份
                gen_checkpoint = CHECKPOINT_DIR / f"checkpoint_gen_{gen}.json"
                save_checkpoint(
                    gen_checkpoint,
                    gen,
                    population,
                    best_overall_ind,
                    best_overall_fitness,
                    history_qwk,
                    history_pearson,
                    history_rmse,
                    history_llm_stats,
                    _LLM_CACHE,
                )
                
                # 清理旧检查点（保留最近 3 个）
                clean_old_checkpoints(CHECKPOINT_DIR, keep_last=3)
            except Exception as e:
                print(f"⚠️  Failed to save checkpoint: {e}")

    print("\n==== Evolution Finished (HF ICL-only) ====")
    if best_overall_ind is None:
        print("ERROR: best_overall_ind is None")
        return

    print(f"🎯 Best Fitness={best_overall_fitness:.4f}")
    print("Best Genome:", best_overall_ind.genome)

    payload = {
        "best_genome": asdict(best_overall_ind.genome),
        "best_metrics": best_overall_ind.metrics,
        "best_fitness": best_overall_fitness,
        "history_qwk": history_qwk,
        "history_pearson": history_pearson,
        "history_rmse": history_rmse,
        "history_llm_stats": history_llm_stats,
        "single_model": SINGLE_MODEL,
    }

    with open(BEST_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    best_prompt_text = build_full_prompt(
        genome=best_overall_ind.genome,
        essay="<<ESSAY_PLACEHOLDER>>",
        icl_examples=[],
        rag_examples=[],
        summary_text=None,
    )
    with open(BEST_TXT, "w", encoding="utf-8") as f:
        f.write(best_prompt_text)

    print(f"📁 Best JSON: {BEST_JSON}")
    print(f"📁 Best prompt txt: {BEST_TXT}")

    gens = list(range(1, N_GENERATIONS + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(gens, history_qwk, marker="o", label="QWK (↑)")
    plt.plot(gens, history_pearson, marker="s", label="Pearson (↑)")
    plt.plot(gens, history_rmse, marker="^", label="RMSE (↓)")
    plt.xlabel("Generation")
    plt.ylabel("Metric Value")
    plt.title("Best Metrics per Generation (HF ICL-only)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(METRIC_FIG, dpi=150)
    plt.close()
    print(f"📊 Metrics curve saved: {METRIC_FIG}")

    print("\n====== Final Evaluation with Best Genome (CSV) ======")
    final_pool = eval_pool

    preds, raws, labels, essays_out = [], [], [], []

    for i, (sid, essay, true_band, prompt_text) in enumerate(final_pool, start=1):
        # 🔥 根据模式选择 ICL 示例
        if best_overall_ind.genome.use_icl_indices and best_overall_ind.genome.icl_indices:
            icl_examples = select_icl_examples(
                train_pool,
                strategy="random",
                k=best_overall_ind.genome.k_shots,
                seed=sid,
                indices=best_overall_ind.genome.icl_indices,
            )
        else:
            icl_examples = select_icl_examples(
                train_pool,
                strategy=best_overall_ind.genome.icl_strategy,
                k=best_overall_ind.genome.k_shots,
                seed=sid,
            )

        full_prompt = build_full_prompt(
            genome=best_overall_ind.genome,
            essay=essay,
            icl_examples=icl_examples,
            rag_examples=[],
            summary_text=None,
        )

        if len(full_prompt) > MAX_CONTEXT_CHARS:
            head = full_prompt[: int(MAX_CONTEXT_CHARS * 0.75)]
            tail = full_prompt[-int(MAX_CONTEXT_CHARS * 0.20):]
            full_prompt = head + "\n\n[Truncated]\n\n" + tail

        cache_key = full_prompt
        reply = _LLM_CACHE.get(cache_key)
        if reply is None:
            throttle()
            reply = call_scoring_llm(
                full_prompt,
                temperature=0.0,
                model=SINGLE_MODEL,
                max_tokens=8,
                max_retries=3,
                timeout=90,
            )
            _LLM_CACHE[cache_key] = reply

        raw_band = parse_band_from_text(reply or "", default=5.0)
        pred_band = round(raw_band * 2) / 2.0

        preds.append(pred_band)
        raws.append(raw_band)
        labels.append(true_band)
        essays_out.append(essay[:2000])

        print(
            f"  Essay {i}/{len(final_pool)} | "
            f"True={true_band:.1f}, Raw={raw_band:.1f}, Pred={pred_band:.1f}"
        )

    final_metrics = compute_metrics(labels, preds)
    print(
        f"\nFinal Evaluation → "
        f"QWK={final_metrics['qwk']:.4f}, "
        f"Pearson={final_metrics['pearson']:.4f}, "
        f"RMSE={final_metrics['rmse']:.4f}, "
        f"Exact={final_metrics['exact_acc']:.3f}, "
        f"Adj={final_metrics['adj_acc']:.3f}"
    )

    df_out = pd.DataFrame(
        {
            "essay": essays_out,
            "true_band": labels,
            "raw_band": raws,
            "pred_band": preds,
        }
    )
    df_out.to_csv(BEST_PRED_CSV, index=False, encoding="utf-8-sig")
    print(f"📄 Predictions saved: {BEST_PRED_CSV}")
    print("🎉 Done!")



if __name__ == "__main__":
    run_evolution_hf_icl_only()
