# evolver/prompt_evolver.py  (Data-Aware + LLM feedback + template pool persistence + LLM stats)
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple
import random
import os
import re
import json
import time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from evolver.data_aware_prompt import PromptGenome, INSTRUCTION_TEMPLATES
from llm_api.openrouter_api import call_scoring_llm

# ======= Individual =======

@dataclass
class Individual:
    genome: PromptGenome
    fitness: Optional[float] = None
    metrics: Optional[Dict[str, float]] = None
    preds: Optional[List[float]] = None
    labels: Optional[List[float]] = None


# ======= Gene spaces =======
INSTRUCTION_IDS = sorted(list(INSTRUCTION_TEMPLATES.keys()))
STRICTNESS_LEVELS = [0, 1]
OUTPUT_FORMATS = ["scalar"]

# 🔥 ICL 模式开关
USE_ICL_INDICES_MODE = os.getenv("USE_ICL_INDICES_MODE", "0") == "1"  # 默认关闭，使用旧策略

# 旧策略参数
ICL_STRATEGIES = [
    "random",
    "score_balanced",
    "near_anchor",
    "quantile_stratified",
    "extreme_balanced",
]

K_SHOTS = [6, 8, 10, 12, 16]

# 新索引模式参数
ICL_K_SHOTS_FIXED = int(os.getenv("ICL_K_SHOTS_FIXED", "8"))  # 固定的 k_shots 数量

RAG_STRATEGIES = ["none"]
USE_SUMMARY = [False]
USE_TEACHER = [False]
TEACHER_WEIGHT = [0.0]

# ======= LLM text-mutation switches =======
USE_LLM_TEXT_MUTATION = os.getenv("USE_LLM_TEXT_MUTATION", "1") == "1"
LLM_TEXT_MUTATION_PROB = float(os.getenv("LLM_TEXT_MUTATION_PROB", "0.8"))
LLM_TEXT_MUTATION_MODEL = os.getenv(
    "LLM_TEXT_MUTATION_MODEL",
    os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
)
LLM_TEXT_MAX_CHARS = int(os.getenv("LLM_TEXT_MAX_CHARS", "900"))
LLM_TEXT_MAX_TRIES = int(os.getenv("LLM_TEXT_MAX_TRIES", "2"))

# ======= LLM ICL-mutation switches =======
USE_LLM_ICL_MUTATION = os.getenv("USE_LLM_ICL_MUTATION", "1") == "1"
LLM_ICL_MUTATION_PROB = float(os.getenv("LLM_ICL_MUTATION_PROB", "0.7"))
LLM_ICL_MUTATION_MODEL = os.getenv(
    "LLM_ICL_MUTATION_MODEL",
    os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
)
LLM_ICL_MAX_TRIES = int(os.getenv("LLM_ICL_MAX_TRIES", "2"))

# ======= Feedback from last generation best =======
# alphaevolve_multi 会在每代结束后 set_llm_feedback(...)
_LLM_FEEDBACK: Dict[str, Any] = {}

def set_llm_feedback(best_instruction: str, bias_stats: Dict[str, Any], best_metrics: Dict[str, float], gen: int):
    """供主流程调用：把当前代最好模板+偏差统计喂给 LLM mutation 用。"""
    global _LLM_FEEDBACK
    _LLM_FEEDBACK = {
        "gen": gen,
        "best_instruction": best_instruction,
        "bias_stats": bias_stats,
        "best_metrics": best_metrics,
        "ts": time.time(),
    }

# ======= Template pool persistence =======

@dataclass
class TemplateRecord:
    text: str
    fitness: float
    metrics: Dict[str, float]
    gen: int
    ts: float

_TEMPLATE_POOL: List[TemplateRecord] = []
_TEMPLATE_POOL_PATH: Optional[Path] = None

def load_template_pool(path: str | Path, max_keep: int = 60) -> List[TemplateRecord]:
    """从 json 加载模板池。"""
    global _TEMPLATE_POOL, _TEMPLATE_POOL_PATH
    _TEMPLATE_POOL_PATH = Path(path)
    if not _TEMPLATE_POOL_PATH.exists():
        _TEMPLATE_POOL = []
        return _TEMPLATE_POOL

    try:
        data = json.loads(_TEMPLATE_POOL_PATH.read_text(encoding="utf-8"))
        recs = []
        for d in data.get("templates", []):
            text = str(d.get("text", "")).strip()
            text = _clean_instruction_text(text)  # ✅ FIX: 载入时也清理尾巴
            if not text:
                continue
            recs.append(
                TemplateRecord(
                    text=text,
                    fitness=float(d.get("fitness", -1e9)),
                    metrics=dict(d.get("metrics", {})),
                    gen=int(d.get("gen", -1)),
                    ts=float(d.get("ts", 0.0)),
                )
            )
        recs.sort(key=lambda r: r.fitness, reverse=True)
        _TEMPLATE_POOL = recs[:max_keep]
    except Exception:
        _TEMPLATE_POOL = []

    return _TEMPLATE_POOL


def save_template_pool(path: str | Path):
    """保存模板池到 json。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "templates": [asdict(r) for r in _TEMPLATE_POOL],
        "updated_at": time.time(),
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def update_template_pool(text: str, fitness: float, metrics: Dict[str, float], gen: int, max_keep: int = 60):
    """加入一个新好模板，去重 + 只保留 topK。"""
    global _TEMPLATE_POOL
    text = _clean_instruction_text(text)  # ✅ FIX: 入池前先清理尾巴
    if not text:
        return

    # 去重：同样文本只留 fitness 更高的
    for i, r in enumerate(_TEMPLATE_POOL):
        if r.text == text:
            if fitness > r.fitness:
                _TEMPLATE_POOL[i] = TemplateRecord(
                    text=text, fitness=fitness, metrics=metrics, gen=gen, ts=time.time()
                )
            break
    else:
        _TEMPLATE_POOL.append(
            TemplateRecord(text=text, fitness=fitness, metrics=metrics, gen=gen, ts=time.time())
        )

    _TEMPLATE_POOL.sort(key=lambda r: r.fitness, reverse=True)
    _TEMPLATE_POOL = _TEMPLATE_POOL[:max_keep]

    if _TEMPLATE_POOL_PATH is not None:
        save_template_pool(_TEMPLATE_POOL_PATH)


def get_template_pool_texts(topk: int = 8) -> List[str]:
    return [r.text for r in _TEMPLATE_POOL[:topk]]


# ======= LLM mutation stats =======
# 用于每代打印 LLM 变异成功率/贡献度
_LLM_STATS: Dict[str, Dict[str, int]] = {
    "text": {
        "triggered": 0,
        "tries": 0,
        "success": 0,
        "fail_empty": 0,
        "fail_short": 0,
        "fail_similar": 0,
        "fail_exception": 0,
    },
    "icl": {
        "triggered": 0,
        "tries": 0,
        "success": 0,
        "fail_empty": 0,
        "fail_parse": 0,
        "fail_invalid": 0,
        "fail_exception": 0,
    },
}

def reset_llm_stats() -> None:
    """每代开始前调用，清空统计。"""
    for k in _LLM_STATS:
        for kk in _LLM_STATS[k]:
            _LLM_STATS[k][kk] = 0

def get_llm_stats() -> Dict[str, Any]:
    """每代结束后调用，获取统计 + 成功率。"""
    out = {}
    for name, d in _LLM_STATS.items():
        trig = d["triggered"]
        succ = d["success"]
        tries = d["tries"]
        out[name] = {
            **d,
            "success_rate_per_trigger": (succ / trig) if trig else 0.0,
            "success_rate_per_try": (succ / tries) if tries else 0.0,
        }
    return out

def _stat_inc(group: str, key: str, n: int = 1) -> None:
    if group in _LLM_STATS and key in _LLM_STATS[group]:
        _LLM_STATS[group][key] += n


# ======= few-shot style guard =======
_TEXT_GUARD = (
    "Write a concise instruction for IELTS Writing Task 2 scoring. "
    "It must be suitable for a real examiner, mention official criteria, "
    "avoid long explanations, and NOT include any examples or scores. "
    "No bullet lists. 1 paragraph only."
)


def random_genome(rng: random.Random, instruction_text: Optional[str] = None, train_pool_size: int = 1000) -> PromptGenome:
    """
    生成随机基因组。
    
    Args:
        rng: 随机数生成器
        instruction_text: 可选的指令文本
        train_pool_size: 训练集大小，用于生成随机索引
    """
    # 🔥 根据模式选择 ICL 参数
    if USE_ICL_INDICES_MODE:
        # 新模式：生成随机索引列表
        k = ICL_K_SHOTS_FIXED
        indices = tuple(rng.sample(range(train_pool_size), min(k, train_pool_size)))
        return PromptGenome(
            instruction_id=rng.choice(INSTRUCTION_IDS),
            instruction_text=instruction_text,
            strictness=rng.choice(STRICTNESS_LEVELS),
            output_format=rng.choice(OUTPUT_FORMATS),
            use_icl_indices=True,
            icl_strategy="random",  # 保留字段但不使用
            k_shots=k,
            icl_indices=indices,
            rag_strategy=rng.choice(RAG_STRATEGIES),
            use_summary=rng.choice(USE_SUMMARY),
            use_teacher=rng.choice(USE_TEACHER),
            teacher_weight=rng.choice(TEACHER_WEIGHT),
        )
    else:
        # 旧模式：使用策略
        return PromptGenome(
            instruction_id=rng.choice(INSTRUCTION_IDS),
            instruction_text=instruction_text,
            strictness=rng.choice(STRICTNESS_LEVELS),
            output_format=rng.choice(OUTPUT_FORMATS),
            use_icl_indices=False,
            icl_strategy=rng.choice(ICL_STRATEGIES),
            k_shots=rng.choice(K_SHOTS),
            icl_indices=None,
            rag_strategy=rng.choice(RAG_STRATEGIES),
            use_summary=rng.choice(USE_SUMMARY),
            use_teacher=rng.choice(USE_TEACHER),
            teacher_weight=rng.choice(TEACHER_WEIGHT),
        )


def build_initial_population(pop_size: int, seed: int = 42, train_pool_size: int = 1000) -> List[Individual]:
    """
    ✅ 初始种群模板池占比降低到 25%，给 LLM 留更多进化空间
    
    Args:
        pop_size: 种群大小
        seed: 随机种子
        train_pool_size: 训练集大小（用于索引模式）
    """
    rng = random.Random(seed)
    pop: List[Individual] = []

    pool_n = max(1, int(pop_size * 0.25))
    pool_texts = get_template_pool_texts(topk=pool_n)
    for t in pool_texts:
        pop.append(Individual(genome=random_genome(rng, instruction_text=t, train_pool_size=train_pool_size)))

    while len(pop) < pop_size:
        pop.append(Individual(genome=random_genome(rng, train_pool_size=train_pool_size)))

    return pop


def tournament_selection(
    population: List[Individual], k: int, num_winners: int
) -> List[Individual]:
    winners: List[Individual] = []
    pool = [p for p in population if p.fitness is not None] or population

    for _ in range(num_winners):
        group = random.sample(pool, k=min(k, len(pool)))
        best = max(group, key=lambda x: x.fitness if x.fitness is not None else float("-inf"))
        winners.append(best)

    return winners


def crossover_genome(g1: PromptGenome, g2: PromptGenome, rng: random.Random) -> PromptGenome:
    """
    ✅ 让包含 LLM 产物的 instruction_text 更容易保留下来
    🔥 支持索引列表的交叉（单点交叉或均匀交叉）
    """
    instruction_text = rng.choice([g1.instruction_text, g2.instruction_text])

    if g1.instruction_text and not g2.instruction_text and rng.random() < 0.7:
        instruction_text = g1.instruction_text
    if g2.instruction_text and not g1.instruction_text and rng.random() < 0.7:
        instruction_text = g2.instruction_text

    instruction_text = _clean_instruction_text(instruction_text or "") or instruction_text  # ✅ FIX

    # 🔥 ICL 索引交叉
    icl_indices = None
    if USE_ICL_INDICES_MODE and g1.icl_indices and g2.icl_indices:
        # 均匀交叉：从两个父代随机选择索引
        len1, len2 = len(g1.icl_indices), len(g2.icl_indices)
        max_len = max(len1, len2)
        child_indices = []
        for i in range(max_len):
            if i < len1 and i < len2:
                child_indices.append(rng.choice([g1.icl_indices[i], g2.icl_indices[i]]))
            elif i < len1:
                child_indices.append(g1.icl_indices[i])
            else:
                child_indices.append(g2.icl_indices[i])
        icl_indices = tuple(child_indices)
    elif g1.icl_indices:
        icl_indices = g1.icl_indices
    elif g2.icl_indices:
        icl_indices = g2.icl_indices

    return PromptGenome(
        instruction_id=rng.choice([g1.instruction_id, g2.instruction_id]),
        instruction_text=instruction_text,
        strictness=rng.choice([g1.strictness, g2.strictness]),
        output_format=rng.choice([g1.output_format, g2.output_format]),
        use_icl_indices=g1.use_icl_indices or g2.use_icl_indices,
        icl_strategy=rng.choice([g1.icl_strategy, g2.icl_strategy]),
        k_shots=rng.choice([g1.k_shots, g2.k_shots]),
        icl_indices=icl_indices,
        rag_strategy=rng.choice([g1.rag_strategy, g2.rag_strategy]),
        use_summary=rng.choice([g1.use_summary, g2.use_summary]),
        use_teacher=rng.choice([g1.use_teacher, g2.use_teacher]),
        teacher_weight=rng.choice([g1.teacher_weight, g2.teacher_weight]),
    )


# ================== LLM text mutation helpers ================== #

def _clean_llm_tail_scores(t: str) -> str:
    """
    ✅ FIX: 清理 LLM 产物末尾“孤立 band 分数”或被截断的分数字尾巴。
    只处理结尾，避免误伤正文中的数字。
    """
    if not t:
        return t
    t = t.strip()

    # a) 如果最后一行只有一个 band 数字（7.0/6.5/8），删掉
    lines = [ln.rstrip() for ln in t.splitlines()]
    while lines:
        last = lines[-1].strip()
        if re.fullmatch(r"[0-9](?:\.5)?", last):
            lines.pop()
            continue
        break
    t = "\n".join(lines).strip()

    # b) 如果末尾是 “... 7.0” 这种尾巴，切掉尾部 band
    t = re.sub(
        r"(\s|[。.!?;:,，；：])([0-9](?:\.5)?)\s*$",
        r"\1",
        t
    ).strip()

    return t


def _clean_instruction_text(t: str) -> str:
    t = (t or "").strip()
    t = re.sub(r"^```.*?\n|\n```$", "", t, flags=re.S).strip()
    t = t.strip('"').strip("'").strip()
    t = re.sub(r"\s+", " ", t).strip()

    # ✅ FIX: 统一清理末尾 band 尾巴
    t = _clean_llm_tail_scores(t)

    if len(t) > LLM_TEXT_MAX_CHARS:
        t = t[:LLM_TEXT_MAX_CHARS].rstrip()

    # 再保险一次（截断后可能又形成尾巴）
    t = _clean_llm_tail_scores(t)

    return t


def _rough_similarity(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _format_bias_stats(bias: Dict[str, Any]) -> str:
    if not bias:
        return "No bias stats."
    lines = []
    if "mean_err" in bias:
        lines.append(f"- overall mean error (pred-true): {bias['mean_err']:+.2f}")
    if "mae" in bias:
        lines.append(f"- overall MAE: {bias['mae']:.2f}")
    if "over_rate" in bias:
        lines.append(f"- over-estimation rate: {bias['over_rate']:.1%}")
    if "under_rate" in bias:
        lines.append(f"- under-estimation rate: {bias['under_rate']:.1%}")

    by_band = bias.get("by_band", {})
    if by_band:
        show_keys = sorted(by_band.keys())[:8]
        lines.append("- per-band mean error:")
        for k in show_keys:
            v = by_band[k]
            lines.append(
                f"  * band {k/2:.1f}: n={v['n']}, mean_err={v['mean_err']:+.2f}, mae={v['mae']:.2f}"
            )
    return "\n".join(lines)


# ================== LLM ICL mutation ================== #

def _parse_json_loose(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```json|^```|```$", "", text, flags=re.I | re.M).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def llm_choose_icl(current: PromptGenome) -> Optional[Tuple[str, int]]:
    prompt = (
        "You are optimizing few-shot prompting for IELTS Task 2 automatic scoring.\n"
        f"Available icl_strategy options: {ICL_STRATEGIES}\n"
        f"Available k_shots options: {K_SHOTS}\n\n"
        "Current genome summary:\n"
        f"- instruction_id={current.instruction_id}\n"
        f"- strictness={current.strictness}\n"
        f"- icl_strategy={current.icl_strategy}\n"
        f"- k_shots={current.k_shots}\n\n"
        "Choose a NEW setting (icl_strategy, k_shots) that is likely to reduce "
        "central-tendency bias and improve extreme-band accuracy.\n"
        "Return ONLY a JSON object like:\n"
        '{"icl_strategy":"extreme_balanced","k_shots":12}\n'
        "Do not add any other text."
    )

    for _ in range(LLM_ICL_MAX_TRIES):
        _stat_inc("icl", "tries", 1)
        try:
            reply = call_scoring_llm(
                prompt,
                temperature=0.4,
                model=LLM_ICL_MUTATION_MODEL,
                max_tokens=80,
                max_retries=2,
                timeout=60,
            )
            if not reply:
                _stat_inc("icl", "fail_empty", 1)
                continue

            d = _parse_json_loose(reply)
            if not d:
                _stat_inc("icl", "fail_parse", 1)
                continue

            s = str(d.get("icl_strategy", "")).strip()
            k = d.get("k_shots", None)

            if s not in ICL_STRATEGIES:
                _stat_inc("icl", "fail_invalid", 1)
                continue
            try:
                k = int(k)
            except Exception:
                _stat_inc("icl", "fail_invalid", 1)
                continue
            if k not in K_SHOTS:
                _stat_inc("icl", "fail_invalid", 1)
                continue

            _stat_inc("icl", "success", 1)
            return (s, k)
        except Exception:
            _stat_inc("icl", "fail_exception", 1)
            continue

    return None


# ================== LLM text mutation ================== #

def llm_generate_new_instruction(current: PromptGenome) -> Optional[str]:
    base_text = (
        current.instruction_text.strip()
        if current.instruction_text and str(current.instruction_text).strip()
        else INSTRUCTION_TEMPLATES.get(current.instruction_id, INSTRUCTION_TEMPLATES[0]).strip()
    )
    base_clean = _clean_instruction_text(base_text)

    feedback_block = ""
    if _LLM_FEEDBACK.get("best_instruction"):
        fb_best = _LLM_FEEDBACK["best_instruction"]
        fb_bias = _LLM_FEEDBACK.get("bias_stats", {})
        fb_metrics = _LLM_FEEDBACK.get("best_metrics", {})
        feedback_block = (
            "\n\n=== Feedback from last generation best template ===\n"
            f"Best template (gen {_LLM_FEEDBACK.get('gen', '?')}):\n{fb_best}\n\n"
            "Bias stats on eval:\n"
            f"{_format_bias_stats(fb_bias)}\n\n"
            f"Best eval metrics: QWK={fb_metrics.get('qwk',0):.3f}, "
            f"Pearson={fb_metrics.get('pearson',0):.3f}, RMSE={fb_metrics.get('rmse',0):.3f}\n"
            "Please rewrite to reduce the bias patterns above."
        )

    prompt = (
        "You are evolving a prompt for automatic IELTS Task 2 scoring.\n"
        f"{_TEXT_GUARD}\n\n"
        "Here is the current instruction:\n"
        f"{base_text}\n"
        f"{feedback_block}\n\n"
        "Rewrite it into a NEW improved variant that may yield more consistent overall band scores.\n"
        "Constraints:\n"
        "- Keep it as ONE paragraph, no bullets.\n"
        "- You MUST change the structure and phrasing substantially (not just synonyms).\n"
        "- Include explicit weighting/priorities among TR, CC, LR, GRA.\n"
        "- Add a short rule about avoiding central-tendency bias (do not overuse 6.0/6.5).\n"
        "- Do NOT output any band score numbers at the end or anywhere (e.g., '7.0', '6.5').\n"
        "Return ONLY the rewritten instruction text."
    )

    for _ in range(LLM_TEXT_MAX_TRIES):
        _stat_inc("text", "tries", 1)
        try:
            reply = call_scoring_llm(
                prompt,
                temperature=0.6,
                model=LLM_TEXT_MUTATION_MODEL,
                max_tokens=220,   # 这里不用改，已足够；尾巴由清理函数兜底
                max_retries=2,
                timeout=60,
            )
            if not reply:
                _stat_inc("text", "fail_empty", 1)
                continue

            new_text = _clean_instruction_text(reply)  # ✅ FIX: 会自动清理尾巴

            if len(new_text) < 40:
                _stat_inc("text", "fail_short", 1)
                continue

            if _rough_similarity(new_text, base_clean) > 0.75:
                _stat_inc("text", "fail_similar", 1)
                continue

            _stat_inc("text", "success", 1)
            return new_text
        except Exception:
            _stat_inc("text", "fail_exception", 1)
            continue

    return None


# ================== mutation ================== #

def mutate_genome(g: PromptGenome, mutation_rate: float, rng: random.Random, train_pool_size: int = 1000) -> PromptGenome:
    """
    变异基因组。
    
    🔥 新增：支持索引列表的变异
    - 替换变异：随机替换某个索引
    - 插入变异：插入新索引
    - 删除变异：删除某个索引
    """
    new_instruction_text = g.instruction_text
    new_instruction_id = g.instruction_id

    # ---- 1) text-level mutation ----
    if USE_LLM_TEXT_MUTATION and rng.random() < LLM_TEXT_MUTATION_PROB:
        _stat_inc("text", "triggered", 1)
        t = llm_generate_new_instruction(g)
        if t is not None:
            new_instruction_text = _clean_instruction_text(t)  # ✅ FIX

    # ---- 2) ICL-level mutation ----
    new_icl_strategy = g.icl_strategy
    new_k_shots = g.k_shots
    new_icl_indices = g.icl_indices
    llm_icl_used = False

    if USE_ICL_INDICES_MODE:
        # 🔥 新模式：变异索引列表
        if g.icl_indices and rng.random() < mutation_rate:
            indices_list = list(g.icl_indices)
            
            # 三种变异操作，随机选择
            mutation_type = rng.choice(["replace", "insert", "delete"])
            
            if mutation_type == "replace" and indices_list:
                # 替换：随机选一个位置，替换为新索引
                pos = rng.randint(0, len(indices_list) - 1)
                new_idx = rng.randint(0, train_pool_size - 1)
                indices_list[pos] = new_idx
                
            elif mutation_type == "insert" and len(indices_list) < ICL_K_SHOTS_FIXED * 2:
                # 插入：添加新索引（限制最大长度）
                new_idx = rng.randint(0, train_pool_size - 1)
                pos = rng.randint(0, len(indices_list))
                indices_list.insert(pos, new_idx)
                
            elif mutation_type == "delete" and len(indices_list) > 1:
                # 删除：移除一个索引（保证至少1个）
                pos = rng.randint(0, len(indices_list) - 1)
                indices_list.pop(pos)
            
            new_icl_indices = tuple(indices_list)
    else:
        # 🔥 旧模式：策略驱动的 ICL 变异
        if USE_LLM_ICL_MUTATION and rng.random() < LLM_ICL_MUTATION_PROB:
            _stat_inc("icl", "triggered", 1)
            sel = llm_choose_icl(g)
            if sel is not None:
                new_icl_strategy, new_k_shots = sel
                llm_icl_used = True

    # ---- 3) discrete fallback ----
    def maybe(v, space):
        return rng.choice(space) if rng.random() < mutation_rate else v

    if not USE_ICL_INDICES_MODE and not llm_icl_used:
        new_icl_strategy = maybe(new_icl_strategy, ICL_STRATEGIES)
        new_k_shots = maybe(new_k_shots, K_SHOTS)

    return PromptGenome(
        instruction_id=maybe(new_instruction_id, INSTRUCTION_IDS),
        instruction_text=_clean_instruction_text(new_instruction_text or "") or new_instruction_text,  # ✅ FIX
        strictness=maybe(g.strictness, STRICTNESS_LEVELS),
        output_format=g.output_format,
        use_icl_indices=g.use_icl_indices,
        icl_strategy=new_icl_strategy,
        k_shots=new_k_shots,
        icl_indices=new_icl_indices,
        rag_strategy=maybe(g.rag_strategy, RAG_STRATEGIES),
        use_summary=maybe(g.use_summary, USE_SUMMARY),
        use_teacher=maybe(g.use_teacher, USE_TEACHER),
        teacher_weight=maybe(g.teacher_weight, TEACHER_WEIGHT),
    )


if __name__ == "__main__":

    print(
        "[ENV CHECK]",
        "USE_LLM_TEXT_MUTATION=", USE_LLM_TEXT_MUTATION,
        "PROB=", LLM_TEXT_MUTATION_PROB,
        "USE_LLM_ICL_MUTATION=", USE_LLM_ICL_MUTATION,
        "PROB=", LLM_ICL_MUTATION_PROB,
    )
