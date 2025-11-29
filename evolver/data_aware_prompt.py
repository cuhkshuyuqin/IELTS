# evolver/data_aware_prompt.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

# ========= Genome =========

@dataclass(frozen=True)
class PromptGenome:
    instruction_id: int = 0
    instruction_text: Optional[str] = None  # ✅ LLM 生成的新模板文本（优先级最高）
    strictness: int = 1
    output_format: str = "scalar"

    # ===== ICL 策略：两种模式 =====
    use_icl_indices: bool = False  # 🔥 开关：True=使用索引列表，False=使用策略
    
    # 模式1：策略驱动（旧方式）
    icl_strategy: str = "random"
    k_shots: int = 0
    
    # 模式2：索引驱动（新方式）
    icl_indices: Optional[tuple] = None  # 🔥 训练集索引列表，如 (12, 45, 78, ...)
    
    # ===== 其他 =====
    rag_strategy: str = "none"
    use_summary: bool = False

    use_teacher: bool = False
    teacher_weight: float = 0.0


# ========= Instruction templates =========
# 静态模板池：instruction_text 为空时才用它
INSTRUCTION_TEMPLATES: Dict[int, str] = {
    0: (
        "You are an IELTS Writing Task 2 examiner. "
        "Assess the essay using the official band descriptors. "
    ),
    1: (
        "You are an IELTS Writing Task 2 examiner. "
        "Evaluate Task Response, Coherence and Cohesion, Lexical Resource, "
        "and Grammatical Range and Accuracy, then decide an overall band. "
    ),
}


STRICTNESS_CLAUSES: Dict[int, str] = {
    0: "Be fair and neutral in scoring.",
    1: "Be strict but fair, avoid band inflation.",
}

OUTPUT_SCALAR_CLAUSE = (
    "Output ONLY the final overall band score as a single number "
    "from 0 to 9 in 0.5 steps (e.g., 6 or 6.5). "
    "Do NOT output any explanation, text, or symbols."
)

UNDERLEN_PENALTY = "If the essay is clearly under 250 words, lower the score by at least 0.5."


def _format_example(ex: Dict[str, Any], max_len: int = 1200) -> str:
    """Few-shot example formatter."""
    p = str(ex.get("prompt", "")).strip()
    e = str(ex.get("essay", "")).strip()
    b = float(ex.get("band", 5.0))

    if len(e) > max_len:
        e = e[:max_len] + " ..."

    return (
        "=== Example ===\n"
        f"Prompt:\n{p}\n\n"
        f"Essay:\n{e}\n\n"
        f"Score: {b:.1f}\n"
    )


def build_full_prompt(
    genome: PromptGenome,
    essay: str,
    icl_examples: Optional[List[Dict[str, Any]]] = None,
    rag_examples: Optional[List[Dict[str, Any]]] = None,
    summary_text: Optional[str] = None,
) -> str:
    """
    Build structured prompt:
    [instruction + strictness + output-format]
    + few-shot examples (ICL)
    + RAG examples (stub)
    + summary (stub)
    + target essay
    """
    # ✅ instruction_text 优先
    if genome.instruction_text and str(genome.instruction_text).strip():
        instruction = str(genome.instruction_text).strip()
    else:
        instruction = INSTRUCTION_TEMPLATES.get(genome.instruction_id, INSTRUCTION_TEMPLATES[0])

    strictness = STRICTNESS_CLAUSES.get(genome.strictness, STRICTNESS_CLAUSES[1])

    parts: List[str] = []
    parts.append(instruction + " " + strictness + " " + UNDERLEN_PENALTY)
    parts.append(OUTPUT_SCALAR_CLAUSE)

    if icl_examples:
        for ex in icl_examples:
            parts.append(_format_example(ex))

    if rag_examples:
        for ex in rag_examples:
            parts.append(_format_example(ex))

    if summary_text:
        parts.append("=== Summary of the essay ===\n" + summary_text.strip())

    parts.append("=== Essay to score ===\n" + (essay or "").strip())
    parts.append("\nFinal overall band score:")

    return "\n\n".join(parts).strip()
