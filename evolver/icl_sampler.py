# evolver/icl_sampler.py
"""
ICL sampler for Data-Aware AlphaEvolve (HF version).

train_pool item format (HF):
    (sid, essay, band, prompt)
or dict with keys:
    {"sid","essay","band","prompt"}

Strategies:
- random
- score_balanced          (half-band balanced)
- near_anchor
- quantile_stratified     (quantile buckets)
- extreme_balanced        (force low/high exposure)
"""

from __future__ import annotations
from typing import List, Dict, Any, Sequence, Optional
import random
import math
import numpy as np


def _safe_float(x, default=None):
    try:
        return float(str(x).strip())
    except Exception:
        return default


def _to_dict(sample: Any) -> Dict[str, Any]:
    """Normalize one sample into dict: {sid, prompt, essay, band}."""
    if isinstance(sample, dict):
        sid = sample.get("sid", sample.get("id", 0))
        essay = str(sample.get("essay", ""))
        prompt = str(sample.get("prompt", ""))
        band = _safe_float(sample.get("band"), default=5.0)
        return {"sid": sid, "prompt": prompt, "essay": essay, "band": band}

    if isinstance(sample, (list, tuple)):
        # Try HF format first: (sid, essay, band, prompt)
        if len(sample) >= 4 and _safe_float(sample[2]) is not None:
            sid = sample[0]
            essay = str(sample[1])
            band = float(sample[2])
            prompt = str(sample[3])
            return {"sid": sid, "prompt": prompt, "essay": essay, "band": band}

        # Fallback Kaggle-like: (sid, essay, band, task_type)
        if len(sample) >= 3 and _safe_float(sample[2]) is not None:
            sid = sample[0]
            essay = str(sample[1])
            band = float(sample[2])
            prompt = ""
            return {"sid": sid, "prompt": prompt, "essay": essay, "band": band}

    return {"sid": 0, "prompt": "", "essay": str(sample), "band": 5.0}


def _half_band(b: float) -> float:
    """Round to nearest 0.5 band (stable bucket key)."""
    return round(float(b) * 2) / 2.0


def _band_bins(pool: List[Dict[str, Any]]) -> Dict[float, List[Dict[str, Any]]]:
    """Bucket by half-band instead of raw float."""
    bins: Dict[float, List[Dict[str, Any]]] = {}
    for s in pool:
        key = _half_band(s["band"])
        bins.setdefault(key, []).append(s)
    return bins


def _safe_sample(rng: random.Random, arr: List[Any], n: int) -> List[Any]:
    n = min(n, len(arr))
    return rng.sample(arr, n) if n > 0 else []


def select_icl_examples(
    train_pool: Sequence[Any],
    strategy: str = "random",
    k: int = 4,
    seed: int = 42,
    anchor_band: Optional[float] = None,
    indices: Optional[Sequence[int]] = None,  # 🔥 新增：直接指定索引列表
) -> List[Dict[str, Any]]:
    """
    Select k ICL examples from train_pool according to strategy or indices.
    
    If indices is provided, use them directly (ignoring strategy).
    Otherwise, use the strategy-based selection.
    
    Return list of dict samples.
    """
    if k <= 0:
        return []

    rng = random.Random(seed)
    pool = [_to_dict(s) for s in train_pool]
    if not pool:
        return []

    # 🔥 模式1：使用索引列表（优先）
    if indices is not None:
        chosen = []
        for idx in indices:
            if 0 <= idx < len(pool):
                chosen.append(pool[idx])
            if len(chosen) >= k:
                break
        return chosen[:k]

    # 🔥 模式2：使用策略（原有逻辑）
    k = min(k, len(pool))

    # -------- random --------
    if strategy == "random":
        return rng.sample(pool, k)

    # -------- score_balanced (half-band round-robin) --------
    if strategy == "score_balanced":
        bins = _band_bins(pool)
        bands = sorted(bins.keys())
        if not bands:
            return rng.sample(pool, k)

        # shuffle within bin for randomness
        for b in bands:
            rng.shuffle(bins[b])

        chosen: List[Dict[str, Any]] = []
        ptr = 0
        while len(chosen) < k:
            b = bands[ptr % len(bands)]
            if bins[b]:
                chosen.append(bins[b].pop())
            ptr += 1
            if ptr > k * 5:  # safety
                break

        if len(chosen) < k:
            rest = [s for s in pool if s not in chosen]
            chosen += rng.sample(rest, k - len(chosen))
        return chosen[:k]

    # -------- near_anchor --------
    if strategy == "near_anchor":
        if anchor_band is None:
            anchor_band = pool[seed % len(pool)]["band"]

        pool_sorted = sorted(
            pool,
            key=lambda s: (abs(s["band"] - anchor_band), rng.random())
        )
        return pool_sorted[:k]

    # -------- quantile_stratified --------
    if strategy == "quantile_stratified":
        bands = np.array([float(s["band"]) for s in pool], dtype=float)
        uniq_n = len(np.unique(bands))
        q = min(5, uniq_n)
        if q < 2:
            return rng.sample(pool, k)

        edges = np.quantile(bands, np.linspace(0, 1, q + 1))
        buckets: List[List[Dict[str, Any]]] = [[] for _ in range(q)]

        for s in pool:
            b = float(s["band"])
            idx = int(np.searchsorted(edges, b, side="right") - 1)
            idx = max(0, min(q - 1, idx))
            buckets[idx].append(s)

        for buc in buckets:
            rng.shuffle(buc)

        chosen: List[Dict[str, Any]] = []
        ptr = 0
        while len(chosen) < k and any(buckets):
            buc = buckets[ptr % q]
            if buc:
                chosen.append(buc.pop())
            ptr += 1
            if ptr > k * 5:
                break

        if len(chosen) < k:
            rest = [s for s in pool if s not in chosen]
            chosen += rng.sample(rest, k - len(chosen))
        rng.shuffle(chosen)
        return chosen[:k]

    # -------- extreme_balanced --------
    if strategy == "extreme_balanced":
        low = [s for s in pool if s["band"] <= 4.5]
        high = [s for s in pool if s["band"] >= 7.0]
        mid = [s for s in pool if 4.5 < s["band"] < 7.0]

        n_low = k // 3
        n_high = k // 3
        n_mid = k - n_low - n_high

        chosen: List[Dict[str, Any]] = []
        chosen += _safe_sample(rng, low, n_low)
        chosen += _safe_sample(rng, high, n_high)
        chosen += _safe_sample(rng, mid, n_mid)

        if len(chosen) < k:
            rest = [s for s in pool if s not in chosen]
            chosen += rng.sample(rest, k - len(chosen))

        rng.shuffle(chosen)
        return chosen[:k]

    # unknown -> fallback random
    return rng.sample(pool, k)
