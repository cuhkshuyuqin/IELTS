# evolver/checkpoint.py
"""
断点续传机制：保存和恢复进化状态
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import asdict

from evolver.prompt_evolver import Individual
from evolver.data_aware_prompt import PromptGenome


def save_checkpoint(
    checkpoint_path: Path,
    generation: int,
    population: List[Individual],
    best_overall_ind: Optional[Individual],
    best_overall_fitness: float,
    history_qwk: List[float],
    history_pearson: List[float],
    history_rmse: List[float],
    history_llm_stats: List[Dict[str, Any]],
    llm_cache: Dict[str, Optional[str]],
) -> None:
    """
    保存当前进化状态到检查点文件。
    
    Args:
        checkpoint_path: 检查点文件路径
        generation: 当前代数
        population: 当前种群
        best_overall_ind: 全局最优个体
        best_overall_fitness: 全局最优适应度
        history_*: 历史记录
        llm_cache: LLM 调用缓存
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 序列化种群
    population_data = []
    for ind in population:
        ind_data = {
            "genome": asdict(ind.genome),
            "fitness": ind.fitness,
            "metrics": ind.metrics,
            "preds": ind.preds,
            "labels": ind.labels,
        }
        population_data.append(ind_data)
    
    # 序列化全局最优
    best_data = None
    if best_overall_ind is not None:
        best_data = {
            "genome": asdict(best_overall_ind.genome),
            "fitness": best_overall_ind.fitness,
            "metrics": best_overall_ind.metrics,
            "preds": best_overall_ind.preds,
            "labels": best_overall_ind.labels,
        }
    
    # 构建检查点数据
    checkpoint = {
        "version": "1.0",
        "timestamp": time.time(),
        "generation": generation,
        "population": population_data,
        "best_overall": best_data,
        "best_overall_fitness": best_overall_fitness,
        "history": {
            "qwk": history_qwk,
            "pearson": history_pearson,
            "rmse": history_rmse,
            "llm_stats": history_llm_stats,
        },
        "llm_cache": llm_cache,
    }
    
    # 写入临时文件，然后原子性重命名（避免写入中断导致损坏）
    temp_path = checkpoint_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    
    temp_path.replace(checkpoint_path)
    print(f"✅ Checkpoint saved: {checkpoint_path} (Gen {generation})")


def load_checkpoint(checkpoint_path: Path) -> Optional[Dict[str, Any]]:
    """
    从检查点文件恢复进化状态。
    
    Args:
        checkpoint_path: 检查点文件路径
    
    Returns:
        检查点数据字典，如果文件不存在则返回 None
    """
    if not checkpoint_path.exists():
        return None
    
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        
        print(f"✅ Checkpoint loaded: {checkpoint_path}")
        print(f"   Generation: {checkpoint['generation']}")
        print(f"   Best fitness: {checkpoint['best_overall_fitness']:.4f}")
        print(f"   Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(checkpoint['timestamp']))}")
        
        return checkpoint
    except Exception as e:
        print(f"⚠️  Failed to load checkpoint: {e}")
        return None


def restore_population(population_data: List[Dict[str, Any]]) -> List[Individual]:
    """
    从序列化数据恢复种群。
    
    Args:
        population_data: 序列化的种群数据
    
    Returns:
        Individual 对象列表
    """
    population = []
    for ind_data in population_data:
        genome_dict = ind_data["genome"]
        
        # 处理 icl_indices：从 list 转为 tuple
        if genome_dict.get("icl_indices") is not None:
            genome_dict["icl_indices"] = tuple(genome_dict["icl_indices"])
        
        genome = PromptGenome(**genome_dict)
        
        ind = Individual(
            genome=genome,
            fitness=ind_data.get("fitness"),
            metrics=ind_data.get("metrics"),
            preds=ind_data.get("preds"),
            labels=ind_data.get("labels"),
        )
        population.append(ind)
    
    return population


def restore_best_individual(best_data: Optional[Dict[str, Any]]) -> Optional[Individual]:
    """
    从序列化数据恢复最优个体。
    
    Args:
        best_data: 序列化的最优个体数据
    
    Returns:
        Individual 对象，如果数据为空则返回 None
    """
    if best_data is None:
        return None
    
    genome_dict = best_data["genome"]
    
    # 处理 icl_indices
    if genome_dict.get("icl_indices") is not None:
        genome_dict["icl_indices"] = tuple(genome_dict["icl_indices"])
    
    genome = PromptGenome(**genome_dict)
    
    return Individual(
        genome=genome,
        fitness=best_data.get("fitness"),
        metrics=best_data.get("metrics"),
        preds=best_data.get("preds"),
        labels=best_data.get("labels"),
    )


def clean_old_checkpoints(checkpoint_dir: Path, keep_last: int = 3) -> None:
    """
    清理旧的检查点文件，只保留最近的几个。
    
    Args:
        checkpoint_dir: 检查点目录
        keep_last: 保留最近的 N 个检查点
    """
    if not checkpoint_dir.exists():
        return
    
    # 查找所有检查点文件
    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_gen_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    # 删除旧的
    for old_cp in checkpoints[keep_last:]:
        try:
            old_cp.unlink()
            print(f"🗑️  Removed old checkpoint: {old_cp.name}")
        except Exception as e:
            print(f"⚠️  Failed to remove {old_cp.name}: {e}")
