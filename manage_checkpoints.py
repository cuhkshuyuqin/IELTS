#!/usr/bin/env python
"""
检查点管理工具：查看、删除、恢复检查点
"""
import sys
import json
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent
CHECKPOINT_DIR = BASE_DIR / "logs" / "checkpoints"


def list_checkpoints():
    """列出所有检查点"""
    if not CHECKPOINT_DIR.exists():
        print("❌ No checkpoint directory found.")
        return
    
    checkpoints = sorted(
        CHECKPOINT_DIR.glob("checkpoint_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    if not checkpoints:
        print("📭 No checkpoints found.")
        return
    
    print("\n📦 Available Checkpoints:")
    print("=" * 80)
    
    for i, cp in enumerate(checkpoints, 1):
        try:
            with open(cp, "r") as f:
                data = json.load(f)
            
            gen = data.get("generation", "?")
            fitness = data.get("best_overall_fitness", 0.0)
            timestamp = data.get("timestamp", 0)
            dt = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            size_mb = cp.stat().st_size / 1024 / 1024
            
            print(f"{i}. {cp.name}")
            print(f"   Generation: {gen}")
            print(f"   Best Fitness: {fitness:.4f}")
            print(f"   Timestamp: {dt}")
            print(f"   Size: {size_mb:.2f} MB")
            print()
        except Exception as e:
            print(f"{i}. {cp.name} (⚠️  corrupted: {e})")
            print()


def show_checkpoint_detail(checkpoint_path: Path):
    """显示检查点详细信息"""
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    try:
        with open(checkpoint_path, "r") as f:
            data = json.load(f)
        
        print("\n📊 Checkpoint Details:")
        print("=" * 80)
        print(f"File: {checkpoint_path.name}")
        print(f"Generation: {data.get('generation', '?')}")
        print(f"Best Fitness: {data.get('best_overall_fitness', 0.0):.4f}")
        print(f"Timestamp: {datetime.fromtimestamp(data.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Population Size: {len(data.get('population', []))}")
        print(f"LLM Cache Size: {len(data.get('llm_cache', {}))}")
        
        history = data.get("history", {})
        print(f"\nHistory Length:")
        print(f"  QWK: {len(history.get('qwk', []))} generations")
        print(f"  Pearson: {len(history.get('pearson', []))} generations")
        print(f"  RMSE: {len(history.get('rmse', []))} generations")
        
        if history.get("qwk"):
            print(f"\nMetrics Trend:")
            print(f"  QWK: {history['qwk']}")
            print(f"  Pearson: {history['pearson']}")
            print(f"  RMSE: {history['rmse']}")
        
        best = data.get("best_overall")
        if best:
            print(f"\nBest Individual:")
            print(f"  Fitness: {best.get('fitness', 0.0):.4f}")
            metrics = best.get("metrics", {})
            if metrics:
                print(f"  QWK: {metrics.get('qwk', 0.0):.4f}")
                print(f"  Pearson: {metrics.get('pearson', 0.0):.4f}")
                print(f"  RMSE: {metrics.get('rmse', 0.0):.4f}")
            
            genome = best.get("genome", {})
            print(f"\n  Genome:")
            print(f"    instruction_id: {genome.get('instruction_id')}")
            print(f"    strictness: {genome.get('strictness')}")
            print(f"    use_icl_indices: {genome.get('use_icl_indices')}")
            print(f"    icl_strategy: {genome.get('icl_strategy')}")
            print(f"    k_shots: {genome.get('k_shots')}")
            if genome.get("icl_indices"):
                indices = genome["icl_indices"]
                print(f"    icl_indices: {indices[:5]}... ({len(indices)} total)")
        
    except Exception as e:
        print(f"❌ Failed to read checkpoint: {e}")


def delete_checkpoint(checkpoint_path: Path):
    """删除检查点"""
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    confirm = input(f"⚠️  Delete {checkpoint_path.name}? (yes/no): ")
    if confirm.lower() in ["yes", "y"]:
        checkpoint_path.unlink()
        print(f"✅ Deleted: {checkpoint_path.name}")
    else:
        print("❌ Cancelled.")


def clear_all_checkpoints():
    """清空所有检查点"""
    if not CHECKPOINT_DIR.exists():
        print("❌ No checkpoint directory found.")
        return
    
    checkpoints = list(CHECKPOINT_DIR.glob("checkpoint_*.json"))
    if not checkpoints:
        print("📭 No checkpoints to clear.")
        return
    
    print(f"⚠️  Found {len(checkpoints)} checkpoint(s):")
    for cp in checkpoints:
        print(f"  - {cp.name}")
    
    confirm = input("\n⚠️  Delete ALL checkpoints? (yes/no): ")
    if confirm.lower() in ["yes", "y"]:
        for cp in checkpoints:
            cp.unlink()
        print(f"✅ Deleted {len(checkpoints)} checkpoint(s).")
    else:
        print("❌ Cancelled.")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python manage_checkpoints.py list              # 列出所有检查点")
        print("  python manage_checkpoints.py show <file>       # 显示检查点详情")
        print("  python manage_checkpoints.py delete <file>     # 删除检查点")
        print("  python manage_checkpoints.py clear             # 清空所有检查点")
        print("\nExamples:")
        print("  python manage_checkpoints.py list")
        print("  python manage_checkpoints.py show checkpoint_latest.json")
        print("  python manage_checkpoints.py delete checkpoint_gen_3.json")
        return
    
    command = sys.argv[1]
    
    if command == "list":
        list_checkpoints()
    
    elif command == "show":
        if len(sys.argv) < 3:
            print("❌ Please specify checkpoint file.")
            return
        checkpoint_path = CHECKPOINT_DIR / sys.argv[2]
        show_checkpoint_detail(checkpoint_path)
    
    elif command == "delete":
        if len(sys.argv) < 3:
            print("❌ Please specify checkpoint file.")
            return
        checkpoint_path = CHECKPOINT_DIR / sys.argv[2]
        delete_checkpoint(checkpoint_path)
    
    elif command == "clear":
        clear_all_checkpoints()
    
    else:
        print(f"❌ Unknown command: {command}")


if __name__ == "__main__":
    main()
