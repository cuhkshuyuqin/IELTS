#!/usr/bin/env python
"""
启动脚本：运行 IELTS 评分 Prompt 进化算法
"""
import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入并运行主函数
from evolver.alphaevolve_multi import run_evolution_hf_icl_only

if __name__ == "__main__":
    run_evolution_hf_icl_only()
