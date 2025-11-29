# IELTS 作文评分 Prompt 进化系统

基于遗传算法的 IELTS 作文自动评分 Prompt 优化框架。

## 功能特性

- **结构化 Prompt 进化**：使用 PromptGenome + 遗传算法优化评分指令
- **Few-shot ICL**：示例随基因一起进化（策略和数量可配置）
- **多指标评估**：QWK / Pearson / RMSE / Accuracy
- **LLM 驱动变异**：基于偏差统计的智能 Prompt 优化
- **模板池管理**：自动积累和复用高质量模板

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境

配置 `.env` 文件

将OPENROUTER_API_KEY加入环境变量
```
OPENROUTER_API_KEY=your_key_here
```

### 运行进化

```bash
python run_evolution.py
```

## 项目结构

```
├── evolver/              # 进化算法核心
│   ├── alphaevolve_multi.py    # 主进化流程
│   ├── prompt_evolver.py       # 遗传算子
│   ├── data_aware_prompt.py    # Prompt 构建
│   └── icl_sampler.py          # ICL 示例采样
├── llm_api/              # LLM API 封装
├── scorer/               # 评分器和特征提取
├── data/                 # 数据集
├── logs/                 # 运行日志和结果
└── run_evolution.py      # 启动脚本
```

## 输出结果

- `logs/best_scoring_prompt_hf.json` - 最佳 Prompt 配置
- `logs/best_scoring_prompt_hf.txt` - 最佳 Prompt 文本
- `logs/best_prompt_predictions_hf.csv` - 预测结果
- `logs/metrics_curve_hf.png` - 指标曲线图
- `logs/template_pool.json` - 模板池

## 超参数配置

在 `evolver/alphaevolve_multi.py` 中可调整：

- `POP_SIZE`: 种群大小（默认 10）
- `N_GENERATIONS`: 进化代数（默认 6）
- `CROSSOVER_RATE`: 交叉率（默认 0.85）
- `MUTATION_RATE`: 变异率（默认 0.35）
- `N_EVAL_SAMPLES`: 评估样本数（默认 64）
