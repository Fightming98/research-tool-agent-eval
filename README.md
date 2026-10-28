# 科研工具 Agent 评测题：微藻培养数据分析

> 用于展示如何设计一套可执行、可评分、可复核的科研工具 Agent 评测任务。

## 目录结构

- `task_prompt_full.md`：提供给被测 Agent 的完整题面。
- `data/microalgae_growth.csv`：Agent 可见的原始数据，含数据质量问题。
- `env/requirements.txt`：参考分析所需 Python 依赖。
- `outputs/`：本次测评流程实际生成的产物。
- `solution/generate_synthetic_data.py`：可复现地重新生成输入 CSV。
- `solution/run_analysis.py`：参考分析程序，生成本项目 outputs 与 gold。
- `solution/verify_template.py`：独立验证器模板。
- `gold/`：标准结果，只用于参考/隐藏评分；**正式评测时不得提供给 Agent**。
- `tests/run.py`：自动验收脚本；**正式评测时不得提供给 Agent**。
- `submission_materials/`：500 字以内笔试答案及嵌入结果图的 Word/Markdown。

## 如何复现参考结果

在安装 `env/requirements.txt` 后，从项目根目录运行：

```bash
python solution/run_analysis.py
python outputs/verify.py
python tests/run.py
```

重新生成模拟原始数据（固定 seed，结果可重复）：

```bash
python solution/generate_synthetic_data.py
python solution/run_analysis.py
```

## 正式评测平台如何拆分

- **Agent 可见：** `task_prompt_full.md`、`data/microalgae_growth.csv` 和必要环境说明。
- **隐藏标准答案 gold：** `gold/` 下标准计数、统计摘要与终点比较结果。
- **隐藏评分 tests：** `tests/run.py`。
- **Agent 应生成：** `quality_report.csv`、`quality_flags.csv`、`summary.csv`、`endpoint_comparison.csv`、`comparison.png`、`report.md`、`verify.py`。

不要将整个 ZIP 一股脑上传到平台的 Agent 可见区，否则会泄露 gold 与 tests。
