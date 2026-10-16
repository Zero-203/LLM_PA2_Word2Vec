# PA2 Word2Vec + FastText

本项目使用 gensim 训练中文 Word2Vec，并以 FastText 完成 THUCNews 分类及人民日报 1998 跨域分析。`PLAN.md` 记录数据隔离约束、阶段验收与完整实验事实；`result/reports/` 保存论文笔记、实验报告和错误分析。

## 环境

项目已在 Windows + Python 3.9.13 64-bit 验证。创建环境并安装依赖：

```powershell
py -3.9 -m venv .venv39
.\.venv39\Scripts\python.exe -m pip install -r requirements.txt
```

如果系统没有 `py -3.9`，需先安装 Python 3.9.13 64-bit。`fasttext-wheel==0.9.2` 的导入名称为 `fasttext`。

## 一键小样本验证与交付物重建

在项目根目录运行：

```powershell
.\scripts\run_all.ps1
```

可选地传入已创建的 Python 3.9 环境路径（例如 CI 临时环境）：

```powershell
.\scripts\run_all.ps1 -PythonPath "$env:TEMP\pa2_m6_clean_env\Scripts\python.exe"
```

该命令分三步执行：

1. 从 `data/THUCNews/` 每类确定性抽取少量文档，在独立拆分上实际训练小型 Word2Vec，并将其向量注入 FastText；断言 U/L/train/val/test 互斥、输入维数正确并完成测试集预测。它不读取原有 `L_test`、人民日报或调用 DeepSeek API。
2. 从本地阶段指标和模型参数中导出字段白名单形式的 `result/metrics/public_summary.json`，不包含本机绝对路径、原始文本或人工审查 ID。
3. 使用本地冻结指标、模型参数和人工审查记录，重建主报告、错误分析及图表。小样本指标只作链路冒烟，不替代或覆盖完整实验结果。

生成或更新：

- `result/metrics/small_sample_smoke.json`
- `result/metrics/public_summary.json`（公开轻量指标汇总）
- `result/small_sample/split_manifest.json`
- `result/small_sample/fasttext_small_sample.bin`
- `result/reports/experiment_report.md`
- `result/reports/error_analysis.md`
- `result/figures/word2vec_convergence.png`
- `result/figures/stage5_confusion_matrix.png`
- `result/figures/class_ladder_decay.png`
- `result/figures/stage5_ablation.png`
- `result/figures/domain_shift_comparison.png`

运行会训练一次小型 Word2Vec/FastText 模型；此模型和小样本分割仅写入 `result/small_sample/`，不会覆盖正式模型、数据分割或阶段 0–5 指标。缺少原始 THUCNews、冻结指标、参数或人工审查记录时会明确失败。

## 完整实验复现边界

小样本模式验证原始 THUCNews 到 Word2Vec、FastText 向量注入和分类预测的核心链路，并重建最终交付物；它不等同于阶段 0–5 全量重训。主报告中的正式实验结果仍来自 `result/metrics/` 的已冻结记录。完整重跑还需要对应原始数据、阶段输入及模型产物；重新标注人民日报另需设置 `DEEPSEEK_API_KEY` 并产生 API 成本。不得将小样本结果解释为正式性能结果。

公开仓库会忽略原始语料、派生数据、模型权重、详细阶段指标、人工审查明细和来源论文 PDF；只发布经字段白名单导出的 `result/metrics/public_summary.json`。因此，克隆后的项目需按 `data/README.md` 配置本地数据，并恢复完整报告脚本所需的冻结指标、模型参数与人工审查记录，之后才能完整运行 `scripts/run_all.ps1`。这些大型/敏感产物不会随 GitHub 仓库分发。

## 结果与论文

- 主指标：`result/metrics/public_summary.json` 中 THUCNews `L_test` 轨道 A。
- 人民日报轨道 B 仅为与 DeepSeek Flash confirmed 标签的一致程度，不是人工真值准确率。
- 公开消融摘要：`result/metrics/public_summary.json`；完整明细保留在本地。
- 论文笔记：`result/reports/notes_paper.md`，覆盖 `documents/reference papers/` 中的 5 篇论文，标注 PDF 页码。
- 计划与验收状态：`PLAN.md`。
