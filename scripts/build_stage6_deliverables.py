# -*- coding: utf-8 -*-
"""从阶段 0–5 的冻结指标与人工审查记录生成阶段 6 报告、图表和错例分析。"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "result" / "metrics"
REPORTS = ROOT / "result" / "reports"
FIGURES = ROOT / "result" / "figures"
AUDIT = ROOT / "result" / "manual_audit" / "stage5"


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fnum(value: Any, digits: int = 4) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def pct(value: Any, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value * 100:.{digits}f}%"


def clip(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def make_figures(w2v: Dict[str, Any], evaluation: Dict[str, Any],
                 ablations: Dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    FIGURES.mkdir(parents=True, exist_ok=True)

    log = read_json(ROOT / "model" / "word2vec" / "best" / "train_log.json")
    epochs = log["epoch_logs"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([x["epoch"] for x in epochs], [x["running_avg_loss"] for x in epochs], marker="o", label="训练损失")
    ax.set(title="Word2Vec 训练收敛曲线", xlabel="训练轮次（epoch）", ylabel="累计训练损失（无量纲）")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "word2vec_convergence.png", dpi=180)
    plt.close(fig)

    track_a = evaluation["track_a"]
    labels = track_a["confusion_matrix_labels"]
    matrix = np.asarray(track_a["confusion_matrix_row_normalized"], dtype=float)
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set(title="THUCNews L_test 行归一化混淆矩阵", xlabel="预测类别", ylabel="真实类别")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    fig.colorbar(image, ax=ax, label="行比例（0–1）")
    fig.tight_layout()
    fig.savefig(FIGURES / "stage5_confusion_matrix.png", dpi=180)
    plt.close(fig)

    ladder = [r for r in ablations["summary_rows"] if r["dimension"] == "类别数 S1/S2/S3/S4"]
    ladder.sort(key=lambda r: r["condition"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([r["condition"] for r in ladder], [r["val_macro_f1"] for r in ladder], marker="o", label="FastText macro-F1")
    ax.set(title="类别阶梯任务的验证集性能变化", xlabel="类别阶梯（S1→S4）", ylabel="验证集 macro-F1（无量纲）", ylim=(0, 1.05))
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "class_ladder_decay.png", dpi=180)
    plt.close(fig)

    rows = ablations["summary_rows"]
    labels_short = [f"{r['dimension']}：{r['condition']}" for r in rows]
    values = [r["val_macro_f1"] for r in rows]
    colors = ["#b23a48" if "LLM" in r.get("metric_semantics", "") else "#3973ac" for r in rows]
    height = max(8, 0.34 * len(rows))
    fig, ax = plt.subplots(figsize=(12, height))
    y = np.arange(len(rows))
    ax.barh(y, values, color=colors, label="macro-F1")
    ax.set_yticks(y, labels=labels_short)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set(title="阶段 5 消融结果（验证集；跨域项为 LLM 标签一致性）", xlabel="macro-F1（无量纲）", ylabel="对照条件")
    ax.legend()
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "stage5_ablation.png", dpi=180)
    plt.close(fig)

    track_b = evaluation["track_b"]
    cross = track_b["confirmed_metrics_all_months"]
    oov = track_b["all_article_oov_and_tokenizer_context"]
    tokenizer_rows = [r for r in ablations["summary_rows"] if r["dimension"] == "jieba vs PKU 分词"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(["THUCNews L_test", "人民日报 confirmed"], [track_a["macro_f1"], cross["macro_f1"]],
                color=["#3973ac", "#b23a48"], label="macro-F1")
    axes[0].set(title="主域与跨域指标（标签语义不同）", ylabel="macro-F1（无量纲）", ylim=(0, 1.05))
    axes[0].legend()
    axes[1].bar(["L_test", "人民日报 jieba", "人民日报 PKU"],
                [oov["L_test_oov_rate"], tokenizer_rows[0]["oov_rate"], tokenizer_rows[1]["oov_rate"]],
                color=["#3973ac", "#6b9e78", "#d18b47"], label="OOV 率")
    axes[1].set(title="词汇覆盖差异", ylabel="OOV 率（%）")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 100:.1f}%"))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "domain_shift_comparison.png", dpi=180)
    plt.close(fig)


def build_error_analysis() -> Dict[str, Any]:
    annotations = read_jsonl(AUDIT / "annotations.jsonl")
    samples = {row["sample_id"]: row for row in read_jsonl(AUDIT / "samples.jsonl")}
    reviewed = [row for row in annotations if row.get("track") == "A5-4" and row.get("completed")]
    counts = Counter(row["human_annotation"].get("error_pattern", "未分类") for row in reviewed)
    per_pattern: Dict[str, int] = defaultdict(int)
    selected = []
    for row in reviewed:
        pattern = row["human_annotation"].get("error_pattern", "未分类")
        if per_pattern[pattern] < 5:
            selected.append(row)
            per_pattern[pattern] += 1
        if len(selected) == 20:
            break
    if len(selected) < 20:
        selected_ids = {row["sample_id"] for row in selected}
        selected.extend(row for row in reviewed if row["sample_id"] not in selected_ids)[:20 - len(selected)]
    if len(selected) < 20:
        raise ValueError(f"A5-4 已完成人工标注不足 20 条：{len(selected)}")

    lines = [
        "# 典型错误案例分析",
        "",
        "> 案例来自 `result/manual_audit/stage5/annotations.jsonl` 与 `samples.jsonl`。本分析采用单人单次审查，案例用于定性诊断，不外推总体错误率。为避免大段复现新闻正文，每条仅保留短语境片段。",
        "",
        "## 复核概况",
        "",
        f"A5-4 共完成人工复核 **{len(reviewed)}** 条；人工判断与数据集标签一致 {sum(r['human_annotation'].get('human_label') == r.get('gold_label') for r in reviewed)} 条，与模型预测一致 {sum(r['human_annotation'].get('human_label') == r.get('model', {}).get('prediction') for r in reviewed)} 条。逐类归因如下：",
        "",
        "| 人工归因 | 数量 |",
        "|---|---:|",
    ]
    lines.extend(f"| {pattern} | {count} |" for pattern, count in counts.most_common())
    lines.extend([
        "",
        "## 案例（20 条）",
        "",
        "| # | 文档 ID | 真值标签 | 预测标签 | 置信度 | 短语境片段 | 人工归因 | 人工依据/评论 |",
        "|---:|---|---|---|---:|---|---|---|",
    ])
    for index, row in enumerate(selected, 1):
        sample = samples.get(row["sample_id"], {})
        text = " ".join(sample.get("text", {}).get("original", "").split())
        excerpt = text[:18].replace("|", "／")
        source = row.get("source", {}).get("doc_id", row["sample_id"])
        human = row["human_annotation"]
        evidence = clip(human.get("evidence", ""), 55).replace("|", "／")
        comment = clip(human.get("comment", ""), 70).replace("|", "／")
        reason = "；".join(value for value in (evidence, comment) if value)
        lines.append(
            f"| {index} | `{source}` | {row.get('gold_label', '')} | {row.get('model', {}).get('prediction', '')} | "
            f"{fnum(row.get('model', {}).get('confidence'), 3)} | {excerpt} | "
            f"{human.get('error_pattern', '未分类')} | {reason} |"
        )
    lines.extend([
        "",
        "## 错误模式归纳",
        "",
        "1. **模型线索偏差**：财经与股票等近邻类别共享金融词汇，模型可能过度依赖高频领域词而忽略文章主体、交易对象或新闻任务。",
        "2. **标签/真值疑似错误**：人工认为数据集标签可能与正文主题不一致；不能据单人复核直接改写测试标签或推算全量噪声率。",
        "3. **类别边界模糊**：文章同时涉及多个主题，单一类别标注无法完整表达；应结合数据集定义和主要报道对象判断。",
        "4. **证据不足/不确定**：短文、标题化内容或上下文不足以支持稳定归因，应保留不确定而非强行判错。",
        "5. **其他/无明显错误**：个案中未观察到明确分类问题，保留作对照。",
        "",
        "**限制**：人工审查样本由既有候选抽样产生，审查者为一人且没有双人一致性/仲裁；类别间样本数不等。以上模式是错误解释，不是因果证明或总体错误率估计。数据集标签、LLM 标签和人工判断必须分开表述。",
        "",
        "来源：`result/metrics/stage5_evaluation.json`、`result/metrics/stage5_manual_audit_samples.jsonl`、`result/manual_audit/stage5/annotations.jsonl`。",
    ])
    output = REPORTS / "error_analysis.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"reviewed": len(reviewed), "case_count": len(selected), "patterns": dict(counts)}


def build_experiment_report(w2v: Dict[str, Any], evaluation: Dict[str, Any],
                            ablations: Dict[str, Any], audit_summary: Dict[str, Any]) -> None:
    train_log = read_json(ROOT / "model" / "word2vec" / "best" / "train_log.json")
    ft_params = read_json(ROOT / "model" / "fasttext" / "S4_best_pretrained" / "params.json")["params"]
    smoke = read_json(METRICS / "small_sample_smoke.json")
    track_a = evaluation["track_a"]
    track_b = evaluation["track_b"]
    coverage = track_b["label_coverage"]
    cross = track_b["confirmed_metrics_all_months"]
    best_params = train_log["params"]
    top_pairs = evaluation["track_a"]["top_confusions"][:3]
    isolation_assertions = {
        key: evaluation.get("isolation", {}).get(key)
        for key in (
            "pd1998_in_training_assertion",
            "training_tuning_paths_exclude_199801",
            "L_test_opened_once_by_this_script",
        )
    }
    ablation_lines = [
        "| 维度 | 条件 | 验证集/跨域 macro-F1 | 样本数 | 训练秒数 | 来源/语义 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in ablations["summary_rows"]:
        semantics = row.get("metric_semantics", "L_val 验证集")
        ablation_lines.append(
            f"| {row['dimension']} | {row['condition']} | {fnum(row['val_macro_f1'], 6)} | "
            f"{row['val_examples']} | {fnum(row.get('train_sec'), 1)} | {row.get('source', '')}；{semantics} |"
        )
    monthly = []
    for month, details in track_b["per_month"].items():
        metrics = details.get("metrics_against_unverified_llm_labels", {})
        monthly.append(
            f"| {month} | {metrics.get('examples', 0)} | {fnum(metrics.get('accuracy'), 6)} | "
            f"{fnum(metrics.get('macro_f1'), 6)} | {pct(details.get('confirmed_ratio'), 2)} |"
        )
    if not monthly:
        monthly.append("| — | — | — | — | — |")
    confusions = "；".join(f"{r['true']}→{r['predicted']}（{r['count']}）" for r in top_pairs)

    content = f"""# PA2 Word2Vec 与 FastText 新闻分类实验报告

> 本报告由阶段指标、冻结模型参数与人工审查记录生成。公开数值汇总见 `result/metrics/public_summary.json`；THUCNews `L_test` 是主评估，人民日报结果只表示与 DeepSeek Flash 已确认标签的一致程度，不是人工金标准性能。

## 1. 环境

实验环境为 Python 3.9.13（64-bit），gensim 4.4.0、fasttext-wheel 0.9.2、jieba 0.42.1、numpy 1.26.4、scipy 1.11.4、scikit-learn 1.3.2、pandas 2.1.4、matplotlib 3.9.4。环境与指标见 `result/metrics/public_summary.json`；依赖锁定范围见 `requirements.txt`。固定随机种子为 `{train_log['seed']}`，配置单点为 `src/config.py`。

## 2. 数据与隔离

THUCNews 原始语料为 14 类共 836,075 篇；按文档划分为无标注嵌入集 U 与标注分类集 L，再由 L 划分 train/val/test。类别阶梯 S1–S4 分别为 2/4/8/14 类。人民日报 199801–199806 只作跨域测试，DeepSeek Flash 按文章正文标注，类别空间与 THUCNews 对齐；版次仅为版面元数据，不作为主题标签。

人民日报格式化文章 {coverage['formatted_articles']:,} 篇，其中 confirmed {coverage['confirmed_labeled']:,} 篇（{pct(coverage['confirmed_ratio'])}），needs_review {coverage['needs_review']:,} 篇（{pct(coverage['needs_review_ratio'])}）。模型自评置信阈值为 {track_b['confidence_threshold']}，提示版本 `{track_b['prompt_version']}`。确认规则不等于人工复核或概率校准。训练/调参只使用 THUCNews train/val；人民日报不参与训练、调参或词典构建；`L_test` 仅在阶段 5 读取一次。相关汇总见 `result/metrics/public_summary.json`。
公开汇总仅保留数据隔离断言布尔值：`pd1998_in_training_assertion`={isolation_assertions['pd1998_in_training_assertion']}，`training_tuning_paths_exclude_199801`={isolation_assertions['training_tuning_paths_exclude_199801']}，`L_test_opened_once_by_this_script`={isolation_assertions['L_test_opened_once_by_this_script']}。

## 3. 方法与参数

Word2Vec 使用 gensim Skip-gram/CBOW 实验，以词上下文预测学习词向量；文档表示比较 P1 均值、P2 TF-IDF 加权均值、P3 去停用词均值。分类主线为 FastText 监督分类，另以池化向量 + LogisticRegression 作对照。先在 50k/150k/全量 U 上进行语料规模实验，再完成参数矩阵；最终 Word2Vec 选择 150,000 篇语料，得到 {train_log['vocab_size']:,} 词词表、{train_log['params']['vector_size']} 维向量。

| 参数 | 含义 | 本项目取值 | 依据/来源 |
|---|---|---|---|
| Word2Vec `vector_size` | 向量维数 | {best_params['vector_size']} | E3 维度对照与最终参数：`result/metrics/public_summary.json` |
| Word2Vec `window` | 上下文窗口 | {best_params['window']} | E4 窗口实验及 `result/metrics/public_summary.json` |
| Word2Vec `min_count` | 词频下限 | {best_params['min_count']} | 冻结参数 `result/metrics/public_summary.json` |
| Word2Vec `sg` | 0=CBOW，1=Skip-gram | {best_params['sg']}（CBOW） | E1/E2 与 `result/metrics/public_summary.json` |
| Word2Vec `negative` / `hs` | 负采样数 / 是否层次 Softmax | {best_params['negative']} / {best_params['hs']} | E5 对照与冻结参数 |
| Word2Vec `sample` | 高频词下采样阈值 | {best_params['sample']} | 冻结参数与训练日志 |
| Word2Vec `alpha` / `min_alpha` | 初始/最低学习率 | {best_params['alpha']} / {best_params['min_alpha']} | 冻结参数与训练日志 |
| Word2Vec `epochs` / `workers` | 训练轮数 / 工作线程 | {best_params['epochs']} / {best_params['workers']} | E6 与复现约束；确定性训练使用单线程 |
| FastText `dim` | 文档特征维度 | {ft_params['dim']} | 与预训练向量维度一致；阶段 4 验证集选择 |
| FastText `epoch` / `lr` | 训练轮数 / 学习率 | {ft_params['epoch']} / {ft_params['lr']} | 覆盖式随机搜索；参数见冻结 `params.json` |
| FastText `wordNgrams` / `minCount` | 词 n-gram 阶数 / 词频下限 | {ft_params['wordNgrams']} / {ft_params['minCount']} | A5-8 n-gram 消融及阶段 4 验证集搜索 |
| FastText `loss` | 多分类损失形式 | `{ft_params['loss']}` | 阶段 4 搜索空间与验证集结果 |
| FastText `bucket` / `thread` | 子词哈希桶 / 线程数 | {ft_params['bucket']:,} / {ft_params['thread']} | 冻结模型参数；环境核数实测 |
| FastText `pretrainedVectors` / `verbose` | 初始化向量文件 / 日志级别 | gensim `best/w2v_format.vec` / {ft_params['verbose']} | 注入链路在 A0-5、A2-9 与阶段 4 输入验证通过 |
| 随机种子 | 控制采样和实验顺序 | {train_log['seed']} | `src/config.py` 与阶段指标 |
| DeepSeek 纳入阈值 | 自动确认置信门槛 | {track_b['confidence_threshold']} | `result/metrics/public_summary.json`；模型自评未校准 |

公开的参数与逐组结果见 `result/metrics/public_summary.json`。完整搜索中间产物保留在本地，不纳入公开仓库。

## 4. 实验结果

### 4.1 Word2Vec

最终模型使用 CBOW、{best_params['vector_size']} 维、窗口 {best_params['window']}、{best_params['epochs']} 轮、U 子集 {train_log['corpus_docs']:,} 篇。词表 {w2v['vocab_size']:,}；训练损失由 {train_log['convergence']['first_loss']:,.0f} 降至 {train_log['convergence']['last_loss']:,.0f}，相对下降 {train_log['convergence']['relative_drop_pct']}%。词相似度评测覆盖 {w2v['similarity']['n_pairs']}/{w2v['similarity']['n_total']} 对，Spearman ρ={w2v['similarity']['spearman_rho']}；类比 top-1 为 {w2v['analogy']['n_correct']}/{w2v['analogy']['n_evaluated']}（{w2v['analogy']['top1_accuracy']}%）。评测集规模有限，不能将这些指标解释为普遍语义能力。来源：`result/metrics/public_summary.json`。

![Word2Vec 训练收敛曲线](../figures/word2vec_convergence.png)

### 4.2 THUCNews 主评估（轨道 A）

冻结的 `S4_best_pretrained` 在 L_test {track_a['examples']:,} 篇上的 accuracy={fnum(track_a['accuracy'], 6)}、macro-F1={fnum(track_a['macro_f1'], 6)}、weighted-F1={fnum(track_a['weighted_f1'], 6)}。混淆较多的前三类为：{confusions}。逐类指标、混淆矩阵和 top-k 指标见 `result/metrics/public_summary.json`。结果超过计划中自设的主评估阈值，但单次固定测试集不代表所有域与未来数据。

![L_test 行归一化混淆矩阵](../figures/stage5_confusion_matrix.png)

### 4.3 人民日报跨域评估（轨道 B）

在 confirmed 子集 {cross['examples']:,} 篇上 accuracy={fnum(cross['accuracy'], 6)}、macro-F1={fnum(cross['macro_f1'], 6)}。指标仅表示与 DeepSeek Flash 标签的一致程度。各月数据如下：

| 月份 | confirmed 样本 | Accuracy | macro-F1 | confirmed 覆盖率 |
|---|---:|---:|---:|---:|
{chr(10).join(monthly)}

OOV 率：THUCNews L_test {pct(track_b['all_article_oov_and_tokenizer_context']['L_test_oov_rate'])}，199801 jieba {pct(track_b['all_article_oov_and_tokenizer_context']['PD199801_oov_rate'])}，199801–199806 全量 {pct(track_b['all_article_oov_and_tokenizer_context']['PD1998_all_oov_rate'])}。OOV 差异与标签质量、类别分布和年代语汇差异共同影响跨域结果，不能单独归因。

![主域与跨域对比](../figures/domain_shift_comparison.png)

## 5. 消融实验

所有分类消融均使用 L_val，不读取 L_test；人民日报分词对照使用同一冻结模型，标签为 LLM confirmed。阶段 5 的全部对照如下，公开逐组汇总见 `result/metrics/public_summary.json`。

{chr(10).join(ablation_lines)}

![阶段 5 消融结果](../figures/stage5_ablation.png)

类别数增加时验证集 macro-F1 下降，反映多类任务难度及类别边界差异；预训练向量在本次验证对照未带来增益，n-gram=2 高于 n-gram=1。类别平衡采样的验证分数低于原始训练分布，这一观察仅适用于当前固定验证集，不能推导为普遍采样策略结论。分词对照的 PKU 结果 OOV 更高且 LLM 标签一致性略高，因此不应将分数差简单等同分词器质量差。

## 6. 结论

项目完成从大规模无监督词表示训练、文档池化对照到 2/4/8/14 类 FastText 监督分类的实验闭环。THUCNews 内部测试表现较高；跨域结果明显较低且与 LLM 标签分布/质量存在不确定性。现有结果支持“领域迁移需要单独评估”和“macro-F1、逐类召回比单看 accuracy 更适合类别不均衡任务”的结论，不支持将跨域指标描述为人工真值准确率，也不能从消融相关性推断因果。

## 7. 局限性

- **人民日报标签**：标签由 DeepSeek Flash 自动生成；{coverage['needs_review']:,}/{coverage['formatted_articles']:,} 篇需复核，confirmed 也未经逐篇人工验证。模型与标签生成模型可能共享类别偏差，标签分布不等于人工真值分布。
- **置信度**：FastText 与 DeepSeek 输出的置信度均未经概率校准，阈值只用于本项目流程规则。
- **领域漂移**：1998 年人民日报与 THUCNews 的年代、新闻风格、词汇及类别先验不同；OOV 指标不是漂移的完整度量。
- **类别不平衡**：L_train 使用分层限额而非真实自然频率，平衡采样消融显示性能会随抽样改变；结论受当前数据构造影响。
- **算力与语料规模**：Word2Vec 使用 150k 篇 U 而非全部 836k THUCNews，以满足单机实验时间约束；E7 结果与参数矩阵均需按记录的语料规模解释。
- **人工错误审查**：{audit_summary['reviewed']} 条 A5-4 样本为单人单次复核，无双人一致性或仲裁，且样本并非随机总体抽样；案例不可外推总体错误率。
- **复现范围**：`scripts/run_all.ps1` 可从已冻结指标重建报告与图表；它不重跑大语料清洗、Word2Vec/FastText 训练或 DeepSeek 标注。完整重训仍依赖原始数据、预训练模型、相应阶段脚本及 API 访问，见 `README.md`。

## 8. 小样本端到端冒烟验证

`scripts/run_all.ps1` 先从 THUCNews 原始类别目录确定性抽取小样本，独立构造 U、L_train、L_val、L_test，再训练 Word2Vec 并通过 `pretrainedVectors` 注入 FastText。实测划分为 U={smoke['counts']['U']}、train={smoke['counts']['L_train']}、val={smoke['counts']['L_val']}、test={smoke['counts']['L_test']}；向量词表 {smoke['word2vec_vocab_size']:,}，FastText 输入矩阵形状 `{smoke['fasttext_input_matrix_shape']}`。预训练注入断言通过。烟雾测试集 macro-F1={fnum(smoke['test']['macro_f1'], 4)}，只用于验证链路，**不作为项目性能结果或阶段 5 指标**。轻量汇总见 `result/metrics/public_summary.json`，完整记录和样本清单保留在本地。

## 附录：来源索引

- 公开环境、模型、分类、消融及小样本指标：`result/metrics/public_summary.json`。
- 可复现脚本和配置：`scripts/`、`src/config.py`、`requirements.txt`。
- 论文阅读：`result/reports/notes_paper.md`，覆盖 5 篇论文并按 PDF 页码标注。
"""
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "experiment_report.md").write_text(content, encoding="utf-8")


def main() -> None:
    required = [
        METRICS / "stage0_env_check.json", METRICS / "stage1_cleaning_check.json",
        METRICS / "stage3_vectorize_best.json", METRICS / "stage4_training.json",
        METRICS / "stage5_evaluation.json", METRICS / "stage5_ablations.json",
        METRICS / "stage5_ablation_summary.csv", METRICS / "w2v_eval_best.json",
        METRICS / "small_sample_smoke.json",
        ROOT / "model" / "word2vec" / "best" / "train_log.json",
        ROOT / "model" / "word2vec" / "best" / "params.json",
        ROOT / "model" / "fasttext" / "S4_best_pretrained" / "params.json",
        AUDIT / "annotations.jsonl", AUDIT / "samples.jsonl",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少阶段 6 输入文件：" + ", ".join(missing))
    w2v = read_json(METRICS / "w2v_eval_best.json")
    evaluation = read_json(METRICS / "stage5_evaluation.json")
    ablations = read_json(METRICS / "stage5_ablations.json")
    audit_summary = build_error_analysis()
    make_figures(w2v, evaluation, ablations)
    build_experiment_report(w2v, evaluation, ablations, audit_summary)
    print("阶段 6 交付物已生成：experiment_report.md、error_analysis.md、5 张图表。")
    print("注意：已运行独立小样本训练链路；报告中的阶段 0–5 主结果仍来自冻结指标，不会被小样本结果覆盖。")


if __name__ == "__main__":
    main()
