# -*- coding: utf-8 -*-
"""阶段 5：使用已冻结的 S4 模型评估 L_test 与人民日报标签语料。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "classify"))

import compat  # noqa: F401,E402
import config  # noqa: E402

MODEL_DIR = config.MODEL_FT / "S4_best_pretrained"
MODEL_PATH = MODEL_DIR / "model.bin"
PARAMS_PATH = MODEL_DIR / "params.json"
STAGE4_PATH = config.RESULT_METRICS / "stage4_training.json"
OUT_PATH = config.RESULT_METRICS / "stage5_evaluation.json"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def metrics(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> Dict[str, Any]:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(labels))
    confusion_pairs = []
    for i, true_label in enumerate(labels):
        for j, predicted_label in enumerate(labels):
            if i != j and matrix[i, j]:
                confusion_pairs.append({"true": true_label, "predicted": predicted_label,
                                        "count": int(matrix[i, j])})
    confusion_pairs.sort(key=lambda item: item["count"], reverse=True)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = matrix / row_sums.clip(min=1)
    return {
        "examples": len(y_true),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=list(labels),
                                          average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=list(labels),
                                             average="weighted", zero_division=0)), 6),
        "per_class": {
            label: {"precision": round(float(p), 6), "recall": round(float(r), 6),
                    "f1": round(float(f), 6), "support": int(n)}
            for label, p, r, f, n in zip(labels, precision, recall, f1, support)
        },
        "confusion_matrix_labels": list(labels),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_row_normalized": normalized.round(6).tolist(),
        "top_confusions": confusion_pairs[:10],
    }


def load_model():
    import fasttext

    # fasttext-wheel 的 Windows C++ 层不能可靠打开 Unicode 路径。
    model_path = MODEL_PATH
    if os.name == "nt" and any(ord(char) > 127 for char in str(model_path)):
        ascii_path = Path(tempfile.gettempdir()) / "pa2_fasttext_ascii" / "S4_best_pretrained.bin"
        os.makedirs(ascii_path.parent, exist_ok=True)
        if not ascii_path.exists() or ascii_path.stat().st_size != model_path.stat().st_size:
            shutil.copyfile(model_path, ascii_path)
        model_path = ascii_path
    return fasttext.load_model(str(model_path))


def predict_texts(model, texts: Sequence[str], k: int = 3,
                  batch_size: int = 512) -> Tuple[List[str], List[str], List[float], List[Dict[str, Any]]]:
    labels: List[str] = []
    top1: List[str] = []
    confidences: List[float] = []
    topk: List[Dict[str, Any]] = []
    for start in range(0, len(texts), batch_size):
        batch = [text if text.strip() else "__empty__" for text in texts[start:start + batch_size]]
        pred_labels, pred_probs = model.predict(batch, k=k)
        for names, probs in zip(pred_labels, pred_probs):
            names = list(names)
            probs = [float(p) for p in probs]
            clean_names = [name.removeprefix("__label__") for name in names]
            labels.append(clean_names[0] if clean_names else "")
            top1.append(clean_names[0] if clean_names else "")
            confidences.append(probs[0] if probs else 0.0)
            topk.append({"labels": clean_names, "probabilities": probs})
    return labels, top1, confidences, topk


def records_from_fasttext(path: Path) -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    labels: List[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.rstrip("\r\n")
            label, separator, text = line.partition(" ")
            if not separator or not label.startswith("__label__") or not text.strip():
                raise ValueError(f"FastText 测试集格式错误: {path}:{line_no}")
            labels.append(label.removeprefix("__label__"))
            texts.append(text)
    return labels, texts


def validate_audit_examples(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """记录人工抽查状态；候选样本的自动生成不代表已完成人工复核。"""
    by_id = {sample["article_id"]: sample for sample in samples}
    checks = {"19980101-01-001", "19980101-01-003",
              "19980101-01-004", "19980101-02-003"}
    return {
        "method": "本脚本仅自动生成抽查候选并检查固定 ID 是否被抽中；未执行人工阅读、标签判定或语义复核",
        "reviewed_examples": [],
        "reviewed_count": 0,
        "requested_fixed_ids": sorted(checks),
        "requested_ids_in_sample": sorted(checks.intersection(by_id)),
        "generalization_limit": "尚无人工复核结果，不能估计全体标签噪声率或独立确定主因排序",
    }


def main() -> Dict[str, Any]:
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if not all(path.is_file() for path in (MODEL_PATH, PARAMS_PATH, config.F_L_TEST,
                                           config.F_PD1998_FORMATTED,
                                           config.F_PD1998_LLM_LABELS, config.F_SPLIT_MANIFEST)):
        raise FileNotFoundError("阶段 5 所需模型、测试集、标注或切分清单文件缺失")
    params = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))
    stage4 = json.loads(STAGE4_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(config.F_SPLIT_MANIFEST.read_text(encoding="utf-8"))
    if params["stage"] != "S4" or params["pretrained_vectors"] is None:
        raise AssertionError("当前模型不是阶段4最佳预训练 S4 模型")
    if manifest.get("isolation_assertions", {}).get("pd1998_in_any_training_set") is not False:
        raise AssertionError("切分清单未能证明人民日报与 THUCNews 训练划分隔离")

    train_tune_paths = []
    for stage_data in stage4["r1"].values():
        train_tune_paths.extend(stage_data["best_model"].get("files", {}).values())
        for check in stage_data["input_validation"].values():
            train_tune_paths.append(check.get("path", ""))
    for stage_data in stage4["r2"].values():
        for route in stage_data.values():
            train_tune_paths.extend(route.get("input_files", []))
    if any("199801" in str(path) for path in train_tune_paths):
        raise AssertionError("A5-9 失败：训练/调参记录路径包含 199801")

    model = load_model()
    labels = list(config.LABELS)
    if set(model.get_labels()) != {"__label__" + label for label in labels}:
        raise AssertionError("已冻结模型标签集与 THUCNews 14 类不一致")

    # 本脚本对 L_test 唯一的一次数据读取：载入标签和文本后不再重开该文件。
    y_test, test_texts = records_from_fasttext(config.F_L_TEST)
    y_test_pred, _, test_confidence, test_topk = predict_texts(model, test_texts)
    test_metrics = metrics(y_test, y_test_pred, labels)
    test_metrics["top2_accuracy"] = round(sum(
        true in pred["labels"][:2] for true, pred in zip(y_test, test_topk)
    ) / max(len(y_test), 1), 6)
    test_metrics["top3_accuracy"] = round(sum(
        true in pred["labels"][:3] for true, pred in zip(y_test, test_topk)
    ) / max(len(y_test), 1), 6)
    test_metrics["mean_top1_confidence"] = round(
        sum(test_confidence) / max(len(test_confidence), 1), 6
    )
    val_macro_f1 = stage4["r1"]["S4"]["best_model"]["val_metrics"]["macro_f1"]
    test_metrics["val_macro_f1_reference"] = val_macro_f1
    test_metrics["macro_f1_absolute_gap_vs_val"] = round(abs(test_metrics["macro_f1"] - val_macro_f1), 6)

    annotations = {row["article_id"]: row for row in read_jsonl(config.F_PD1998_LLM_LABELS)}
    formatted = read_jsonl(config.F_PD1998_FORMATTED)
    pd_texts = [" ".join(row.get("tokens_jieba") or []) for row in formatted]
    pd_pred, _, pd_confidence, pd_topk = predict_texts(model, pd_texts)
    pd_labels = [annotations.get(row["article_id"], {}).get("label")
                 if annotations.get(row["article_id"], {}).get("status") == "confirmed" else None
                 for row in formatted]
    pd_status = [annotations.get(row["article_id"], {}).get("status", "missing")
                 for row in formatted]
    pd_candidate_labels = [annotations.get(row["article_id"], {}).get("candidate_label")
                           for row in formatted]
    pd_all_distribution = dict(Counter(pd_pred))
    pd_confirmed = [i for i, label in enumerate(pd_labels) if label in labels]
    pd_confirmed_metrics = metrics([pd_labels[i] for i in pd_confirmed],
                                   [pd_pred[i] for i in pd_confirmed], labels)
    pd_confirmed_metrics["mean_prediction_confidence"] = round(
        sum(pd_confidence[i] for i in pd_confirmed) / max(len(pd_confirmed), 1), 6
    )

    per_month: Dict[str, Any] = {}
    for month in sorted({row["month"] for row in formatted}):
        ids = [i for i, row in enumerate(formatted) if row["month"] == month]
        confirmed_ids = [i for i in ids if pd_labels[i] in labels]
        review_ids = [i for i in ids if pd_status[i] == "needs_review"]
        month_metrics = metrics([pd_labels[i] for i in confirmed_ids],
                                [pd_pred[i] for i in confirmed_ids], labels) if confirmed_ids else None
        per_month[month] = {
            "articles": len(ids),
            "confirmed": len(confirmed_ids),
            "needs_review": len(review_ids),
            "missing_or_unlabeled": sum(pd_status[i] == "missing" for i in ids),
            "confirmed_ratio": round(len(confirmed_ids) / max(len(ids), 1), 6),
            "needs_review_ratio": round(len(review_ids) / max(len(ids), 1), 6),
            "mean_confirmed_label_confidence": round(
                sum(float(annotations[formatted[i]["article_id"]].get("confidence") or 0)
                    for i in confirmed_ids) / max(len(confirmed_ids), 1), 6
            ),
            "label_distribution_confirmed": dict(Counter(pd_labels[i] for i in confirmed_ids)),
            "prediction_distribution_all": dict(Counter(pd_pred[i] for i in ids)),
            "metrics_against_unverified_llm_labels": month_metrics,
        }

    confirmed_label_distribution = dict(Counter(pd_labels[i] for i in pd_confirmed))
    banci_distribution = dict(Counter(str(row.get("banci", "unknown")) for row in formatted))
    mismatch_indices = [i for i in pd_confirmed if pd_pred[i] != pd_labels[i]]
    review_indices = [i for i, status in enumerate(pd_status) if status == "needs_review"]
    audit_indices = sorted(mismatch_indices,
                           key=lambda i: abs(pd_confidence[i] - 0.5))[:30]
    audit_indices += sorted(review_indices,
                            key=lambda i: abs(pd_confidence[i] - 0.5))[:10]
    audit_samples = []
    for i in audit_indices:
        row = formatted[i]
        annotation = annotations.get(row["article_id"], {})
        audit_samples.append({
            "article_id": row["article_id"], "month": row["month"],
            "banci": row.get("banci"), "status": pd_status[i],
            "llm_label": pd_labels[i], "candidate_label": pd_candidate_labels[i],
            "llm_confidence": annotation.get("confidence"),
            "model_prediction": pd_pred[i], "model_confidence": pd_confidence[i],
            "top3": pd_topk[i],
            "text_excerpt": " ".join((row.get("title_text", ""),
                                       row.get("body_text", ""))).strip()[:240],
            "manual_review": "待人工复核；模型自评 confidence 未校准",
        })

    oov_baseline = json.loads((config.RESULT_METRICS / "a1_6_oov_baseline.json").read_text(encoding="utf-8"))
    stage3 = json.loads((config.RESULT_METRICS / "stage3_vectorize_best.json").read_text(encoding="utf-8"))
    oov = {
        "L_test_oov_rate": oov_baseline["L_test"]["oov_rate"],
        "PD199801_oov_rate": oov_baseline["T_pd199801"]["oov_rate"],
        "PD199801_minus_L_test": oov_baseline["oov_rate_difference_pd199801_minus_L_test"],
        "PD1998_all_oov_rate": stage3["splits"]["T_pd1998_all"]["P1_mean"]["stats"]["oov_token_ratio"],
        "tokenizer_mismatch_isolated_experiment": "未执行，不能单独量化其贡献",
        "tokenizer_comparison_source": "formatted corpus contains both tokens_jieba and tokens_pku; classifier evaluated only jieba",
    }
    ablation = [
        {"dimension": "预训练向量", "result": "已执行", "metric": "R1/noVec 见阶段4结果"},
        {"dimension": "CBOW vs Skip-gram", "result": "词向量 intrinsic 对照已执行；下游分类对照 ⚠️未执行", "metric": "result/metrics/w2v_eval_best.json；下游 Δ 未执行"},
        {"dimension": "Word2Vec dim 100/200/300", "result": "阶段2训练/评估已执行；下游 FastText 对照 ⚠️未执行", "metric": "无阶段5下游消融值"},
        {"dimension": "P1/P2/P3 pooling", "result": "阶段3向量已生成；R2 分类对照 ⚠️未执行", "metric": "阶段4仅训练 P1_mean R2"},
        {"dimension": "wordNgrams 1 vs 2", "result": "阶段4搜索覆盖；严格单变量对照 ⚠️未执行", "metric": "见 tuning_S1-S4.csv；配置还同时改变其他参数"},
        {"dimension": "类别数 S1-S4", "result": "已执行", "metric": stage4["acceptance"]["A4-7"]["macro_f1_by_ladder"]},
        {"dimension": "PKU vs jieba resegmentation", "result": "未执行", "metric": "⚠️未执行"},
        {"dimension": "balanced vs original imbalance", "result": "未执行", "metric": "⚠️未执行"},
    ]

    small_recall = {label: test_metrics["per_class"][label]["recall"] for label in labels}
    label_coverage = {
        "formatted_articles": len(formatted),
        "confirmed_labeled": len(pd_confirmed),
        "needs_review": len(review_indices),
        "missing_status": sum(status == "missing" for status in pd_status),
        "candidate_label_present_but_unconfirmed": sum(
            pd_status[i] == "needs_review" and pd_candidate_labels[i] in labels
            for i in range(len(formatted))
        ),
        "confirmed_ratio": round(len(pd_confirmed) / max(len(formatted), 1), 6),
        "needs_review_ratio": round(len(review_indices) / max(len(formatted), 1), 6),
    }
    result = {
        "stage": "stage5-evaluation",
        "started_at": started,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": config.SEED,
        "model": {"path": str(MODEL_PATH), "params_path": str(PARAMS_PATH),
                  "params_sha256": hashlib.sha256(PARAMS_PATH.read_bytes()).hexdigest(),
                  "params": params["params"]},
        "test_data_reads": {"L_test.txt": 1},
        "l_test_used": True,
        "track_a": test_metrics,
        "track_b": {
            "semantics": "相对于 DeepSeek Flash status=confirmed 标签的一致性，不代表人工真值准确率",
            "model": config.DEEPSEEK_MODEL,
            "prompt_version": config.PD1998_LABEL_PROMPT_VERSION,
            "confidence_threshold": config.PD1998_LABEL_CONFIDENCE_THRESHOLD,
            "label_coverage": label_coverage,
            "confirmed_metrics_all_months": pd_confirmed_metrics,
            "confirmed_label_distribution": confirmed_label_distribution,
            "model_prediction_distribution_all_articles": pd_all_distribution,
            "banci_distribution_as_layout_metadata_only": banci_distribution,
            "per_month": per_month,
            "all_article_oov_and_tokenizer_context": oov,
        },
        "manual_audit_samples": audit_samples,
        "manual_audit_status": "包含模型与 confirmed LLM 标签不一致样例，以及 needs_review 样例；自动结果不当作人工真值",
        "human_review": validate_audit_examples(audit_samples),
        "ablation_summary": ablation,
        "acceptance": {},
        "isolation": {"pd1998_in_training_assertion": False,
                      "training_tuning_paths_exclude_199801": True,
                      "L_test_opened_once_by_this_script": True},
    }
    a = result["acceptance"]
    a["A5-1"] = {"ok": test_metrics["macro_f1"] >= config.THRESHOLDS["A5-1_Ltest_S4_macro_f1_min"]
                  and test_metrics["accuracy"] >= config.THRESHOLDS["A5-1_Ltest_S4_acc_min"],
                  "macro_f1": test_metrics["macro_f1"], "accuracy": test_metrics["accuracy"]}
    a["A5-2"] = {"ok": test_metrics["macro_f1_absolute_gap_vs_val"] <
                  config.THRESHOLDS["A5-2_test_val_gap_max"],
                  "absolute_macro_f1_gap": test_metrics["macro_f1_absolute_gap_vs_val"]}
    a["A5-3"] = {"ok": len(pd_confirmed) > 0 and label_coverage["confirmed_ratio"] > 0,
                  "confirmed_articles": len(pd_confirmed),
                  "metrics_are_llm_agreement_not_human_truth": True,
                  "per_month_metrics_available": all(v["metrics_against_unverified_llm_labels"] is not None
                                                     for v in per_month.values())}
    confusion_interpretations = {
        ("财经", "股票"): "两类共享公司、市场、证券等词汇，股票报道也常落在财经语境；存在相邻类别边界重叠。",
        ("股票", "财经"): "财经新闻常包含股价或证券市场内容，单篇文章主旨可能兼具行业与金融市场。",
        ("教育", "社会"): "校园管理、升学就业等议题兼有教育制度与社会民生属性，标题/短篇上下文会弱化主类别信号。",
    }
    top3 = test_metrics["top_confusions"][:3]
    a["A5-4"] = {"ok": False,
                  "top_confusions": [
                      {**item, "interpretation": confusion_interpretations.get(
                          (item["true"], item["predicted"]), "需结合具体错例进一步判断")}
                      for item in top3
                  ],
                  "interpretation_scope": "类别语义层面的合理性假设，不是经人工错例审读验证的结论；具体错例分析留待阶段6"}
    recall_min = config.THRESHOLDS["A5-5_per_class_recall_min"]
    a["A5-5"] = {"ok": all(value >= recall_min for value in small_recall.values()),
                  "threshold": recall_min, "per_class_recall": small_recall,
                  "below_threshold": {k: v for k, v in small_recall.items() if v < recall_min}}
    a["A5-6"] = {"ok": bool(pd_all_distribution and confirmed_label_distribution),
                  "model_prediction_distribution": pd_all_distribution,
                  "confirmed_llm_label_distribution": confirmed_label_distribution,
                  "banci_is_layout_metadata_not_ground_truth": True}
    reviewed_n = result["human_review"]["reviewed_count"]
    a["A5-7"] = {"ok": False, "oov_comparison": oov,
                  "attribution": {"tokenizer_mismatch": "现有 NFKC+jieba 分词与 PKU tokens 均可用，但未做同模型 tokenizer 配对实验",
                                  "vocabulary_drift": "有 OOV 差异证据：PD199801 3.97% vs L_test 2.81%（+1.15pp）",
                                  "llm_label_noise_bias": f"confirmed 为模型自评阈值标签；本次人工复核 {reviewed_n} 条，不足以估计总体噪声率",
                                  "model_capacity": "THUCNews L_test 达标而跨域 LLM 一致性较低；容量/域差异与标签噪声仍未因果分离"},
                  "preliminary_order_not_causal": ["词汇/时代领域漂移", "LLM 标签噪声或口径差异", "分词差异（未量化）", "模型容量（当前证据较弱）"],
                  "status": "给出暂定解释顺序；须通过 tokenizer 配对实验和人工错例抽样验证后才能满足完整归因要求"}
    a["A5-8"] = {"ok": False, "ablations": ablation,
                  "missing": [row["dimension"] for row in ablation if "未执行" in row["result"]]}
    a["A5-9"] = {"ok": result["isolation"]["pd1998_in_training_assertion"] is False
                  and result["isolation"]["training_tuning_paths_exclude_199801"]
                  and result["isolation"]["L_test_opened_once_by_this_script"],
                  **result["isolation"]}
    a["A5-10"] = {"ok": bool(result["model"]["params_sha256"] and result["started_at"]
                              and result["completed_at"]),
                   "params_sha256": result["model"]["params_sha256"]}
    result["all_A5_pass"] = all(item["ok"] for item in a.values())
    OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    samples_path = config.RESULT_METRICS / "stage5_manual_audit_samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as fh:
        for sample in audit_samples:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"阶段5指标 -> {OUT_PATH}", flush=True)
    print(f"待人工抽查样例 -> {samples_path}", flush=True)
    print(f"A5验收：{sum(item['ok'] for item in a.values())}/10 PASS；阶段6未启动", flush=True)
    return result


if __name__ == "__main__":
    main()
