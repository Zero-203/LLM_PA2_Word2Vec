# -*- coding: utf-8 -*-
"""导出不含本机绝对路径、样例正文和审查 ID 的公开汇总指标。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "result" / "metrics"


def load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select(source: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    return {key: source[key] for key in keys if key in source}


def metric_tree(source: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for key, value in source.items():
        if isinstance(value, dict):
            child = metric_tree(value)
            if child:
                result[key] = child
        elif isinstance(value, (int, float, bool)) or value is None:
            result[key] = value
    return result


def metric_only(source: Dict[str, Any]) -> Dict[str, Any]:
    return select(source, ("examples", "accuracy", "macro_f1", "weighted_f1"))


def numeric_counts(source: Dict[str, Any]) -> Dict[str, int]:
    return {key: value for key, value in source.items()
            if isinstance(value, int) and not isinstance(value, bool)}


def main() -> None:
    env = load(ROOT / "result" / "env_snapshot.json")
    w2v = load(METRICS / "w2v_eval_best.json")
    train_log = load(ROOT / "model" / "word2vec" / "best" / "train_log.json")
    evaluation = load(METRICS / "stage5_evaluation.json")
    ablations = load(METRICS / "stage5_ablations.json")
    scaling = load(METRICS / "e7_corpus_scaling.json")
    smoke = load(METRICS / "small_sample_smoke.json")
    fasttext_params = load(ROOT / "model" / "fasttext" / "S4_best_pretrained" / "params.json")

    track_a = evaluation["track_a"]
    track_b = evaluation["track_b"]
    monthly = {}
    for month, details in track_b["per_month"].items():
        month_metrics = details.get("metrics_against_unverified_llm_labels", {})
        monthly[month] = {
            **metric_only(month_metrics),
            "confirmed_ratio": details.get("confirmed_ratio"),
            "mean_confirmed_label_confidence": details.get("mean_confirmed_label_confidence"),
        }

    summary = {
        "schema": "pa2-public-summary-v1",
        "generated_by": "scripts/export_public_metrics.py",
        "privacy": {
            "absolute_local_paths_included": False,
            "raw_text_or_manual_audit_ids_included": False,
            "full_stage_metrics_are_local_only": True,
        },
        "environment": {
            "python_version": env.get("python_version"),
            "platform": env.get("platform"),
            "packages": env.get("packages", {}),
            "seed": env.get("seed"),
        },
        "word2vec": {
            "exp_id": w2v.get("exp_id"),
            "corpus_docs": train_log.get("corpus_docs"),
            "corpus_word_count": train_log.get("corpus_word_count"),
            "vocab_size": w2v.get("vocab_size"),
            "vector_size": w2v.get("vector_size"),
            "params": train_log.get("params", {}),
            "convergence": train_log.get("convergence", {}),
            "epoch_logs": train_log.get("epoch_logs", []),
            "training_seconds": train_log.get("total_sec"),
            "similarity": select(w2v.get("similarity", {}),
                                 ("n_pairs", "n_total", "n_skipped", "coverage", "spearman_rho", "p_value")),
            "analogy": select(w2v.get("analogy", {}),
                              ("n_total", "n_evaluated", "n_correct", "top1_accuracy", "n_skipped")),
            "verdict": w2v.get("verdict"),
        },
        "corpus_scaling": {
            "corpus_total_docs": scaling.get("corpus_total_docs"),
            "results": [select(row, ("exp_id", "corpus_docs", "vocab_size", "corpus_word_count",
                                     "last_loss", "relative_drop_pct", "total_sec"))
                        for row in scaling.get("results", [])],
        },
        "classifier": {
            "model_id": evaluation.get("model", {}).get("params", {}).get("exp_id", "S4_best_pretrained"),
            "params": fasttext_params.get("params", {}),
            "track_a": {
                **metric_only(track_a),
                "per_class": metric_tree(track_a.get("per_class", {})),
                "confusion_matrix_labels": track_a.get("confusion_matrix_labels", []),
                "confusion_matrix": track_a.get("confusion_matrix", []),
                "confusion_matrix_row_normalized": track_a.get("confusion_matrix_row_normalized", []),
                "top_confusions": track_a.get("top_confusions", []),
                "top_k_accuracy": select(track_a, ("top2_accuracy", "top3_accuracy", "mean_top1_confidence")),
                "test_data_read_count": evaluation.get("test_data_reads", {}).get("L_test.txt"),
            },
            "track_b": {
                "semantics": track_b.get("semantics"),
                "model": track_b.get("model"),
                "prompt_version": track_b.get("prompt_version"),
                "confidence_threshold": track_b.get("confidence_threshold"),
                "label_coverage": select(track_b.get("label_coverage", {}), (
                    "formatted_articles", "confirmed_labeled", "needs_review",
                    "confirmed_ratio", "needs_review_ratio",
                )),
                "confirmed_metrics_all_months": metric_only(track_b.get("confirmed_metrics_all_months", {})),
                "per_month": monthly,
                "confirmed_label_distribution": numeric_counts(track_b.get("confirmed_label_distribution", {})),
                "model_prediction_distribution_all_articles": numeric_counts(
                    track_b.get("model_prediction_distribution_all_articles", {})),
                "banci_distribution_as_layout_metadata_only": numeric_counts(
                    track_b.get("banci_distribution_as_layout_metadata_only", {})),
                "oov": select(track_b.get("all_article_oov_and_tokenizer_context", {}), (
                    "L_test_oov_rate", "PD199801_oov_rate", "PD199801_minus_L_test",
                    "PD1998_all_oov_rate",
                )),
                "isolation_assertions": select(evaluation.get("isolation", {}), (
                    "pd1998_in_training_assertion",
                    "training_tuning_paths_exclude_199801",
                    "L_test_opened_once_by_this_script",
                )),
            },
        },
        "ablations": [
            select(row, ("dimension", "condition", "reference", "val_examples", "val_accuracy",
                        "val_macro_f1", "val_weighted_f1", "delta_macro_f1_vs_reference",
                        "train_sec", "oov_rate", "metric_semantics"))
            for row in ablations.get("summary_rows", [])
        ],
        "small_sample_smoke": {
            "data_scope": smoke.get("data_scope"),
            "counts": smoke.get("counts", {}),
            "labels": smoke.get("labels", []),
            "word2vec_vocab_size": smoke.get("word2vec_vocab_size"),
            "fasttext_input_matrix_shape": smoke.get("fasttext_input_matrix_shape"),
            "pretrained_vector_injection_ok": smoke.get("pretrained_vector_injection_ok"),
            "validation": metric_only(smoke.get("validation", {})),
            "test": metric_only(smoke.get("test", {})),
            "parameters": smoke.get("parameters", {}),
        },
    }
    output = METRICS / "public_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Public metric summary written: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
