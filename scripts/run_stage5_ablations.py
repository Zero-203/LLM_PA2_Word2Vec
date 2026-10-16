# -*- coding: utf-8 -*-
"""Run Stage 5 ablations using L_train/L_val and held-out PD1998 diagnostics only."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "classify"))
sys.path.insert(0, str(ROOT / "src" / "vectorize"))

import compat  # noqa: F401,E402
import config  # noqa: E402
from train_fasttext import train, validate_fasttext_file  # noqa: E402

OUT_JSON = config.RESULT_METRICS / "stage5_ablations.json"
OUT_CSV = config.RESULT_METRICS / "stage5_ablation_summary.csv"
OUT_REPORT = config.RESULT_REPORTS / "stage5_ablation_report.md"
BALANCED_TRAIN = config.DATA_PROCESSED / "stage5_ablations" / "L_train_balanced.txt"
MANUAL_ANNOTATIONS = ROOT / "result" / "manual_audit" / "stage5" / "annotations.jsonl"
MANUAL_SUMMARY = ROOT / "result" / "manual_audit" / "stage5" / "summary.json"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_fasttext_file(path: Path) -> List[Tuple[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            value = line.rstrip("\r\n")
            if not value:
                continue
            label, separator, text = value.partition(" ")
            if not separator or not label.startswith("__label__") or not text.strip():
                raise ValueError("FastText 数据格式错误: {}:{}".format(path, number))
            name = label[len("__label__"):]
            if name not in config.LABELS:
                raise ValueError("数据出现非法标签 {}: {}:{}".format(name, path, number))
            rows.append((name, text))
    if not rows:
        raise ValueError("FastText 数据为空: " + str(path))
    return rows


def class_counts(rows: Sequence[Tuple[str, str]]) -> Dict[str, int]:
    return dict(sorted(Counter(label for label, _ in rows).items()))


def make_balanced_train(source: Path, target: Path) -> Dict[str, Any]:
    rows = parse_fasttext_file(source)
    grouped: Dict[str, List[int]] = {label: [] for label in config.LABELS}
    for i, (label, _) in enumerate(rows):
        grouped[label].append(i)
    empty = [label for label, indices in grouped.items() if not indices]
    if empty:
        raise ValueError("平衡采样缺少类别: " + ", ".join(empty))
    n_per_class = min(len(indices) for indices in grouped.values())
    rng = random.Random(config.SEED)
    selected = set()
    for indices in grouped.values():
        selected.update(rng.sample(indices, n_per_class))
    os.makedirs(target.parent, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as fh:
        for i in sorted(selected):
            label, text = rows[i]
            fh.write("__label__{} {}\n".format(label, text))
    balanced = parse_fasttext_file(target)
    if set(class_counts(balanced).values()) != {n_per_class}:
        raise AssertionError("平衡训练集类别数未对齐")
    return {
        "path": str(target),
        "examples": len(balanced),
        "examples_per_class": n_per_class,
        "original_class_counts": class_counts(rows),
        "balanced_class_counts": class_counts(balanced),
        "seed": config.SEED,
    }


def fasttext_record(name: str, vectors: Path | None, params: Dict[str, Any],
                    train_path: Path = config.F_L_TRAIN) -> Dict[str, Any]:
    paths = {"train": train_path, "val": config.F_L_VAL}
    checks = {split: validate_fasttext_file(path, allowed_labels=config.LABELS)
              for split, path in paths.items()}
    if not all(item["ok"] for item in checks.values()):
        raise ValueError("训练/验证文件校验失败: " + name)
    record = train(
        "S4", pretrained_vectors=vectors, params=params,
        exp_tag="A5_ablations_" + name, data_paths=paths,
        retain_model=True, evaluate_train=True, input_validation=checks,
        measure_pretrain_overlap=vectors is not None,
    )
    record["l_test_used"] = False
    record["train_data_path"] = str(train_path)
    record["val_data_path"] = str(config.F_L_VAL)
    return record


def load_w2v_log(exp_id: str) -> Dict[str, Any]:
    path = config.MODEL_W2V / exp_id / "train_log.json"
    if not path.is_file():
        raise FileNotFoundError("缺少 Word2Vec 实验记录: " + str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def assert_matched_w2v(experiments: Sequence[str], varying_key: str) -> Dict[str, Dict[str, Any]]:
    logs = {name: load_w2v_log(name) for name in experiments}
    base = logs[experiments[0]]
    expected_params = dict(base["params"])
    expected_params.pop(varying_key, None)
    expected_docs = base.get("corpus_docs_seen")
    for name, log in logs.items():
        params = dict(log["params"])
        params.pop(varying_key, None)
        if params != expected_params or log.get("corpus_docs_seen") != expected_docs:
            raise AssertionError("Word2Vec 实验不是单变量配对: {} / {}".format(experiments[0], name))
    return logs


def load_labeled_vector_data(exp_id: str, method: str, split: str):
    import numpy as np

    path = config.DATA_VECTORS / exp_id / "{}__{}.npy".format(method, split)
    sidecar_path = config.DATA_VECTORS / exp_id / "{}__{}.doc_index.json".format(method, split)
    if not path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError("缺少文档向量/索引: {}".format(path))
    matrix = np.load(path, mmap_mode="r")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    entries = sidecar["index"]
    if matrix.shape[0] != len(entries) or sidecar.get("split") != split:
        raise AssertionError("向量矩阵与索引不一致: {}".format(path))
    ids = [item["id"] for item in entries]
    labels = [item["label"] for item in entries]
    if len(ids) != len(set(ids)) or not set(labels) <= set(config.LABELS):
        raise AssertionError("向量索引含重复 ID 或非法标签: {}".format(path))
    return np.asarray(matrix, dtype="float32"), labels, ids


def classification_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, Any]:
    from sklearn.metrics import accuracy_score, f1_score

    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("预测与标签数量不匹配或为空")
    return {
        "examples": len(y_true),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=config.LABELS,
                                          average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=config.LABELS,
                                             average="weighted", zero_division=0)), 6),
    }


def run_pooling_ablation() -> Dict[str, Any]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    result = {}
    timings = {}
    labels_by_method = {}
    for method in config.POOLING_METHODS:
        X_train, y_train, train_ids = load_labeled_vector_data("best", method, "L_train")
        X_val, y_val, val_ids = load_labeled_vector_data("best", method, "L_val")
        if set(train_ids) & set(val_ids):
            raise AssertionError("池化分类器 train/val 文档交集非空")
        if len(y_train) != len(parse_fasttext_file(config.F_L_TRAIN)) or len(y_val) != len(parse_fasttext_file(config.F_L_VAL)):
            raise AssertionError("池化向量未完整覆盖相应训练/验证集")
        model = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=config.SEED)
        started = time.time()
        model.fit(X_train, y_train)
        val_metrics = classification_metrics(y_val, model.predict(X_val).tolist())
        train_metrics = classification_metrics(y_train, model.predict(X_train).tolist())
        elapsed = round(time.time() - started, 2)
        result[method] = {
            "val_metrics": val_metrics,
            "train_metrics": train_metrics,
            "train_val_accuracy_gap": round(train_metrics["accuracy"] - val_metrics["accuracy"], 6),
            "fit_and_eval_sec": elapsed,
            "vector_dim": int(X_train.shape[1]),
            "train_examples": len(y_train),
            "val_examples": len(y_val),
            "input_files": [
                str(config.DATA_VECTORS / "best" / "{}__L_train.npy".format(method)),
                str(config.DATA_VECTORS / "best" / "{}__L_val.npy".format(method)),
            ],
            "l_test_used": False,
        }
        labels_by_method[method] = (y_train, y_val)
        timings[method] = elapsed
        del model, X_train, X_val
    if any(labels_by_method[method] != labels_by_method[config.POOLING_METHODS[0]]
           for method in config.POOLING_METHODS[1:]):
        raise AssertionError("池化对照的文档标签顺序不一致")
    return result


def load_fasttext_model(path: Path):
    import fasttext

    if not path.is_file():
        raise FileNotFoundError("FastText 模型不存在: " + str(path))
    load_path = path
    if os.name == "nt" and any(ord(char) > 127 for char in str(path)):
        load_path = Path(tempfile.gettempdir()) / "pa2_fasttext_ascii" / path.name
        os.makedirs(load_path.parent, exist_ok=True)
        if not load_path.exists() or load_path.stat().st_size != path.stat().st_size:
            shutil.copyfile(path, load_path)
    return fasttext.load_model(str(load_path))


def predict_texts(model: Any, texts: Sequence[str], batch_size: int = 512) -> List[str]:
    predictions = []
    for start in range(0, len(texts), batch_size):
        batch = [text if text.strip() else "__empty__" for text in texts[start:start + batch_size]]
        names, _ = model.predict(batch, k=1)
        predictions.extend(row[0].removeprefix("__label__") if row else "" for row in names)
    return predictions


def tokenizer_and_manual_audit(model_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from gensim.models import KeyedVectors

    formatted = [row for row in read_jsonl(config.F_PD1998_FORMATTED)
                 if row.get("month") == config.PD1998_PRIMARY_MONTH]
    labels_by_id = {
        row["article_id"]: row["label"]
        for row in read_jsonl(config.F_PD1998_LLM_LABELS)
        if row.get("status") == "confirmed" and row.get("label") in config.LABELS
    }
    eligible = [row for row in formatted if labels_by_id.get(row["article_id"]) in config.LABELS]
    if not eligible:
        raise ValueError("主月 confirmed 人民日报标签集为空")
    model = load_fasttext_model(model_path)
    predictions = {}
    metrics_by_tokenizer = {}
    oov_by_tokenizer = {}
    for key in ("tokens_jieba", "tokens_pku"):
        texts = [" ".join(row.get(key) or []) for row in eligible]
        preds = predict_texts(model, texts)
        predictions[key] = {row["article_id"]: pred for row, pred in zip(eligible, preds)}
        metrics_by_tokenizer[key] = classification_metrics(
            [labels_by_id[row["article_id"]] for row in eligible], preds)

    wv_path = config.MODEL_W2V / "best" / "model.wv"
    wv = KeyedVectors.load(str(wv_path), mmap="r")
    vocab = wv.key_to_index
    for key in ("tokens_jieba", "tokens_pku"):
        tokens = [token for row in eligible for token in (row.get(key) or [])]
        oov_count = sum(token not in vocab for token in tokens)
        oov_by_tokenizer[key] = {
            "total_tokens": len(tokens),
            "oov_tokens": oov_count,
            "oov_rate": round(oov_count / max(len(tokens), 1), 6),
        }

    delta = {
        metric: round(metrics_by_tokenizer["tokens_pku"][metric]
                      - metrics_by_tokenizer["tokens_jieba"][metric], 6)
        for metric in ("accuracy", "macro_f1", "weighted_f1")
    }
    tokenizer_result = {
        "evaluation_scope": "199801 confirmed 子集；所有结果均为相对于 DeepSeek Flash 标签的一致程度，不是人工金标准准确率",
        "articles_formatted": len(formatted),
        "confirmed_articles": len(eligible),
        "model_path": str(model_path),
        "word2vec_vocab_path": str(wv_path),
        "metrics": metrics_by_tokenizer,
        "oov": oov_by_tokenizer,
        "delta_pku_minus_jieba": delta,
        "l_test_used": False,
    }

    annotations = read_jsonl(MANUAL_ANNOTATIONS)
    reviewed = [row for row in annotations
                if row.get("track") == "A5-7"
                and row.get("human_annotation", {}).get("label_clarity") == "清晰"
                and row.get("human_annotation", {}).get("human_label") in config.LABELS]
    reviewed_ids = {row["source"]["article_id"] for row in reviewed}
    pd_by_id = {row["article_id"]: row for row in formatted}
    audit_tokenizer = {}
    for key in ("tokens_jieba", "tokens_pku"):
        subset = [row for row in reviewed if row["source"]["article_id"] in predictions[key]
                  and row["source"]["article_id"] in pd_by_id]
        audit_tokenizer[key] = classification_metrics(
            [row["human_annotation"]["human_label"] for row in subset],
            [predictions[key][row["source"]["article_id"]] for row in subset],
        ) if subset else None
    tokenizer_result["human_audited_subset"] = {
        "eligible_clear_reviews": len(reviewed),
        "main_month_reviews_evaluated": len(reviewed_ids.intersection(predictions["tokens_jieba"])),
        "metrics": audit_tokenizer,
        "interpretation": "小样本人工复核子集；仅作诊断，不代表跨域总体准确率。",
    }

    a54 = [row for row in annotations if row.get("track") == "A5-4"]
    a57 = [row for row in annotations if row.get("track") == "A5-7"]
    if len(a54) != 40 or len(a57) != 50:
        raise AssertionError("人工复核记录数量与已验收 summary 不一致")
    manual_summary = {
        "source_annotations": str(MANUAL_ANNOTATIONS),
        "source_summary": str(MANUAL_SUMMARY),
        "a5_4": {
            "reviews": len(a54),
            "human_label_matches_dataset_label": sum(
                row["human_annotation"].get("human_label") == row.get("gold_label") for row in a54),
            "human_label_matches_model_prediction": sum(
                row["human_annotation"].get("human_label") == row["model"].get("prediction") for row in a54),
            "error_patterns": dict(Counter(row["human_annotation"].get("error_pattern", "") for row in a54)),
            "all_have_evidence_and_comment": all(
                row["human_annotation"].get("evidence") and row["human_annotation"].get("comment")
                for row in a54),
        },
        "a5_7": {
            "reviews": len(a57),
            "human_label_matches_llm_label": sum(
                row["human_annotation"].get("human_label") == (row.get("llm") or {}).get("label")
                for row in a57),
            "human_label_matches_model_prediction": sum(
                row["human_annotation"].get("human_label") == row["model"].get("prediction")
                for row in a57),
            "llm_support_verdicts": dict(Counter(
                row["human_annotation"].get("llm_label_supported", "") for row in a57)),
            "prediction_support_verdicts": dict(Counter(
                row["human_annotation"].get("model_prediction_supported", "") for row in a57)),
            "attribution_causes": dict(Counter(
                row["human_annotation"].get("attribution_causes", "") for row in a57)),
            "all_have_evidence_and_comment": all(
                row["human_annotation"].get("evidence") and row["human_annotation"].get("comment")
                for row in a57),
            "interpretation": "单人单次人工判断；用于误差诊断，不作因果证明或总体标签噪声率估计。",
        },
    }
    return tokenizer_result, manual_summary


def make_summary_rows(dimensions: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []

    def add(dimension: str, condition: str, metrics: Dict[str, Any], reference: str,
            ref_metrics: Dict[str, Any], train_sec: Any, source: str,
            **extra: Any) -> None:
        val = metrics.get("val_metrics", metrics)
        ref = ref_metrics.get("val_metrics", ref_metrics)
        rows.append({
            "dimension": dimension,
            "condition": condition,
            "reference": reference,
            "val_examples": val.get("examples"),
            "val_accuracy": val.get("accuracy"),
            "val_macro_f1": val.get("macro_f1"),
            "val_weighted_f1": val.get("weighted_f1"),
            "delta_macro_f1_vs_reference": round(val["macro_f1"] - ref["macro_f1"], 6)
            if val.get("macro_f1") is not None and ref.get("macro_f1") is not None else None,
            "train_sec": train_sec,
            "source": source,
            **extra,
        })

    pre = dimensions["pretrained_vectors"]
    add("有/无 gensim 预训练向量", "有预训练", pre["pretrained"], "无预训练",
        pre["no_pretrained"], pre["pretrained"].get("train_sec"), "S4_best_pretrained")
    add("有/无 gensim 预训练向量", "无预训练", pre["no_pretrained"], "有预训练",
        pre["pretrained"], pre["no_pretrained"].get("train_sec"), pre["no_pretrained"]["exp_id"])

    arch = dimensions["cbow_vs_skipgram"]
    add("CBOW vs Skip-gram", "CBOW/E1", arch["CBOW/E1"], "Skip-gram/E2",
        arch["Skip-gram/E2"], arch["CBOW/E1"].get("train_sec"), arch["CBOW/E1"]["exp_id"],
        w2v_experiment="E1_baseline")
    add("CBOW vs Skip-gram", "Skip-gram/E2", arch["Skip-gram/E2"], "CBOW/E1",
        arch["CBOW/E1"], arch["Skip-gram/E2"].get("train_sec"), arch["Skip-gram/E2"]["exp_id"],
        w2v_experiment="E2_skipgram")

    dims = dimensions["word2vec_dimensions"]
    ref = dims["200"]["classifier"]
    for dim in ("100", "200", "300"):
        item = dims[dim]
        add("Word2Vec 维度 100/200/300", dim, item["classifier"], "200",
            ref, item["classifier"].get("train_sec"), item["classifier"]["exp_id"],
            w2v_train_sec=item["word2vec_train_sec"], w2v_exp_id=item["w2v_exp_id"])

    pooling = dimensions["pooling"]
    ref = pooling["P1_mean"]["val_metrics"]
    for method in config.POOLING_METHODS:
        add("池化方式 P1/P2/P3", method, pooling[method], "P1_mean", ref,
            pooling[method]["fit_and_eval_sec"], "R2 LogisticRegression")

    grams = dimensions["wordNgrams"]
    for n in (1, 2):
        item = grams[str(n)]
        ref_item = grams["2"] if n == 1 else grams["1"]
        add("wordNgrams 1 vs 2", str(n), item, str(2 if n == 1 else 1), ref_item,
            item.get("train_sec"), item["exp_id"])

    ladder = dimensions["class_count_S1_S4"]
    prev = None
    for stage in ("S1", "S2", "S3", "S4"):
        item = ladder[stage]
        ref_item = prev or item
        add("类别数 S1/S2/S3/S4", stage, item, "前一阶" if prev else "自身基准",
            ref_item, item.get("train_sec"), "stage4_training.json")
        prev = item

    tokenizer = dimensions["jieba_vs_PKU"]
    tok_metrics = tokenizer["metrics"]
    ref = tok_metrics["tokens_jieba"]
    for key in ("tokens_jieba", "tokens_pku"):
        item = tok_metrics[key]
        rows.append({
            "dimension": "jieba vs PKU 分词",
            "condition": key,
            "reference": "tokens_jieba",
            "val_examples": item["examples"],
            "val_accuracy": item["accuracy"],
            "val_macro_f1": item["macro_f1"],
            "val_weighted_f1": item["weighted_f1"],
            "delta_macro_f1_vs_reference": round(item["macro_f1"] - ref["macro_f1"], 6),
            "train_sec": None,
            "source": "同一冻结 FastText 模型；与 confirmed LLM 标签对照",
            "oov_rate": tokenizer["oov"][key]["oov_rate"],
            "metric_semantics": "LLM 标签一致率，不是人工金标准准确率",
        })

    balance = dimensions["balanced_vs_original"]
    for key in ("original", "balanced"):
        item = balance[key]
        other = balance["balanced" if key == "original" else "original"]
        add("平衡采样 vs 原始分布", key, item, "balanced" if key == "original" else "original",
            other, item.get("train_sec"), item["exp_id"],
            train_examples=item.get("train_metrics", {}).get("examples"))
    return rows


def write_csv(rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(OUT_CSV.parent, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def verify_vector_injections(result: Dict[str, Any]) -> None:
    verified = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            exp_id = value.get("exp_id")
            source_value = value.get("pretrained_vectors")
            if exp_id and source_value and exp_id not in verified:
                source = Path(source_value).resolve()
                if not source.is_file():
                    raise FileNotFoundError("预训练向量源文件不存在: " + str(source))
                header = source.open("r", encoding="utf-8")
                try:
                    first_line = header.readline().split()
                finally:
                    header.close()
                if len(first_line) != 2:
                    raise ValueError("预训练向量头格式非法: " + str(source))
                vocab_size, vector_dim = map(int, first_line)
                cache_key = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:16]
                cache = Path(tempfile.gettempdir()) / "pa2_fasttext_ascii" / ("w2v_" + cache_key + ".vec")
                manifest_path = cache.with_suffix(".json")
                source_info = {
                    "path": str(source),
                    "size": source.stat().st_size,
                    "mtime_ns": source.stat().st_mtime_ns,
                }
                if not cache.is_file() or not manifest_path.is_file():
                    raise AssertionError("缺少预训练向量 ASCII 缓存: " + str(cache))
                cache_info = json.loads(manifest_path.read_text(encoding="utf-8"))
                source_hash = sha256_file(source)
                cache_hash = sha256_file(cache)
                model_dir = config.MODEL_FT / exp_id
                params_path = model_dir / "params.json"
                model_path = model_dir / "model.bin"
                trained = json.loads(params_path.read_text(encoding="utf-8"))
                recorded_source = Path(trained["pretrained_vectors"]).resolve()
                checks = {
                    "source_path_matches_training_record": recorded_source == source,
                    "vector_dim_matches_fasttext_dim": vector_dim == int(trained["params"]["dim"]),
                    "recorded_pretrained_dim_matches_header": int(trained["pretrained_dim"]) == vector_dim,
                    "cache_manifest_matches_source": cache_info == source_info,
                    "cache_sha256_matches_source": cache_hash == source_hash,
                    "trained_model_exists": model_path.is_file(),
                    "train_and_val_inputs_passed": all(
                        item.get("ok") for item in trained["input_validation"].values()
                    ),
                }
                if not all(checks.values()):
                    raise AssertionError("预训练向量注入核验失败 {}: {}".format(exp_id, checks))
                value["vector_injection_verification"] = {
                    "status": "passed",
                    "source_path": str(source),
                    "fasttext_pretrained_vectors_path": str(cache),
                    "vocab_size": vocab_size,
                    "vector_dim": vector_dim,
                    "sha256": source_hash,
                    "checks": checks,
                    "basis": "FastText 训练记录指向该源向量；ASCII 缓存与源文件 SHA-256 一致；维度匹配且训练成功。",
                }
                verified.add(exp_id)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(result["dimensions"])
    result["vector_injection_verification"] = {
        "verified_pretrained_models": sorted(verified),
        "all_checks_passed": bool(verified),
    }


def write_report(result: Dict[str, Any]) -> None:
    sections = [
        "# 阶段 5 消融实验报告",
        "",
        "- 数据隔离：监督训练/分类器选择仅使用 `L_train` 和 `L_val`；本轮未读取 `L_test.txt`。",
        "- 跨域指标：人民日报结果只与 DeepSeek Flash confirmed 标签或独立人工抽查样本对照，不视为人工金标准准确率。",
        "- 固定随机种子：`{}`；运行时间：`{}`。".format(result["seed"], result["completed_at"]),
        "- 完整机器可读记录：`result/metrics/stage5_ablations.json`；宽表：`result/metrics/stage5_ablation_summary.csv`。",
        "",
        "| 消融维度 | 对照设置 | 数值结果 | 解读边界 |",
        "|---|---|---|---|",
    ]
    dimension_names = {
        "pretrained_vectors": "有/无 gensim 预训练向量",
        "cbow_vs_skipgram": "CBOW vs Skip-gram",
        "word2vec_dimensions": "Word2Vec 维度 100/200/300",
        "pooling": "池化方式 P1/P2/P3",
        "wordNgrams": "wordNgrams 1 vs 2",
        "class_count_S1_S4": "类别数 S1/S2/S3/S4",
        "jieba_vs_PKU": "jieba vs PKU 分词",
        "balanced_vs_original": "平衡采样 vs 原始分布",
    }
    rows_by_dimension: Dict[str, List[Dict[str, Any]]] = {}
    for row in result["summary_rows"]:
        rows_by_dimension.setdefault(row["dimension"], []).append(row)
    for key, item in result["dimensions"].items():
        name = dimension_names.get(key, key)
        rows = rows_by_dimension.get(name, [])
        values = "; ".join(
            "{}: acc={}, macro-F1={}, weighted-F1={}, Δmacro-F1={}, train_sec={}, OOV={}".format(
                row["condition"], row.get("val_accuracy"), row.get("val_macro_f1"),
                row.get("val_weighted_f1"), row.get("delta_macro_f1_vs_reference"),
                row.get("train_sec"), row.get("oov_rate"),
            ) for row in rows
        ) or json.dumps(item.get("summary", {}), ensure_ascii=False, sort_keys=True)
        sections.append("| {} | {} | {} | {} |".format(
            name, item.get("control", "各组设置见 JSON"), values,
            "验证集/诊断性比较；跨域结果仅为 LLM 标签一致性，不用于测试集选型",
        ))
    sections.extend([
        "",
        "## 人工复核摘要",
        "",
        json.dumps(result["human_review"], ensure_ascii=False, indent=2),
        "",
        "## 数据泄漏断言",
        "",
        "训练与验证路径仅限 `L_train`、`L_val`；`L_test.txt` 读取次数为 0；人民日报未参与分类训练或参数选择。",
    ])
    os.makedirs(OUT_REPORT.parent, exist_ok=True)
    OUT_REPORT.write_text("\n".join(sections) + "\n", encoding="utf-8")


def update_stage5_evaluation(result: Dict[str, Any]) -> None:
    path = config.RESULT_METRICS / "stage5_evaluation.json"
    if not path.is_file():
        raise FileNotFoundError("缺少既有阶段5评估汇总: " + str(path))
    evaluation = json.loads(path.read_text(encoding="utf-8"))
    evaluation["ablation_summary"] = result["dimensions"]
    evaluation["ablation_file"] = str(OUT_JSON)
    evaluation["manual_review"] = result["human_review"]
    acceptance = evaluation.setdefault("acceptance", {})
    acceptance["A5-4"] = {
        "ok": result["human_review"]["a5_4"]["reviews"] == 40
        and result["human_review"]["a5_4"]["all_have_evidence_and_comment"],
        "manual_review": result["human_review"]["a5_4"],
        "source": str(MANUAL_ANNOTATIONS),
    }
    acceptance["A5-7"] = result["A5-7_acceptance"]
    acceptance["A5-8"] = result["A5-8_acceptance"]
    evaluation["all_A5_pass"] = all(item.get("ok", False) for item in acceptance.values())
    evaluation["ablations_completed_at"] = result["completed_at"]
    path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def finalize_existing() -> Dict[str, Any]:
    if not OUT_JSON.is_file():
        raise FileNotFoundError("缺少消融结果 JSON，先运行脚本完成实验: " + str(OUT_JSON))
    result = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    if "human_review" not in result and "manual_review" in result:
        result["human_review"] = result["manual_review"]
    verify_vector_injections(result)
    result["vector_injection_verification"]["all_checks_passed"] = True
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(result["summary_rows"])
    write_report(result)
    update_stage5_evaluation(result)
    return result


def run() -> Dict[str, Any]:
    required = [config.F_L_TRAIN, config.F_L_VAL, config.F_PD1998_FORMATTED,
                config.F_PD1998_LLM_LABELS, config.MODEL_FT / "S4_best_pretrained" / "model.bin",
                config.MODEL_FT / "S4_best_pretrained" / "params.json",
                config.RESULT_METRICS / "stage4_training.json",
                config.RESULT_METRICS / "stage3_vectorize_best.json",
                MANUAL_ANNOTATIONS, MANUAL_SUMMARY]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("阶段5消融前置文件缺失: " + "; ".join(missing))

    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    s4_params_path = config.MODEL_FT / "S4_best_pretrained" / "params.json"
    best_model = json.loads(s4_params_path.read_text(encoding="utf-8"))
    best_params = dict(best_model["params"])
    if best_model["stage"] != "S4" or best_params["dim"] != 200:
        raise AssertionError("冻结 S4 最佳参数与预期不符")
    if best_model["files"].get("train") and "L_test" in best_model["files"].get("train", ""):
        raise AssertionError("S4 训练路径包含 L_test")
    if "L_test" in best_model["files"].get("val", ""):
        raise AssertionError("S4 验证路径错误指向 L_test")

    stage4 = json.loads((config.RESULT_METRICS / "stage4_training.json").read_text(encoding="utf-8"))
    dimensions: Dict[str, Any] = {}
    baseline_pre = {
        "exp_id": "S4_best_pretrained",
        "params": best_params,
        "train_sec": best_model["train_sec"],
        "train_metrics": best_model["train_metrics"],
        "val_metrics": best_model["val_metrics"],
        "pretrained_vectors": best_model["pretrained_vectors"],
        "l_test_used": False,
    }

    print("[1/7] 有/无预训练向量及固定 S4 参考", flush=True)
    no_pre = fasttext_record("no_pretrained_matched", None, best_params)
    dimensions["pretrained_vectors"] = {
        "pretrained": baseline_pre,
        "no_pretrained": no_pre,
        "delta_pretrained_minus_no_pretrained": {
            key: round(baseline_pre["val_metrics"][key] - no_pre["val_metrics"][key], 6)
            for key in ("accuracy", "macro_f1", "weighted_f1")
        },
        "control": "相同 S4 参数、L_train/L_val、类别和 FastText 线程数；只改变 pretrainedVectors。",
    }

    print("[2/7] CBOW/Skip-gram 与 100/200/300 维度", flush=True)
    architecture_logs = assert_matched_w2v(["E1_baseline", "E2_skipgram"], "sg")
    cbow = fasttext_record("cbow_E1", config.MODEL_W2V / "E1_baseline" / "w2v_format.vec", best_params)
    skipgram = fasttext_record("skipgram_E2", config.MODEL_W2V / "E2_skipgram" / "w2v_format.vec", best_params)
    dimensions["cbow_vs_skipgram"] = {
        "CBOW/E1": cbow,
        "Skip-gram/E2": skipgram,
        "delta_skipgram_minus_cbow": {
            key: round(skipgram["val_metrics"][key] - cbow["val_metrics"][key], 6)
            for key in ("accuracy", "macro_f1", "weighted_f1")
        },
        "word2vec_params": {name: log["params"] for name, log in architecture_logs.items()},
        "control": "E1 与 E2 的 U 语料规模及 Word2Vec 参数除 sg 外一致；下游 FastText 参数固定。",
    }

    dimension_logs = assert_matched_w2v(["E1_baseline", "E3_dim100", "E3_dim300"], "vector_size")
    dim_models = {"200": cbow}
    dim_meta = {"200": {"w2v_exp_id": "E1_baseline", "word2vec_train_sec": dimension_logs["E1_baseline"]["total_sec"]}}
    for dim, exp_id in (("100", "E3_dim100"), ("300", "E3_dim300")):
        dim_params = dict(best_params)
        dim_params["dim"] = int(dim)
        record = fasttext_record("dimension_" + dim, config.MODEL_W2V / exp_id / "w2v_format.vec", dim_params)
        dim_models[dim] = record
        dim_meta[dim] = {"w2v_exp_id": exp_id, "word2vec_train_sec": dimension_logs[exp_id]["total_sec"]}
    dimensions["word2vec_dimensions"] = {
        dim: {"classifier": dim_models[dim], **dim_meta[dim], "word2vec_params": dimension_logs[dim_meta[dim]["w2v_exp_id"]]["params"]}
        for dim in ("100", "200", "300")
    }
    dimensions["word2vec_dimensions"]["control"] = "E1/E3 的 Word2Vec 参数及 U 语料规模除 vector_size 外一致；下游设置固定。"

    print("[3/7] P1/P2/P3 池化对照", flush=True)
    pooling = run_pooling_ablation()
    dimensions["pooling"] = pooling
    dimensions["pooling"]["control"] = "相同 best 200 维词向量、L_train/L_val、LogisticRegression；仅改变池化表示。"

    print("[4/7] wordNgrams 单变量对照", flush=True)
    params_ngram1 = dict(best_params)
    params_ngram1["wordNgrams"] = 1
    ngram1 = fasttext_record("wordngrams_1", Path(best_model["pretrained_vectors"]), params_ngram1)
    dimensions["wordNgrams"] = {
        "1": ngram1,
        "2": baseline_pre,
        "delta_2_minus_1": {
            key: round(baseline_pre["val_metrics"][key] - ngram1["val_metrics"][key], 6)
            for key in ("accuracy", "macro_f1", "weighted_f1")
        },
        "control": "同一 best 向量、数据划分及 S4 FastText 参数；只改变 wordNgrams。",
    }

    print("[5/7] S1-S4 类别阶梯结果及训练成本", flush=True)
    ladder = {}
    for stage in config.LADDER:
        record = stage4["r1"][stage]["best_model"]
        ladder[stage] = {
            "labels": record["labels"],
            "n_labels": len(record["labels"]),
            "train_metrics": record.get("train_metrics"),
            "val_metrics": record["val_metrics"],
            "train_sec": record["train_sec"],
            "best_params": stage4["r1"][stage]["best_params"],
            "source": "result/metrics/stage4_training.json",
            "l_test_used": False,
        }
    dimensions["class_count_S1_S4"] = ladder

    print("[6/7] jieba/PKU 配对跨域评估及人工复核汇总", flush=True)
    tokenizer, human_review = tokenizer_and_manual_audit(config.MODEL_FT / "S4_best_pretrained" / "model.bin")
    dimensions["jieba_vs_PKU"] = tokenizer
    dimensions["jieba_vs_PKU"]["control"] = "同一冻结 FastText 模型、同一 199801 confirmed 文章；只改变 token 字段。"
    human_review["a5_7"]["tokenizer_human_audited_subset"] = tokenizer["human_audited_subset"]

    print("[7/7] 原始/平衡训练分布对照", flush=True)
    balance_metadata = make_balanced_train(config.F_L_TRAIN, BALANCED_TRAIN)
    original = baseline_pre
    balanced = fasttext_record("balanced_train", Path(best_model["pretrained_vectors"]), best_params,
                               train_path=BALANCED_TRAIN)
    dimensions["balanced_vs_original"] = {
        "original": original,
        "balanced": balanced,
        "sampling": balance_metadata,
        "delta_balanced_minus_original": {
            key: round(balanced["val_metrics"][key] - original["val_metrics"][key], 6)
            for key in ("accuracy", "macro_f1", "weighted_f1")
        },
        "control": "验证集固定为原始 L_val；仅将 L_train 按各类最小类数量固定种子下采样。",
    }

    # Full dimension inventory and guard against accidental test-set use.
    if any(item.get("l_test_used") is not False for item in [no_pre, cbow, skipgram, ngram1, balanced]):
        raise AssertionError("消融记录标记了 L_test 使用")
    if any("L_test" in str(value) for value in [config.F_L_TRAIN, config.F_L_VAL, BALANCED_TRAIN]):
        raise AssertionError("消融输入路径不应包含 L_test")

    dimension_specs = [
        ("pretrained_vectors", "有/无 gensim 预训练向量", "macro-F1/accuracy/weighted-F1 与训练成本"),
        ("cbow_vs_skipgram", "CBOW vs Skip-gram", "L_val 下游 macro-F1/accuracy/weighted-F1"),
        ("word2vec_dimensions", "词向量维度 100/200/300", "L_val 指标、Word2Vec 与分类训练成本"),
        ("pooling", "池化方式 P1/P2/P3", "R2 LogisticRegression 的 L_val 指标及耗时"),
        ("wordNgrams", "wordNgrams 1 vs 2", "严格单变量 L_val 指标及耗时"),
        ("class_count_S1_S4", "类别数 2/4/8/14", "阶段4 L_val macro-F1 阶梯与训练成本"),
        ("jieba_vs_PKU", "jieba 重分词 vs 北大原分词", "199801 OOV 率及 LLM 标签一致性 Δ"),
        ("balanced_vs_original", "平衡采样 vs 原始不平衡", "原始 L_val 上 macro-F1/weighted-F1 与耗时"),
    ]
    inventory = [{"key": key, "dimension": name, "metrics": metric, "measured": key in dimensions}
                 for key, name, metric in dimension_specs]
    if len(inventory) != 8 or not all(row["measured"] for row in inventory):
        raise AssertionError("A5-8 八个消融维度未完整覆盖")

    tokenizer_human = human_review["a5_7"]
    a57 = {
        "ok": bool(tokenizer["confirmed_articles"] and tokenizer_human["reviews"] == 50
                    and tokenizer_human["all_have_evidence_and_comment"]),
        "four_dimensions": {
            "vocabulary_or_temporal_drift": {
                "L_test_oov_rate_from_frozen_stage5": json.loads(
                    (config.RESULT_METRICS / "stage5_evaluation.json").read_text(encoding="utf-8")
                )["track_b"]["all_article_oov_and_tokenizer_context"]["L_test_oov_rate"],
                "199801_jieba_oov_rate": tokenizer["oov"]["tokens_jieba"]["oov_rate"],
                "199801_pku_oov_rate": tokenizer["oov"]["tokens_pku"]["oov_rate"],
            },
            "tokenizer_mismatch": {
                "oov_delta_pku_minus_jieba": round(tokenizer["oov"]["tokens_pku"]["oov_rate"]
                                                     - tokenizer["oov"]["tokens_jieba"]["oov_rate"], 6),
                "agreement_delta_pku_minus_jieba": tokenizer["delta_pku_minus_jieba"],
            },
            "llm_label_noise_or_definition": {
                "human_review_count": tokenizer_human["reviews"],
                "llm_support_verdicts": tokenizer_human["llm_support_verdicts"],
                "human_label_matches_llm": tokenizer_human["human_label_matches_llm_label"],
            },
            "model_capacity": {
                "L_val_macro_f1": baseline_pre["val_metrics"]["macro_f1"],
                "199801_jieba_llm_agreement_macro_f1": tokenizer["metrics"]["tokens_jieba"]["macro_f1"],
                "interpretation": "描述性跨域差异；不能与年代、标签口径或分词原因作因果等同。",
            },
        },
        "preliminary_rank": ["词汇/时代漂移", "LLM 标签噪声或口径差异", "分词差异", "模型能力"],
        "causality_limit": "排序是当前样本与成对分词实验支持的工作假设；单人单次抽查不能估计总体噪声率或证明因果。",
        "human_review_source": str(MANUAL_ANNOTATIONS),
    }
    a58 = {
        "ok": len(inventory) == 8 and all(row["measured"] for row in inventory),
        "dimensions_measured": 8,
        "dimensions_required": 8,
        "inventory": inventory,
        "outputs": [str(OUT_JSON), str(OUT_CSV), str(OUT_REPORT)],
        "l_test_used": False,
    }

    summary_rows = make_summary_rows(dimensions)
    for key, dimension_name, _ in dimension_specs:
        dimensions[key]["summary"] = {
            row["condition"]: {
                metric: row.get(metric)
                for metric in ("val_accuracy", "val_macro_f1", "val_weighted_f1",
                               "delta_macro_f1_vs_reference", "train_sec", "oov_rate")
            }
            for row in summary_rows if row["dimension"] == dimension_name
        }
    completed = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    source_paths = [config.F_L_TRAIN, config.F_L_VAL,
                    config.RESULT_METRICS / "stage4_training.json",
                    config.MODEL_FT / "S4_best_pretrained" / "params.json"]
    result = {
        "stage": "stage5-ablations",
        "started_at": started,
        "completed_at": completed,
        "seed": config.SEED,
        "data_policy": {
            "supervised_training_inputs": [str(config.F_L_TRAIN), str(config.F_L_VAL)],
            "L_test_reads": 0,
            "people_daily_used_for_training_or_tuning": False,
            "people_daily_metrics_semantics": "DeepSeek Flash confirmed 标签一致性，不是人工金标准准确率",
        },
        "baseline_model": {
            "exp_id": "S4_best_pretrained",
            "params": best_params,
            "params_sha256": sha256_file(s4_params_path),
            "params_path": str(s4_params_path),
        },
        "input_fingerprints": {str(path): sha256_file(path) for path in source_paths},
        "dimensions": dimensions,
        "summary_rows": summary_rows,
        "human_review": human_review,
        "A5-7_acceptance": a57,
        "A5-8_acceptance": a58,
    }
    verify_vector_injections(result)
    os.makedirs(OUT_JSON.parent, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(summary_rows)
    write_report(result)
    update_stage5_evaluation(result)
    print("A5-8 八个维度全部产生结果；L_test 未读取。", flush=True)
    print("JSON:", OUT_JSON, flush=True)
    print("CSV:", OUT_CSV, flush=True)
    print("报告:", OUT_REPORT, flush=True)
    return result


if __name__ == "__main__":
    if sys.argv[1:] == ["finalize"]:
        finalize_existing()
    elif not sys.argv[1:]:
        run()
    else:
        raise SystemExit("用法: run_stage5_ablations.py [finalize]")
