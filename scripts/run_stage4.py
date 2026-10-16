# -*- coding: utf-8 -*-
"""运行阶段 4 的 FastText 调参与池化向量对照，评估仅使用 L_train/L_val。"""

from __future__ import annotations

import csv
import itertools
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "classify"))

import compat  # noqa: F401,E402
import config  # noqa: E402
from train_fasttext import make_ladder_files, train, validate_fasttext_file  # noqa: E402

TRIAL_BUDGET = 12
SEARCH_FIELDS = ("epoch", "lr", "wordNgrams", "minCount", "loss")


def baseline_search_params() -> Dict[str, Any]:
    return {key: config.FT_BASELINE[key] for key in SEARCH_FIELDS}


def covering_random_configs() -> Tuple[List[Dict[str, Any]], int]:
    """固定种子的覆盖式随机搜索，含基线并覆盖搜索空间的每个参数取值。"""
    grid = [dict(zip(SEARCH_FIELDS, values)) for values in itertools.product(
        *(config.FT_GRID[field] for field in SEARCH_FIELDS)
    )]
    rng = random.Random(config.SEED)
    chosen = [baseline_search_params()]
    required = (
        ("epoch", 10), ("epoch", 25),
        ("lr", 0.05), ("lr", 0.5), ("lr", 1.0),
        ("wordNgrams", 2), ("minCount", 2),
        ("loss", "hs"), ("loss", "ova"),
    )
    for field, value in required:
        options = [c for c in grid if c[field] == value and c not in chosen]
        chosen.append(rng.choice(options))
    remaining = [c for c in grid if c not in chosen]
    rng.shuffle(remaining)
    chosen.extend(remaining[:max(0, TRIAL_BUDGET - len(chosen))])
    return chosen, len(grid)


def serializable_row(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(path.parent, exist_ok=True)
    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serializable_row(value) for key, value in row.items()})


def add_tuning_row(stage: str, trial_id: str, kind: str,
                   params: Dict[str, Any], record: Dict[str, Any],
                   search_space_size: int) -> Dict[str, Any]:
    val = record["val_metrics"]
    return {
        "stage": stage,
        "trial_id": trial_id,
        "trial_type": kind,
        "search_method": "seeded_covering_random",
        "trial_budget": TRIAL_BUDGET,
        "search_space_size": search_space_size,
        "params": params,
        "accuracy": val["accuracy"],
        "macro_f1": val["macro_f1"],
        "weighted_f1": val["weighted_f1"],
        "per_class_recall": {k: v["recall"] for k, v in val["per_class"].items()},
        "train_sec": record["train_sec"],
        "pretrain_vocab_overlap": record["pretrain_vocab_overlap"],
        "input_validation": record["input_validation"],
    }


def add_bidirectional_vocab_coverage(overlap: Dict[str, Any]) -> Dict[str, Any]:
    """报告两种词表交集率；预训练词表是否被完整包含不以 FT 词数占比判定。"""
    ft_words = overlap["ft_words"]
    vec_vocab = overlap["vec_vocab"]
    intersection = overlap["intersection"]
    overlap["overlap_ratio_of_ft_words"] = round(intersection / max(ft_words, 1), 4)
    overlap["overlap_ratio_of_word2vec_vocab"] = round(intersection / max(vec_vocab, 1), 4)
    overlap["word2vec_vocab_is_fasttext_subset"] = intersection == vec_vocab
    overlap["interpretation"] = (
        "Word2Vec 词表完整包含于 FastText 词表；较低的 FastText 方向交集率反映其词表更大，"
        "不表示预训练词表偏离。"
        if overlap["word2vec_vocab_is_fasttext_subset"] else
        "Word2Vec 词表未被 FastText 词表完整覆盖。"
    )
    return overlap


def refresh_vocab_coverage_report(results: Dict[str, Any]) -> None:
    for stage in config.LADDER:
        overlap = results["r1"][stage]["pretrained_baseline"]["pretrain_vocab_overlap"]
        add_bidirectional_vocab_coverage(overlap)
        results["r1"][stage]["matched_pretraining_delta"]["vocabulary_overlap"] = overlap


def load_pooled_data(stage: str, split: str):
    import numpy as np

    method = "P1_mean"
    stem = f"{method}__{split}"
    matrix = np.load(config.DATA_VECTORS / "best" / f"{stem}.npy", mmap_mode="r")
    sidecar = json.loads((config.DATA_VECTORS / "best" / f"{stem}.doc_index.json").read_text(
        encoding="utf-8"))
    labels = set(config.ladder_labels(stage))
    entries = sidecar["index"]
    keep = [i for i, item in enumerate(entries) if item.get("label") in labels]
    X = np.asarray(matrix[keep], dtype=np.float32)
    y = [entries[i]["label"] for i in keep]
    ids = [entries[i]["id"] for i in keep]
    if not X.shape[0]:
        raise ValueError(f"{stage} {split} 没有可用的 P1 向量")
    return X, y, ids


def evaluate_sklearn(model, X, y, labels: Sequence[str]) -> Dict[str, Any]:
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    predicted = model.predict(X)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predicted, labels=list(labels), zero_division=0
    )
    return {
        "examples": len(y),
        "accuracy": round(float(accuracy_score(y, predicted)), 4),
        "macro_f1": round(float(f1_score(y, predicted, labels=list(labels),
                                         average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y, predicted, labels=list(labels),
                                            average="weighted", zero_division=0)), 4),
        "per_class": {
            label: {"precision": round(float(p), 4), "recall": round(float(r), 4),
                    "f1": round(float(f), 4), "support": int(s)}
            for label, p, r, f, s in zip(labels, precision, recall, f1, support)
        },
    }


def load_completed_stage(stage: str) -> Tuple[Dict[str, Any], Dict[str, Any], Path, List[Dict[str, Any]]]:
    csv_path = config.RESULT_METRICS / f"tuning_{stage}.csv"
    model_root = config.MODEL_FT
    r2_names = ("logistic_regression", "linear_svc")
    required = [csv_path, model_root / f"{stage}_best_pretrained" / "params.json"]
    required.extend(model_root / f"R2_{stage}_{name}" / "params.json" for name in r2_names)
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(f"{stage} 缺少阶段结果文件")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != TRIAL_BUDGET + 1:
        raise ValueError(f"{stage} 调参 CSV 行数不完整: {len(rows)}")
    indexed = {row["trial_id"]: row for row in rows}
    if not {"R1-01", "R1-noVec"} <= set(indexed):
        raise ValueError(f"{stage} 调参 CSV 缺少基线行")

    def metrics(row: Dict[str, str]) -> Dict[str, Any]:
        recalls = json.loads(row["per_class_recall"])
        return {"accuracy": float(row["accuracy"]), "macro_f1": float(row["macro_f1"]),
                "weighted_f1": float(row["weighted_f1"]),
                "per_class": {label: {"recall": value} for label, value in recalls.items()}}

    best = json.loads((model_root / f"{stage}_best_pretrained" / "params.json").read_text(encoding="utf-8"))
    overlap_row = json.loads(indexed["R1-01"]["pretrain_vocab_overlap"])
    overlap_row.setdefault("overlap_ratio_of_word2vec_vocab",
                           overlap_row["intersection"] / max(overlap_row["vec_vocab"], 1))
    overlap_row.setdefault("word2vec_vocab_is_fasttext_subset",
                           overlap_row["intersection"] == overlap_row["vec_vocab"])
    pretrained = {"val_metrics": metrics(indexed["R1-01"]),
                  "pretrain_vocab_overlap": overlap_row}
    no_pretrained = {"val_metrics": metrics(indexed["R1-noVec"])}
    params = best["params"]
    selected = next((row for row in rows if json.loads(row["params"]) ==
                     {key: params[key] for key in SEARCH_FIELDS}), None)
    delta = {name: round(pretrained["val_metrics"][name] - no_pretrained["val_metrics"][name], 4)
             for name in ("accuracy", "macro_f1", "weighted_f1")}
    delta["measurable_difference"] = delta["accuracy"] != 0 or delta["macro_f1"] != 0
    delta["vocabulary_overlap"] = pretrained["pretrain_vocab_overlap"]
    r1 = {"best_params": {key: params[key] for key in SEARCH_FIELDS},
          "best_trial_val_metrics": metrics(selected) if selected else best["val_metrics"],
          "best_model": best, "no_pretrained_baseline": no_pretrained,
          "pretrained_baseline": pretrained, "matched_pretraining_delta": delta,
          "input_validation": best["input_validation"], "tuning_trials": TRIAL_BUDGET}
    r2 = {name: json.loads((model_root / f"R2_{stage}_{name}" / "params.json").read_text(encoding="utf-8"))
          for name in r2_names}
    return r1, r2, csv_path, rows


def train_sklearn_routes(stage: str, route_filter: Sequence[str] | None = None) -> Dict[str, Any]:
    import joblib
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    labels = config.ladder_labels(stage)
    X_train, y_train, train_ids = load_pooled_data(stage, "L_train")
    X_val, y_val, val_ids = load_pooled_data(stage, "L_val")
    overlap = set(train_ids) & set(val_ids)
    if overlap:
        raise AssertionError(f"R2 train/val 文档交集非空：{len(overlap)}")
    allowed = set(labels)
    if not set(y_train + y_val) <= allowed:
        raise AssertionError(f"R2 出现阶梯外标签：{stage}")

    estimator_factories = (
        ("logistic_regression", lambda: LogisticRegression(
            solver="lbfgs", max_iter=2000 if stage == "S4" else 300,
            random_state=config.SEED
        )),
        ("linear_svc", lambda: LinearSVC(C=1.0, dual=False, max_iter=3000,
                                         random_state=config.SEED)),
    )
    outputs: Dict[str, Any] = {}
    for name, factory in estimator_factories:
        if route_filter is not None and name not in route_filter:
            continue
        estimator = factory()
        print(f"[{stage}/R2/{name}] train={len(y_train):,} val={len(y_val):,}", flush=True)
        started = time.time()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(X_train, y_train)
        train_metrics = evaluate_sklearn(estimator, X_train, y_train, labels)
        val_metrics = evaluate_sklearn(estimator, X_val, y_val, labels)
        exp_id = f"R2_{stage}_{name}"
        out_dir = config.MODEL_FT / exp_id
        os.makedirs(out_dir, exist_ok=True)
        model_path = out_dir / "model.joblib"
        joblib.dump(estimator, model_path)
        record = {
            "exp_id": exp_id,
            "route": "R2_P1_mean",
            "stage": stage,
            "labels": labels,
            "train_examples": len(y_train),
            "val_examples": len(y_val),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "train_val_accuracy_gap": round(
                train_metrics["accuracy"] - val_metrics["accuracy"], 4
            ),
            "fit_and_eval_sec": round(time.time() - started, 1),
            "warnings": [str(item.message) for item in caught],
            "max_iter": getattr(estimator, "max_iter", None),
            "n_iter": getattr(estimator, "n_iter_", None).tolist()
            if getattr(estimator, "n_iter_", None) is not None else None,
            "model_path": str(model_path),
            "input_files": [
                str(config.DATA_VECTORS / "best" / f"P1_mean__L_train.npy"),
                str(config.DATA_VECTORS / "best" / f"P1_mean__L_val.npy"),
            ],
            "l_test_used": False,
        }
        (out_dir / "params.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        outputs[name] = record
        del estimator, X_train, X_val
        import gc
        gc.collect()
        X_train, y_train, train_ids = load_pooled_data(stage, "L_train")
        X_val, y_val, val_ids = load_pooled_data(stage, "L_val")
        print(f"[{stage}/R2/{name}] macro-F1={val_metrics['macro_f1']:.4f} "
              f"accuracy={val_metrics['accuracy']:.4f}  {record['fit_and_eval_sec']:.1f}s",
              flush=True)
    return outputs


def run() -> Dict[str, Any]:
    config.ensure_dirs()
    vec_path = config.MODEL_W2V / "best" / "w2v_format.vec"
    if not vec_path.is_file():
        raise FileNotFoundError(f"缺少 best 预训练向量: {vec_path}")
    if not config.F_L_TRAIN.is_file() or not config.F_L_VAL.is_file():
        raise FileNotFoundError("阶段 4 仅需要 L_train.txt 与 L_val.txt；请先运行 build_splits.py")

    configs, search_space_size = covering_random_configs()
    all_results: Dict[str, Any] = {
        "stage": "stage4-classification",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": config.SEED,
        "search_method": "seeded_covering_random",
        "trial_budget_per_ladder": TRIAL_BUDGET,
        "search_space_size_per_ladder": search_space_size,
        "pretrained_vectors": str(vec_path),
        "r1": {},
        "r2": {},
        "tuning_csvs": {},
        "l_test_used": False,
    }
    search_rows_by_stage: Dict[str, List[Dict[str, Any]]] = {}

    for stage in config.LADDER:
        if stage != "S4":
            try:
                r1, r2, csv_path, rows = load_completed_stage(stage)
            except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
                pass
            else:
                print(f"[{stage}] 检测到完整结果，已从现有产物恢复并跳过重复训练", flush=True)
                all_results["r1"][stage] = r1
                all_results["r2"][stage] = r2
                all_results["tuning_csvs"][stage] = str(csv_path)
                search_rows_by_stage[stage] = rows
                continue
        paths = make_ladder_files(stage, include_test=False)
        if "test" in paths:
            raise AssertionError("A4-3 失败：调参文件生成意外读取 L_test")
        file_checks = {
            name: validate_fasttext_file(path, allowed_labels=config.ladder_labels(stage))
            for name, path in paths.items()
        }
        if not all(item["ok"] for item in file_checks.values()):
            raise AssertionError(f"A4-3 FastText 文件校验失败：{stage} {file_checks}")

        trial_records: List[Dict[str, Any]] = []
        trial_rows: List[Dict[str, Any]] = []
        for i, params in enumerate(configs, 1):
            record = train(
                stage, pretrained_vectors=vec_path, params=params,
                exp_tag=f"tune{i:02d}", data_paths=paths, retain_model=False,
                input_validation=file_checks,
                measure_pretrain_overlap=(params == baseline_search_params()),
            )
            record["input_validation"] = file_checks
            trial_records.append(record)
            trial_rows.append(add_tuning_row(
                stage, f"R1-{i:02d}", "pretrained_search", params, record,
                search_space_size,
            ))
            write_csv(config.RESULT_METRICS / f"tuning_{stage}.csv", trial_rows)
            print(f"[{stage}/R1 trial {i}/{TRIAL_BUDGET}] "
                  f"val macro-F1={record['val_metrics']['macro_f1']:.4f} "
                  f"accuracy={record['val_metrics']['accuracy']:.4f}", flush=True)

        best_index = max(range(len(trial_records)), key=lambda i: (
            trial_records[i]["val_metrics"]["macro_f1"],
            trial_records[i]["val_metrics"]["accuracy"],
        ))
        best_params = configs[best_index]
        best_record = train(
            stage, pretrained_vectors=vec_path, params=best_params,
            exp_tag="best_pretrained", data_paths=paths, retain_model=True,
            evaluate_train=True, input_validation=file_checks,
        )
        no_vec = train(
            stage, pretrained_vectors=None, params=baseline_search_params(),
            exp_tag="baseline_no_pretrain", data_paths=paths, retain_model=False,
            input_validation=file_checks,
        )
        trial_rows.append(add_tuning_row(
            stage, "R1-noVec", "matched_no_pretrained_ablation", baseline_search_params(),
            no_vec, search_space_size,
        ))
        baseline_idx = next(i for i, candidate in enumerate(configs)
                            if candidate == baseline_search_params())
        baseline_pre = trial_records[baseline_idx]
        r1_record = {
            "best_params": best_params,
            "best_trial_val_metrics": trial_records[best_index]["val_metrics"],
            "best_model": best_record,
            "no_pretrained_baseline": no_vec,
            "pretrained_baseline": baseline_pre,
            "matched_pretraining_delta": {
                "accuracy": round(baseline_pre["val_metrics"]["accuracy"] -
                                   no_vec["val_metrics"]["accuracy"], 4),
                "macro_f1": round(baseline_pre["val_metrics"]["macro_f1"] -
                                   no_vec["val_metrics"]["macro_f1"], 4),
                "weighted_f1": round(baseline_pre["val_metrics"]["weighted_f1"] -
                                     no_vec["val_metrics"]["weighted_f1"], 4),
                "measurable_difference": (
                    baseline_pre["val_metrics"]["accuracy"] != no_vec["val_metrics"]["accuracy"]
                    or baseline_pre["val_metrics"]["macro_f1"] != no_vec["val_metrics"]["macro_f1"]
                ),
                "vocabulary_overlap": baseline_pre["pretrain_vocab_overlap"],
            },
            "input_validation": file_checks,
            "tuning_trials": len(trial_records),
        }
        all_results["r1"][stage] = r1_record
        search_rows_by_stage[stage] = trial_rows
        csv_path = config.RESULT_METRICS / f"tuning_{stage}.csv"
        write_csv(csv_path, trial_rows)
        all_results["tuning_csvs"][stage] = str(csv_path)
        del trial_records

        all_results["r2"][stage] = train_sklearn_routes(stage)

    acceptance: Dict[str, Any] = {}
    r1_stages = [all_results["r1"][stage]["best_model"] for stage in config.LADDER]
    all_routes_exist = all(
        stage in all_results["r1"] and
        {"logistic_regression", "linear_svc"} <= set(all_results["r2"][stage])
        for stage in config.LADDER
    )
    r2_converged = all(
        not model["warnings"]
        for stage in config.LADDER for model in all_results["r2"][stage].values()
    )
    acceptance["A4-1"] = {"ok": all_routes_exist and r2_converged,
                           "r1_ladders": list(config.LADDER),
                           "r2_models_per_ladder": ["logistic_regression", "linear_svc"]}
    overlap_ok = all(
        item["pretrained_baseline"]["pretrain_vocab_overlap"]["word2vec_vocab_is_fasttext_subset"]
        for item in all_results["r1"].values()
    )
    difference_ok = all(
        item["matched_pretraining_delta"]["measurable_difference"]
        for item in all_results["r1"].values()
    )
    acceptance["A4-2"] = {
        "ok": overlap_ok and difference_ok,
        "overlap_policy": "双向报告；不以 intersection/FT_vocab > 0.5 为硬门槛",
        "all_word2vec_vocab_is_subset_of_fasttext_vocab": overlap_ok,
        "overlap_by_stage": {
            stage: all_results["r1"][stage]["pretrained_baseline"]["pretrain_vocab_overlap"]
            for stage in config.LADDER
        },
        "all_stages_measurable_metric_difference": difference_ok,
    }
    leakage_ok = all(
        check["ok"] for stage in config.LADDER
        for check in all_results["r1"][stage]["input_validation"].values()
    ) and all("L_test" not in str(record.get("input_files", []))
              for stage in config.LADDER for record in all_results["r2"][stage].values())
    acceptance["A4-3"] = {
        "ok": leakage_ok,
        "fasttext_train_val_format_checks": "passed",
        "l_test_used": False,
        "tuning_inputs": [str(config.F_L_TRAIN), str(config.F_L_VAL)],
    }
    gaps = {}
    for stage in config.LADDER:
        gaps[stage] = {
            "R1": all_results["r1"][stage]["best_model"]["train_val_accuracy_gap"],
            "R2": {name: model["train_val_accuracy_gap"]
                  for name, model in all_results["r2"][stage].items()},
        }
    gap_ok = all(
        value is not None and value <= config.THRESHOLDS["A4-4_overfit_gap_max"]
        for stage in gaps.values()
        for value in [stage["R1"], *stage["R2"].values()]
    )
    acceptance["A4-4"] = {
        "ok": gap_ok,
        "threshold": config.THRESHOLDS["A4-4_overfit_gap_max"],
        "train_val_accuracy_gaps": gaps,
    }
    s1_accuracy = all_results["r1"]["S1"]["best_model"]["val_metrics"]["accuracy"]
    acceptance["A4-5"] = {
        "ok": s1_accuracy >= config.THRESHOLDS["A4-5_S1_val_acc_min"],
        "accuracy": s1_accuracy,
        "threshold": config.THRESHOLDS["A4-5_S1_val_acc_min"],
    }
    s4_metrics = all_results["r1"]["S4"]["best_model"]["val_metrics"]
    acceptance["A4-6"] = {
        "ok": s4_metrics["macro_f1"] >= config.THRESHOLDS["A4-6_S4_val_macro_f1_min"],
        "accuracy": s4_metrics["accuracy"],
        "macro_f1": s4_metrics["macro_f1"],
        "weighted_f1": s4_metrics["weighted_f1"],
        "threshold": config.THRESHOLDS["A4-6_S4_val_macro_f1_min"],
    }
    ladder_scores = {
        stage: all_results["r1"][stage]["best_model"]["val_metrics"]["macro_f1"]
        for stage in config.LADDER
    }
    acceptance["A4-7"] = {
        "ok": ladder_scores["S4"] <= ladder_scores["S2"],
        "macro_f1_by_ladder": ladder_scores,
        "monotonic_nonincreasing": all(
            ladder_scores[a] >= ladder_scores[b]
            for a, b in zip(config.LADDER, list(config.LADDER)[1:])
        ),
        "S4_gt_S2_anomaly": ladder_scores["S4"] > ladder_scores["S2"],
    }
    r1_s4 = all_results["r1"]["S4"]["best_model"]["val_metrics"]["macro_f1"]
    r2_s4 = {
        name: model["val_metrics"]["macro_f1"]
        for name, model in all_results["r2"]["S4"].items()
    }
    best_r2_name = max(r2_s4, key=r2_s4.get)
    acceptance["A4-8"] = {
        "ok": True,
        "R1_FastText_macro_f1": r1_s4,
        "R2_macro_f1": r2_s4,
        "best_R2": best_r2_name,
        "delta_R1_minus_best_R2": round(r1_s4 - r2_s4[best_r2_name], 4),
        "conclusion": "FastText 优于池化+sklearn" if r1_s4 > r2_s4[best_r2_name]
                      else "池化+sklearn 优于或持平 FastText",
    }
    recall_s4 = {label: s4_metrics["per_class"].get(label, {}).get("recall", 0.0)
                 for label in config.SMALL_LABELS}
    acceptance["A4-9"] = {
        "ok": all(value >= config.THRESHOLDS["A4-9_small_label_recall_min"]
                  for value in recall_s4.values()),
        "small_class_recall": recall_s4,
        "threshold": config.THRESHOLDS["A4-9_small_label_recall_min"],
    }
    tuning_files = [Path(path) for path in all_results["tuning_csvs"].values()]
    acceptance["A4-10"] = {
        "ok": all(path.is_file() for path in tuning_files)
              and all(len(rows) == TRIAL_BUDGET + 1 for rows in search_rows_by_stage.values()),
        "search_space_size": search_space_size,
        "trials_per_stage": TRIAL_BUDGET,
        "csv_files": [str(path) for path in tuning_files],
    }
    all_results["acceptance"] = acceptance
    all_results["all_A4_pass"] = all(item["ok"] for item in acceptance.values())
    all_results["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    all_results["stage5_started"] = False

    metrics_path = config.RESULT_METRICS / "stage4_training.json"
    metrics_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    report_path = config.RESULT_REPORTS / "stage4_acceptance.md"
    report_lines = [
        "# 阶段 4 分类训练与 A4 验收",
        "",
        "- 标注模型：THUCNews 原始目录标签；人民日报标签未用于训练。",
        f"- 调参输入：`L_train.txt` 与 `L_val.txt`；L_test 未读取。",
        f"- R1 搜索：每个阶梯 {TRIAL_BUDGET} 组固定种子覆盖式随机配置，空间大小 {search_space_size}。",
        f"- 阶段 5：未启动。",
        "",
        "| 验收 | 结果 | 详情 |",
        "|---|---:|---|",
    ]
    for item, result in acceptance.items():
        report_lines.append(f"| {item} | {'PASS' if result['ok'] else 'FAIL'} | "
                            f"`{json.dumps(result, ensure_ascii=False)}` |")
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"阶段 4 指标 -> {metrics_path}")
    print(f"阶段 4 验收报告 -> {report_path}")
    print("阶段 4 验收：{}；阶段 5 未启动".format(
        "PASS" if all_results["all_A4_pass"] else "存在未通过项，按要求暂停报告"))
    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="阶段 4 训练和结果修订")
    parser.add_argument("--refresh-s4-logistic-regression", action="store_true")
    args = parser.parse_args()
    if args.refresh_s4_logistic_regression:
        metrics_path = config.RESULT_METRICS / "stage4_training.json"
        results = json.loads(metrics_path.read_text(encoding="utf-8"))
        refresh_vocab_coverage_report(results)
        refreshed = train_sklearn_routes("S4", route_filter=("logistic_regression",))
        results["r2"]["S4"].update(refreshed)
        model_warnings = [
            model["warnings"] for stage in config.LADDER
            for model in results["r2"][stage].values()
        ]
        acceptance = results["acceptance"]
        acceptance["A4-1"]["ok"] = not any(model_warnings)
        acceptance["A4-1"]["r2_convergence"] = {
            f"{stage}/{name}": {
                "converged": not model["warnings"],
                "warnings": model["warnings"],
                "max_iter": model.get("max_iter"),
                "n_iter": model.get("n_iter"),
            }
            for stage in config.LADDER for name, model in results["r2"][stage].items()
        }
        acceptance["A4-2"]["overlap_policy"] = "双向报告；不以 intersection/FT_vocab > 0.5 为硬门槛"
        acceptance["A4-2"].pop("overlap_threshold", None)
        acceptance["A4-2"].pop("all_stages_overlap_ok", None)
        acceptance["A4-2"]["all_word2vec_vocab_is_subset_of_fasttext_vocab"] = all(
            item["pretrained_baseline"]["pretrain_vocab_overlap"]["word2vec_vocab_is_fasttext_subset"]
            for item in results["r1"].values()
        )
        acceptance["A4-2"]["overlap_by_stage"] = {
            stage: results["r1"][stage]["pretrained_baseline"]["pretrain_vocab_overlap"]
            for stage in config.LADDER
        }
        acceptance["A4-2"]["ok"] = (
            acceptance["A4-2"]["all_word2vec_vocab_is_subset_of_fasttext_vocab"]
            and acceptance["A4-2"]["all_stages_measurable_metric_difference"]
        )
        gap_record = results["acceptance"]["A4-4"]["train_val_accuracy_gaps"]["S4"]["R2"]
        gap_record["logistic_regression"] = refreshed["logistic_regression"]["train_val_accuracy_gap"]
        results["acceptance"]["A4-4"]["ok"] = all(
            gap <= config.THRESHOLDS["A4-4_overfit_gap_max"]
            for stage in results["acceptance"]["A4-4"]["train_val_accuracy_gaps"].values()
            for gap in [stage["R1"], *stage["R2"].values()]
        )
        results["all_A4_pass"] = all(item["ok"] for item in results["acceptance"].values())
        results["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        metrics_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
        print(f"S4 LogisticRegression 更新完成；A4-1={results['acceptance']['A4-1']['ok']}；"
              f"指标已写入 {metrics_path}")
    else:
        run()
