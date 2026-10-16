# -*- coding: utf-8 -*-
"""用独立的 THUCNews 小样本验证 Word2Vec→FastText 预训练向量链路。"""

from __future__ import annotations

import json
import random
import re
import shutil
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import compat  # noqa: E402,F401
import config  # noqa: E402

SEED = config.SEED
DOCS_PER_LABEL = 12
UNLABELED_PER_LABEL = 4
VECTOR_SIZE = 50
MIN_TOKENS = 20
OUTPUT_DIR = ROOT / "result" / "small_sample"
METRICS_PATH = config.RESULT_METRICS / "small_sample_smoke.json"


def read_tokens(path: Path) -> List[str]:
    try:
        text = path.read_text(encoding="utf-8-sig").replace("\x00", " ")
    except (OSError, UnicodeError):
        return []
    text = re.sub(r"\s+", " ", text)
    import jieba

    tokens = [
        token.strip()
        for token in jieba.cut(text, HMM=False)
        if token.strip() and re.search(r"[\u4e00-\u9fffA-Za-z0-9]", token)
    ]
    return tokens if len(tokens) >= MIN_TOKENS else []


def sample_documents() -> Dict[str, List[Dict[str, Any]]]:
    rng = random.Random(SEED)
    result: Dict[str, List[Dict[str, Any]]] = {}
    for label in config.LABELS:
        label_dir = config.DATA_THUCNEWS / label
        if not label_dir.is_dir():
            raise FileNotFoundError(f"THUCNews 类别目录不存在: {label_dir}")
        paths = sorted(label_dir.rglob("*.txt"))
        rng.shuffle(paths)
        selected: List[Dict[str, Any]] = []
        for path in paths:
            tokens = read_tokens(path)
            if tokens:
                selected.append({
                    "id": path.relative_to(config.DATA_THUCNEWS).as_posix(),
                    "label": label,
                    "tokens": tokens,
                })
            if len(selected) == DOCS_PER_LABEL:
                break
        if len(selected) != DOCS_PER_LABEL:
            raise RuntimeError(
                f"类别 {label} 仅找到 {len(selected)} 篇有效文档；需要 {DOCS_PER_LABEL} 篇"
            )
        rng.shuffle(selected)
        result[label] = selected
    return result


def write_fasttext_file(path: Path, docs: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for doc in docs:
            fh.write(f"__label__{doc['label']} {' '.join(doc['tokens'])}\n")


def evaluate(model: Any, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    from sklearn.metrics import accuracy_score, f1_score

    y_true = [doc["label"] for doc in docs]
    y_pred = [model.predict(" ".join(doc["tokens"]), k=1)[0][0].removeprefix("__label__")
              for doc in docs]
    return {
        "examples": len(y_true),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=config.LABELS,
                                           average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=config.LABELS,
                                              average="weighted", zero_division=0)), 6),
        "predictions": y_pred,
    }


def main() -> None:
    if sys.version_info[:2] != (3, 9) or struct.calcsize("P") * 8 != 64:
        raise RuntimeError("小样本复现要求 Python 3.9 64-bit；请使用 .venv39。")
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if not config.DATA_THUCNEWS.is_dir():
        raise FileNotFoundError(f"THUCNews 数据目录不存在: {config.DATA_THUCNEWS}")

    grouped = sample_documents()
    unlabeled: List[Dict[str, Any]] = []
    train_docs: List[Dict[str, Any]] = []
    val_docs: List[Dict[str, Any]] = []
    test_docs: List[Dict[str, Any]] = []
    for docs in grouped.values():
        unlabeled.extend(docs[:UNLABELED_PER_LABEL])
        labeled = docs[UNLABELED_PER_LABEL:]
        train_docs.extend(labeled[:6])
        val_docs.extend(labeled[6:7])
        test_docs.extend(labeled[7:8])

    all_ids = [doc["id"] for docs in grouped.values() for doc in docs]
    if len(all_ids) != len(set(all_ids)):
        raise AssertionError("小样本出现重复文档 ID")
    split_ids = {
        "U": {doc["id"] for doc in unlabeled},
        "L_train": {doc["id"] for doc in train_docs},
        "L_val": {doc["id"] for doc in val_docs},
        "L_test": {doc["id"] for doc in test_docs},
    }
    for left, right in (("U", "L_train"), ("U", "L_val"), ("U", "L_test"),
                        ("L_train", "L_val"), ("L_train", "L_test"), ("L_val", "L_test")):
        if split_ids[left] & split_ids[right]:
            raise AssertionError(f"小样本划分泄漏: {left} 与 {right}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": SEED,
        "source": "THUCNews raw documents only; no People’s Daily or original L_test input",
        "counts": {key: len(value) for key, value in {
            "U": unlabeled, "L_train": train_docs, "L_val": val_docs, "L_test": test_docs
        }.items()},
        "doc_ids": {key: sorted(value) for key, value in split_ids.items()},
    }
    (OUTPUT_DIR / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    try:
        import fasttext
        from gensim.models import Word2Vec
        from gensim.models.word2vec import LineSentence
    except ImportError as exc:
        raise RuntimeError("缺少 gensim / fasttext-wheel；请先按 README 安装 requirements.txt") from exc

    with tempfile.TemporaryDirectory(prefix="pa2_small_sample_") as temp_name:
        temp = Path(temp_name)
        corpus_path = temp / "u_corpus.txt"
        with corpus_path.open("w", encoding="utf-8", newline="\n") as fh:
            for doc in unlabeled:
                fh.write(" ".join(doc["tokens"]) + "\n")

        w2v = Word2Vec(
            sentences=LineSentence(str(corpus_path)),
            vector_size=VECTOR_SIZE,
            window=5,
            min_count=1,
            workers=1,
            sg=0,
            seed=SEED,
            epochs=5,
        )
        vec_path = temp / "pretrained.vec"
        w2v.wv.save_word2vec_format(str(vec_path), binary=False)
        w2v_vocab_size = len(w2v.wv)

        train_path = temp / "train.txt"
        val_path = temp / "val.txt"
        test_path = temp / "test.txt"
        write_fasttext_file(train_path, train_docs)
        write_fasttext_file(val_path, val_docs)
        write_fasttext_file(test_path, test_docs)
        model = fasttext.train_supervised(
            input=str(train_path),
            pretrainedVectors=str(vec_path),
            dim=VECTOR_SIZE,
            epoch=10,
            lr=0.5,
            wordNgrams=2,
            minCount=1,
            loss="ova",
            bucket=5000,
            thread=1,
            verbose=0,
        )
        input_matrix = model.get_input_matrix()
        if input_matrix.ndim != 2 or input_matrix.shape[0] <= 0 or input_matrix.shape[1] != VECTOR_SIZE:
            raise AssertionError(f"预训练向量注入矩阵异常: {input_matrix.shape}")
        val_metrics = evaluate(model, val_docs)
        test_metrics = evaluate(model, test_docs)
        temp_model = temp / "small_sample_fasttext.bin"
        model.save_model(str(temp_model))
        shutil.copyfile(temp_model, OUTPUT_DIR / "fasttext_small_sample.bin")

    params = {
        "word2vec": {
            "vector_size": VECTOR_SIZE, "window": 5, "min_count": 1,
            "sg": 0, "seed": SEED, "epochs": 5, "workers": 1,
        },
        "fasttext": {
            "dim": VECTOR_SIZE, "epoch": 10, "lr": 0.5,
            "wordNgrams": 2, "minCount": 1, "loss": "ova",
            "bucket": 5000, "thread": 1, "pretrained_vectors": True,
        },
    }
    record = {
        "stage": "small-sample-end-to-end-smoke",
        "started_at": started,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": SEED,
        "data_scope": "独立的小样本流程验证；不是阶段 0–5 的主实验指标",
        "counts": manifest["counts"],
        "labels": config.LABELS,
        "word2vec_vocab_size": w2v_vocab_size,
        "fasttext_input_matrix_shape": list(input_matrix.shape),
        "pretrained_vector_injection_ok": True,
        "parameters": params,
        "validation": {key: value for key, value in val_metrics.items() if key != "predictions"},
        "test": {key: value for key, value in test_metrics.items() if key != "predictions"},
        "test_predictions": test_metrics["predictions"],
        "test_labels": [doc["label"] for doc in test_docs],
        "artifacts": {
            "manifest": "result/small_sample/split_manifest.json",
            "fasttext_model": "result/small_sample/fasttext_small_sample.bin",
        },
    }
    config.RESULT_METRICS.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        "小样本端到端验证通过："
        f"U={len(unlabeled)}, train={len(train_docs)}, val={len(val_docs)}, test={len(test_docs)}, "
        f"W2V vocab={w2v_vocab_size}, FastText input={list(input_matrix.shape)}, "
        f"test macro-F1={test_metrics['macro_f1']:.4f}。"
    )


if __name__ == "__main__":
    main()
