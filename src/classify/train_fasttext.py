# -*- coding: utf-8 -*-
"""阶段 4 · FastText 端到端分类（PLAN.md §5.1，路线 R1）。

核心链路
--------
``fasttext.train_supervised(input=..., pretrainedVectors=<gensim 导出的 .vec>)``
阶段 0 的验收项 A0-5 已实测验证该链路可用（gensim → save_word2vec_format
→ fasttext 加载，``get_input_matrix()`` 非空）。

关键约束
--------
1. ``dim`` **必须等于**预训练向量维度，否则 gensim 向量无法注入，
   fasttext 会静默退化为随机初始化——这是最隐蔽的失效模式。
   本模块强制校验并断言（验收 A4-2）。
2. 输入文件为 FastText 格式 ``__label__X w1 w2 ...``，
   标签**只能出现在行首**（验收 A4-3 防标签泄漏）。
3. 超参搜索只用 L_train + L_val，L_test 与人民日报测试集**不参与调参**
   （PLAN.md §0.4-4）。
4. 类别阶梯 S1–S4 需按类别子集过滤训练/验证文件，
   过滤在行级完成，不重新分词。
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import compat  # noqa: F401,E402
import config  # noqa: E402

# FastText 格式：行首标签 + 空格 + token 串
_RE_LABEL = re.compile(r"^__label__(\S+)\s+(.*)$")


# --------------------------------------------------------------------------
# 类别阶梯子集过滤
# --------------------------------------------------------------------------
def filter_by_labels(src: Path, dst: Path, labels: Iterable[str]) -> int:
    """从 FastText 格式文件中筛出指定类别的行，返回写出行数。

    用于类别阶梯实验 S1–S4：同一份 L_train 派生出不同类别数的训练文件，
    保证各档实验的底层分词与切分完全一致（只有类别集合不同），
    否则"类别数增加导致指标下降"的结论会混入分词差异的干扰。
    """
    keep = set(labels)
    n = 0
    os.makedirs(dst.parent, exist_ok=True)
    with src.open("r", encoding="utf-8") as fin, \
            dst.open("w", encoding="utf-8", newline="\n") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            m = _RE_LABEL.match(line)
            if m is None:
                continue                      # 畸形行由 A4-3 的校验统一报告
            if m.group(1) in keep:
                fout.write(line + "\n")
                n += 1
    return n


def make_ladder_files(stage: str, include_test: bool = False) -> Dict[str, Path]:
    """为阶梯实验派生文件；调参默认只读取 train/val，防止触碰 L_test。"""
    labels = config.ladder_labels(stage)
    out_dir = config.DATA_PROCESSED / f"ladder_{stage}"
    paths = {}
    counts = {}
    sources = [("train", config.F_L_TRAIN), ("val", config.F_L_VAL)]
    if include_test:
        sources.append(("test", config.F_L_TEST))
    for name, src in sources:
        if not src.exists():
            raise FileNotFoundError(
                f"{src} 不存在，请先运行 scripts/build_splits.py")
        dst = out_dir / f"{stage}_{name}.txt"
        counts[name] = filter_by_labels(src, dst, labels)
        paths[name] = dst
    summary = f"[{stage}] {len(labels)} 类  train={counts['train']:,} val={counts['val']:,}"
    if "test" in counts:
        summary += f" test={counts['test']:,}"
    print(summary, flush=True)
    return paths


# --------------------------------------------------------------------------
# 输入文件校验（验收 A4-3：防标签泄漏）
# --------------------------------------------------------------------------
def validate_fasttext_file(path: Path, max_report: int = 5,
                           allowed_labels: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """校验 FastText 输入文件格式与标签卫生。"""
    n = 0
    malformed: List[str] = []
    label_leak: List[str] = []          # 正文中出现 __label__ 的行
    empty_tokens: List[str] = []
    unexpected_labels: List[str] = []
    labels: Dict[str, int] = {}
    allowed = set(allowed_labels) if allowed_labels is not None else None

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            n += 1
            m = _RE_LABEL.match(line)
            if m is None:
                if len(malformed) < max_report:
                    malformed.append(line[:120])
                continue
            lab, body = m.group(1), m.group(2)
            labels[lab] = labels.get(lab, 0) + 1
            if allowed is not None and lab not in allowed and len(unexpected_labels) < max_report:
                unexpected_labels.append(lab)
            # 标签泄漏：正文里不应再出现 __label__
            if "__label__" in body:
                if len(label_leak) < max_report:
                    label_leak.append(line[:120])
            if not body.split():
                if len(empty_tokens) < max_report:
                    empty_tokens.append(line[:120])

    return {
        "path": str(path),
        "lines": n,
        "malformed": malformed,
        "n_malformed": len(malformed),
        "label_leak": label_leak,
        "n_label_leak": len(label_leak),
        "empty_tokens": empty_tokens,
        "n_empty_tokens": len(empty_tokens),
        "unexpected_labels": unexpected_labels,
        "labels": labels,
        "n_labels": len(labels),
        "ok": not (malformed or label_leak or empty_tokens or unexpected_labels),
    }


# --------------------------------------------------------------------------
# 训练
# --------------------------------------------------------------------------
def train(stage: str,
          pretrained_vectors: Optional[Path] = None,
          params: Optional[Dict[str, Any]] = None,
          exp_tag: str = "",
          data_paths: Optional[Dict[str, Path]] = None,
          retain_model: bool = True,
          evaluate_train: bool = False,
          input_validation: Optional[Dict[str, Any]] = None,
          measure_pretrain_overlap: bool = True) -> Dict[str, Any]:
    """训练一个 FastText 分类器。

    Parameters
    ----------
    stage : S1–S4
    pretrained_vectors : gensim 导出的 .vec 路径；None 表示不用预训练向量
        （A4-2 的关键消融对照）
    params : 覆盖 config.FT_BASELINE
    exp_tag : 实验后缀，用于区分同 stage 的不同配置
    """
    import fasttext

    cfg = dict(config.FT_BASELINE)
    if params:
        cfg.update(params)

    paths = data_paths or make_ladder_files(stage)
    if "train" not in paths or "val" not in paths:
        raise ValueError("FastText 训练需要 train 和 val 输入路径")
    expected_labels = set(config.ladder_labels(stage))
    input_checks = input_validation or {
        name: validate_fasttext_file(paths[name], allowed_labels=expected_labels)
        for name in ("train", "val")
    }
    failed_inputs = [name for name, check in input_checks.items() if not check["ok"]]
    if failed_inputs:
        raise ValueError(f"A4-3 输入校验失败: {failed_inputs}; details={input_checks}")

    # --- 关键校验：dim 必须与预训练向量一致 ---
    vec_dim = None
    if pretrained_vectors is not None:
        pretrained_vectors = Path(pretrained_vectors)
        if not pretrained_vectors.exists():
            raise FileNotFoundError(f"预训练向量不存在: {pretrained_vectors}")
        with pretrained_vectors.open("r", encoding="utf-8") as fh:
            header = fh.readline().split()
        if len(header) != 2:
            raise ValueError(
                f"预训练向量文件头格式错误（应为 '<词数> <维度>'）: {header}")
        vec_dim = int(header[1])
        if cfg["dim"] != vec_dim:
            # ⚠️ 维度不一致时 fasttext **不会报错**，而是静默忽略预训练向量、
            #    退化为随机初始化。这会让"预训练向量是否有效"的消融结论
            #    完全失真，故此处强制拦截。
            raise ValueError(
                f"dim({cfg['dim']}) 与预训练向量维度({vec_dim}) 不一致。"
                f"FastText 会静默忽略预训练向量并退化为随机初始化，"
                f"导致消融结论失真。请使 cfg['dim'] == {vec_dim}。")

    tag = exp_tag or ("pretrained" if pretrained_vectors else "nopretrain")
    exp_id = f"{stage}_{tag}" + (f"_{params.get('_suffix')}" if params and params.get("_suffix") else "")
    out_dir = config.MODEL_FT / exp_id
    model_path = out_dir / "model.bin"
    if retain_model:
        os.makedirs(out_dir, exist_ok=True)

    train_kwargs = {k: v for k, v in cfg.items() if not k.startswith("_")}
    if pretrained_vectors is not None:
        fasttext_vectors = pretrained_vectors
        if os.name == "nt" and any(ord(char) > 127 for char in str(fasttext_vectors)):
            ascii_dir = Path(os.environ.get("TEMP", os.getcwd())) / "pa2_fasttext_ascii"
            os.makedirs(ascii_dir, exist_ok=True)
            source_path = pretrained_vectors.resolve()
            cache_key = hashlib.sha256(str(source_path).encode("utf-8")).hexdigest()[:16]
            fasttext_vectors = ascii_dir / f"w2v_{cache_key}.vec"
            manifest_path = fasttext_vectors.with_suffix(".json")
            source_info = {
                "path": str(source_path),
                "size": source_path.stat().st_size,
                "mtime_ns": source_path.stat().st_mtime_ns,
            }
            cached_info = None
            if manifest_path.is_file():
                try:
                    cached_info = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    cached_info = None
            if (not fasttext_vectors.is_file()
                    or fasttext_vectors.stat().st_size != source_info["size"]
                    or cached_info != source_info):
                shutil.copyfile(pretrained_vectors, fasttext_vectors)
                manifest_path.write_text(json.dumps(source_info, ensure_ascii=False), encoding="utf-8")
        train_kwargs["pretrainedVectors"] = str(fasttext_vectors)

    print(f"[{exp_id}] 训练参数 = {train_kwargs}", flush=True)
    t0 = time.time()
    fasttext_input = paths["train"]
    if os.name == "nt" and any(ord(char) > 127 for char in str(fasttext_input)):
        ascii_dir = Path(os.environ.get("TEMP", os.getcwd())) / "pa2_fasttext_ascii"
        os.makedirs(ascii_dir, exist_ok=True)
        fasttext_input = ascii_dir / f"{stage}_train.txt"
        shutil.copyfile(paths["train"], fasttext_input)
    model = fasttext.train_supervised(input=str(fasttext_input), **train_kwargs)
    train_sec = time.time() - t0
    save_path = model_path
    if retain_model:
        if os.name == "nt" and any(ord(char) > 127 for char in str(model_path)):
            save_path = Path(os.environ.get("TEMP", os.getcwd())) / "pa2_fasttext_ascii" / f"{exp_id}.bin"
        model.save_model(str(save_path))
        if save_path != model_path:
            shutil.copyfile(save_path, model_path)

    val_metrics = evaluate_file(model, paths["val"], expected_labels)
    train_metrics = (evaluate_file(model, paths["train"], expected_labels)
                     if evaluate_train else None)

    # --- A4-2：确认预训练向量真的被载入 ---
    ft_words = model.get_words()
    vec_overlap = None
    input_matrix_shape = None
    if pretrained_vectors is not None and measure_pretrain_overlap:
        vec_vocab = _read_vec_vocab(pretrained_vectors)
        inter = len(vec_vocab & set(ft_words))
        vec_overlap = {
            "ft_words": len(ft_words),
            "vec_vocab": len(vec_vocab),
            "intersection": inter,
            "overlap_ratio_of_ft_words": round(inter / max(len(ft_words), 1), 4),
            "overlap_ratio_of_word2vec_vocab": round(inter / max(len(vec_vocab), 1), 4),
            "word2vec_vocab_is_fasttext_subset": inter == len(vec_vocab),
        }
    if retain_model and measure_pretrain_overlap:
        input_matrix_shape = list(model.get_input_matrix().shape)

    record = {
        "exp_id": exp_id,
        "stage": stage,
        "n_labels": len(config.ladder_labels(stage)),
        "labels": config.ladder_labels(stage),
        "params": cfg,
        "pretrained_vectors": str(pretrained_vectors) if pretrained_vectors else None,
        "pretrained_dim": vec_dim,
        "files": {k: str(v) for k, v in paths.items()},
        "input_validation": input_checks,
        "train_sec": round(train_sec, 1),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "train_val_accuracy_gap": (
            round(train_metrics["accuracy"] - val_metrics["accuracy"], 4)
            if train_metrics is not None else None
        ),
        "input_matrix_shape": input_matrix_shape,
        "n_ft_words": len(ft_words),
        "pretrain_vocab_overlap": vec_overlap,
        "model_path": str(model_path) if retain_model else None,
    }
    if retain_model:
        (out_dir / "params.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[{exp_id}] 完成 {train_sec:.1f}s  words={len(ft_words):,}", flush=True)
    if vec_overlap:
        print(f"[{exp_id}] 预训练词表重叠率 = "
              f"{vec_overlap['overlap_ratio_of_ft_words']:.4f} "
              f"({vec_overlap['intersection']:,}/{vec_overlap['ft_words']:,})  "
              f"达标={vec_overlap['meets_threshold']}", flush=True)
    return record


def evaluate_file(model, path: Path, expected_labels: Iterable[str],
                  batch_size: int = 512) -> Dict[str, Any]:
    """流式计算准确率、macro/weighted-F1 与逐类 recall。"""
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    labels = sorted("__label__" + label for label in expected_labels)
    y_true: List[str] = []
    y_pred: List[str] = []
    texts: List[str] = []
    batch_labels: List[str] = []

    def predict_batch() -> None:
        if not texts:
            return
        predicted, _ = model.predict(texts, k=1)
        y_true.extend(batch_labels)
        y_pred.extend(items[0] if items else "__label__<NONE>" for items in predicted)
        texts.clear()
        batch_labels.clear()

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            match = _RE_LABEL.match(line.rstrip("\n"))
            if match is None:
                raise ValueError(f"评估文件含畸形行: {path}")
            label, text = "__label__" + match.group(1), match.group(2)
            if label not in labels:
                raise ValueError(f"评估文件含阶梯外标签 {label}: {path}")
            batch_labels.append(label)
            texts.append(text)
            if len(texts) >= batch_size:
                predict_batch()
        predict_batch()

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "examples": len(y_true),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels,
                                         average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=labels,
                                            average="weighted", zero_division=0)), 4),
        "per_class": {
            label.replace("__label__", ""): {
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "f1": round(float(f), 4),
                "support": int(s),
            }
            for label, p, r, f, s in zip(labels, precision, recall, f1, support)
        },
    }


@lru_cache(maxsize=4)
def _read_vec_vocab(path: Path) -> set:
    """读取 word2vec 文本格式的词表（跳过首行头信息）。"""
    vocab = set()
    with path.open("r", encoding="utf-8") as fh:
        fh.readline()                          # 头：<词数> <维度>
        for line in fh:
            parts = line.rstrip("\n").split(" ")
            if parts and parts[0]:
                vocab.add(parts[0])
    return vocab


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="FastText 分类训练")
    ap.add_argument("--stage", required=True, choices=list(config.LADDER))
    ap.add_argument("--vectors", default=None,
                    help="gensim 导出的 .vec 路径；省略则不注入预训练向量")
    ap.add_argument("--suffix", default=None, help="实验后缀")
    args = ap.parse_args()

    params = {"_suffix": args.suffix} if args.suffix else None
    train(args.stage,
          pretrained_vectors=Path(args.vectors) if args.vectors else None,
          params=params)
