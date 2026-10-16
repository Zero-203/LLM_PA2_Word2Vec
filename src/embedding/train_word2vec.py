# -*- coding: utf-8 -*-
"""阶段 2 · gensim Word2Vec 训练（PLAN.md §3.1）。

按决议 D3 的强制顺序执行：
    第 1 轮  E7 语料量消融（5万 / 15万 / 全量），固定其余参数
             → 判定"指标随语料量的饱和点"
    第 2 轮  在饱和点语料量上跑 E1–E6 完整矩阵
    第 3 轮  用最优配置导出向量，供阶段 3/4 使用

每组实验独立落盘：
    model/word2vec/<exp_id>/
        model.wv              KeyedVectors（供 most_similar / 向量查询）
        w2v_format.vec        save_word2vec_format 文本格式（供 FastText 注入）
        params.json           完整超参（可复现依据）
        train_log.json        逐 epoch 训练损失、耗时、词表规模

内存策略
--------
U 语料约 24 万篇 / 2 亿 token，全量载入内存不现实。
使用 gensim 的 ``LineSentence`` 流式读取 ``U_corpus.txt``（每行一篇），
多 epoch 时每轮重新迭代文件，内存占用恒定。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# 本文件位于 <ROOT>/src/embedding/，故 parents[2] 才是项目根。
# ⚠️ 实测更正（2026-10-12）：初版误写 parents[1]，拼出不存在的
#    <ROOT>/src/src，导致 `python -m src.embedding.train_word2vec` 报
#    ModuleNotFoundError: No module named 'compat'。
#    此前的冒烟测试因外层已手动 sys.path.insert(0,'src') 而掩盖了该缺陷。
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import compat  # noqa: F401,E402  (必须先于任何第三方库)
import config  # noqa: E402


class SubsampledCorpus:
    """按篇数上限流式截取 ``U_corpus.txt`` 的前 N 篇。

    E7 语料量消融需要在同一语料文件上取不同规模的子集。
    刻意取**前 N 篇**而非随机抽样：U_corpus.txt 由 build_splits 按
    L 优先/U 尾部规则确定性写出，前 N 篇即固定子集，
    保证同档位在不同实验中完全一致（可复现，A2-10）。
    """

    def __init__(self, path: Path, max_docs: Optional[int] = None):
        self.path = path
        self.max_docs = max_docs

    def __iter__(self) -> Iterable[List[str]]:
        n = 0
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if self.max_docs is not None and n >= self.max_docs:
                    return
                line = line.rstrip("\n")
                if not line:
                    continue
                yield line.split(" ")
                n += 1


def count_corpus_docs(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as fh:
        for _ in fh:
            n += 1
    return n


# --------------------------------------------------------------------------
# 单次实验
# --------------------------------------------------------------------------
def run_experiment(exp_id: str,
                   corpus_docs: Optional[int] = None,
                   params: Optional[Dict[str, Any]] = None,
                   corpus_path: Optional[Path] = None,
                   epochs_progress: bool = True) -> Dict[str, Any]:
    """训练一次 Word2Vec 并落盘全部产物，返回实验记录。

    Parameters
    ----------
    exp_id : 实验目录名，如 ``E7_corpus50k``
    corpus_docs : 取前 N 篇（None = 全量）
    params : 覆盖 ``config.W2V_BASELINE`` 的超参
    epochs_progress : True 时按 epoch 分段训练，以记录逐 epoch 损失
        （gensim 的 ``get_latest_training_loss`` 只在增量 train 时有意义）
    """
    from gensim.models import Word2Vec

    corpus_path = corpus_path or config.F_U_CORPUS
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"U 语料不存在: {corpus_path}\n请先运行 scripts/build_splits.py")

    cfg = dict(config.W2V_BASELINE)
    if params:
        cfg.update(params)
    total_epochs = int(cfg.pop("epochs"))
    workers = cfg.pop("workers", config.W2V_WORKERS)

    out_dir = config.MODEL_W2V / exp_id
    os.makedirs(out_dir, exist_ok=True)

    corpus = SubsampledCorpus(corpus_path, corpus_docs)

    print(f"[{exp_id}] 参数 = {cfg} epochs={total_epochs} "
          f"corpus_docs={corpus_docs or 'ALL'}", flush=True)

    t0 = time.time()
    model = Word2Vec(
        sentences=None,            # 先建空模型，再分段 train 以记录损失
        workers=workers,
        epochs=0,
        **cfg,
    )
    # 第一遍：构建词表（min_count / sample 生效）
    model.build_vocab(corpus, progress_per=10.0)
    vocab_size = len(model.wv)
    # ⚠️ 实测更正（2026-10-12）：``corpus_count`` 是**文档数**（实测取 800 篇
    # 时该值恰为 800），不是词数。真实语料词数在 ``corpus_total_words``。
    # 用错字段会让 train(total_examples=...) 的进度与词表统计全部失真。
    corpus_docs_seen = model.corpus_count
    corpus_words = model.corpus_total_words
    print(f"[{exp_id}] 词表 = {vocab_size:,}  文档数 = {corpus_docs_seen:,}  "
          f"语料词数 = {corpus_words:,}", flush=True)

    # ⚠️ 实测更正（2026-10-12）：``get_latest_training_loss()`` 返回的是
    # gensim 内部累积的**平均损失**（running average），随训练整体递减，
    # 但**并不逐 epoch 单调**（实测 5 epoch 序列为
    # 216264 → 156571 → 140561 → 112720 → 132095，末轮反弹）。
    # 故：
    #   1) 不得把相邻差值当作"本轮损失"（差值可为负，语义不成立）；
    #   2) A2-3 的收敛判定应基于"整体趋势下降 + 首末降幅"，
    #      而非要求逐 epoch 单调，否则会误判正常训练为失败。
    epoch_logs: List[Dict[str, Any]] = []
    for ep in range(1, total_epochs + 1):
        # 每轮重新迭代语料文件（流式，内存恒定）
        model.train(SubsampledCorpus(corpus_path, corpus_docs),
                    total_examples=model.corpus_count,
                    epochs=1,
                    compute_loss=True)
        cum_loss = model.get_latest_training_loss()
        elapsed = time.time() - t0
        epoch_logs.append({
            "epoch": ep,
            "running_avg_loss": round(cum_loss, 3),
            "elapsed_sec": round(elapsed, 1),
        })
        if epochs_progress:
            print(f"[{exp_id}]   epoch {ep}/{total_epochs}  "
                  f"running_avg_loss={cum_loss:.1f}  ({elapsed:.0f}s)", flush=True)

    total_sec = time.time() - t0

    # 收敛性摘要：首末降幅 + 单调性（单调性仅作记录，不作硬性判据）
    losses = [e["running_avg_loss"] for e in epoch_logs]
    convergence = {
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "min_loss": min(losses),
        "absolute_drop": round(losses[0] - losses[-1], 3),
        "relative_drop_pct": round(
            100.0 * (losses[0] - losses[-1]) / max(losses[0], 1e-9), 2),
        "is_monotonic_decreasing": all(
            losses[i] >= losses[i + 1] for i in range(len(losses) - 1)),
        "n_increases": sum(
            1 for i in range(len(losses) - 1) if losses[i + 1] > losses[i]),
    }

    # 落盘
    wv_path = out_dir / "model.wv"
    vec_path = out_dir / "w2v_format.vec"
    model.wv.save(str(wv_path))
    model.wv.save_word2vec_format(str(vec_path), binary=False)

    record = {
        "exp_id": exp_id,
        "corpus_path": str(corpus_path),
        "corpus_docs": corpus_docs,
        "params": {**cfg, "epochs": total_epochs, "workers": workers},
        "vocab_size": vocab_size,
        "corpus_docs_seen": corpus_docs_seen,
        "corpus_word_count": corpus_words,
        "convergence": convergence,
        "total_sec": round(total_sec, 1),
        "epoch_logs": epoch_logs,
        "artifacts": {
            "model_wv": str(wv_path),
            "w2v_format_vec": str(vec_path),
            "params": str(out_dir / "params.json"),
            "train_log": str(out_dir / "train_log.json"),
        },
        "seed": config.SEED,
    }
    (out_dir / "params.json").write_text(
        json.dumps(record["params"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    (out_dir / "train_log.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 导出 U 词表，供 A1-6 的 OOV 率基线核验
    if exp_id.startswith("E1") or exp_id == "best":
        vocab_path = config.DATA_PROCESSED / "u_vocab.txt"
        vocab_path.write_text("\n".join(model.wv.index_to_key) + "\n",
                              encoding="utf-8")
        record["u_vocab_file"] = str(vocab_path)
        print(f"[{exp_id}] U 词表导出 -> {vocab_path}", flush=True)

    print(f"[{exp_id}] 完成：vocab={vocab_size:,}  "
          f"last_loss={convergence['last_loss']:.1f}  "
          f"降幅={convergence['relative_drop_pct']:.1f}%  "
          f"耗时 {total_sec:.0f}s -> {out_dir}", flush=True)
    return record


# --------------------------------------------------------------------------
# 第 1 轮：E7 语料量消融（决议 D3）
# --------------------------------------------------------------------------
def run_e7_corpus_scaling(corpus_path: Optional[Path] = None) -> Dict[str, Any]:
    """E7：固定 CBOW/dim=200/win=5/ep=5，只变语料量，判定饱和点。"""
    corpus_path = corpus_path or config.F_U_CORPUS
    total = count_corpus_docs(corpus_path)
    print(f"U 语料总篇数 = {total:,}")

    sizes = [s for s in config.E7_CORPUS_SIZES if s is not None and s < total]
    sizes.append(None)                       # None = 全量
    results = []
    for s in sizes:
        exp_id = f"E7_corpus{'all' if s is None else f'{s//1000}k'}"
        rec = run_experiment(exp_id, corpus_docs=s, corpus_path=corpus_path)
        results.append(rec)

    summary = {
        "round": 1,
        "experiment": "E7 语料量消融",
        "corpus_total_docs": total,
        "note": (
            "loss 为 gensim 的 running average（整体递减但非逐 epoch 单调），"
            "饱和点判定须结合下游词向量质量指标，不能只看 loss"
        ),
        "results": [
            {
                "exp_id": r["exp_id"],
                "corpus_docs": r["corpus_docs"] or total,
                "vocab_size": r["vocab_size"],
                "corpus_word_count": r["corpus_word_count"],
                "last_loss": r["convergence"]["last_loss"],
                "min_loss": r["convergence"]["min_loss"],
                "relative_drop_pct": r["convergence"]["relative_drop_pct"],
                "total_sec": r["total_sec"],
            }
            for r in results
        ],
    }
    out = config.RESULT_METRICS / "e7_corpus_scaling.json"
    os.makedirs(out.parent, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"\nE7 汇总 -> {out}")
    return summary


# --------------------------------------------------------------------------
# 第 2 轮：E1–E6 完整矩阵
# --------------------------------------------------------------------------
def build_matrix(corpus_docs: Optional[int]) -> List[Dict[str, Any]]:
    """按 §3.1 构造 E1–E6 实验清单。"""
    g = config.W2V_GRID
    exps: List[Dict[str, Any]] = []

    # E1 基线：CBOW / dim=200 / win=5 / neg=5 / ep=5
    exps.append({"exp_id": "E1_baseline", "params": {}})

    # E2 架构：CBOW vs Skip-gram
    exps.append({"exp_id": "E2_skipgram", "params": {"sg": 1}})

    # E3 维度
    for d in g["vector_size"]:
        if d == config.W2V_BASELINE["vector_size"]:
            continue                       # 与 E1 重复
        exps.append({"exp_id": f"E3_dim{d}", "params": {"vector_size": d}})

    # E4 窗口
    for w in g["window"]:
        if w == config.W2V_BASELINE["window"]:
            continue
        exps.append({"exp_id": f"E4_win{w}", "params": {"window": w}})

    # E5 训练法：负采样 vs 层次 softmax
    exps.append({"exp_id": "E5_hierarchical_softmax",
                 "params": {"hs": 1, "negative": 0}})

    # E6 轮次
    for e in g["epochs"]:
        if e == config.W2V_BASELINE["epochs"]:
            continue
        exps.append({"exp_id": f"E6_epochs{e}", "params": {"epochs": e}})

    return [{"corpus_docs": corpus_docs, **x} for x in exps]


def run_matrix(corpus_docs: Optional[int],
               only: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    exps = build_matrix(corpus_docs)
    if only:
        exps = [e for e in exps if e["exp_id"] in only]
    out = []
    print(f"\n=== 第 2 轮：{len(exps)} 组实验（语料档位 = "
          f"{corpus_docs or '全量'}）===")
    for e in exps:
        out.append(run_experiment(e["exp_id"], corpus_docs=e["corpus_docs"],
                                  params=e["params"]))
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Word2Vec 训练")
    ap.add_argument("--round", choices=["1", "2"], default=None,
                    help="1=E7 语料量消融；2=E1–E6 完整矩阵")
    ap.add_argument("--exp", nargs="*", default=None,
                    help="只跑指定 exp_id（第 2 轮用）")
    ap.add_argument("--corpus-docs", type=int, default=None,
                    help="第 2 轮的语料档位（取 E7 判定的饱和点）")
    args = ap.parse_args()

    if args.round == "1":
        run_e7_corpus_scaling()
    elif args.round == "2":
        run_matrix(args.corpus_docs, only=args.exp)
    else:
        print("请指定 --round 1 或 --round 2（决议 D3：先定饱和点，再跑完整矩阵）")
