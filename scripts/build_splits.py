# -*- coding: utf-8 -*-
"""阶段 1-C：数据集划分与格式导出（PLAN.md §2.4）。

输入
----
- ``data/interim/thucnews_clean.jsonl``   THUCNews 清洗产物（含 split 字段 L/U）
- ``data/interim/pd1998_formatted.jsonl`` 人民日报格式清洗产物（无标签）

输出
----
- ``data/processed/U_corpus.txt``         Word2Vec 训练语料，每行一篇分词文本
- ``data/processed/L_train.txt``          FastText 监督格式 ``__label__X w1 w2``
- ``data/processed/L_val.txt``
- ``data/processed/L_test.txt``
- ``data/processed/T_pd199801_confirmed.txt`` 有确认标签的主评估月 FastText 输入
- ``data/processed/T_pd1998_all_confirmed.txt`` 有确认标签的跨月 FastText 输入
- ``data/processed/split_manifest.json``  划分清单，供 A1-5/A5-9 审计

设计要点
--------
1. **L 三分法按文档划分**，类别分层，固定种子（A1-5 / §0.4-2）。
2. **U 与 L 已在清洗阶段物理隔离**（L 取文件号头部、U 取尾部），
   此处再做一次集合断言，防止任何回归。
3. 轨道 B 输出两份：``*_high.txt`` 仅高置信弱标签（主报告口径），
   ``*_all.txt`` 全量（探索性）。
4. 停用词表**只能由 U 统计生成**（§0.4-3），本脚本顺带产出
   ``data/processed/stopwords_from_U.txt`` 与 U 的 IDF 统计，
   供阶段 3 的 P2/P3 池化使用。
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import compat  # noqa: F401,E402
import config  # noqa: E402


# --------------------------------------------------------------------------
# 载入
# --------------------------------------------------------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"缺少清洗产物: {path}\n请先运行 scripts/run_clean_thucnews.py 与 "
            f"python -m src.data.clean_pd1998"
        )
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# --------------------------------------------------------------------------
# L 三分法
# --------------------------------------------------------------------------
def split_L(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """把 L 侧记录按类别分层切分为 train/val/test（8:1:1）。"""
    rng = random.Random(config.SEED)
    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r)

    buckets = {"train": [], "val": [], "test": []}
    per_label_counts: Dict[str, Dict[str, int]] = {}

    for label in config.LABELS:
        items = by_label.get(label, [])
        items = sorted(items, key=lambda r: r["doc_id"])   # 先排序保证确定性
        rng.shuffle(items)                                  # 再用固定种子打乱
        n = len(items)
        n_test = int(round(n * config.L_TEST_RATIO))
        n_val = int(round(n * config.L_VAL_RATIO))
        # 保证 train 至少占剩余，且小类不被 val/test 掏空
        n_test = min(n_test, max(0, n - 2))
        n_val = min(n_val, max(0, n - n_test - 1))
        test = items[:n_test]
        val = items[n_test:n_test + n_val]
        train = items[n_test + n_val:]
        buckets["test"].extend(test)
        buckets["val"].extend(val)
        buckets["train"].extend(train)
        per_label_counts[label] = {
            "total": n, "train": len(train), "val": len(val), "test": len(test),
        }

    return {"buckets": buckets, "per_label": per_label_counts}


# --------------------------------------------------------------------------
# 格式导出
# --------------------------------------------------------------------------
def write_u_corpus(records: List[Dict[str, Any]], path: Path) -> Dict[str, Any]:
    """导出 Word2Vec 语料：每行一篇，token 以空格分隔。"""
    n = 0
    tok = 0
    os.makedirs(path.parent, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            tokens = r["tokens"]
            if not tokens:
                continue
            fh.write(" ".join(tokens) + "\n")
            n += 1
            tok += len(tokens)
    return {"docs": n, "tokens": tok, "path": str(path)}


def write_fasttext(records: List[Dict[str, Any]], path: Path,
                   label_key: str = "label",
                   token_key: str = "tokens") -> Dict[str, Any]:
    """导出 FastText 监督格式：``__label__<类别> <token...>``。"""
    n = 0
    os.makedirs(path.parent, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            lab = r.get(label_key)
            tokens = r.get(token_key) or []
            if not lab or not tokens:
                continue
            fh.write(f"__label__{lab} " + " ".join(tokens) + "\n")
            n += 1
    return {"lines": n, "path": str(path)}


# --------------------------------------------------------------------------
# 停用词与 IDF（只由 U 统计，§0.4-3）
# --------------------------------------------------------------------------
def build_stopwords(u_records: List[Dict[str, Any]], top_n: int = 200) -> Dict[str, Any]:
    """从 U 统计高频词作为停用词候选，并产出 IDF 供 P2 池化使用。"""
    df: Counter = Counter()
    for r in u_records:
        df.update(set(r["tokens"]))       # 文档频次：每篇每词计一次
    N = max(len(u_records), 1)
    stopwords = [w for w, _ in df.most_common(top_n)]
    # sklearn 风格的平滑 IDF：log((1+N)/(1+df)) + 1
    idf = {w: math.log((1.0 + N) / (1.0 + c)) + 1.0 for w, c in df.items()}

    sw_path = config.DATA_PROCESSED / "stopwords_from_U.txt"
    sw_path.write_text("\n".join(stopwords) + "\n", encoding="utf-8")
    idf_path = config.DATA_PROCESSED / "idf_from_U.json"
    with idf_path.open("w", encoding="utf-8") as fh:
        json.dump(idf, fh, ensure_ascii=False)

    return {
        "vocab_size_from_U": len(df),
        "stopwords_n": len(stopwords),
        "stopwords_file": str(sw_path),
        "idf_file": str(idf_path),
        "top20_preview": stopwords[:20],
    }


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def run() -> Dict[str, Any]:
    config.ensure_dirs()
    print("载入清洗产物 ...", flush=True)
    thuc = load_jsonl(config.F_THUCNEWS_CLEAN)
    pd1998 = load_jsonl(config.F_PD1998_FORMATTED)
    print(f"  THUCNews = {len(thuc):,} 篇   人民日报 = {len(pd1998):,} 篇")

    L = [r for r in thuc if r["split"] == "L"]
    U = [r for r in thuc if r["split"] == "U"]
    print(f"  L = {len(L):,}   U = {len(U):,}")

    # --- §0.4 数据隔离红线断言 ---
    ids_L = {r["doc_id"] for r in L}
    ids_U = {r["doc_id"] for r in U}
    assert not (ids_L & ids_U), "数据隔离被破坏：U 与 L 存在交集"
    print("  断言通过：U ∩ L = ∅")

    # --- L 三分 ---
    print("\n切分 L → train/val/test ...", flush=True)
    sp = split_L(L)
    buckets, per_label = sp["buckets"], sp["per_label"]
    ids_train = {r["doc_id"] for r in buckets["train"]}
    ids_val = {r["doc_id"] for r in buckets["val"]}
    ids_test = {r["doc_id"] for r in buckets["test"]}
    assert not (ids_train & ids_val), "train/val 有交集"
    assert not (ids_train & ids_test), "train/test 有交集"
    assert not (ids_val & ids_test), "val/test 有交集"
    assert (ids_train | ids_val | ids_test) == ids_L, "三分未覆盖全部 L"
    print("  断言通过：三分互斥且并集 = L")

    for split in ("train", "val", "test"):
        print(f"    {split:5s} = {len(buckets[split]):>7,} 篇")

    # --- 导出 ---
    print("\n导出各数据集 ...", flush=True)
    u_info = write_u_corpus(U, config.F_U_CORPUS)
    print(f"  U_corpus.txt      {u_info['docs']:,} 篇 / {u_info['tokens']:,} tokens")

    ft_infos = {}
    docid_files = {}
    for split, path in (("train", config.F_L_TRAIN),
                        ("val", config.F_L_VAL),
                        ("test", config.F_L_TEST)):
        info = write_fasttext(buckets[split], path)
        ft_infos[split] = info
        print(f"  {path.name:18s} {info['lines']:,} 行")

        # 同步导出 doc_id 归属清单。
        # ⚠️ 这份文件是下游按划分取数据的**唯一凭证**：FastText 输入格式
        #    （__label__ + tokens）不含 doc_id，下游无法从中反推归属。
        #    若缺失，下游只能整表载入 L，导致 L_test 混入训练数据（R8 泄漏）。
        #    实测初版遗漏了此导出，而下游的"回退整表载入"分支会静默掩盖它。
        did_path = config.DATA_PROCESSED / f"doc_ids_{split}.txt"
        did_path.write_text(
            "\n".join(sorted(r["doc_id"] for r in buckets[split])) + "\n",
            encoding="utf-8")
        docid_files[split] = str(did_path)
        print(f"  {did_path.name:18s} {len(buckets[split]):,} 个 doc_id")

    # 人民日报 FastText 文件只包含已有独立确认标签的文章；本脚本不生成标签。
    def pd_path(stem: str) -> Path:
        return config.DATA_PROCESSED / f"{stem}.txt"

    labels_by_id = {}
    if config.F_PD1998_LLM_LABELS.exists():
        labels_by_id = {
            r["article_id"]: r for r in load_jsonl(config.F_PD1998_LLM_LABELS)
            if r.get("status") == "confirmed" and r.get("label") in config.LABELS
        }
    labelled = [dict(r, llm_label=labels_by_id[r["article_id"]]["label"])
                for r in pd1998 if r["article_id"] in labels_by_id]
    m1 = [r for r in labelled if r["month"] == config.PD1998_PRIMARY_MONTH]
    t_infos = {
        "T_pd199801_confirmed": write_fasttext(
            m1, config.F_T_PD1998,
            label_key="llm_label", token_key="tokens_jieba"),
        "T_pd1998_all_confirmed": write_fasttext(
            labelled, config.F_T_PD1998_ALL,
            label_key="llm_label", token_key="tokens_jieba"),
    }
    for k, v in t_infos.items():
        print(f"  {k:20s} {v['lines']:,} 行 -> {Path(v['path']).name}")

    # 停用词与 IDF（只由 U 统计）
    print("\n统计停用词与 IDF（仅由 U，§0.4-3）...", flush=True)
    sw = build_stopwords(U)
    print(f"  U 词表 = {sw['vocab_size_from_U']:,}  停用词 top{sw['stopwords_n']}")
    print(f"  预览: {' '.join(sw['top20_preview'])}")

    # --- 划分清单 ---
    manifest = {
        "seed": config.SEED,
        "split_strategy": (
            "U/L 互斥：L 取文件号头部、U 取尾部；"
            "各类别按 L 优先 + 小类补足 500 篇分配（config.allocate_split）"
        ),
        "L_trisection": "按类别分层，doc_id 排序后用固定种子打乱，8:1:1 切分",
        "isolation_assertions": {
            "U_intersect_L": 0,
            "train_intersect_val": 0,
            "train_intersect_test": 0,
            "val_intersect_test": 0,
            "train_val_test_union_equals_L": True,
            "pd1998_in_any_training_set": False,
        },
        "counts": {
            "thucnews_clean_total": len(thuc),
            "L": len(L), "U": len(U),
            "L_train": len(buckets["train"]),
            "L_val": len(buckets["val"]),
            "L_test": len(buckets["test"]),
            "pd1998_total": len(pd1998),
            "pd1998_199801_confirmed": len(m1),
            "pd1998_confirmed": len(labelled),
        },
        "per_label_L_split": per_label,
        "label_distribution": {
            "L_train": dict(Counter(r["label"] for r in buckets["train"])),
            "L_val": dict(Counter(r["label"] for r in buckets["val"])),
            "L_test": dict(Counter(r["label"] for r in buckets["test"])),
            "U": dict(Counter(r["label"] for r in U)),
        },
        "llm_label_distribution": {
            "pd1998": dict(Counter(r["llm_label"] for r in labelled)),
            "pd1998_199801": dict(Counter(r["llm_label"] for r in m1)),
        },
        "outputs": {
            "U_corpus": u_info["path"],
            **{f"L_{k}": v["path"] for k, v in ft_infos.items()},
            **{k: v["path"] for k, v in t_infos.items()},
            # doc_id 归属清单：下游按划分取数的唯一凭证。
            # FastText 输入格式不含 doc_id，故必须单独导出，
            # 否则下游无法区分 train/val/test（曾因此埋下泄漏隐患）。
            **{f"doc_ids_{k}": v for k, v in docid_files.items()},
            "stopwords": sw["stopwords_file"],
            "idf": sw["idf_file"],
        },
        "stopwords": sw,
        "pd1998_months": sorted({r["month"] for r in pd1998}),
    }

    config.F_SPLIT_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n划分清单 -> {config.F_SPLIT_MANIFEST}")
    return manifest


if __name__ == "__main__":
    run()
