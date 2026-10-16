# -*- coding: utf-8 -*-
"""阶段 3 · 文档向量化（PLAN.md §4.1，验收 A3-1 ~ A3-7）。

Word2Vec 产出的是**词向量**，分类器需要**文档向量**，
本模块实现三种池化并对比：

| 方案 | 公式 | 备注 |
|---|---|---|
| P1_mean       | mean(wv[t]) | 基线 |
| P2_tfidf      | Σ tfidf(t)·wv[t] / Σ tfidf(t) | 抑制高频虚词主导 |
| P3_nostop_mean| 先剔停用词再平均 | 与 P1 对照，验证停用词作用 |

统一约束
--------
- 输出做 **L2 归一化**（A3-3 验收 norm ∈ 1±1e-5）
- OOV 词跳过并计数（A3-5）
- 全 OOV 导致零向量的文档必须报警（A3-5）
- **索引对齐凭证**：同时输出 ``doc_index.json``，行号 ↔ doc_id 一一映射。
  A3-2 用它专项抽检，防止"向量与标签错位"这类静默致命 bug（R9）。

⚠️ 数据隔离（§0.4-3）
-------------------
IDF 与停用词表**只能由 U 集统计**，L 与人民日报测试集均不参与。
本模块直接读取 build_splits.py 产出的 ``idf_from_U.json`` 与
``stopwords_from_U.txt``，不在内部重新统计，从代码层面杜绝泄漏。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import compat  # noqa: F401,E402
import config  # noqa: E402

IDF_PATH = config.DATA_PROCESSED / "idf_from_U.json"
STOPWORDS_PATH = config.DATA_PROCESSED / "stopwords_from_U.txt"


# --------------------------------------------------------------------------
# 向量查询
# --------------------------------------------------------------------------
def _build_lookup(wv) -> Tuple[Dict[str, int], Any]:
    """返回 (key_to_index 映射, vectors 矩阵)，用于高效查询。

    ⚠️ 实测更正（2026-10-12）：初版调用 ``wv.get(t)``，但 gensim 的
    ``KeyedVectors`` **没有 get 方法**（会抛 AttributeError），单元测试
    第一时间捕获。实测可用的方式有三种：
      - ``t in wv`` / ``wv.has_index_for(t)``  → 判存在
      - ``wv.get_vector(t)``                   → 取值（缺词抛 KeyError）
      - ``wv.key_to_index`` dict               → 直接映射到行号
    这里选第三种：一次 dict 查找 + 一次数组索引，避免 has_index_for +
    get_vector 的两次哈希查找。

    ⚠️ 性能更正（同日）：本函数会**复制**整个 key_to_index 字典。
    百万级词表的复制成本极高，绝不可在逐文档循环里调用。
    批量向量化必须调用一次后用 ``lookup`` 参数传入池化函数。
    ``wv.key_to_index`` 本身已是 dict，直接复用即可，无需再复制。
    """
    return wv.key_to_index, wv.vectors


def _vec(t: str, index: Dict[str, int], vectors) -> Optional[Any]:
    """查词的向量；不在词表中返回 None（调用方据此统计 OOV）。"""
    i = index.get(t)
    if i is None:
        return None
    return vectors[i]


# --------------------------------------------------------------------------
# 资源载入
# --------------------------------------------------------------------------
def load_idf() -> Dict[str, float]:
    """载入**仅由 U 统计**的 IDF 表（§0.4-3）。"""
    if not IDF_PATH.exists():
        raise FileNotFoundError(
            f"IDF 表不存在: {IDF_PATH}\n请先运行 scripts/build_splits.py"
            f"（IDF 必须只由 U 集统计，禁止用 L 或测试集）")
    with IDF_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_stopwords() -> set:
    """载入**仅由 U 统计**的停用词表（§0.4-3）。"""
    if not STOPWORDS_PATH.exists():
        raise FileNotFoundError(
            f"停用词表不存在: {STOPWORDS_PATH}\n请先运行 scripts/build_splits.py")
    return {w.strip() for w in
            STOPWORDS_PATH.read_text(encoding="utf-8").split("\n") if w.strip()}


# --------------------------------------------------------------------------
# 池化实现
# --------------------------------------------------------------------------
def pool_mean(tokens: Sequence[str], wv, oov: set,
              lookup: Optional[Tuple[Dict[str, int], Any]] = None) -> Optional[Any]:
    """P1：平均池化。全 OOV 返回 None。

    ``lookup`` 为预建的 (index, vectors)；批量场景必须传入，
    否则每篇文档都会重建词表映射。
    """
    index, vectors = lookup if lookup is not None else _build_lookup(wv)
    acc = None
    n = 0
    for t in tokens:
        v = _vec(t, index, vectors)
        if v is None:
            oov.add(t)
            continue
        acc = v.copy() if acc is None else acc + v
        n += 1
    if n == 0:
        return None
    return acc / n


def pool_tfidf(tokens: Sequence[str], wv, idf: Dict[str, float],
               oov: set,
               lookup: Optional[Tuple[Dict[str, int], Any]] = None,
               min_idf: Optional[float] = None) -> Optional[Any]:
    """P2：TF-IDF 加权平均。

    TF 用文档内词频（归一化），IDF 来自 U 统计。
    未出现在 IDF 表中的词（U 中未见）按最小 IDF 处理，
    避免给 OOV 词异常高的权重。
    """
    from collections import Counter

    index, vectors = lookup if lookup is not None else _build_lookup(wv)
    tf = Counter(tokens)
    total = sum(tf.values()) or 1
    if min_idf is None:
        min_idf = min(idf.values()) if idf else 1.0

    acc = None
    wsum = 0.0
    for t, c in tf.items():
        v = _vec(t, index, vectors)
        if v is None:
            oov.add(t)
            continue
        w = (c / total) * idf.get(t, min_idf)
        acc = (v * w) if acc is None else acc + v * w
        wsum += w
    if acc is None or wsum <= 0:
        return None
    return acc / wsum


def pool_nostop_mean(tokens: Sequence[str], wv, stopwords: set,
                     oov: set,
                     lookup: Optional[Tuple[Dict[str, int], Any]] = None) -> Optional[Any]:
    """P3：剔停用词后平均池化。"""
    filtered = [t for t in tokens if t not in stopwords]
    return pool_mean(filtered, wv, oov, lookup=lookup)


def l2_normalize(vec) -> Tuple[Any, bool]:
    """L2 归一化。返回 (向量, 是否为零向量)。"""
    import numpy as np

    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return vec, True
    return vec / norm, False


# --------------------------------------------------------------------------
# 批量向量化
# --------------------------------------------------------------------------
def vectorize(records: List[Dict[str, Any]], wv, method: str,
              token_key: str = "tokens",
              idf: Optional[Dict[str, float]] = None,
              stopwords: Optional[set] = None) -> Dict[str, Any]:
    """对一批文档做向量化。

    返回 dict 含 ``matrix``（n×d，行序与 ``doc_index`` 严格对应）、
    ``doc_index``（行号 → 标识）、``stats``。

    ⚠️ 索引对齐是本模块最容易出静默致命错误的地方（R9）：
    matrix 的第 i 行必须对应 doc_index[i]。此处用**同一次循环内
    同步 append** 保证对齐，并在返回前做长度断言。
    """
    import numpy as np

    if method == "P2_tfidf" and idf is None:
        idf = load_idf()
    if method == "P3_nostop_mean" and stopwords is None:
        stopwords = load_stopwords()

    dim = wv.vector_size
    # 查找表只建一次，传入各池化函数，避免逐文档重建（见 _build_lookup 的性能更正）
    lookup = _build_lookup(wv)
    index, vectors = lookup
    min_idf = min(idf.values()) if method == "P2_tfidf" and idf else 1.0

    rows: List[Any] = []
    doc_index: List[Dict[str, str]] = []
    oov_types: set = set()
    zero_vec_docs: List[str] = []
    dropped_docs: List[str] = []
    total_tokens = 0
    oov_tokens = 0          # OOV 的 token **出现次数**（区别于 oov_types 的种类数）

    for rec in records:
        tokens = rec.get(token_key) or []
        total_tokens += len(tokens)
        # A3-5 要求报告 OOV token 占比，须按出现次数统计，
        # 只统计种类数无法反映实际信息损失量。
        oov_tokens += sum(1 for t in tokens if index.get(t) is None)
        did = rec.get("doc_id") or rec.get("article_id")
        if not did:
            raise ValueError(f"记录缺少 doc_id/article_id: {rec.keys()}")

        if method == "P1_mean":
            vec = pool_mean(tokens, wv, oov_types, lookup=lookup)
        elif method == "P2_tfidf":
            vec = pool_tfidf(tokens, wv, idf, oov_types, lookup=lookup,
                             min_idf=min_idf)
        elif method == "P3_nostop_mean":
            vec = pool_nostop_mean(tokens, wv, stopwords, oov_types, lookup=lookup)
        else:
            raise ValueError(f"未知池化方案: {method}")

        if vec is None:
            dropped_docs.append(did)
            continue

        vec, is_zero = l2_normalize(np.asarray(vec, dtype="float32"))
        if is_zero:
            zero_vec_docs.append(did)
            dropped_docs.append(did)
            continue

        rows.append(vec)
        doc_index.append({
            "row": len(rows) - 1,
            "id": did,
            "label": rec.get("label") or rec.get("llm_label"),
        })

    if not rows:
        raise RuntimeError(f"{method}: 没有任何文档成功向量化")

    matrix = np.vstack(rows).astype("float32")

    # --- 索引对齐硬断言（A3-2 的自动化部分）---
    assert matrix.shape[0] == len(doc_index), (
        f"索引错位：matrix {matrix.shape[0]} 行 vs doc_index {len(doc_index)} 条")
    for i, d in enumerate(doc_index):
        assert d["row"] == i, f"索引错位：第 {i} 行标记为 row={d['row']}"

    stats = {
        "method": method,
        "n_input_records": len(records),
        "n_vectors": matrix.shape[0],
        "dim": matrix.shape[1],
        "dropped_docs": len(dropped_docs),
        "zero_vector_docs": len(zero_vec_docs),
        "dropped_examples": dropped_docs[:20],
        "total_tokens": total_tokens,
        "oov_tokens": oov_tokens,
        "oov_token_ratio": round(oov_tokens / max(total_tokens, 1), 4),
        "unique_oov_types": len(oov_types),
        "oov_examples": sorted(oov_types)[:30],
    }
    return {"matrix": matrix, "doc_index": doc_index, "stats": stats}


# --------------------------------------------------------------------------
# 落盘
# --------------------------------------------------------------------------
def save_vectors(result: Dict[str, Any], split_name: str,
                 exp_id: str) -> Dict[str, str]:
    """保存矩阵与索引凭证。路径含池化方案与词向量实验 ID，便于追溯。"""
    import numpy as np

    method = result["stats"]["method"]
    out_dir = config.DATA_VECTORS / f"{exp_id}"
    os.makedirs(out_dir, exist_ok=True)

    npy_path = out_dir / f"{method}__{split_name}.npy"
    idx_path = out_dir / f"{method}__{split_name}.doc_index.json"

    np.save(npy_path, result["matrix"])
    payload = {
        "split": split_name,
        "exp_id": exp_id,
        "method": method,
        "shape": list(result["matrix"].shape),
        "index": result["doc_index"],
        "stats": result["stats"],
    }
    idx_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    return {"npy": str(npy_path), "doc_index": str(idx_path)}


# --------------------------------------------------------------------------
# 校验（A3-1 ~ A3-7）
# --------------------------------------------------------------------------
def verify(result: Dict[str, Any], records: List[Dict[str, Any]],
           token_key: str = "tokens",
           idf: Optional[Dict[str, float]] = None,
           stopwords: Optional[set] = None,
           wv=None) -> Dict[str, Any]:
    """执行 A3 系列可自动化的验收断言。"""
    import numpy as np

    m = result["matrix"]
    stats = result["stats"]
    checks: Dict[str, Any] = {}

    # A3-1 维度与行数
    checks["A3-1_dim"] = {
        "ok": m.ndim == 2 and m.shape == (len(records), stats["dim"]),
        "shape": list(m.shape),
        "expected_docs": len(records),
    }

    # A3-2 索引对齐：随机抽检 20 例，比对 id 与 label
    rng = np.random.RandomState(config.SEED)
    n = len(result["doc_index"])
    sample = rng.choice(n, size=min(20, n), replace=False)
    lookup = _build_lookup(wv) if wv is not None else None
    min_idf = min(idf.values()) if idf else 1.0
    rec_by_id = {}
    for r in records:
        did = r.get("doc_id") or r.get("article_id")
        rec_by_id[did] = r
    mismatches = []
    for i in sample:
        entry = result["doc_index"][int(i)]
        rec = rec_by_id.get(entry["id"])
        if rec is None:
            mismatches.append((int(i), entry["id"], "记录不存在"))
            continue
        if entry["label"] != (rec.get("label") or rec.get("llm_label")):
            mismatches.append((int(i), entry["id"], "标签不一致"))
            continue
        tokens = rec.get(token_key) or []
        method = stats["method"]
        if lookup is None:
            continue
        if method == "P1_mean":
            expected = pool_mean(tokens, wv, set(), lookup=lookup)
        elif method == "P2_tfidf":
            if idf is None:
                continue
            expected = pool_tfidf(tokens, wv, idf, set(), lookup=lookup,
                                  min_idf=min_idf)
        else:
            if stopwords is None:
                continue
            expected = pool_nostop_mean(tokens, wv, stopwords, set(),
                                        lookup=lookup)
        if expected is not None:
            expected, is_zero = l2_normalize(expected)
            if is_zero or not np.allclose(m[int(i)], expected, rtol=1e-5, atol=1e-6):
                mismatches.append((int(i), entry["id"], "向量与文档 token 不一致"))
    checks["A3-2_alignment"] = {
        "ok": not mismatches,
        "sampled": len(sample),
        "mismatches": mismatches,
    }

    # A3-3 归一化
    norms = np.linalg.norm(m, axis=1)
    tol = config.THRESHOLDS["A3-3_norm_tolerance"]
    dev = float(np.max(np.abs(norms - 1.0)))
    checks["A3-3_normalized"] = {
        "ok": dev <= tol,
        "max_abs_deviation": dev,
        "tolerance": tol,
    }

    # A3-4 无 NaN/Inf
    checks["A3-4_finite"] = {
        "ok": bool(np.isfinite(m).all()),
        "n_nan": int(np.isnan(m).sum()),
        "n_inf": int(np.isinf(m).sum()),
    }

    # A3-5 零向量与 OOV
    checks["A3-5_no_zero_vector"] = {
        "ok": stats["zero_vector_docs"] == 0,
        "zero_vector_docs": stats["zero_vector_docs"],
        "dropped_docs": stats["dropped_docs"],
        "unique_oov_types": stats["unique_oov_types"],
    }

    return checks


def compare_methods(vectors: Dict[str, Any]) -> Dict[str, Any]:
    """A3-6：三方案两两平均余弦须 < 0.99（若 ≈1 说明实现无差异，有 bug）。"""
    import numpy as np

    methods = sorted(vectors)
    out: Dict[str, Any] = {"pairwise_mean_cosine": {}, "ok": True,
                           "threshold": config.THRESHOLDS["A3-6_pooling_pairwise_cos_max"]}
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            a, b = methods[i], methods[j]
            ma, mb = vectors[a]["matrix"], vectors[b]["matrix"]
            if ma.shape != mb.shape:
                out["pairwise_mean_cosine"][f"{a}|{b}"] = None
                out["ok"] = False
                continue
            # 已 L2 归一化，点积即余弦
            cos = (ma * mb).sum(axis=1)
            mean_cos = float(cos.mean())
            out["pairwise_mean_cosine"][f"{a}|{b}"] = round(mean_cos, 4)
            if mean_cos >= out["threshold"]:
                out["ok"] = False
    return out


def domain_separability(records: List[Dict[str, Any]], matrix,
                        doc_index: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A3-7：各方案的类内 − 类间余弦差，作为 P1/P2/P3 选型依据。"""
    import numpy as np
    from collections import defaultdict

    by_label: Dict[str, List[int]] = defaultdict(list)
    for i, d in enumerate(doc_index):
        if d.get("label"):
            by_label[d["label"]].append(i)

    label_counts = {label: len(indices) for label, indices in sorted(by_label.items())}
    labels = [l for l, idx in by_label.items() if len(idx) >= 2]
    if len(labels) < 2:
        return {"error": "类别数不足，无法计算类间距离",
                "n_labels": len(labels),
                "n_labeled_articles": sum(label_counts.values()),
                "label_counts": label_counts}

    # 每类取质心（样本量上限，控制计算量）
    rng = np.random.RandomState(config.SEED)
    centroids = {}
    for l in labels:
        idx = by_label[l]
        if len(idx) > 2000:
            idx = list(rng.choice(idx, size=2000, replace=False))
        vecs = matrix[idx]
        c = vecs.mean(axis=0)
        nc = np.linalg.norm(c)
        centroids[l] = c / nc if nc > 0 else c

    C = np.vstack([centroids[l] for l in labels])
    sim = C @ C.T
    iu = np.triu_indices(len(labels), k=1)
    inter = float(sim[iu].mean())

    # 类内：每类抽样文档与其质心的余弦
    intra_vals = []
    for l in labels:
        idx = by_label[l]
        if len(idx) > 500:
            idx = list(rng.choice(idx, size=500, replace=False))
        c = centroids[l]
        cos = matrix[idx] @ c
        intra_vals.append(float(cos.mean()))
    intra = float(np.mean(intra_vals))

    return {
        "n_labels": len(labels),
        "n_labeled_articles": sum(label_counts.values()),
        "label_counts": label_counts,
        "intra_cosine_mean": round(intra, 4),
        "inter_cosine_mean": round(inter, 4),
        "intra_minus_inter": round(intra - inter, 4),
    }


if __name__ == "__main__":
    print(__doc__)
    print("本模块由 scripts/run_vectorize.py 调用。")
