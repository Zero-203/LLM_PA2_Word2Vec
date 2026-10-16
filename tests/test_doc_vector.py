# -*- coding: utf-8 -*-
"""doc_vector 模块单元测试（对应验收 A3-1 ~ A3-7）。

用**合成向量**做精确断言：三个正交单位向量 a/b/c，
手工推导 P1/P2/P3 的期望结果，逐项比对。
这比用真实词向量更能验证池化公式本身是否正确。

运行：
    .\\.venv39\\Scripts\\python.exe tests\\test_doc_vector.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import compat  # noqa: F401  (必须先于任何第三方库)
import config  # noqa: E402

import numpy as np
from gensim.models import KeyedVectors

from vectorize.doc_vector import (
    compare_methods,
    domain_separability,
    l2_normalize,
    pool_mean,
    pool_nostop_mean,
    pool_tfidf,
    vectorize,
    verify,
)

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, got, want, tol: float = 1e-6) -> None:
    ok = False
    try:
        if isinstance(want, (list, tuple, np.ndarray)) or isinstance(got, (list, tuple, np.ndarray)):
            ok = bool(np.allclose(np.asarray(got, dtype=float),
                                  np.asarray(want, dtype=float), atol=tol))
        elif isinstance(want, float) or isinstance(got, float):
            ok = abs(float(got) - float(want)) <= tol
        else:
            ok = (got == want)
    except Exception:
        ok = (got == want)
    if ok:
        PASS.append(name)
        print("  [ok]   %s" % name)
    else:
        FAIL.append(name)
        print("  [FAIL] %s\n         got : %r\n         want: %r" % (name, got, want))


def section(t: str) -> None:
    print()
    print("=" * 72)
    print(t)
    print("=" * 72)


# --------------------------------------------------------------------------
# 合成词表：a/b/c 为正交单位向量，z 为零向量，用于精确推导
# --------------------------------------------------------------------------
def make_wv() -> KeyedVectors:
    wv = KeyedVectors(vector_size=3)
    wv.add_vectors(
        ["a", "b", "c", "z"],
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype="float32"),
    )
    return wv


WV = make_wv()
IDF = {"a": 1.0, "b": 2.0, "c": 1.0, "z": 1.0}
STOP = {"a"}

# 期望值手工推导（a/b/c 为正交单位向量）：
# P1 mean   = (2*[1,0,0] + [0,1,0]) / 3 = [0.6667, 0.3333, 0]
#             L2 归一化后 = [2/√5, 1/√5, 0] = [0.8944, 0.4472, 0]
# P2 tfidf  : tf(a)=2/3, tf(b)=1/3 ; w_a=(2/3)*1=0.6667, w_b=(1/3)*2=0.6667
#             acc=[0.6667,0.6667,0], wsum=1.3333 -> [0.5,0.5,0]
#             L2 归一化后 = [1/√2, 1/√2, 0] = [0.7071, 0.7071, 0]
# P3 nostop : 剔除停用词 a -> [b] -> [0,1,0]（已归一化）
EXP_P1 = [2 / math.sqrt(5), 1 / math.sqrt(5), 0.0]
EXP_P2 = [1 / math.sqrt(2), 1 / math.sqrt(2), 0.0]
EXP_P3 = [0.0, 1.0, 0.0]

DOC = {"doc_id": "d1", "label": "L1", "tokens": ["a", "a", "b"]}


section("1. 池化函数的数学正确性（合成正交向量，精确推导）")
oov: set = set()
check("P1 pool_mean", pool_mean(DOC["tokens"], WV, oov), [2 / 3, 1 / 3, 0.0])
oov2: set = set()
check("P2 pool_tfidf", pool_tfidf(DOC["tokens"], WV, IDF, oov2), [0.5, 0.5, 0.0])
oov3: set = set()
check("P3 pool_nostop_mean（剔停用词 a）",
      pool_nostop_mean(DOC["tokens"], WV, STOP, oov3), [0.0, 1.0, 0.0])

section("2. L2 归一化")
v, is_zero = l2_normalize(np.array([3.0, 4.0, 0.0]))
check("归一化后模长=1", float(np.linalg.norm(v)), 1.0)
check("归一化结果", v, [0.6, 0.8, 0.0])
check("非零向量标记", is_zero, False)
zv, z_is_zero = l2_normalize(np.array([0.0, 0.0, 0.0]))
check("零向量被识别", z_is_zero, True)

section("3. OOV 与零向量处理")
oov4: set = set()
check("全 OOV 文档返回 None", pool_mean(["q", "w"], WV, oov4), None)
check("OOV 类型被收集", sorted(oov4), ["q", "w"])

records_edge = [
    {"doc_id": "ok1", "label": "L1", "tokens": ["a", "b"]},
    {"doc_id": "alloov", "label": "L1", "tokens": ["q", "w", "e"]},
    {"doc_id": "zerovec", "label": "L2", "tokens": ["z", "z"]},
    {"doc_id": "partial", "label": "L2", "tokens": ["a", "q", "c"]},
]
res = vectorize(records_edge, WV, "P1_mean")
st = res["stats"]
check("零向量文档被计入", st["zero_vector_docs"], 1)
check("丢弃文档数（全OOV + 零向量）", st["dropped_docs"], 2)
check("留存向量数", st["n_vectors"], 2)
# total_tokens = 2+3+2+3 = 10 ; OOV = 3(q,w,e) + 1(q) = 4
check("OOV token 计数（按出现次数）", st["oov_tokens"], 4)
check("token 总数", st["total_tokens"], 10)
check("OOV token 占比", st["oov_token_ratio"], round(4 / 10, 4))
check("OOV 类型数（去重）", st["unique_oov_types"], 3)
check("留存文档正是非丢弃的两个",
      [d["id"] for d in res["doc_index"]], ["ok1", "partial"])

section("4. 索引对齐（R9 防线，A3-2）")
for i, d in enumerate(res["doc_index"]):
    check(f"doc_index[{i}].row == {i}", d["row"], i)
check("matrix 行数 == doc_index 条数", res["matrix"].shape[0], len(res["doc_index"]))
check("标签随索引正确携带",
      [d["label"] for d in res["doc_index"]], ["L1", "L2"])

section("5. 批量向量化的三方案输出")
records = [
    {"doc_id": "x1", "label": "体育", "tokens": ["a", "a", "b"]},
    {"doc_id": "x2", "label": "体育", "tokens": ["a", "b", "c"]},
    {"doc_id": "x3", "label": "科技", "tokens": ["b", "c", "c"]},
    {"doc_id": "x4", "label": "科技", "tokens": ["c", "a", "b"]},
]
results = {}
for m, kw in (("P1_mean", {}), ("P2_tfidf", {"idf": IDF}),
              ("P3_nostop_mean", {"stopwords": STOP})):
    results[m] = vectorize(records, WV, m, **kw)
    check(f"{m} 输出形状 == (4,3)", list(results[m]["matrix"].shape), [4, 3])

check("P1 首行 == 手工推导值", results["P1_mean"]["matrix"][0], EXP_P1)
check("P2 首行 == 手工推导值", results["P2_tfidf"]["matrix"][0], EXP_P2)
check("P3 首行 == 手工推导值", results["P3_nostop_mean"]["matrix"][0], EXP_P3)

section("6. verify() 的 A3 系列断言")
chk = verify(results["P1_mean"], records)
for key in ("A3-1_dim", "A3-2_alignment", "A3-3_normalized",
            "A3-4_finite", "A3-5_no_zero_vector"):
    check(f"{key} 通过", chk[key]["ok"], True)
norms = np.linalg.norm(results["P1_mean"]["matrix"], axis=1)
check("A3-3 最大偏差在容差内",
      float(np.max(np.abs(norms - 1.0))) <= config.THRESHOLDS["A3-3_norm_tolerance"],
      True)

section("7. compare_methods()（A3-6：三方案须可区分）")
cmp_res = compare_methods(results)
print("  两两平均余弦:", cmp_res["pairwise_mean_cosine"])
check("A3-6 判定通过（两两余弦 < 阈值）", cmp_res["ok"], True)
check("阈值取自 config", cmp_res["threshold"],
      config.THRESHOLDS["A3-6_pooling_pairwise_cos_max"])

section("8. domain_separability()（A3-7）")
sep = domain_separability(records, results["P1_mean"]["matrix"],
                          results["P1_mean"]["doc_index"])
print("  ", sep)
check("类别数 = 2", sep["n_labels"], 2)
check("类内余弦在 [-1,1]", -1.0 <= sep["intra_cosine_mean"] <= 1.0, True)
check("类间余弦在 [-1,1]", -1.0 <= sep["inter_cosine_mean"] <= 1.0, True)
check("类内 > 类间（合成数据下应成立）",
      sep["intra_cosine_mean"] > sep["inter_cosine_mean"], True)

section("9. 异常输入")
try:
    vectorize(records, WV, "不存在的方案")
    check("未知方案应抛 ValueError", False, True)
except ValueError:
    check("未知方案抛 ValueError", True, True)

try:
    vectorize([{"doc_id": "only", "label": "L", "tokens": ["q"]}], WV, "P1_mean")
    check("全部文档失败应抛 RuntimeError", False, True)
except RuntimeError:
    check("全部文档失败抛 RuntimeError", True, True)

try:
    vectorize([{"label": "L", "tokens": ["a"]}], WV, "P1_mean")
    check("缺少 doc_id 应抛 ValueError", False, True)
except ValueError:
    check("缺少 doc_id 抛 ValueError", True, True)

print()
print("=" * 72)
print("结果: %d 通过, %d 失败" % (len(PASS), len(FAIL)))
if FAIL:
    print("失败项: %s" % FAIL)
print("=" * 72)
sys.exit(1 if FAIL else 0)
