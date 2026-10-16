"""阶段 2 后运行：使用 U 词表验收 A1-6 OOV 率基线。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import compat  # noqa: F401,E402
import config  # noqa: E402


def main() -> int:
    vocab_path = config.DATA_PROCESSED / "u_vocab.txt"
    test_ids_path = config.DATA_PROCESSED / "doc_ids_test.txt"
    required = (vocab_path, test_ids_path,
                config.F_THUCNEWS_CLEAN, config.F_PD1998_CLEAN)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("A1-6 尚不可验收，缺少依赖：")
        for path in missing:
            print(f"  {path}")
        return 2

    vocabulary = {word for word in vocab_path.read_text(encoding="utf-8").splitlines()
                  if word}
    if not vocabulary:
        print(f"U 词表为空：{vocab_path}")
        return 2

    test_ids = {doc_id for doc_id in test_ids_path.read_text(encoding="utf-8").splitlines()
                if doc_id}
    found_ids: set[str] = set()
    l_total = l_oov = l_docs = 0
    with config.F_THUCNEWS_CLEAN.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            doc_id = record.get("doc_id")
            if doc_id not in test_ids:
                continue
            found_ids.add(doc_id)
            tokens = record.get("tokens") or []
            l_docs += 1
            l_total += len(tokens)
            l_oov += sum(token not in vocabulary for token in tokens)

    missing_ids = test_ids - found_ids
    if missing_ids:
        print(f"L_test doc_id 清单有 {len(missing_ids)} 个 ID 未在清洗总表中找到。")
        return 1

    pd_total = pd_oov = pd_docs = 0
    with config.F_PD1998_CLEAN.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("month") != config.PD1998_PRIMARY_MONTH:
                continue
            tokens = record.get("tokens_jieba") or []
            pd_docs += 1
            pd_total += len(tokens)
            pd_oov += sum(token not in vocabulary for token in tokens)

    if not l_total or not pd_total:
        print("A1-6 无有效 token，无法计算 OOV 率。")
        return 1

    l_rate = l_oov / l_total
    pd_rate = pd_oov / pd_total
    result: Dict[str, Any] = {
        "item": "A1-6",
        "u_vocab_size": len(vocabulary),
        "L_test": {
            "documents": l_docs, "tokens": l_total,
            "oov_tokens": l_oov, "oov_rate": l_rate,
        },
        "T_pd199801": {
            "documents": pd_docs, "tokens": pd_total,
            "oov_tokens": pd_oov, "oov_rate": pd_rate,
        },
        "oov_rate_difference_pd199801_minus_L_test": pd_rate - l_rate,
        "interpretation": "人民日报 OOV 偏高属预期，反映 1998 与 2010s 的领域漂移。",
        "passed": True,
    }
    out = config.RESULT_METRICS / "a1_6_oov_baseline.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")

    print(f"A1-6 OOV 率基线（U 词表 {len(vocabulary):,} 词）")
    for name in ("L_test", "T_pd199801"):
        stats = result[name]
        print(f"  {name}: {stats['documents']:,} 篇，{stats['tokens']:,} tokens，"
              f"OOV {stats['oov_tokens']:,}，率 {stats['oov_rate']:.4f}")
    print(f"  差值（人民日报 − L_test）= {pd_rate - l_rate:+.4f}")
    print(f"验收记录 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())