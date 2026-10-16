# -*- coding: utf-8 -*-
"""阶段 3 · 文档向量化入口（PLAN.md §4，验收 A3-1 ~ A3-7）。

用法：
    .\\.venv39\\Scripts\\python.exe scripts\\run_vectorize.py --exp-id E1_baseline
    .\\.venv39\\Scripts\\python.exe scripts\\run_vectorize.py --exp-id best

对 U / L_train / L_val / L_test / 人民日报测试集分别做 P1/P2/P3 三种池化，
逐划分逐方案执行 A3 系列验收断言，并输出方案间对比（A3-6）与
领域可分性（A3-7）供选型。

内存提示
--------
L_train 约 19 万篇 × dim=200 × float32 ≈ 155 MB / 方案；
三方案同时驻留约 465 MB。故**逐方案处理并及时释放**，不一次性全部驻留。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import compat  # noqa: F401,E402
import config  # noqa: E402

from vectorize.doc_vector import (  # noqa: E402
    compare_methods,
    domain_separability,
    load_idf,
    load_stopwords,
    save_vectors,
    vectorize,
    verify,
)

# 各划分的数据来源与 token 字段名
SPLITS: Dict[str, Dict[str, Any]] = {
    "U": {"source": config.F_THUCNEWS_CLEAN, "filter_split": "U",
          "token_key": "tokens"},
    "L_train": {"source": config.F_THUCNEWS_CLEAN, "filter_split": "L",
                "token_key": "tokens"},
    "L_val": {"source": config.F_THUCNEWS_CLEAN, "filter_split": "L",
              "token_key": "tokens"},
    "L_test": {"source": config.F_THUCNEWS_CLEAN, "filter_split": "L",
               "token_key": "tokens"},
    "T_pd199801": {"source": config.F_PD1998_FORMATTED, "filter_split": None,
                   "token_key": "tokens_jieba"},
    "T_pd1998_all": {"source": config.F_PD1998_FORMATTED, "filter_split": None,
                     "token_key": "tokens_jieba"},
}


def load_keyed_vectors(exp_id: str):
    from gensim.models import KeyedVectors

    path = config.MODEL_W2V / exp_id / "model.wv"
    if not path.exists():
        raise FileNotFoundError(
            f"词向量不存在: {path}\n请先运行 train_word2vec（阶段 2）")
    return KeyedVectors.load(str(path))


def load_records_for_split(split: str, manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """按划分载入记录。L_train/val/test 依赖 split_manifest 的 doc_id 归属。"""
    spec = SPLITS[split]
    src = spec["source"]
    if not src.exists():
        raise FileNotFoundError(f"缺少清洗产物: {src}")

    token_key = spec["token_key"]

    # 人民日报：从独立标签文件连接已确认标签；格式文本本身不推断类别。
    if src == config.F_PD1998_FORMATTED:
        labels_by_id = {}
        if config.F_PD1998_LLM_LABELS.exists():
            with config.F_PD1998_LLM_LABELS.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    label_record = json.loads(line)
                    if (label_record.get("status") == "confirmed"
                            and label_record.get("label") in config.LABELS):
                        labels_by_id[label_record["article_id"]] = label_record["label"]
        out = []
        with src.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if split == "T_pd199801" and r["month"] != config.PD1998_PRIMARY_MONTH:
                    continue
                out.append({"article_id": r["article_id"],
                            "label": labels_by_id.get(r["article_id"]),
                            "llm_label": labels_by_id.get(r["article_id"]),
                            "month": r["month"],
                            token_key: r.get(token_key) or []})
        return out

    # THUCNews：按 doc_id 归属清单过滤
    if split == "U":
        wanted = None                      # U 侧整表即所需，无需清单
        keep_split = "U"
    else:
        sub = {"L_train": "train", "L_val": "val", "L_test": "test"}[split]
        ids_file = config.DATA_PROCESSED / f"doc_ids_{sub}.txt"
        if not ids_file.exists():
            # ⚠️ 绝不可回退为"整表载入 L"：那会让 L_test 包含 L_train 的
            #    全部文档，构成数据泄漏（PLAN.md §0.4-4 / R8），且结果看似
            #    正常（指标会虚高），属于最难察觉的错误类型。
            raise FileNotFoundError(
                f"缺少 doc_id 归属清单: {ids_file}\n"
                f"该清单由 scripts/build_splits.py 导出，是按划分取数的唯一凭证。\n"
                f"请重新运行 build_splits.py（勿用整表载入绕过，否则 L_test 会泄漏）。")
        wanted = set(ids_file.read_text(encoding="utf-8").split("\n")) - {""}
        if not wanted:
            raise ValueError(f"doc_id 归属清单为空: {ids_file}")
        keep_split = "L"

    out = []
    with src.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("split") != keep_split:
                continue
            if wanted is not None and r["doc_id"] not in wanted:
                continue
            out.append({"doc_id": r["doc_id"], "label": r["label"],
                        token_key: r.get(token_key) or []})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="阶段 3 文档向量化")
    ap.add_argument("--exp-id", required=True,
                    help="词向量实验目录名，如 E1_baseline / best")
    ap.add_argument("--splits", nargs="*", default=None,
                    help="只处理指定划分，默认全部")
    ap.add_argument("--methods", nargs="*", default=None,
                    choices=config.POOLING_METHODS,
                    help="只处理指定池化方案，默认 P1/P2/P3")
    args = ap.parse_args()

    config.ensure_dirs()

    if not config.F_SPLIT_MANIFEST.exists():
        print("缺少 split_manifest.json，请先运行 scripts/build_splits.py")
        return 2

    manifest = json.loads(config.F_SPLIT_MANIFEST.read_text(encoding="utf-8"))
    wv = load_keyed_vectors(args.exp_id)
    print(f"词向量 exp_id={args.exp_id}  词表={len(wv):,}  维度={wv.vector_size}")

    idf = load_idf()
    stopwords = load_stopwords()
    print(f"IDF 表 = {len(idf):,} 词（仅由 U 统计）")
    print(f"停用词 = {len(stopwords):,} 词（仅由 U 统计）")

    splits = args.splits or list(SPLITS)
    methods = args.methods or config.POOLING_METHODS

    summary: Dict[str, Any] = {
        "exp_id": args.exp_id,
        "vocab_size": len(wv),
        "vector_size": wv.vector_size,
        "splits": {},
        "method_comparison": {},
        "label_dependency_status": {},
        "pd_unlabelled_articles": {},
    }
    all_ok = True

    for split in splits:
        if split not in SPLITS:
            raise ValueError(f"未知划分 {split!r}；可选值: {', '.join(SPLITS)}")
        print(f"\n=== 划分 {split} ===", flush=True)
        try:
            records = load_records_for_split(split, manifest)
        except FileNotFoundError as exc:
            print(f"  [SKIP] {exc}")
            summary["splits"][split] = {"skipped": str(exc)}
            continue
        print(f"  记录数 = {len(records):,}", flush=True)
        if not records:
            print("  [SKIP] 无记录")
            summary["splits"][split] = {"skipped": "无记录"}
            if split.startswith("T_"):
                summary["label_dependency_status"][split] = {
                    "ok": False,
                    "reason": "无人日文章记录，不能验收 A3-7",
                }
            continue

        # A3-7 只使用已确认文章；未确认文章仍参与向量覆盖验收。
        if split.startswith("T_"):
            print("  人民日报标签依赖检查（A3-7）:", flush=True)
            labelled = [r for r in records if r.get("label") in config.LABELS]
            unlabelled_count = len(records) - len(labelled)
            label_counts = Counter(r["label"] for r in labelled)
            eligible_label_count = sum(1 for n in label_counts.values() if n >= 2)
            summary["pd_unlabelled_articles"][split] = unlabelled_count
            summary["label_dependency_status"][split] = {
                "ok": eligible_label_count >= 2,
                "reason": "A3-7 使用 confirmed 子集；未确认文章不计入类间可分性"
                          if eligible_label_count >= 2
                          else "confirmed 子集中不足两个至少含两篇文章的类别",
                "labelled_articles": len(labelled),
                "unlabelled_articles": unlabelled_count,
                "confirmed_label_counts": dict(sorted(label_counts.items())),
                "eligible_label_count": eligible_label_count,
            }
            print(f"  confirmed={len(labelled):,}；复核/无标签={unlabelled_count:,}；"
                  "向量化覆盖所有格式清洗文章", flush=True)

        per_method: Dict[str, Any] = {}
        vectors_for_compare: Dict[str, Any] = {}

        for method in methods:
            t0 = time.time()
            try:
                res = vectorize(records, wv, method,
                                token_key=SPLITS[split]["token_key"],
                                idf=idf, stopwords=stopwords)
            except Exception as exc:  # noqa: BLE001
                print(f"  [FAIL] {method}: {type(exc).__name__}: {exc}")
                all_ok = False
                continue

            paths = save_vectors(res, split, args.exp_id)
            chk = verify(res, records,
                         token_key=SPLITS[split]["token_key"],
                         idf=idf, stopwords=stopwords, wv=wv)
            if split == "U":
                sep = {"note": "U 无标签，跳过可分性计算"}
            elif split.startswith("T_") and not summary["label_dependency_status"][split]["ok"]:
                sep = {"note": "A3-7 pending：confirmed 子集中可用类别不足"}
            else:
                sep = domain_separability(records, res["matrix"], res["doc_index"])
                if split.startswith("T_"):
                    sep["n_confirmed_articles"] = summary["label_dependency_status"][split]["labelled_articles"]
                    sep["n_excluded_review_or_unlabelled"] = summary["label_dependency_status"][split]["unlabelled_articles"]

            ok = all(v["ok"] for v in chk.values()
                     if isinstance(v, dict) and "ok" in v)
            all_ok &= ok
            st = res["stats"]
            print(f"  {method:<15} 向量 {st['n_vectors']:>7,} × {st['dim']}  "
                  f"OOV率 {st['oov_token_ratio']:.4f}  "
                  f"丢弃 {st['dropped_docs']}  "
                  f"验收 {'PASS' if ok else 'FAIL'}  "
                  f"{time.time()-t0:.1f}s")
            if not ok:
                for k, v in chk.items():
                    if isinstance(v, dict) and not v.get("ok", True):
                        print(f"      [FAIL] {k}: {v}")

            per_method[method] = {
                "stats": st,
                "checks": chk,
                "domain_separability": sep,
                "paths": paths,
                "elapsed_sec": round(time.time() - t0, 1),
            }
            vectors_for_compare[method] = res
            del res
            gc.collect()

        # A3-6：方案间对比
        if len(vectors_for_compare) >= 2:
            cmp_res = compare_methods(vectors_for_compare)
            summary["method_comparison"][split] = cmp_res
            all_ok &= cmp_res["ok"]
            print(f"  A3-6 方案可区分性: {'PASS' if cmp_res['ok'] else 'FAIL'}  "
                  f"{cmp_res['pairwise_mean_cosine']}")

        # 释放该划分的矩阵
        vectors_for_compare.clear()
        gc.collect()

        summary["splits"][split] = per_method

    out = config.RESULT_METRICS / f"stage3_vectorize_{args.exp_id}.json"
    if args.splits and out.exists():
        previous = json.loads(out.read_text(encoding="utf-8"))
        for key in ("splits", "method_comparison", "label_dependency_status",
                    "pd_unlabelled_articles"):
            previous.setdefault(key, {}).update(summary.get(key, {}))
        previous["last_rerun_splits"] = splits
        summary = previous
    os.makedirs(out.parent, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"\n汇总 -> {out}")
    labels_pending = any(not item.get("ok") for item in summary["label_dependency_status"].values())
    if labels_pending:
        print("人民日报标签验收检查: PENDING（confirmed 标签类别数不足）")
    print(f"整体验收: {'PASS' if all_ok and not labels_pending else 'INCOMPLETE/FAIL'}")
    return 0 if all_ok and not labels_pending else 1


if __name__ == "__main__":
    raise SystemExit(main())
