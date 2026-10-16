# -*- coding: utf-8 -*-
"""THUCNews 清洗（PLAN.md §2.1 C1–C8）+ 划分选择（§2.4）。

设计要点
--------
1. **流式写盘**：48 万篇选中样本的 token 总量约 2.7 亿，若全部驻留内存
   需 16 GB+，故每类别由独立 worker 边清洗边写入分片 JSONL，主进程仅聚合计数。
2. **按类别并行**：14 个类别 → 最多 14 个进程，各写各的分片，无共享状态，
   避免 Windows spawn 模式下的序列化与竞态问题。
3. **先选文件再清洗**：只处理被选中的 48 万篇（占全量 57.5%），
   其余 35.5 万篇不读盘，显著降低耗时。
4. **U 取文件号尾部、L 取头部**（PLAN.md §2.4）：物理隔离，便于审计 U∩L=∅。
5. **执行顺序**：规范化 → 分词 → token 过滤（顺序不可颠倒，见 normalize 模块头部）。
6. **全局去重在合并阶段做**：跨类别重复稿也能被捕获。

产物
----
- ``data/interim/thucnews_parts/{U|L}__<label>.jsonl``  各类别分片
- ``data/interim/thucnews_clean.jsonl``                  合并+全局去重后的总表
- ``data/interim/thucnews_clean_report.json``            清洗报告
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import compat  # noqa: F401,E402  (必须先于任何第三方库导入)
import config  # noqa: E402
import jieba  # noqa: E402

jieba.setLogLevel(60)

from utils.normalize import clean_tokens, normalize_text  # noqa: E402
from utils.textio import DecodingError, read_text_strict  # noqa: E402

PARTS_DIR = config.DATA_INTERIM / "thucnews_parts"

# 进度与异常观测参数（全量清洗单类可达数万篇，串行模式下单类可能数十分钟）
PROGRESS_EVERY = 2000          # 每处理 N 篇打印一次进度
SLOW_DOC_THRESHOLD_SEC = 0.5   # 单篇超过该秒数即打印文件名，便于定位异常文档

# 冒烟模式限量：通过**环境变量**而非 monkey-patch 传递。
# Windows 的 ProcessPoolExecutor 使用 spawn 启动子进程，子进程会重新 import
# 本模块与 config，主进程对 config 函数的 monkey-patch **不会传播过去**，
# 会导致 --limit 静默失效、冒烟测试变成全量运行。环境变量可被继承。
ENV_LIMIT = "PA2_LIMIT_PER_LABEL"


def _limit_per_label() -> Optional[int]:
    """从环境变量读取冒烟限量（无则为 None，即全量）。"""
    v = os.environ.get(ENV_LIMIT)
    if not v:
        return None
    try:
        n = int(v)
    except ValueError:
        return None
    return n if n > 0 else None


def _alloc_for(label: str) -> Dict[str, int]:
    """取某类别的 U/L 分配篇数，冒烟模式下按 limit 收紧。"""
    alloc = config.allocate_split()[label]
    lim = _limit_per_label()
    if lim:
        alloc = {"total": alloc["total"],
                 "l": min(alloc["l"], lim),
                 "u": min(alloc["u"], lim)}
    return alloc


# --------------------------------------------------------------------------
# 文件枚举与划分选择
# --------------------------------------------------------------------------
def _file_sort_key(p: Path) -> Tuple[int, str]:
    """按文件名的整数序排序。

    THUCNews 文件名为 ``0.txt``、``1.txt`` … ``131603.txt``，
    若按字符串排序会得到 10.txt < 2.txt 的错误顺序，
    导致"U 取尾部、L 取头部"的隔离失效。
    """
    stem = p.stem
    return (int(stem), "") if stem.isdigit() else (1 << 62, stem)


def list_label_files(label: str) -> List[Path]:
    """列出某类别下全部 .txt 文件，按整数序升序。"""
    d = config.DATA_THUCNEWS / label
    if not d.is_dir():
        raise FileNotFoundError(f"类别目录不存在: {d}")
    files = [Path(e.path) for e in os.scandir(d) if e.name.endswith(".txt")]
    files.sort(key=_file_sort_key)
    return files


def select_split_files(label: str) -> Tuple[List[Path], List[Path], int]:
    """返回 (L 文件列表, U 文件列表, 该类别总文件数)。

    L 取头部、U 取尾部，二者不重叠（由 ``allocate_split`` 保证 u+l <= n）。
    """
    files = list_label_files(label)
    alloc = _alloc_for(label)
    n_l, n_u = alloc["l"], alloc["u"]
    assert n_l + n_u <= len(files), (
        f"{label}: L({n_l}) + U({n_u}) = {n_l+n_u} > 实际文件数 {len(files)}"
    )
    return files[:n_l], files[len(files) - n_u:] if n_u else [], len(files)


# --------------------------------------------------------------------------
# 单篇清洗
# --------------------------------------------------------------------------
def clean_document(path: Path, label: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """清洗单篇文档。返回 (记录, 丢弃原因)；成功时原因为 None。"""
    try:
        raw_text = read_text_strict(path, config.ENCODING_THUCNEWS)
    except (DecodingError, OSError):
        return None, "decode_error"

    lines = [ln.strip() for ln in raw_text.splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return None, "empty_file"

    title_raw = lines[0]
    body_raw = "".join(lines[1:]) if len(lines) > 1 else ""

    # 顺序：规范化 → 分词 → 过滤（见模块头部说明）
    body_norm = normalize_text(body_raw)
    title_norm = normalize_text(title_raw)

    # jieba 已在模块级导入并关闭日志，此处直接使用（避免每篇重复导入查找）
    body_tokens = clean_tokens(jieba.cut(body_norm))
    title_tokens = clean_tokens(jieba.cut(title_norm))

    if len(body_tokens) < config.MIN_DOC_TOKENS:
        return None, "too_short"

    content_hash = hashlib.sha1(
        " ".join(body_tokens).encode("utf-8")
    ).hexdigest()

    record = {
        "doc_id": f"{label}/{path.name}",
        "label": label,
        "title_tokens": title_tokens,
        "tokens": body_tokens,
        "token_count": len(body_tokens),
        "content_hash": content_hash,
    }
    return record, None


# --------------------------------------------------------------------------
# 每类别 worker
# --------------------------------------------------------------------------
def process_label(label: str) -> Dict[str, Any]:
    """清洗单个类别的 L 与 U 文件，流式写入分片。

    带分阶段计时与进度输出：全量清洗单类可达数万篇、串行模式下单次运行
    可能数十分钟，若无进度信息则无法区分"正在正常推进"与"已挂住"。
    ``SLOW_DOC_THRESHOLD_SEC`` 以上的单篇会立即打印文件名，
    使异常文档可直接定位，而不必事后猜测。
    """
    # jieba 已在模块级导入（子进程 spawn 时会重新 import 本模块，故同样生效）。
    # 每进程独立 jieba 缓存目录，避免多进程同时写 jieba.cache 造成损坏。
    cache_dir = config.DATA_INTERIM / "jieba_cache" / str(os.getpid())
    os.makedirs(cache_dir, exist_ok=True)
    jieba.dt.tmp_dir = str(cache_dir)

    t0 = time.time()
    t = time.time()
    l_files, u_files, total = select_split_files(label)
    t_select = round(time.time() - t, 3)

    stats: Dict[str, Any] = {
        "label": label,
        "total_files_on_disk": total,
        "allocated_l": len(l_files),
        "allocated_u": len(u_files),
        "not_selected": total - len(l_files) - len(u_files),
        "drop_reasons": Counter(),
        "kept_l": 0,
        "kept_u": 0,
        "tokens_l": 0,
        "tokens_u": 0,
        "decode_error_examples": [],
        "elapsed_sec": 0.0,
        "timing": {"select_split_files_sec": t_select,
                   "clean_loop_sec": 0.0,
                   "slowest_docs": []},
        "part_files": [],
    }

    n_all = len(l_files) + len(u_files)
    done = 0
    slowest: List[Tuple[float, str]] = []
    t_loop = time.time()

    for split, files, kept_key, tok_key in (
        ("L", l_files, "kept_l", "tokens_l"),
        ("U", u_files, "kept_u", "tokens_u"),
    ):
        if not files:
            continue
        out_path = PARTS_DIR / f"{split}__{label}.jsonl"
        stats["part_files"].append(str(out_path))
        with out_path.open("w", encoding="utf-8", newline="\n") as fh:
            for fp in files:
                t_doc = time.time()
                rec, reason = clean_document(fp, label)
                dt = time.time() - t_doc

                if dt >= SLOW_DOC_THRESHOLD_SEC:
                    print(f"  [SLOW] {label}/{fp.name} = {dt:.2f}s", flush=True)
                slowest.append((dt, f"{split}/{fp.name}"))

                done += 1
                if done % PROGRESS_EVERY == 0 or done == n_all:
                    elapsed = time.time() - t_loop
                    rate = done / max(elapsed, 1e-9)
                    eta = (n_all - done) / max(rate, 1e-9)
                    print(f"  [{label}] {done}/{n_all} "
                          f"{rate:.0f}篇/s  ETA {eta/60:.1f}min", flush=True)

                if rec is None:
                    stats["drop_reasons"][reason] += 1
                    if reason == "decode_error" and len(stats["decode_error_examples"]) < 5:
                        stats["decode_error_examples"].append(str(fp))
                    continue
                rec["split"] = split
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats[kept_key] += 1
                stats[tok_key] += rec["token_count"]

    slowest.sort(reverse=True)
    stats["timing"]["clean_loop_sec"] = round(time.time() - t_loop, 3)
    stats["timing"]["slowest_docs"] = [
        {"doc": name, "sec": round(dt, 3)} for dt, name in slowest[:10]
    ]
    stats["drop_reasons"] = dict(stats["drop_reasons"])
    stats["elapsed_sec"] = round(time.time() - t0, 1)
    return stats


# --------------------------------------------------------------------------
# 合并 + 全局去重
# --------------------------------------------------------------------------
def merge_parts(label_order: List[str], min_docs: int) -> Dict[str, Any]:
    """合并各类别分片，执行全局去重（C7）与最小长度过滤，写出总表。"""
    seen_hashes: set[str] = set()
    out_path = config.F_THUCNEWS_CLEAN
    os.makedirs(out_path.parent, exist_ok=True)

    kept = 0
    dup = 0
    short = 0
    total_tokens = 0
    per_label = Counter()
    per_split = Counter()

    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        for split in ("L", "U"):
            for label in label_order:
                part = PARTS_DIR / f"{split}__{label}.jsonl"
                if not part.exists():
                    continue
                with part.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        h = rec["content_hash"]
                        if h in seen_hashes:
                            dup += 1
                            continue
                        seen_hashes.add(h)
                        if rec["token_count"] < min_docs:
                            short += 1
                            continue
                        out.write(line + "\n")
                        kept += 1
                        total_tokens += rec["token_count"]
                        per_label[label] += 1
                        per_split[split] += 1

    return {
        "kept": kept,
        "duplicate_dropped": dup,
        "too_short_dropped": short,
        "total_tokens": total_tokens,
        "per_label": dict(per_label),
        "per_split": dict(per_split),
        "output_file": str(out_path),
    }


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def run(max_workers: Optional[int] = None, limit_per_label: Optional[int] = None) -> Dict[str, Any]:
    """执行 THUCNews 清洗。

    Parameters
    ----------
    limit_per_label : 仅用于小样本冒烟，每类每侧最多处理 N 篇。
    """
    config.ensure_dirs()
    os.makedirs(PARTS_DIR, exist_ok=True)

    labels = config.LABELS
    workers = max_workers or min(len(labels), config.W2V_WORKERS)

    # 冒烟限量必须走环境变量：Windows spawn 子进程会重新 import config，
    # 主进程对 config 函数的 monkey-patch 不会传播，会导致限量静默失效。
    if limit_per_label:
        os.environ[ENV_LIMIT] = str(limit_per_label)
    else:
        os.environ.pop(ENV_LIMIT, None)

    eff_limit = _limit_per_label()
    print(f"清洗模式: {'冒烟（每类每侧 <= %d 篇）' % eff_limit if eff_limit else '全量'}"
          f"  目标并行度 = {workers}", flush=True)

    t0 = time.time()
    label_stats: List[Dict[str, Any]] = []

    def _report(st: Dict[str, Any]) -> None:
        label_stats.append(st)
        print(f"  [{st['label']}] 磁盘 {st['total_files_on_disk']:>7d} | "
              f"L 选 {st['allocated_l']:>6d} 留 {st['kept_l']:>6d} | "
              f"U 选 {st['allocated_u']:>6d} 留 {st['kept_u']:>6d} | "
              f"{st['elapsed_sec']:>6.1f}s", flush=True)

    # 执行模式：优先多进程；若环境禁止创建进程池（受限沙箱下
    # mp.Pipe 的 CreateFile 会抛 PermissionError: [WinError 5]），
    # 自动降级为单进程串行，保证清洗在任何环境都能完成。
    used_workers = workers
    mode = "parallel"
    if workers <= 1:
        mode = "serial"
    else:
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(process_label, lab): lab for lab in labels}
                for fut in as_completed(futures):
                    lab = futures[fut]
                    try:
                        _report(fut.result())
                    except Exception as exc:  # noqa: BLE001
                        print(f"  [ERROR] {lab}: {type(exc).__name__}: {exc}")
                        raise
        except (PermissionError, OSError) as exc:
            print(f"  [WARN] 进程池不可用（{type(exc).__name__}: {exc}），"
                  f"降级为单进程串行执行。", flush=True)
            mode = "serial (auto-fallback)"
            used_workers = 1
            label_stats.clear()

    if mode.startswith("serial"):
        for lab in labels:
            _report(process_label(lab))

    print(f"\n执行模式 = {mode}，实际并行度 = {used_workers}")

    label_stats.sort(key=lambda s: config.LABELS.index(s["label"]))

    print("\n合并分片并执行全局去重 ...")
    merged = merge_parts(labels, config.MIN_DOC_TOKENS)

    total_elapsed = round(time.time() - t0, 1)
    report = {
        "stage": "1-thucnews-cleaning",
        "config": {
            "seed": config.SEED,
            "encoding": config.ENCODING_THUCNEWS,
            "min_doc_tokens": config.MIN_DOC_TOKENS,
            "u_cap_per_label": config.U_CAP_PER_LABEL,
            "l_cap_per_label": config.L_CAP_PER_LABEL,
            "u_min_per_label": config.U_MIN_PER_LABEL,
            "split_strategy": "L 优先 + 小类补足（config.allocate_split）",
            "selection": "L 取文件号头部，U 取尾部，物理隔离",
            "token_filter_rule": "含汉字或字母→保留；纯标点/纯阿拉伯数字→剔除",
            "order": "规范化 → jieba 分词 → token 过滤",
            "limit_per_label": limit_per_label,
            "workers_requested": workers,
            "workers_actual": used_workers,
            "execution_mode": mode,
        },
        "totals": {
            "labels": len(labels),
            "files_on_disk": sum(s["total_files_on_disk"] for s in label_stats),
            "files_selected": sum(s["allocated_l"] + s["allocated_u"] for s in label_stats),
            "files_not_selected": sum(s["not_selected"] for s in label_stats),
            "kept_after_merge": merged["kept"],
            "duplicate_dropped": merged["duplicate_dropped"],
            "total_tokens": merged["total_tokens"],
            "elapsed_sec": total_elapsed,
        },
        "per_label": label_stats,
        "merged": merged,
    }

    config.F_THUCNEWS_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n报告 -> {config.F_THUCNEWS_REPORT}")
    print(f"总表 -> {merged['output_file']}  ({merged['kept']} 篇, "
          f"{merged['total_tokens']:,} tokens, 去重丢弃 {merged['duplicate_dropped']})")
    print(f"总耗时 {total_elapsed}s")
    return report


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="THUCNews 清洗")
    ap.add_argument("--limit", type=int, default=None,
                    help="冒烟模式：每类每侧最多处理 N 篇")
    ap.add_argument("--workers", type=int, default=None, help="并行进程数")
    args = ap.parse_args()
    run(max_workers=args.workers, limit_per_label=args.limit)
