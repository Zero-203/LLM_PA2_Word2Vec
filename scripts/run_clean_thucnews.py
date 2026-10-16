# -*- coding: utf-8 -*-
"""阶段 1-A：THUCNews 全量清洗入口。

用法（全量）：
    .\\.venv39\\Scripts\\python.exe scripts\\run_clean_thucnews.py
用法（冒烟，每类每侧 N 篇）：
    .\\.venv39\\Scripts\\python.exe scripts\\run_clean_thucnews.py --limit 50

说明
----
- 受限环境下 ProcessPoolExecutor 创建命名管道会被拒
  （PermissionError: [WinError 5]），clean_thucnews.run() 会自动降级为
  单进程串行；本机 20 核，若能创建进程池则自动并行加速。
- 进度按类别与每 2000 篇打印，单篇超过 0.5s 会打印 [SLOW] 与文件名。
- 结束后校验报告中的关键计数，防止"跑完但数据不对"。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import compat  # noqa: F401,E402  (必须先于任何第三方库)
import config  # noqa: E402
from data import clean_thucnews  # noqa: E402


def verify(report: dict) -> int:
    """校验清洗报告的关键计数，返回失败项数量。"""
    problems: list[str] = []
    per_label = {s["label"]: s for s in report["per_label"]}
    alloc = config.allocate_split()

    print("\n" + "=" * 78)
    print("清洗结果校验")
    print("=" * 78)

    # 1) 类别完整性（验收 A1-9，防 R11 回归）
    for lab in config.LABELS:
        if lab not in per_label:
            problems.append(f"{lab}: 报告中缺失该类别")
            continue
        st = per_label[lab]
        if st["allocated_l"] != alloc[lab]["l"]:
            problems.append(
                f"{lab}: L 选中 {st['allocated_l']} != 计划 {alloc[lab]['l']}")
        if st["allocated_u"] != alloc[lab]["u"]:
            problems.append(
                f"{lab}: U 选中 {st['allocated_u']} != 计划 {alloc[lab]['u']}")
        if st["kept_l"] == 0:
            problems.append(f"{lab}: L 侧留存 0 篇，该类无法参与分类")
        if st["kept_u"] == 0:
            problems.append(f"{lab}: U 侧留存 0 篇，该类领域词汇未进入嵌入")

    # 2) 文件数守恒
    disk = sum(s["total_files_on_disk"] for s in report["per_label"])
    if disk != config.THUCNEWS_TOTAL_EXPECTED:
        problems.append(
            f"磁盘文件总数 {disk} != 预期 {config.THUCNEWS_TOTAL_EXPECTED}")

    # 3) 解码错误必须为 0（编码已实测确认为 UTF-8）
    decode_err = sum(
        s["drop_reasons"].get("decode_error", 0)
        for s in report["per_label"])
    if decode_err:
        problems.append(f"存在 {decode_err} 篇解码错误，须核查编码声明")

    # 4) 总表行数与报告一致
    total_file = config.F_THUCNEWS_CLEAN
    if total_file.exists():
        n_lines = sum(1 for _ in total_file.open("r", encoding="utf-8"))
        if n_lines != report["merged"]["kept"]:
            problems.append(
                f"总表行数 {n_lines} != 报告 kept {report['merged']['kept']}")
        print(f"总表行数 = {n_lines}")
    else:
        problems.append(f"总表不存在: {total_file}")

    print(f"磁盘文件总数 = {disk} / 预期 {config.THUCNEWS_TOTAL_EXPECTED}")
    print(f"选中文件数   = {report['totals']['files_selected']}")
    print(f"清洗留存     = {report['merged']['kept']}"
          f"（去重丢弃 {report['merged']['duplicate_dropped']}）")
    print(f"token 总量   = {report['merged']['total_tokens']:,}")
    print(f"解码错误     = {decode_err}")
    print(f"各类别 L/U 留存：")
    for lab in config.LABELS:
        st = per_label.get(lab, {})
        print(f"  {lab:<4} L {st.get('kept_l',0):>6d} / {st.get('allocated_l',0):>6d}"
              f"   U {st.get('kept_u',0):>6d} / {st.get('allocated_u',0):>6d}"
              f"   drop={st.get('drop_reasons',{})}")

    if problems:
        print("\n校验未通过：")
        for p in problems:
            print("  [FAIL]", p)
    else:
        print("\n校验全部通过。")
    return len(problems)


def main() -> int:
    ap = argparse.ArgumentParser(description="THUCNews 全量清洗")
    ap.add_argument("--limit", type=int, default=None,
                    help="冒烟模式：每类每侧最多处理 N 篇")
    ap.add_argument("--workers", type=int, default=None, help="目标并行进程数")
    args = ap.parse_args()

    mode = "冒烟" if args.limit else "全量"
    print(f"=== 阶段 1-A THUCNews 清洗（{mode}）===")
    print(f"seed={config.SEED}  encoding={config.ENCODING_THUCNEWS}  "
          f"min_doc_tokens={config.MIN_DOC_TOKENS}")
    alloc = config.allocate_split()
    print(f"划分策略: L 优先 + 小类补足")
    print(f"  L 合计 = {sum(v['l'] for v in alloc.values()):,} 篇")
    print(f"  U 合计 = {sum(v['u'] for v in alloc.values()):,} 篇")
    print(flush=True)

    t0 = time.time()
    report = clean_thucnews.run(max_workers=args.workers,
                                limit_per_label=args.limit)
    elapsed = time.time() - t0
    print(f"\n清洗总耗时 = {elapsed/60:.1f} 分钟")

    n_problems = verify(report)
    return 1 if n_problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
