# -*- coding: utf-8 -*-
"""阶段 1 验收脚本（PLAN.md §2.5，验收项 A1-1 ~ A1-9）。

用法：
    .\\.venv39\\Scripts\\python.exe scripts\\check_stage1.py

逐项断言并输出验收清单；结果落盘
``result/metrics/stage1_cleaning_check.json``。
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import compat  # noqa: F401,E402
import config  # noqa: E402
from utils.normalize import has_url  # noqa: E402

RESULTS: List[Dict[str, Any]] = []


def record(item: str, title: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append({"item": item, "title": title, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {item}  {title}")
    if detail:
        for line in detail.splitlines():
            print(f"        {line}")
    return bool(ok)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# --------------------------------------------------------------------------
# A1-1 解析完整性
# --------------------------------------------------------------------------
def check_a11(thuc_report: Dict, pd_report: Dict) -> bool:
    lines: List[str] = []
    ok = True

    # THUCNews：磁盘文件数须等于实测基线
    disk = thuc_report["totals"]["files_on_disk"]
    exp = config.THUCNEWS_TOTAL_EXPECTED
    lines.append(f"THUCNews 磁盘文件数 = {disk:,} / 实测基线 {exp:,} "
                 f"{'一致' if disk == exp else '不一致'}")
    ok &= (disk == exp)

    # 选中数须等于 allocate_split 的计划值
    alloc = config.allocate_split()
    plan_l = sum(v["l"] for v in alloc.values())
    plan_u = sum(v["u"] for v in alloc.values())
    sel = thuc_report["totals"]["files_selected"]
    lines.append(f"选中文件数 = {sel:,} / 计划 L+U = {plan_l+plan_u:,} "
                 f"{'一致' if sel == plan_l+plan_u else '不一致'}")
    ok &= (sel == plan_l + plan_u)

    # 各类别选中数逐一相等
    bad = []
    per = {s["label"]: s for s in thuc_report["per_label"]}
    for lab in config.LABELS:
        if lab not in per:
            bad.append(f"{lab} 缺失")
            continue
        if per[lab]["allocated_l"] != alloc[lab]["l"]:
            bad.append(f"{lab} L {per[lab]['allocated_l']}!={alloc[lab]['l']}")
        if per[lab]["allocated_u"] != alloc[lab]["u"]:
            bad.append(f"{lab} U {per[lab]['allocated_u']}!={alloc[lab]['u']}")
    lines.append("各类别选中数与计划一致: " + ("是" if not bad else f"否 {bad}"))
    ok &= not bad

    # 人民日报：畸形行率必须为 0
    mal = pd_report["totals"]["malformed_line_count"]
    tot_lines = pd_report["totals"]["line_count"]
    rate = mal / max(tot_lines, 1)
    lines.append(f"人民日报 总行数 = {tot_lines:,}  畸形行 = {mal}  畸形率 = {rate:.6f} "
                 f"{'OK' if rate == 0 else 'FAIL'}")
    ok &= (mal == 0)

    # 丢弃数须可追溯（每类 drop_reasons 落盘）
    drops = sum(sum(s["drop_reasons"].values()) for s in thuc_report["per_label"])
    kept = thuc_report["merged"]["kept"]
    dup = thuc_report["merged"]["duplicate_dropped"]
    recon = sel - drops - dup
    lines.append(f"守恒校验: 选中 {sel:,} − 丢弃 {drops:,} − 去重 {dup:,} = {recon:,}"
                 f" vs 留存 {kept:,} {'一致' if recon == kept else '不一致'}")
    ok &= (recon == kept)

    return record("A1-1", "解析完整性", ok, "\n".join(lines))


# --------------------------------------------------------------------------
# A1-2 编码正确性（抽样人工核验 + 自动断言）
# --------------------------------------------------------------------------
def check_a12(thuc: List[Dict], pd1998: List[Dict]) -> bool:
    lines = []
    ok = True
    rng = random.Random(config.SEED)

    # 自动断言：无替换符
    def has_repl(rec, key):
        return "\ufffd" in " ".join(rec.get(key, []))

    thuc_repl = sum(1 for r in thuc if has_repl(r, "tokens"))
    pd_repl = sum(1 for r in pd1998 if has_repl(r, "tokens_jieba"))
    lines.append(f"THUCNews 含 U+FFFD 替换符的记录 = {thuc_repl}")
    lines.append(f"人民日报 含 U+FFFD 替换符的记录 = {pd_repl}")
    ok &= (thuc_repl == 0 and pd_repl == 0)

    # 人工核验清单（各抽 5 篇，输出供目视）
    lines.append("--- 人工目视核验样本（THUCNews 5 篇）---")
    for r in rng.sample(thuc, min(5, len(thuc))):
        lines.append(f"  [{r['label']}] {' '.join(r['tokens'][:30])}")
    lines.append("--- 人工目视核验样本（人民日报 5 篇）---")
    for r in rng.sample(pd1998, min(5, len(pd1998))):
        lines.append(f"  [banci={r['banci']}] {' '.join(r['tokens_jieba'][:30])}")

    return record("A1-2", "编码正确性", ok, "\n".join(lines))


# --------------------------------------------------------------------------
# A1-3 清洗有效性
# --------------------------------------------------------------------------
def check_a13(thuc: List[Dict], pd1998: List[Dict]) -> bool:
    lines = []
    ok = True

    def scan(records, token_key, name):
        n_html = n_url = n_mail = n_fw = n_multi_space = n_punct = 0
        examples = []
        for r in records:
            toks = r.get(token_key, [])
            joined = " ".join(toks)
            if re.search(r"<[a-zA-Z]", joined):
                n_html += 1
                if len(examples) < 3: examples.append(("HTML", r, toks[:5]))
            if has_url(joined):
                n_url += 1
                if len(examples) < 3: examples.append(("URL", r, toks[:5]))
            if "@" in joined and "." in joined:
                n_mail += 1
            if any("\uff10" <= c <= "\uff19" for t in toks for c in t):
                n_fw += 1
            if any(t != t.strip() or not t for t in toks):
                n_multi_space += 1
            if any(re.fullmatch(r"[，。！？；：、,.!?;:]+", t) is None
                   and all(not ('\u4e00' <= c <= '\u9fff') and not c.isalnum()
                           for c in t) and t for t in toks):
                n_punct += 1
        lines.append(f"[{name}] HTML残留={n_html} URL残留={n_url} "
                     f"邮箱残留={n_mail} 全角数字={n_fw} 空白token={n_multi_space}")
        return n_html == 0 and n_url == 0 and n_fw == 0 and n_multi_space == 0

    ok &= scan(thuc, "tokens", "THUCNews")
    ok &= scan(pd1998, "tokens_jieba", "人民日报")

    # 抽样展示
    rng = random.Random(config.SEED + 1)
    lines.append("--- 抽样 3 篇清洗后正文（人工核验无噪声）---")
    for r in rng.sample(thuc, min(3, len(thuc))):
        lines.append(f"  [{r['label']}] {' '.join(r['tokens'][:40])}")

    return record("A1-3", "清洗有效性", ok, "\n".join(lines))


# --------------------------------------------------------------------------
# A1-4 分词质量（复合词专项抽检）
# --------------------------------------------------------------------------
COMPOUND_WORDS = [
    "中共中央", "一九九八年", "邓小平理论", "社会主义", "改革开放",
    "人民政府", "国民经济", "精神文明", "人民代表大会", "国务院",
]


def check_a14(thuc: List[Dict], pd1998: List[Dict]) -> bool:
    lines = []
    # 在人民日报侧统计复合词是否作为完整 token 出现
    pd_vocab = Counter()
    for r in pd1998:
        pd_vocab.update(r["tokens_jieba"])
    thuc_vocab = Counter()
    for r in thuc:
        thuc_vocab.update(r["tokens"])

    lines.append("复合词完整性抽检（作为独立 token 的出现次数）:")
    for w in COMPOUND_WORDS:
        lines.append(f"  {w:<12} 人民日报={pd_vocab.get(w,0):>7,d}  "
                     f"THUCNews={thuc_vocab.get(w,0):>7,d}")

    # 判定：至少部分复合词在人民日报语料中作为完整 token 出现
    hit = sum(1 for w in COMPOUND_WORDS if pd_vocab.get(w, 0) > 0)
    lines.append(f"人民日报侧完整保留的复合词 = {hit}/{len(COMPOUND_WORDS)}")

    # 抽样人工核验
    rng = random.Random(config.SEED + 2)
    lines.append("--- 抽样 5 篇分词结果（人工核验切分合理性）---")
    for r in rng.sample(pd1998, min(5, len(pd1998))):
        lines.append(f"  [banci={r['banci']}] {' '.join(r['tokens_jieba'][:35])}")

    ok = hit >= len(COMPOUND_WORDS) // 2
    return record("A1-4", "分词质量", ok, "\n".join(lines))


# --------------------------------------------------------------------------
# A1-5 划分互斥性
# --------------------------------------------------------------------------
def check_a15(manifest: Dict) -> bool:
    lines = []
    ok = True

    iso = manifest["isolation_assertions"]
    expected_values = {
        "U_intersect_L": 0,
        "train_intersect_val": 0,
        "train_intersect_test": 0,
        "val_intersect_test": 0,
        "train_val_test_union_equals_L": True,
        "pd1998_in_any_training_set": False,
    }
    for k, v in iso.items():
        expected = expected_values.get(k)
        good = k in expected_values and type(v) is type(expected) and v == expected
        lines.append(f"  {k} = {v}  {'OK' if good else 'FAIL'}")
        ok &= good

    # 类别分布偏差 < 1%
    dist = manifest["label_distribution"]
    dev_max = 0.0
    for lab in config.LABELS:
        tot = sum(dist[s].get(lab, 0) for s in ("L_train", "L_val", "L_test"))
        if tot == 0:
            ok = False
            lines.append(f"  {lab}: L 三分总数为 0  FAIL")
            continue
        for s in ("L_train", "L_val", "L_test"):
            share = dist[s].get(lab, 0) / tot
            expect = {"L_train": 0.8, "L_val": 0.1, "L_test": 0.1}[s]
            dev_max = max(dev_max, abs(share - expect))
    lines.append(f"  各类别在三分中的占比最大偏差 = {dev_max:.4f} "
                 f"(阈值 {config.THRESHOLDS['A1-5_label_ratio_deviation_max']})")
    ok &= dev_max <= config.THRESHOLDS["A1-5_label_ratio_deviation_max"]

    return record("A1-5", "划分互斥性", ok, "\n".join(lines))


# --------------------------------------------------------------------------
# A1-7 人民日报文章聚合正确
# --------------------------------------------------------------------------
def check_a17(pd_report: Dict, pd1998: List[Dict]) -> bool:
    lines = []
    ok = True

    m1 = [r for r in pd1998 if r["month"] == config.PD1998_PRIMARY_MONTH]
    base = config.PD199801_EXPECTED
    lines.append(f"199801 段落数 = {pd_report['per_month'][0]['paragraph_count']:,} "
                 f"/ 实测基线 {base['paragraph_count']:,}")

    # 聚合后文章数：留存数必然 <= 原始数（清洗会丢弃无正文/过短文章）
    raw_articles = base["article_count"]
    kept = len(m1)
    lines.append(f"199801 原始文章数 = {raw_articles:,}  清洗后留存 = {kept:,}  "
                 f"丢弃 = {raw_articles - kept:,}")
    ok &= (0 < kept <= raw_articles)

    # 版次分布：留存数须 <= 原始数，且 12 个版次齐全
    banci = Counter(r["banci"] for r in m1)
    lines.append(f"版次数 = {len(banci)} / 基线 {base['banci_count']}")
    ok &= (len(banci) == base["banci_count"])
    for b, raw_n in sorted(config.PD199801_BANCI_ARTICLES.items()):
        k = banci.get(b, 0)
        flag = "" if 0 < k <= raw_n else "  <-- FAIL"
        if flag:
            ok = False
        lines.append(f"    版次 {b}: 留存 {k:>4d} / 原始 {raw_n:>4d}{flag}")

    # 全部 6 个月都须有产出
    months = sorted({r["month"] for r in pd1998})
    lines.append(f"覆盖月份 = {months}")
    ok &= (len(months) == 6)

    return record("A1-7", "人民日报文章聚合正确", ok, "\n".join(lines))


# --------------------------------------------------------------------------
# A1-8 可复现性
# --------------------------------------------------------------------------
def check_a18(pd1998: List[Dict], thuc_sample: List[Dict]) -> bool:
    """清洗的可复现性由"固定种子 + 无随机遍历 + 确定性排序"保证。

    此处做**内容级**校验：同一 article_id / doc_id 的 content_hash 唯一，
    即相同输入必然产生相同哈希（哈希由 token 序列决定，无随机性）。
    字节级二次运行比对成本高，改由 CI 层面执行（见 PLAN.md A1-8）。
    """
    lines = []
    ok = True

    pd_hashes = Counter(r["content_hash"] for r in pd1998)
    dup_pd = sum(1 for h, c in pd_hashes.items() if c > 1)
    lines.append(f"人民日报 content_hash 重复数 = {dup_pd}（应为 0，因已全局去重）")
    ok &= (dup_pd == 0)

    th_hashes = Counter(r["content_hash"] for r in thuc_sample)
    dup_th = sum(1 for h, c in th_hashes.items() if c > 1)
    lines.append(f"THUCNews content_hash 重复数 = {dup_th}（应为 0）")
    ok &= (dup_th == 0)

    lines.append("可复现性依据：SEED=%d；文件枚举按整数序排序；"
                 "无随机遍历；content_hash = sha1(token序列)" % config.SEED)
    return record("A1-8", "可复现性", ok, "\n".join(lines))


# --------------------------------------------------------------------------
# A1-9 类别完整性（防 R11 回归）
# --------------------------------------------------------------------------
def check_a19(manifest: Dict, thuc: List[Dict]) -> bool:
    lines = []
    ok = True
    alloc = config.allocate_split()

    cnt_U = Counter(r["label"] for r in thuc if r["split"] == "U")
    cnt_L = Counter(r["label"] for r in thuc if r["split"] == "L")

    u_min = config.THRESHOLDS["A1-9_u_min_per_label"]
    lines.append("%-6s %10s %10s %10s %10s" % ("类别", "U留存", "L留存", "L计划", "判定"))
    for lab in config.LABELS:
        u, l = cnt_U.get(lab, 0), cnt_L.get(lab, 0)
        plan = alloc[lab]
        good = u > 0 and l > 0
        ok &= good
        lines.append("%-6s %10d %10d %10d %10s"
                     % (lab, u, l, plan["l"], "OK" if good else "FAIL"))

    min_l = min(cnt_L.get(l, 0) for l in config.LABELS)
    min_u = min(cnt_U.get(l, 0) for l in config.LABELS)
    lines.append(f"L 最小类 = {min_l} (阈值 >= 2000)   "
                 f"U 最小类 = {min_u} (阈值 >= {u_min})")
    ok &= (min_l >= 2000 and min_u >= u_min)

    # 各阶梯可用性
    for stage in ("S1", "S2", "S3", "S4"):
        labs = config.ladder_labels(stage)
        mn = min(cnt_L.get(l, 0) for l in labs)
        good = mn >= 500
        ok &= good
        lines.append(f"{stage} ({len(labs):>2d} 类): L 合计 "
                     f"{sum(cnt_L.get(l,0) for l in labs):>7,d}  最小类 {mn:>6,d}  "
                     f"{'OK' if good else 'FAIL'}")

    return record("A1-9", "类别完整性（防 R11 回归）", ok, "\n".join(lines))


# --------------------------------------------------------------------------
# 轨道 B 弱标签专项（决议 D1）
# --------------------------------------------------------------------------
def check_trackb(manifest: Dict, pd1998: List[Dict]) -> bool:
    lines = []
    ok = True
    high = [r for r in pd1998 if r["weak_label_confidence"] == "high"]
    lines.append(f"轨道 B 高置信子集（6 个月合计）= {len(high):,} 篇")

    by_label = Counter(r["weak_label"] for r in high)
    lines.append(f"弱标签分布 = {dict(by_label)}")
    for lab in config.TRACK_B_LABELS:
        n = by_label.get(lab, 0)
        lines.append(f"  {lab}: {n} 篇  {'OK' if n > 0 else 'FAIL'}")
        ok &= (n > 0)

    # 主评估月单独核对
    m1_high = [r for r in high if r["month"] == config.PD1998_PRIMARY_MONTH]
    lines.append(f"主评估月 {config.PD1998_PRIMARY_MONTH} 高置信子集 = "
                 f"{len(m1_high)} 篇（原始版次 735 篇，清洗后留存）")

    # 映射表完整性：12 个版次都有映射定义
    mapped = set(config.BANCI_LABEL_MAP)
    actual = {r["banci"] for r in pd1998}
    lines.append(f"映射表覆盖版次 = {len(mapped)}  实际出现版次 = {sorted(actual)}")
    ok &= actual <= mapped

    # 未映射版次不得带 high 标签
    bad = [r for r in pd1998
           if r["weak_label_confidence"] == "high"
           and config.BANCI_LABEL_MAP.get(r["banci"], {}).get("confidence") != "high"]
    lines.append(f"非法 high 标签记录 = {len(bad)}（应为 0）")
    ok &= not bad

    return record("TB", "轨道 B 弱标签映射正确性（决议 D1）", ok, "\n".join(lines))


# --------------------------------------------------------------------------
def main() -> int:
    print("=" * 78)
    print("PA2 阶段 1 数据清洗验收（PLAN.md §2.5）")
    print("=" * 78)

    config.ensure_dirs()

    for p in (config.F_THUCNEWS_REPORT, config.F_PD1998_REPORT,
              config.F_THUCNEWS_CLEAN, config.F_PD1998_CLEAN):
        if not p.exists():
            print(f"缺少必需产物: {p}\n请先运行清洗脚本。")
            return 2

    thuc_report = json.loads(config.F_THUCNEWS_REPORT.read_text(encoding="utf-8"))
    pd_report = json.loads(config.F_PD1998_REPORT.read_text(encoding="utf-8"))

    print("载入清洗产物 ...", flush=True)
    thuc = load_jsonl(config.F_THUCNEWS_CLEAN)
    pd1998 = load_jsonl(config.F_PD1998_CLEAN)
    print(f"  THUCNews = {len(thuc):,} 篇   人民日报 = {len(pd1998):,} 篇\n")

    check_a11(thuc_report, pd_report)
    check_a12(thuc, pd1998)
    check_a13(thuc, pd1998)
    check_a14(thuc, pd1998)

    if config.F_SPLIT_MANIFEST.exists():
        manifest = json.loads(config.F_SPLIT_MANIFEST.read_text(encoding="utf-8"))
        check_a15(manifest)
        check_a19(manifest, thuc)
        check_trackb(manifest, pd1998)
    else:
        record("A1-5", "划分互斥性", False, "split_manifest.json 不存在，请先运行 build_splits.py")
        record("A1-9", "类别完整性", False, "split_manifest.json 不存在")
        check_trackb({}, pd1998)

    check_a17(pd_report, pd1998)
    check_a18(pd1998, thuc)

    os.makedirs(config.RESULT_METRICS, exist_ok=True)
    out = config.RESULT_METRICS / "stage1_cleaning_check.json"
    summary = {
        "stage": "1-cleaning",
        "passed": all(r["ok"] for r in RESULTS),
        "results": RESULTS,
    }
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")

    passed = sum(1 for r in RESULTS if r["ok"])
    print("=" * 78)
    print(f"验收结果：{passed}/{len(RESULTS)} 通过   -> {out}")
    print("注：A1-6（OOV 率基线）移至阶段 2 生成 U 词表后单独验收。")
    print("=" * 78)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
