# -*- coding: utf-8 -*-
"""人民日报 1998 语料清洗（PLAN.md §2.2 D1–D7）。

语料格式
--------
每行一个段落，前缀为段落编号，正文是**北大 PFR 标准的已分词+词性标注**文本：

    19980101-01-001-004/m  中共中央/nt  总书记/n  、/w  国家/n  主席/n  ...
    └─日期──┘└版次┘└文章┘└段┘/m

- 版次(banci) = 第 2 段编号，是**报纸版面**，不是新闻主题类别（PLAN.md §0.3）
- 成分标注形如 ``[中央/n 人民/n 广播/vn 电台/n]nt``：方括号内是词，
  方括号外紧跟的是短语级标签，须一并剥离

处理要点
--------
1. **D7 统一分词器**：本语料自带北大分词，但 THUCNews 侧用 jieba。
   两套切分粒度不同（如 ``中共中央`` vs ``中共/中央``）会造成词表错配，
   虚高 OOV 率，使跨域评估无法归因（PLAN.md §2.3）。
   故主方案：剥离词性得到纯文本后**用 jieba 重新分词**，
   同时保留北大原分词 ``tokens_pku`` 供消融对照。
2. **D4 文章级聚合**：分类对象是文章，不是段落。按 article_id 聚合。
3. **D5 版面噪声过滤**：图片说明、记者署名、日期括注等对主题分类无信息量。
4. **执行顺序**：剥词性 → 规范化 → jieba 分词 → token 过滤
   （顺序不可颠倒，见 normalize 模块头部说明）。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import compat  # noqa: F401,E402  (必须先于任何第三方库)
import config  # noqa: E402
import jieba  # noqa: E402

jieba.setLogLevel(60)

from utils.normalize import clean_tokens, normalize_text  # noqa: E402
from utils.textio import DecodingError, decode_bytes_strict  # noqa: E402

# --------------------------------------------------------------------------
# D2：段落编号解析
# --------------------------------------------------------------------------
# 形如 19980101-01-001-004/m  正文...
_RE_RECORD = re.compile(
    r"^(?P<date>\d{8})-(?P<banci>\d{2})-(?P<article>\d{3})-(?P<para>\d{3})/m"
    r"(?:\s+(?P<body>.*))?$"
)

# --------------------------------------------------------------------------
# D3：剥离词性与成分标注
# --------------------------------------------------------------------------
# 成分标注的短语级标签，如 ]nt ]ns ]nz。剥离方括号与该标签。
_RE_CLOSE_BRACKET = re.compile(r"\][a-zA-Z]*")
# 开方括号
_RE_OPEN_BRACKET = re.compile(r"\[")
_RE_POS_BEFORE_PUNCT = re.compile(r"/[a-zA-Z]+(?=[，。！？；：、,.!?;:])")


def strip_pos(raw_body: str) -> Tuple[List[str], int]:
    """D3：剥离词性标注与成分标注，返回 (词列表, 标注异常计数)。

    对每个空格分隔的 token：
    - 去掉开头的 ``[``
    - 去掉 ``]xx`` 形式的短语标签（如 ``电台/n]nt`` → ``电台/n`` → ``电台``）
    - 用 rsplit('/', 1) 取词（词本身可能含 ``/``，故用 rsplit 而非 split）
    - 无 ``/`` 的 token 视为标注缺失，原样保留并计数
    """
    words: List[str] = []
    untagged = 0

    for raw in raw_body.split():
        tok = _RE_CLOSE_BRACKET.sub("", raw)
        tok = _RE_OPEN_BRACKET.sub("", tok)
        tok = _RE_POS_BEFORE_PUNCT.sub("", tok)
        tok = tok.strip()
        if not tok:
            continue
        if "/" in tok:
            word, tag = tok.rsplit("/", 1)
            if not word or not re.fullmatch(r"[a-zA-Z]+", tag):
                untagged += 1
                words.append(tok)
                continue
            if word:
                words.append(word)
        else:
            untagged += 1
            words.append(tok)

    return words, untagged


# --------------------------------------------------------------------------
# D5：版面噪声过滤
# --------------------------------------------------------------------------
# 全角括号内的版面噪声：图片说明、记者署名、电讯头、日期括注
_RE_PAREN_IMAGE = re.compile(r"（[^（）]*附[^（）]*图片[^（）]*）")
_RE_PAREN_BYLINE = re.compile(
    r"（[^（）]*(?:新华社|本报|中新社|本报记者|摄|电)[^（）]*）"
)
_RE_PAREN_DATE_ONLY = re.compile(
    r"^（[^（）]*(?:年|月|日)[^（）]*）$"
)
# 半角括号版本（语料中少量存在）
_RE_HALF_PAREN_IMAGE = re.compile(r"\([^()]*图片[^()]*\)")

# 整段即为日期括注（如"（一九九七年十二月三十一日）"）
_RE_PARA_DATE_ONLY = re.compile(r"^（[^（）]*）$|^[\(][^()]*[\)]$")
# 整段即为记者署名/电讯头（如"新华社北京12月31日电"、"本报记者XXX"）
_RE_PARA_BYLINE = re.compile(
    r"^(?:据|本报|新华|中新)"            # 以通讯社/报社开头
    r"|记者.{0,12}(?:报道|摄|讯)"        # 记者…报道/摄/讯
    r"|^\s*[\(（][^（）()]*电[\)）]\s*$"  # 整段为 (…电)
)


def strip_layout_noise(text: str) -> Tuple[str, int]:
    """D5：剔除段内版面噪声括注，返回 (清洗后文本, 剔除的括注数)。

    只剔除**可识别的噪声模式**，不做无差别的括号删除——
    正文中的引语、补充说明常带括号，删掉会破坏语义完整性。
    """
    removed = 0
    for pat in (_RE_PAREN_IMAGE, _RE_PAREN_BYLINE, _RE_HALF_PAREN_IMAGE):
        text, n = pat.subn(" ", text)
        removed += n
    return text, removed


def is_layout_paragraph(text: str) -> bool:
    """判断**整段**是否为版面噪声段，应整段丢弃。

    ⚠️ 必须逐段判断，不能对拼接后的正文判断：
    实测若把全部段落拼接后再匹配 ``^（…）$``，该锚定模式永远不触发，
    日期括注段（如"（一九九七年十二月三十一日）"）会残留在正文里。
    """
    t = text.strip()
    if not t:
        return True
    if _RE_PARA_DATE_ONLY.match(t):
        return True
    if _RE_PARA_BYLINE.search(t) and len(t) <= 40:
        # 署名/电讯头通常很短；限制长度避免误删含"记者"的正文长段
        return True
    return False


# --------------------------------------------------------------------------
# 段落级解析
# --------------------------------------------------------------------------
def parse_paragraph(line: str) -> Optional[Dict[str, str]]:
    """解析单行段落。畸形行返回 None（由调用方计数，不静默丢弃）。"""
    m = _RE_RECORD.match(line.strip())
    if m is None:
        return None
    g = m.groupdict()
    article_id = f"{g['date']}-{g['banci']}-{g['article']}"
    return {
        "date": g["date"],
        "banci": g["banci"],
        "article_no": g["article"],
        "para_no": g["para"],
        "article_id": article_id,
        "paragraph_id": f"{article_id}-{g['para']}",
        "body": g["body"] or "",
    }


def iter_records(path: Path) -> Iterator[Tuple[Optional[Dict[str, str]], str]]:
    """流式逐行解析，返回 (记录或None, 原始行)。"""
    raw = path.read_bytes()
    text = decode_bytes_strict(raw, config.ENCODING_PD1998, path)
    for line in text.splitlines():
        if not line.strip():
            continue
        yield parse_paragraph(line), line


# --------------------------------------------------------------------------
# 文章聚合与清洗
# --------------------------------------------------------------------------
def clean_month(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """清洗单个月份文件，返回 (文章记录列表, 统计信息)。"""
    t0 = time.time()

    # 1) 按 article_id 聚合段落
    articles: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    stats: Dict[str, Any] = {
        "source_file": path.name,
        "month": path.stem,
        "line_count": 0,
        "blank_line_count": 0,
        "malformed_line_count": 0,
        "malformed_examples": [],
        "paragraph_count": 0,
        "empty_body_paragraph_count": 0,
        "untagged_token_count": 0,
        "layout_noise_removed": 0,
        "layout_paragraph_dropped": 0,
        "article_count": 0,
        "article_dropped_too_short": 0,
        "article_dropped_no_body": 0,
        "banci_distribution": Counter(),
        "token_count_jieba": 0,
        "token_count_pku": 0,
        "elapsed_sec": 0.0,
    }

    raw = path.read_bytes()
    text = decode_bytes_strict(raw, config.ENCODING_PD1998, path)
    for line in text.splitlines():
        if not line.strip():
            stats["blank_line_count"] += 1
            continue
        stats["line_count"] += 1
        rec = parse_paragraph(line)
        if rec is None:
            stats["malformed_line_count"] += 1
            if len(stats["malformed_examples"]) < 5:
                stats["malformed_examples"].append(line[:200])
            continue

        stats["paragraph_count"] += 1
        words_pku, untagged = strip_pos(rec["body"])
        stats["untagged_token_count"] += untagged
        if not rec["body"].strip():
            stats["empty_body_paragraph_count"] += 1

        aid = rec["article_id"]
        if aid not in articles:
            articles[aid] = {
                "article_id": aid,
                "date": rec["date"],
                "banci": rec["banci"],
                "paragraphs": [],
            }
            order.append(aid)
            stats["banci_distribution"][rec["banci"]] += 0  # 占位，稍后按文章计
        articles[aid]["paragraphs"].append((rec["para_no"], words_pku))

    # 2) 逐文章清洗
    out_records: List[Dict[str, Any]] = []
    banci_articles: Counter = Counter()

    for aid in order:
        art = articles[aid]
        paras = sorted(art["paragraphs"], key=lambda x: x[0])

        # 段落 -001 通常视为标题，其余为正文（语料惯例）。
        # ⚠️ 实测修正（2026-10-12）：本语料存在大量**单段成篇**的新闻简讯
        # （多为 03 版国际简讯，60–170 字），其完整内容就在段落 -001 内，
        # 并无独立标题。若一律把 -001 当标题，这些文章正文为空，
        # 会被 article_dropped_no_body 丢弃——实测误丢 351 篇，
        # 且恰好是时政类的重要内容。
        # 故：当文章只有 1 个段落时，该段落**同时**作为标题与正文。
        if len(paras) == 1:
            title_words_pku = paras[0][1]
            body_paras = paras          # 单段：标题段兼作正文
            single_para = True
        else:
            title_words_pku = paras[0][1]
            body_paras = paras[1:]
            single_para = False

        # D5：版面噪声过滤。
        # **逐段**判断整段是否为署名/电讯头/日期括注段（is_layout_paragraph），
        # 再对保留段落做段内括注剔除（strip_layout_noise）。
        # 顺序不可颠倒：若先拼接整篇再匹配锚定模式 ^（…）$，则永不触发。
        kept_paras: List[Tuple[str, List[str]]] = []
        body_parts: List[str] = []
        pku_parts: List[List[str]] = []
        formatted_paragraphs: List[Dict[str, str]] = []
        for para_no, words in body_paras:
            para_text = "".join(words)
            if is_layout_paragraph(para_text):
                stats["layout_paragraph_dropped"] += 1
                continue
            cleaned, n_removed = strip_layout_noise(para_text)
            stats["layout_noise_removed"] += n_removed
            if not cleaned.strip():
                stats["layout_paragraph_dropped"] += 1
                continue
            kept_paras.append((para_no, words))
            normalized_para = normalize_text(cleaned)
            body_parts.append(normalized_para)
            pku_parts.append(words)
            formatted_paragraphs.append({
                "paragraph_no": para_no,
                "text": normalized_para,
            })

        title_text, n1 = strip_layout_noise("".join(title_words_pku))
        stats["layout_noise_removed"] += n1
        body_text = "".join(body_parts)

        # D3 保留的北大原分词（剥词性后、未重新分词），同样只取保留段落
        tokens_pku = clean_tokens(title_words_pku + [
            t for ws in pku_parts for t in ws
        ])

        # D7：规范化 → jieba 重新分词 → 过滤
        title_norm = normalize_text(title_text)
        body_norm = body_text
        title_tokens = clean_tokens(jieba.cut(title_norm))
        body_tokens = clean_tokens(jieba.cut(body_norm))

        if not body_tokens:
            stats["article_dropped_no_body"] += 1
            continue
        if len(body_tokens) < config.MIN_DOC_TOKENS:
            stats["article_dropped_too_short"] += 1
            continue

        content_hash = hashlib.sha1(
            " ".join(body_tokens).encode("utf-8")).hexdigest()

        banci_articles[art["banci"]] += 1
        stats["token_count_jieba"] += len(body_tokens)
        stats["token_count_pku"] += len(tokens_pku)

        out_records.append({
            "source_file": path.name,
            "month": path.stem,
            "article_id": aid,
            "date": art["date"],
            "banci": art["banci"],
            # 同时记录原始段落数与过滤后保留段落数：
            # 差值即被 D5 丢弃的署名/电讯头/日期括注段，便于审计过滤强度。
            # 单段成篇的文章标题段兼作正文，不再额外 +1，避免重复计数。
            "paragraph_count": len(paras),
            "paragraph_count_kept": len(kept_paras) + (0 if single_para else 1),
            "single_para_article": single_para,
            "title_text": title_norm,
            "body_text": body_norm,
            "paragraphs_cleaned": formatted_paragraphs,
            "title_tokens_jieba": title_tokens,
            "title_tokens_pku": clean_tokens(title_words_pku),
            "tokens_jieba": body_tokens,
            "tokens_pku": tokens_pku,
            "token_count": len(body_tokens),
            "content_hash": content_hash,
        })

    stats["article_count"] = len(out_records)
    stats["banci_distribution"] = dict(banci_articles)
    stats["elapsed_sec"] = round(time.time() - t0, 1)
    return out_records, stats


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def run(months: Optional[List[str]] = None) -> Dict[str, Any]:
    """清洗人民日报 1998 语料（默认全部 6 个月）。"""
    config.ensure_dirs()
    months = months or config.PD1998_FILES

    all_records: List[Dict[str, Any]] = []
    all_stats: List[Dict[str, Any]] = []
    t0 = time.time()

    for name in months:
        path = config.DATA_PD1998 / name
        if not path.exists():
            raise FileNotFoundError(f"语料文件缺失: {path}")
        print(f"  清洗 {name} ...", flush=True)
        recs, st = clean_month(path)
        all_records.extend(recs)
        all_stats.append(st)
        print(f"    段落 {st['paragraph_count']:>6d} | 文章 {st['article_count']:>5d} "
              f"(丢弃 短{st['article_dropped_too_short']} 空{st['article_dropped_no_body']}) "
              f"| 畸形行 {st['malformed_line_count']} "
              f"| {st['elapsed_sec']}s", flush=True)

    # 全局去重（跨月重复稿）
    seen: set = set()
    unique: List[Dict[str, Any]] = []
    dup = 0
    duplicate_by_month: Counter = Counter()
    for r in all_records:
        h = r["content_hash"]
        if h in seen:
            dup += 1
            duplicate_by_month[r["month"]] += 1
            continue
        seen.add(h)
        unique.append(r)

    out = config.F_PD1998_FORMATTED
    os.makedirs(out.parent, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for r in unique:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "stage": "pd1998-format-cleaning",
        "acceptance": {
            "all_six_months_processed": len(all_stats) == len(config.PD1998_FILES),
            "malformed_line_count_zero": sum(
                s["malformed_line_count"] for s in all_stats) == 0,
            "format_only_no_labels": True,
        },
        "config": {
            "seed": config.SEED,
            "encoding": config.ENCODING_PD1998,
            "min_doc_tokens": config.MIN_DOC_TOKENS,
            "months": list(months),
            "tokenizer": "jieba（统一分词器，D7）；同时保留北大原分词供消融",
            "order": "剥词性 → 版面噪声过滤 → 规范化 → jieba 分词 → token 过滤",
            "labels": "未执行；格式清洗中间数据不包含主题标签",
        },
        "totals": {
            "months_processed": len(all_stats),
            "line_count": sum(s["line_count"] for s in all_stats),
            "paragraph_count": sum(s["paragraph_count"] for s in all_stats),
            "malformed_line_count": sum(s["malformed_line_count"] for s in all_stats),
            "articles_before_dedup": len(all_records),
            "duplicate_dropped": dup,
            "articles_kept": len(unique),
            "token_count_jieba": sum(s["token_count_jieba"] for s in all_stats),
            "token_count_pku": sum(s["token_count_pku"] for s in all_stats),
            "layout_noise_removed": sum(s["layout_noise_removed"] for s in all_stats),
            "layout_paragraph_dropped": sum(
                s["layout_paragraph_dropped"] for s in all_stats),
            "untagged_token_count": sum(s["untagged_token_count"] for s in all_stats),
            "elapsed_sec": round(time.time() - t0, 1),
        },
        "per_month": [
            {**{k: v for k, v in s.items() if k != "malformed_examples"},
             "duplicate_dropped": duplicate_by_month[s["month"]]}
            for s in all_stats
        ],
        "malformed_examples": [
            {"file": s["source_file"], "examples": s["malformed_examples"]}
            for s in all_stats if s["malformed_examples"]
        ],
        "output_file": str(out),
    }

    config.F_PD1998_FORMAT_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n格式清洗报告 -> {config.F_PD1998_FORMAT_REPORT}")
    print(f"总表 -> {out}  ({len(unique)} 篇, 去重丢弃 {dup})")
    print("标签未生成；格式清洗与标签数据分离。")
    print(f"总耗时 {report['totals']['elapsed_sec']}s")
    return report


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="人民日报 1998 语料清洗")
    ap.add_argument("--months", nargs="*", default=None,
                    help="指定月份文件名，默认全部 6 个月")
    args = ap.parse_args()
    run(months=args.months)
