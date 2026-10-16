# -*- coding: utf-8 -*-
"""人民日报 1998 格式清洗的独立验收（不验收尚未生成的 LLM 标签）。"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import compat  # noqa: F401,E402
import config  # noqa: E402
from utils.textio import decode_bytes_strict  # noqa: E402


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSONL 第 {line_no} 行无效: {exc}") from exc
    return rows


def main() -> int:
    if not config.F_PD1998_FORMATTED.exists() or not config.F_PD1998_FORMAT_REPORT.exists():
        raise FileNotFoundError("请先运行 python -m src.data.clean_pd1998")

    report = json.loads(config.F_PD1998_FORMAT_REPORT.read_text(encoding="utf-8"))
    rows = load_jsonl(config.F_PD1998_FORMATTED)
    checks: Dict[str, Dict[str, Any]] = {}

    def record(name: str, ok: bool, detail: Any) -> None:
        checks[name] = {"ok": bool(ok), "detail": detail}
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    months = set(config.PD1998_FILES)
    expected_months = {Path(name).stem for name in config.PD1998_FILES}
    report_months = {item["month"] for item in report.get("per_month", [])}
    row_months = {row.get("month") for row in rows}
    record("six_month_coverage", report_months == expected_months and row_months == expected_months,
           {"report": sorted(report_months), "output": sorted(row_months)})
    per_month_count = Counter(row.get("month") for row in rows)
    report_count_match = all(per_month_count.get(item["month"])
                             == item["article_count"] - item.get("duplicate_dropped", 0)
                             for item in report.get("per_month", []))
    record("per_month_article_counts", report_count_match,
           {"output": dict(sorted(per_month_count.items())),
            "report_after_dedup": {item["month"]: item["article_count"]
                                   - item.get("duplicate_dropped", 0)
                       for item in report.get("per_month", [])}})

    raw_unchanged = all((config.DATA_PD1998 / name).is_file() for name in months)
    record("raw_sources_present", raw_unchanged, [name for name in months if not (config.DATA_PD1998 / name).is_file()])

    total_lines = sum(item["line_count"] for item in report.get("per_month", []))
    parsed_paragraphs = sum(item["paragraph_count"] for item in report.get("per_month", []))
    malformed = sum(item["malformed_line_count"] for item in report.get("per_month", []))
    totals = report.get("totals", {})
    record("paragraph_accounting", total_lines == parsed_paragraphs + malformed
           and parsed_paragraphs == totals.get("paragraph_count")
           and malformed == totals.get("malformed_line_count"),
           {"nonblank_lines": total_lines, "parsed_paragraphs": parsed_paragraphs,
            "malformed": malformed})
    record("no_malformed_lines", malformed == 0,
           {"malformed": malformed, "examples": report.get("malformed_examples", [])})

    ids = [row.get("article_id") for row in rows]
    record("unique_article_ids", len(ids) == len(set(ids)) and all(ids),
           {"rows": len(ids), "unique_ids": len(set(ids))})
    record("output_count_accounting", len(rows) == totals.get("articles_kept"),
           {"jsonl_rows": len(rows), "report_articles_kept": totals.get("articles_kept")})

    bad_fields = []
    too_short = []
    malformed_id = []
    residual_tags = []
    text_mismatch = []
    bad_tokens = []
    for row in rows:
        aid = row.get("article_id", "")
        if not re.fullmatch(r"\d{8}-\d{2}-\d{3}", aid):
            malformed_id.append(aid)
        if not row.get("title_text") or not row.get("body_text") or not row.get("paragraphs_cleaned"):
            bad_fields.append(aid)
        if row.get("token_count", 0) < config.MIN_DOC_TOKENS or row.get("token_count") != len(row.get("tokens_jieba", [])):
            too_short.append(aid)
        if any(token in {"[", "]"} for token in row.get("tokens_pku", [])):
            residual_tags.append(aid)
        elif any(re.search(r"\[[^\]]*\][a-zA-Z]+", p.get("text", ""))
                 or re.search(r"/[a-zA-Z]+(?=\s|$)", p.get("text", ""))
                 for p in row.get("paragraphs_cleaned", [])):
            residual_tags.append(aid)
        if any(not isinstance(token, str) or not token.strip() for token in row.get("tokens_jieba", [])):
            bad_tokens.append(aid)
        paragraph_text = "".join(p.get("text", "") for p in row.get("paragraphs_cleaned", []))
        if paragraph_text != row.get("body_text"):
            text_mismatch.append(aid)
    record("article_schema_and_text", not bad_fields and not malformed_id and not text_mismatch,
           {"missing_text_fields": bad_fields[:10], "bad_article_ids": malformed_id[:10],
            "paragraph_body_mismatch": text_mismatch[:10]})
    record("minimum_length_and_tokens", not too_short and not bad_tokens,
           {"too_short_or_wrong_count": too_short[:10], "bad_tokens": bad_tokens[:10]})
    record("pos_and_phrase_tags_removed", not residual_tags,
           {"article_ids": residual_tags[:10], "count": len(residual_tags)})
    label_fields = [row.get("article_id") for row in rows
                    if any(key in row for key in ("label", "weak_label", "weak_label_confidence"))]
    record("format_data_unlabelled", not label_fields, {"label_fields": label_fields[:10], "count": len(label_fields)})

    duplicate_hashes = len(rows) - len({row.get("content_hash") for row in rows})
    record("content_deduplicated", duplicate_hashes == 0,
           {"duplicate_content_hashes": duplicate_hashes,
            "reported_cross_month_duplicates": totals.get("duplicate_dropped")})

    strict_utf8 = True
    for name in config.PD1998_FILES:
        path = config.DATA_PD1998 / name
        raw = path.read_bytes()
        try:
            decode_bytes_strict(raw, config.ENCODING_PD1998, path)
        except UnicodeDecodeError:
            strict_utf8 = False
    record("strict_utf8_raw_files", strict_utf8,
           {"encoding": config.ENCODING_PD1998, "source_files": len(config.PD1998_FILES)})

    stable_ids = sorted(ids)
    digest = hashlib.sha256("\n".join(stable_ids).encode("utf-8")).hexdigest()
    record("id_manifest_digest", bool(digest), {"sha256": digest, "count": len(stable_ids)})

    result = {
        "stage": "pd1998-format-clean-acceptance",
        "passed": all(item["ok"] for item in checks.values()),
        "checks": checks,
        "articles": len(rows),
        "months": sorted(row_months),
        "format_only": True,
        "llm_annotation": "pending; this checker does not accept labels",
    }
    out = config.RESULT_METRICS / "pd1998_format_acceptance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"验收报告 -> {out}")
    print(f"格式验收: {'PASS' if result['passed'] else 'FAIL'}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
