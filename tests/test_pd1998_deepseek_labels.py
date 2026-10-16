# -*- coding: utf-8 -*-
"""DeepSeek 人民日报标注响应校验与复核规则测试。"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import config
from label_pd1998_deepseek import (
    LabelingError,
    annotate_batch,
    export_fasttext,
    make_messages,
    parse_model_output,
)


class DeepSeekLabelTests(unittest.TestCase):
    def setUp(self):
        self.records = [{
            "article_id": "19980101-01-001",
            "month": "199801",
            "source_file": "199801.txt",
            "banci": "01",
            "title_text": "新型计算机技术取得进展",
            "body_text": "研究人员介绍了计算机系统和芯片技术的新进展。",
        }]

    def test_parser_accepts_configured_category_and_null_review(self):
        parsed = parse_model_output(
            '{"items":[{"article_id":"19980101-01-001","label":"科技",'
            '"confidence":0.91,"reason":"计算机与芯片技术"}]}',
            self.records,
        )
        self.assertEqual(parsed[0]["label"], "科技")
        self.assertEqual(parsed[0]["confidence"], 0.91)

        review = parse_model_output(
            '{"items":[{"article_id":"19980101-01-001","label":null,'
            '"confidence":0.42,"reason":"主题不清"}]}', self.records,
        )
        self.assertIsNone(review[0]["label"])

    def test_parser_rejects_invalid_label_and_missing_article(self):
        with self.assertRaises(LabelingError):
            parse_model_output(
                '{"items":[{"article_id":"19980101-01-001","label":"01",'
                '"confidence":0.9,"reason":""}]}', self.records,
            )
        with self.assertRaises(LabelingError):
            parse_model_output(
                '{"items":[{"article_id":"other","label":"科技",'
                '"confidence":0.9,"reason":""}]}', self.records,
            )

    def test_prompt_uses_article_content_not_banci_for_labeling(self):
        messages = make_messages(self.records)
        user_message = messages[-1]["content"]
        self.assertIn("14类", user_message)
        self.assertIn("计算机技术", user_message)
        self.assertNotIn('"banci"', user_message)

    def test_confidence_threshold_controls_confirmed_status(self):
        parsed = [{"label": "科技", "confidence": config.PD1998_LABEL_CONFIDENCE_THRESHOLD,
                   "reason": "技术主题"}]
        with patch("label_pd1998_deepseek.request_batch",
                   return_value=(parsed, '{"items":[]}', {}, "req-test")):
            accepted = annotate_batch(self.records, "not-a-real-key")
        self.assertEqual(accepted[0]["status"], "confirmed")
        self.assertEqual(accepted[0]["confirmation_method"], "model_confidence_threshold")

        parsed[0]["confidence"] = config.PD1998_LABEL_CONFIDENCE_THRESHOLD - 0.01
        with patch("label_pd1998_deepseek.request_batch",
                   return_value=(parsed, '{"items":[]}', {}, "req-test")):
            review = annotate_batch(self.records, "not-a-real-key")
        self.assertEqual(review[0]["status"], "needs_review")

    def test_fasttext_high_and_all_exports_separate_review_candidates(self):
        rows = [
            {"article_id": "a", "month": "199801", "tokens_jieba": ["科技", "芯片"]},
            {"article_id": "b", "month": "199801", "tokens_jieba": ["财经", "市场"]},
            {"article_id": "c", "month": "199802", "tokens_jieba": ["不确定", "文章"]},
        ]
        annotations = {
            "a": {"status": "confirmed", "label": "科技", "candidate_label": "科技"},
            "b": {"status": "needs_review", "label": "财经", "candidate_label": "财经"},
            "c": {"status": "needs_review", "label": None, "candidate_label": None},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {
                "F_T_PD1998": root / "primary_confirmed.txt",
                "F_T_PD1998_ALL": root / "all_confirmed.txt",
                "F_T_PD1998_HIGH": root / "primary_high.txt",
                "F_T_PD1998_CANDIDATES": root / "primary_all.txt",
                "F_T_PD1998_ALL_HIGH": root / "all_high.txt",
                "F_T_PD1998_ALL_CANDIDATES": root / "all_candidates.txt",
            }
            with patch.multiple(config, **paths):
                counts = export_fasttext(rows, annotations)

            self.assertEqual(counts["primary_month_high_lines"], 1)
            self.assertEqual(counts["primary_month_all_lines"], 2)
            self.assertEqual(counts["all_months_high_lines"], 1)
            self.assertEqual(counts["all_months_all_lines"], 2)
            self.assertEqual(counts["all_months_all_needs_review_without_candidate_label"], 1)
            self.assertEqual((root / "primary_all.doc_ids.txt").read_text(encoding="utf-8"), "a\nb\n")


if __name__ == "__main__":
    unittest.main()
