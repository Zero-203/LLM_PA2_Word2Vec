# -*- coding: utf-8 -*-
"""回归测试：人民日报 D2/D3/POS 标签剥离及格式清洗口径。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.clean_pd1998 import parse_paragraph, strip_pos


class PeopleDailyFormatTests(unittest.TestCase):
    def test_parse_paragraph_extracts_article_metadata(self):
        parsed = parse_paragraph("19980101-01-001-004/m  中共中央/nt")
        self.assertEqual(parsed["article_id"], "19980101-01-001")
        self.assertEqual(parsed["para_no"], "004")
        self.assertEqual(parsed["body"], "中共中央/nt")

    def test_strip_pos_removes_trailing_tag_before_punctuation(self):
        words, _ = strip_pos("腾空而起/l。/w")
        self.assertEqual(words, ["腾空而起。"])

    def test_strip_pos_removes_phrase_tags(self):
        words, _ = strip_pos("[中国/ns 质量/n 万里行/n]nt 企业/n")
        self.assertEqual(words, ["中国", "质量", "万里行", "企业"])


if __name__ == "__main__":
    unittest.main()
