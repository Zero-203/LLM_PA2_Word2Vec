# -*- coding: utf-8 -*-
"""normalize 模块单元测试（对应验收 A1-3 / A1-4 的可自动化部分）。

运行：
    .\\.venv39\\Scripts\\python.exe tests\\test_normalize.py

契约说明（避免误判预期）
------------------------
- ``strip_noise``：只负责**剔除噪声并用单个空格占位**，不负责压缩空格。
  因此 "详情见 <URL> 报道" → "详情见   报道"（3 个空格）是**正确行为**。
  空格压缩由 ``normalize_spaces`` / ``normalize_text`` 负责。
- ``normalize_width``：**只**转换全角数字与字母，中文标点必须原样保留。

本文件同时是以下已修复问题的**回归测试**（见各处注释）：
1. 实现 bug：normalize_width 曾把全角标点 U+FF01/FF0C 等一并转成 ASCII，
   会导致 48 万篇语料的中文标点被改写成西文标点，并与人民日报侧
   保留的中文标点形成系统性不一致，污染跨域评估。
2. 实现 bug：HTML 正则两处漏洞——闭合 '>' 曾设为可选，"A<B比较"
   被吞；收紧后属性部分 (?:\\s[^<>]*)? 仍能匹配中文，使
   "若 a<b 且 c>d 则成立" 被吞成 "若 a d 则成立"。
   现要求属性匹配真实 ASCII 属性形式且标签必须闭合。
3. 规则歧义（非实现 bug）：PLAN.md C4 "剔除纯数字，保留中文数字与年份"
   未界定裸 '1998' 是否算年份。已厘清为单一可判定标准：
   **含汉字或字母→保留；纯标点/纯阿拉伯数字→剔除**。
   据此 '1998年'、'１２月'、'一九九八年' 保留，裸 '1998'、'89%' 剔除。
   另：初版实现用 strip(".,%") 判纯数字，会剥掉 '%' 使逻辑不可靠，
   已改为整体正则匹配。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import compat  # noqa: F401,E402  (必须在任何第三方库之前)

from utils.normalize import (  # noqa: E402
    clean_tokens,
    collapse_punct,
    has_cjk,
    has_url,
    is_pure_arabic_number,
    is_pure_cn_number,
    is_pure_punct,
    keep_token,
    normalize_spaces,
    normalize_text,
    normalize_width,
    strip_noise,
)

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, got, want) -> None:
    if got == want:
        PASS.append(name)
        print("  [ok]   %s" % name)
    else:
        FAIL.append(name)
        print("  [FAIL] %s\n         got : %r\n         want: %r" % (name, got, want))


def check_true(name: str, cond) -> None:
    check(name, bool(cond), True)


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


section("1. C2 宽度与空格规范化")
check("全角数字→半角", normalize_width("１２月３１日"), "12月31日")
check("全角字母→半角", normalize_width("ＡＢＣ新闻"), "ABC新闻")
check("全角小写字母→半角", normalize_width("ａｂｃ"), "abc")
check("全角空格→半角", normalize_width("　　正文"), "  正文")
# --- 关键回归 1：中文标点绝不可被转换 ---
check("中文逗号保留", normalize_width("，"), "，")
check("中文叹号保留", normalize_width("！"), "！")
check("中文问号保留", normalize_width("？"), "？")
check("中文冒号保留", normalize_width("："), "：")
check("中文分号保留", normalize_width("；"), "；")
check("中文顿号保留", normalize_width("、"), "、")
check("中文标点整体不被转换", normalize_width("，。！？；：、《》"), "，。！？；：、《》")
check("中文不被转换", normalize_width("迈向新世纪"), "迈向新世纪")
check("混合串只转数字字母", normalize_width("１９９８年，ＡＢＣ！"), "1998年，ABC！")
check("连续空白压缩", normalize_spaces("a\u3000\u3000b   c"), "a b c")
check("去首尾空白", normalize_spaces("  正文  "), "正文")

section("2. C3 噪声剔除（strip_noise 只剔噪，不压空格）")
check("URL 被剔除", strip_noise("详情见 http://www.a.com/x?id=1 报道"), "详情见   报道")
check("https URL 被剔除", strip_noise("访问 https://abc.cn/p 了解"), "访问   了解")
check("全角标点 URL 被剔除并保留相邻正文",
      normalize_text("教育部公布(http：／www．cscse．edu．cn)，用人单位可查询"),
      "教育部公布( )，用人单位可查询")
check("邮箱被剔除", strip_noise("联系 news@example.com 投稿"), "联系   投稿")
check("HTML 标签被剔除", strip_noise("<p>正文<br/>内容</p>"), " 正文 内容 ")
check("HTML 实体被剔除", strip_noise("A&nbsp;B&amp;C"), "A B C")
check("重复标点压缩", collapse_punct("好。。。。确实！！！"), "好。确实！")
# --- 关键回归 4：`\bwww\.` 在中文后失效（Python `\w` 匹配 CJK）---
# 冒烟清洗实测发现 5 篇残留，均为此形态
check("中文后紧跟 www 域名被剔除",
      strip_noise("报名网站www.73774.org"), "报名网站 ")
check("中文后紧跟 http URL 被剔除",
      strip_noise("详情见http://a.cn/x"), "详情见 ")
check("全角 URL 可被验收检测", has_url("http：／www．cscse．edu．cn"), True)
check("孤立协议文本不误判为 URL", has_url("网站说明中出现 http 一词"), False)
# --- 关键回归 5：URL 尾部字符类曾未排除汉字，静默吞掉正文 ---
# 实测 `(http:/www.sxkszx.cn)在栏内` 曾被吞成 `( `，正文 `)在栏内` 丢失
check("括号内畸形单斜杠 URL 被剔除且右括号保留",
      strip_noise("登录山西招生考试网(http:/www.sxkszx.cn)在栏内"),
      "登录山西招生考试网( )在栏内")
check("URL 后正文汉字完整保留",
      normalize_text("登录山西招生考试网(http:/www.sxkszx.cn)在栏内点击"),
      "登录山西招生考试网( )在栏内点击")
# --- 关键回归 6：含 www 子串的普通域名不得误伤 ---
check("子域名 www 不被误伤", strip_noise("访问 abc-www.cn"), "访问 abc-www.cn")
check("行首 www 域名被剔除", normalize_text("www.a.cn 报道"), "报道")
check("无分隔协议前缀的 www URL 被剔除",
      normalize_text("访问httpwww.worldcup98.net.cn获取信息"), "访问 获取信息")
check("紧邻协议与 www 的 URL 被验收检测",
      has_url("访问httpwww.worldcup98.net.cn获取信息"), True)
# --- 标点归还：strip_noise 不压空格（由 normalize_text 负责），故分别断言 ---
check("strip_noise 归还 URL 后中文句号",
      strip_noise("详见 www.a.cn。后续正文"), "详见  。后续正文")
check("normalize_text 收敛空格并保留中文句号",
      normalize_text("详见 www.a.cn。后续正文"), "详见 。后续正文")
check("normalize_text 归还 URL 后英文句点",
      normalize_text("see http://a.cn/x. next"), "see . next")
# --- 关键回归 2：含数学比较符的正文不可被 HTML 正则误删 ---
check("数学比较符不误删", strip_noise("A<B比较"), "A<B比较")
check("未闭合尖括号不吞正文", strip_noise("若 a<b 且 c>d 则成立"), "若 a<b 且 c>d 则成立")
check("normalize_text 剔 URL 后收敛空格",
      normalize_text("详情见 http://www.a.com/x 报道"), "详情见 报道")
check("normalize_text 剔 HTML 后收敛空格",
      normalize_text("<p>正文<br/>内容</p>"), "正文 内容")

section("3. C4 token 过滤（判定标准：含汉字或字母→保留；纯标点/纯阿拉伯数字→剔除）")
# --- 必须保留：含汉字的表达（PLAN.md C4 "保留中文数字与年份" 的完整覆盖）---
check("保留 '一九九八年'", keep_token("一九九八年"), True)
check("保留 '十五大'", keep_token("十五大"), True)
check("保留 '三中全会'", keep_token("三中全会"), True)
check("保留 '中共中央'", keep_token("中共中央"), True)
check("保留 '12月'(含汉字月)", keep_token("12月"), True)
check("保留 '1998年'(含汉字年)", keep_token("1998年"), True)
check("保留 '３１日'(含汉字日)", keep_token("31日"), True)
check("保留 '89%增长'(含汉字)", keep_token("89%增长"), True)
check("保留 '1,234亿元'(含汉字)", keep_token("1,234亿元"), True)
check("保留 英文 'CNN'", keep_token("CNN"), True)
check("保留 中英混排 'GDP增速'", keep_token("GDP增速"), True)
# --- 必须剔除：纯标点 / 纯阿拉伯数字 ---
check("丢弃 纯标点 '，'", keep_token("，"), False)
check("丢弃 纯标点 '《》'", keep_token("《》"), False)
check("丢弃 纯标点 '——'", keep_token("——"), False)
check("丢弃 纯整数 '2005'", keep_token("2005"), False)
check("丢弃 纯整数 '123'", keep_token("123"), False)
check("丢弃 纯百分数 '89%'", keep_token("89%"), False)
check("丢弃 纯小数 '56.7%'", keep_token("56.7%"), False)
check("丢弃 纯小数 '3.14'", keep_token("3.14"), False)
check("丢弃 纯千分位 '1,234'", keep_token("1,234"), False)
check("丢弃 空串", keep_token(""), False)
check("丢弃 空白串", keep_token("   "), False)
# --- 字符类判定函数 ---
check("is_pure_punct('——')", is_pure_punct("——"), True)
check("is_pure_punct('新华社')", is_pure_punct("新华社"), False)
check("is_pure_arabic_number('1998')", is_pure_arabic_number("1998"), True)
check("is_pure_arabic_number('一九九八')", is_pure_arabic_number("一九九八"), False)
check("is_pure_arabic_number('89%')", is_pure_arabic_number("89%"), True)
check("is_pure_arabic_number('89%增长')", is_pure_arabic_number("89%增长"), False)
check("is_pure_arabic_number('1,234')", is_pure_arabic_number("1,234"), True)
check("is_pure_cn_number('一九九八')", is_pure_cn_number("一九九八"), True)
check("is_pure_cn_number('1998')", is_pure_cn_number("1998"), False)

section("4. clean_tokens 序列过滤")
check("token 序列过滤",
      clean_tokens(["中共中央", "，", "总书记", "1998", "发表",
                    "一九九八年", "新年", "讲话", "。"]),
      ["中共中央", "总书记", "发表", "一九九八年", "新年", "讲话"])
check("含汉字的数值表达保留、纯数字剔除",
      clean_tokens(["增长", "56.7%", "达", "1,234亿元", "一九九八年", "89%"]),
      ["增长", "达", "1,234亿元", "一九九八年"])

section("5. normalize_text 端到端（真实语料样本）")
raw = ("　　记者傅亚雨沈阳报道 来到沈阳，国奥队依然没有摆脱雨水的困扰。"
       "<br/>详情 http://news.sina.com.cn/a.html 联系 a@b.com "
       "7月31日下午6点。。。。")
got = normalize_text(raw)
print("  输入: %s" % raw[:70])
print("  输出: %s" % got[:90])
check_true("无 HTML 残留", "<br" not in got and "<p" not in got)
check_true("无 URL 残留", "http" not in got and "sina" not in got)
check_true("无邮箱残留", "@b.com" not in got)
check_true("无全角空格", "\u3000" not in got)
check_true("无连续重复句号", "。。" not in got)
check_true("无连续空格", "  " not in got)
check_true("保留中文数字 '7月31日'", "7月31日" in got)
check_true("保留正文 '国奥队'", "国奥队" in got)
check_true("中文逗号未被转成西文", "，" in got and "," not in got)

section("6. has_cjk")
check("has_cjk('新华社')", has_cjk("新华社"), True)
check("has_cjk('123')", has_cjk("123"), False)
check("has_cjk('ABC')", has_cjk("ABC"), False)

print()
print("=" * 72)
print("结果: %d 通过, %d 失败" % (len(PASS), len(FAIL)))
if FAIL:
    print("失败项: %s" % FAIL)
print("=" * 72)
sys.exit(1 if FAIL else 0)
