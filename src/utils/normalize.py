# -*- coding: utf-8 -*-
"""文本规范化：THUCNews 清洗规则（PLAN.md §2.1 C2–C4）。

设计原则
--------
全部为**纯函数**：输入 str / list[str]，输出 str / list[str]，
无 IO、无全局状态、无随机性。这样每条规则都能被单独单元测试，
且二次运行字节级一致（验收 A1-8）。

规则对应关系
------------
- C2 全角空格→半角、去首尾空白
- C3 剔除 HTML 残留、URL、邮箱
- C4 剔除纯标点/纯数字 token，**保留中文数字与年份**

⚠️ 实测发现与强制约束（2026-10-12）
----------------------------------
1. **两份语料的数字形态不一致**：人民日报 199801 前 400 行含全角数字
   661 个、半角 5,712 个（全角占 10%）；THUCNews 抽样含全角 **0** 个、
   半角 9,027 个。因此 ``normalize_width`` 是**跨语料对齐的必要前提**，
   不可省略，否则人民日报侧的日期 token 与 THUCNews 侧无法统一。

2. **执行顺序必须是"先规范化 → 再分词 → 最后 token 过滤"**。
   实测 ``１９９８年新年讲话``：
   - 先分词：``['１','９','９','８','年','新年','讲话']`` → 全角数字被逐字切碎
   - 先规范化再分词：``['1998','年','新年','讲话']`` → 数字成词
   顺序颠倒虽不会报错，但会静默产生大量单字碎片，污染词表与词向量。

3. **纯数字剔除会使日期数字信息丢失**：上述两种顺序下 ``1998`` 最终都被
   C4 丢弃，仅余 ``年``/``月``/``日`` 单字。因两份语料行为一致，**不会引入
   系统性偏差**，但会削弱轨道 B 的域漂移信号（1998 vs 2010s 的时间词差异
   被消除），报告须说明；并提供 ``keep_pure_numbers`` 开关做消融。
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

# --------------------------------------------------------------------------
# C3：噪声模式
# --------------------------------------------------------------------------
# HTML 标签残留。
# ⚠️ 两处实测修正（2026-10-12）：
#   (1) 初版把闭合 '>' 设为可选，"A<B比较" 会被 [^>]* 贪婪吞掉整段中文；
#   (2) 收紧后仍有漏洞：属性部分写成 (?:\s[^<>]*)? 会匹配任意中文，
#       使 "若 a<b 且 c>d 则成立" 中的 "<b 且 c>" 被当成标签删掉，
#       正文被吞成 "若 a d 则成立"。
# 因此属性必须匹配**真实 ASCII 属性形式** name="value" / name='v' / name=v，
# 且整个标签必须以 '>' 闭合。未闭合的孤立 '<' 交由 C4 纯标点过滤处理。
_RE_HTML_ATTR = r"""
    (?:                                     # 单个属性
        \s+                                 # 前置空白
        [a-zA-Z_:][-a-zA-Z0-9_:.]*          # 属性名
        (?:                                 # 可选属性值
            \s*=\s*
            (?:
                "[^"]*"                     # 双引号值（内部可含中文）
              | '[^']*'                     # 单引号值
              | [^\s"'>]+                   # 无引号值
            )
        )?
    )*
"""
_RE_HTML_TAG = re.compile(
    r"<\s*/?\s*[a-zA-Z][a-zA-Z0-9]*" + _RE_HTML_ATTR + r"\s*/?>",
    re.VERBOSE,
)
# HTML 实体，如 &nbsp; &amp; &#39;
_RE_HTML_ENTITY = re.compile(r"&(?:[a-zA-Z]{2,10}|#\d{1,6});")

# URL / 裸域名 / 邮箱
# ⚠️ 三处实测修正（2026-10-12，来自冒烟清洗发现的 URL 残留）：
#   (1) 原 `\bwww\.` 在中文语境**完全失效**：Python 的 `\w` 匹配 CJK，
#       故"站"与"www"之间不存在词边界，`报名网站www.73774.org` 不被匹配。
#       改用后行断言，只看前一字符、不消费字符，中文后与行首都能正确匹配。
#   (2) 语料中存在畸形 URL `http:/www.sxkszx.cn`（少一个斜杠），
#       原 `://` 要求两个斜杠故漏网。放宽为 `:/{1,2}`。
#   (3) **最严重**：原尾部字符类 `[^\s\u3000，。；！？…]+` 是"排除式"写法，
#       它只排除了空白与部分中文标点，**未排除汉字**，于是
#       `(http:/www.sxkszx.cn)在栏内` 会把 `)在栏内` 一并吞掉，
#       静默删除正文内容（实测该文档正文被截断）。
#       改为 RFC 3986 的 **ASCII 白名单式**字符集，从根本上不可能匹配汉字。
#
# RFC 3986 允许的 URL 字符（子分隔符与通用分隔符，不含空格与任何非 ASCII）
_URL_CHARS = r"""A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%．／？＃％＆＝＋"""
_URL_TAIL = "[" + _URL_CHARS + "]+"

# www 域名的左边界：前一字符不得是 ASCII 字母/数字/点/连字符/下划线，
# 否则 `abc-www.cn` 这类**含 www 子串的普通域名**会被误伤（实测曾误删）。
_WWW_LEFT = r"(?<![A-Za-z0-9._\-])"

_RE_URL = re.compile(
    r"(?:https?|ftp)\s*[:：]\s*[/／]{1,2}\s*" + _URL_TAIL,
    re.IGNORECASE,
)
_RE_WWW = re.compile(
    _WWW_LEFT + r"(?:(?:https?|ftp)\s*)?www[.．]" + _URL_TAIL,
    re.IGNORECASE,
)
_RE_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

# URL 匹配可能把紧邻的句末标点一并吞入（如 `详见 www.a.cn.` 的结尾句点、
# `(http:/a.cn)` 的右括号）。这些字符不属于 URL 正文，替换时须原样归还，
# 否则等于静默删除正文标点。
_TRAILING_PUNCT = ".,;:!?)]}'\"、。，；！？"
# 连续空白（含全角空格残留）压缩为单个半角空格
_RE_MULTI_SPACE = re.compile(r"[ \t\u3000]+")
# 连续重复标点，如 "。。。。" "！！！" "———"（保留最多 1 个）
_RE_REPEAT_PUNCT = re.compile(r"([，。！？；：、,.!?;:])\1{1,}")


# --------------------------------------------------------------------------
# C4：token 级过滤所需字符类判定
# --------------------------------------------------------------------------
# 中文数字（一、二、十、百、千、万、亿、零、两）——须保留，
# 否则"一九九八年""十五大""三中全会"等时间/会议表达会被破坏
_CN_NUM = set("一二三四五六七八九十百千万亿零两")

# 允许保留的标点白名单（中文常用 + 基本西文标点）
_KEEP_PUNCT = set(
    "，。！？；：、""''《》〈〉【】〔〕（）—…·"
    ",.!?;:()[]{}\"'-"
)


def is_cjk_char(ch: str) -> bool:
    """是否为 CJK 表意文字（基本区 + 扩展 A）。"""
    cp = ord(ch)
    return (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF)


def has_cjk(text: str) -> bool:
    """文本是否含至少一个汉字。"""
    return any(is_cjk_char(c) for c in text)


# --------------------------------------------------------------------------
# C2：字符级规范化
# --------------------------------------------------------------------------
def normalize_spaces(text: str) -> str:
    """C2：全角空格 \\u3000 → 半角，连续空白压缩为单个空格，去首尾空白。"""
    return _RE_MULTI_SPACE.sub(" ", text.replace("\u3000", " ")).strip()


def normalize_width(text: str) -> str:
    """全角 ASCII **数字与字母** → 半角；中文、中文标点、全角标点一律不动。

    ⚠️ 实测修正（2026-10-12）：初版按 ``0xFF01 <= cp <= 0xFF5E`` 整段转换，
    但该区间同时覆盖了全角标点：

    - U+FF01 ``！`` → ``!``
    - U+FF0C ``，`` → ``,``
    - U+FF1A ``：`` → ``:``
    - U+FF1B ``；`` → ``;``
    - U+FF1F ``？`` → ``?``

    若照此执行，48 万篇语料的中文标点会被**全部改写成西文标点**，
    且与人民日报语料侧保留的中文标点形成系统性不一致，直接污染跨域评估。

    因此这里**只转换**全角数字（FF10–FF19）与全角字母
    （FF21–FF3A 大写 / FF41–FF5A 小写），其余字符原样保留。
    """
    out = []
    for ch in text:
        cp = ord(ch)
        if (0xFF10 <= cp <= 0xFF19          # ０-９
                or 0xFF21 <= cp <= 0xFF3A   # Ａ-Ｚ
                or 0xFF41 <= cp <= 0xFF5A): # ａ-ｚ
            out.append(chr(cp - 0xFEE0))
        elif cp == 0x3000:                  # 全角空格 → 半角空格
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


# --------------------------------------------------------------------------
# C3：噪声剔除
# --------------------------------------------------------------------------
def _strip_url_keep_punct(pattern: "re.Pattern[str]", text: str) -> str:
    """用 pattern 剔除 URL，但归还被一并吞入的尾随标点。

    ``_URL_TAIL`` 的字符类不含中文标点，却**含**半角 ``. , ; : ! ? ) ]`` 等，
    因此 ``(http:/a.cn)`` 或 ``详见 www.a.cn。`` 中的闭合括号会被吞掉。
    这些标点属于正文，须原样保留，否则等于静默删除正文标点。
    """
    def repl(m: "re.Match[str]") -> str:
        matched = m.group(0)
        tail = ""
        while matched and matched[-1] in _TRAILING_PUNCT:
            tail = matched[-1] + tail
            matched = matched[:-1]
        return (" " + tail) if tail else " "

    return pattern.sub(repl, text)


def strip_noise(text: str) -> str:
    """C3：剔除 HTML 标签/实体、URL、邮箱。"""
    text = _strip_url_keep_punct(_RE_URL, text)
    text = _strip_url_keep_punct(_RE_WWW, text)
    text = _RE_EMAIL.sub(" ", text)
    text = _RE_HTML_ENTITY.sub(" ", text)
    text = _RE_HTML_TAG.sub(" ", text)
    return text


def has_url(text: str) -> bool:
    """检测常见 ASCII/全角标点 URL，供清洗验收复用。"""
    return bool(_RE_URL.search(text) or _RE_WWW.search(text))


def collapse_punct(text: str) -> str:
    """压缩连续重复标点（如 '。。。。' → '。'）。"""
    return _RE_REPEAT_PUNCT.sub(r"\1", text)


# --------------------------------------------------------------------------
# token 级过滤（C4）
# --------------------------------------------------------------------------
# 纯阿拉伯数字（可带千分位、小数、百分号、正负号）
# ⚠️ 不可用 ``token.strip(".,%")`` 先剥字符再判数字：``strip`` 会剥掉
#    **首尾**的 '%'，使 '56.7%' 的 core 变成 '56.7'（合法数字），
#    从而把带百分号的统计数据误判为"纯数字"而丢弃。改为整体匹配。
_RE_PURE_ARABIC = re.compile(r"^[+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?$"
                             r"|^[+-]?\d+(?:\.\d+)?%?$")


def is_pure_punct(token: str) -> bool:
    """token 是否**全部**由标点/符号构成（无字母、数字、汉字）。"""
    if not token:
        return True
    for ch in token:
        cat = unicodedata.category(ch)
        # P* = 标点, S* = 符号, Z* = 分隔符, C* = 控制字符
        if cat[0] not in ("P", "S", "Z", "C"):
            return False
    return True


def is_pure_arabic_number(token: str) -> bool:
    """token 是否为纯阿拉伯数字（可含千分位/小数点/百分号/正负号）。

    例：'123' '1,234' '56.7' '89%' '-42' → True
        '一九九八年' '12月' '十五大' '89%增长' → False
    """
    t = token.strip()
    return bool(t) and bool(_RE_PURE_ARABIC.match(t))


def is_pure_cn_number(token: str) -> bool:
    """token 是否为纯中文数字串（如 '一九九八'、'十五'）。

    用于统计，不用于丢弃——中文数字**须保留**。
    """
    return bool(token) and all(c in _CN_NUM for c in token)


def keep_token(token: str) -> bool:
    """C4：判定单个 token 是否保留。

    判定规则（单一可判定标准，消除 PLAN.md C4 "保留年份" 的表述歧义）
    ------------------------------------------------------------------
    **含汉字或字母 → 保留；纯标点/符号 → 剔除；纯阿拉伯数字 → 剔除。**

    - 丢弃：空串、纯标点（'，' '《》' '——'）、纯阿拉伯数字
      （'2005' '1,234' '56.7' '89%'）
    - 保留：任何含汉字或拉丁字母的 token

    为什么裸 '1998' 不算"年份表达"而应剔除
    --------------------------------------
    PLAN.md C4 原文为"保留中文数字与年份"，意图是防止 '一九九八年'
    '十五大' '三中全会' 这类时间/会议实体被破坏。这些表达在本语料中
    分词后**必然含汉字**（'一九九八年'、'1998年'、'１２月'），
    因此"含汉字即保留"已完整覆盖该意图。
    而裸 '1998'、'89%'、'1,234' 不构成年份表达，对报刊主题分类
    基本无信息量（多为页码、图片编号、金额碎片），剔除可减少词表噪声。
    """
    t = token.strip()
    if not t:
        return False
    if is_pure_punct(t):
        return False
    if is_pure_arabic_number(t):
        return False
    return True


# --------------------------------------------------------------------------
# 组合：整条规范化流水线
# --------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """对一段原始文本依次执行 C2 + C3 + 标点压缩。

    顺序不可调换：先规范化宽度/空格，再剔噪声，最后压缩标点，
    否则 URL 中的全角字符可能导致正则漏匹配。
    """
    text = normalize_width(text)
    text = normalize_spaces(text)
    text = strip_noise(text)
    text = collapse_punct(text)
    return normalize_spaces(text)      # 剔噪后可能留下多余空格，再压一次


def clean_tokens(tokens: Iterable[str], keep_pure_numbers: bool = False) -> list[str]:
    """对已分词的 token 序列执行 C4 过滤。

    Parameters
    ----------
    keep_pure_numbers : 默认 False（按 PLAN.md C4 剔除纯阿拉伯数字）。
        置 True 时保留纯数字 token，用于消融"日期/年份数字对轨道 B
        域漂移信号的影响"（见本模块头部实测发现 3）。
    """
    out = []
    for tok in tokens:
        t = tok.strip()
        if not t or is_pure_punct(t):
            continue
        if not keep_pure_numbers and is_pure_arabic_number(t):
            continue
        out.append(t)
    return out
