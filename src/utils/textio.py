"""文本 IO 工具：严格解码与安全读取。

设计约束（PLAN.md §1.3 实测更正）
--------------------------------
1. 两份语料实测均为 **UTF-8 无 BOM**；解码走严格模式，
   禁止 ``errors="replace"`` 静默替换非法字节——那会把编码错误
   伪装成"能跑通"，直到指标异常才暴露。
2. 解码失败必须抛异常并带上文件路径与字节位置，便于人工核查。
3. 保留 ``gb18030`` 兜底探测，以应对后续可能混入的其他来源语料，
   但探测结果必须与 ``config`` 声明一致，否则视为异常。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import config


class DecodingError(RuntimeError):
    """语料解码失败。携带路径与字节位置，不做静默替换。"""


def read_text_strict(path: Path, expected_encoding: str) -> str:
    """按指定编码严格读取整个文件。

    Parameters
    ----------
    path : 文件路径
    expected_encoding : 期望编码（来自 config）。若严格解码失败，
        会尝试 ``config.DECODING_CANDIDATES`` 中的其他编码，
        但只要有一个候选能解码而 expected 不能，就抛出
        ``DecodingError`` 并说明"编码与声明不符"，而不是悄悄用它。
    """
    raw = path.read_bytes()
    return decode_bytes_strict(raw, expected_encoding, path)


def decode_bytes_strict(raw: bytes, expected_encoding: str,
                        origin: object = None) -> str:
    """严格解码字节串。失败时探测真实编码并给出可操作的错误信息。

    刻意不把探测到的其他编码当作"可用结果"返回——那会掩盖 config 声明
    与真实编码不符的问题，使错误延迟到指标异常时才暴露。
    """
    try:
        return raw.decode(expected_encoding)
    except UnicodeDecodeError as exc:
        where = f"{origin}: " if origin is not None else ""
        detail = (f"{where}以 {expected_encoding} 解码失败"
                  f"（byte {exc.start} 处 0x{raw[exc.start]:02x} 非法）")
        actual = _detect_encoding(raw)
        if actual is not None and actual != expected_encoding:
            raise DecodingError(
                detail + f"，探测真实编码为 {actual}。请核对 config 编码声明。"
            ) from exc
        raise DecodingError(detail + "，且无候选编码可解码。须人工核查。") from exc


def _detect_encoding(raw: bytes) -> Optional[str]:
    """BOM 感知地探测编码名。

    刻意不采用"逐个候选试解码取第一个成功的"做法：
    ``utf-8-sig`` 能成功解码无 BOM 的纯 UTF-8（只是不剥离 BOM），
    会把编码名误报为 ``utf-8-sig``。必须先判 BOM，再定编码名。
    """
    if raw.startswith(b"\xef\xbb\xbf"):
        candidate_order = ("utf-8-sig",)
    else:
        candidate_order = ("utf-8", "gb18030")
    for enc in candidate_order:
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return None


def read_lines_strict(path: Path, expected_encoding: str) -> Tuple[list[str], str]:
    """严格读取并按行切分，返回 (行列表, 实际编码)。只读盘一次。"""
    raw = path.read_bytes()
    text = decode_bytes_strict(raw, expected_encoding, path)
    actual = _detect_encoding(raw) or expected_encoding
    return text.splitlines(), actual


def iter_lines_strict(path: Path, expected_encoding: str):
    """流式逐行读取（避免大文件一次性载入内存）。"""
    with path.open("r", encoding=expected_encoding, errors="strict",
                   newline="") as fh:
        for line in fh:
            yield line.rstrip("\r\n")
