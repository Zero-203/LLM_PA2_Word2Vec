"""阶段 0 环境验收脚本（PLAN.md §1.2，验收项 A0-1 ~ A0-6）。

用法：
    .\\.venv39\\Scripts\\python.exe scripts\\check_env.py

逐项断言并输出验收清单；任一项失败以非零退出码结束。
其中 A0-5 是全项目技术咽喉（gensim 向量能否被 FastText 加载），
必须最先验证。
"""

from __future__ import annotations

import json
import os
import platform
import struct
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config  # noqa: E402

RESULTS: list[dict] = []


def record(item: str, title: str, ok: bool, detail: str) -> bool:
    """记录一条验收结果。"""
    RESULTS.append({"item": item, "title": title, "ok": ok, "detail": detail})
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {item}  {title}")
    if detail:
        for line in detail.splitlines():
            print(f"        {line}")
    return ok


# --------------------------------------------------------------------------
# A0-1 虚拟环境与解释器
# --------------------------------------------------------------------------
def check_a01() -> bool:
    ver = sys.version_info
    bits = struct.calcsize("P") * 8
    exe = sys.executable
    ok = (ver.major, ver.minor) == (3, 9) and bits == 64
    detail = (
        f"executable = {exe}\n"
        f"version    = {platform.python_version()}\n"
        f"bits       = {bits}\n"
        f"platform   = {platform.platform()}\n"
        f"cpu_count  = {os.cpu_count()}"
    )
    if "venv" not in exe.lower() and ".venv39" not in exe.lower():
        ok = False
        detail += "\nWARNING: 未在虚拟环境 .venv39 内运行"
    return record("A0-1", "虚拟环境与解释器版本", ok, detail)


# --------------------------------------------------------------------------
# A0-2 / A0-3 依赖导入与版本快照
# --------------------------------------------------------------------------
def check_a02_a03() -> bool:
    specs = [
        ("gensim", "gensim"),
        ("fasttext", "fasttext-wheel"),
        ("jieba", "jieba"),
        ("sklearn", "scikit-learn"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
    ]
    versions: dict[str, str] = {}
    failed: list[str] = []
    for import_name, dist_name in specs:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", None)
            if ver is None:
                # fasttext 等包无 __version__ 属性，回退到发行元数据
                from importlib import metadata
                ver = metadata.version(dist_name)
            versions[import_name] = str(ver)
        except Exception as exc:                       # noqa: BLE001
            failed.append(f"{import_name}: {type(exc).__name__}: {exc}")

    ok = not failed
    detail = "\n".join(f"{k:12s} = {v}" for k, v in sorted(versions.items()))
    if failed:
        detail += "\n导入失败：\n" + "\n".join(failed)

    a02 = record("A0-2", "全部依赖可导入", ok, detail)

    # A0-3 版本落盘
    snapshot = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "packages": versions,
        "seed": config.SEED,
    }
    try:
        os.makedirs(config.RESULT_DIR, exist_ok=True)
        config.F_ENV_SNAPSHOT.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        a03 = record("A0-3", "环境快照落盘", True, f"-> {config.F_ENV_SNAPSHOT}")
    except Exception as exc:                           # noqa: BLE001
        a03 = record("A0-3", "环境快照落盘", False, f"{type(exc).__name__}: {exc}")

    # numpy 2.x 与 gensim 4.4.0 存在兼容性风险，显式校验
    np_major = int(versions.get("numpy", "0").split(".")[0] or 0)
    if np_major >= 2:
        record("A0-3b", "numpy 主版本 < 2", False,
               f"检测到 numpy {versions.get('numpy')}，gensim 4.4.0 可能不兼容")
        return False
    return a02 and a03


# --------------------------------------------------------------------------
# A0-4 gensim Word2Vec 冒烟训练
# --------------------------------------------------------------------------
TOY_SENTENCES = [
    "体育 足球 比赛 训练 教练 球员 冠军 联赛 进球 球场",
    "财经 股票 市场 投资 基金 上市 公司 收益 经济 金融",
    "科技 计算机 网络 数据 软件 硬件 芯片 研发 算法 系统",
    "娱乐 电影 明星 演出 音乐 导演 观众 剧场 票房 艺人",
    "球员 在 球场 上 进球 帮助 球队 获得 冠军",
    "公司 在 市场 上 投资 股票 获得 收益",
    "研发 团队 用 算法 处理 数据 优化 系统",
    "明星 在 剧场 演出 获得 观众 好评",
]


def check_a04() -> bool:
    try:
        from gensim.models import Word2Vec

        model = Word2Vec(
            sentences=[s.split() for s in TOY_SENTENCES],
            vector_size=5,
            window=3,
            min_count=1,
            workers=1,
            epochs=20,
            seed=config.SEED,
        )
        vocab = len(model.wv)
        similar = model.wv.most_similar("球员", topn=3)
        vec = model.wv["球员"]
        ok = vocab > 0 and len(similar) == 3 and vec.shape == (5,)
        detail = (
            f"vocab_size = {vocab}\n"
            f"vector['球员'].shape = {vec.shape}\n"
            f"most_similar('球员') = {[(w, round(float(s), 4)) for w, s in similar]}"
        )
    except Exception:                                  # noqa: BLE001
        return record("A0-4", "gensim Word2Vec 冒烟训练", False, traceback.format_exc())
    return record("A0-4", "gensim Word2Vec 冒烟训练", ok, detail)


# --------------------------------------------------------------------------
# A0-5 关键技术咽喉：gensim 向量注入 FastText
# --------------------------------------------------------------------------
def check_a05() -> bool:
    """验证 save_word2vec_format 产出的向量能被 fasttext 的
    pretrainedVectors 正确加载，且输入矩阵非空。"""
    try:
        import fasttext
        from gensim.models import Word2Vec

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            vec_path = tmpdir / "toy.vec"
            train_path = tmpdir / "toy_train.txt"
            model_path = tmpdir / "toy_ft.bin"

            # 1) 用玩具语料训练 gensim 并导出 w2v 文本格式
            w2v = Word2Vec(
                sentences=[s.split() for s in TOY_SENTENCES],
                vector_size=10, window=3, min_count=1,
                workers=1, epochs=20, seed=config.SEED,
            )
            w2v.wv.save_word2vec_format(str(vec_path), binary=False)

            # 校验 vec 文件头：<词数> <维度>
            with vec_path.open("r", encoding="utf-8") as fh:
                header = fh.readline().split()
            header_ok = len(header) == 2 and header[1] == "10"

            # 2) 构造 FastText 监督训练格式
            ft_lines = [
                "__label__体育 球员 球场 进球 冠军 联赛",
                "__label__体育 球队 训练 教练 比赛 球员",
                "__label__财经 公司 股票 市场 投资 收益",
                "__label__财经 基金 上市 经济 金融 股票",
                "__label__科技 算法 数据 系统 芯片 软件",
                "__label__科技 研发 计算机 网络 硬件 系统",
            ]
            train_path.write_text("\n".join(ft_lines) + "\n", encoding="utf-8")

            # 3) 带预训练向量训练 FastText（核心链路）
            ft = fasttext.train_supervised(
                input=str(train_path),
                pretrainedVectors=str(vec_path),
                dim=10, epoch=5, lr=0.1, wordNgrams=1,
                minCount=1, thread=1, verbose=0,
            )

            input_matrix = ft.get_input_matrix()
            words = ft.get_words()
            labels = ft.get_labels()

            # 4) 预测冒烟
            pred_labels, pred_probs = ft.predict("球员 进球 冠军")

            ok = (
                header_ok
                and input_matrix.shape[0] > 0
                and input_matrix.shape[1] == 10
                and len(words) > 0
                and len(labels) == 3
                and len(pred_labels) == 1
            )
            detail = (
                f"vec 文件头 = {' '.join(header)}  (期望 '<词数> 10')  header_ok={header_ok}\n"
                f"ft.get_input_matrix().shape = {input_matrix.shape}\n"
                f"ft.get_words() 数量 = {len(words)}\n"
                f"ft.get_labels() = {labels}\n"
                f"predict('球员 进球 冠军') = {pred_labels} p={float(pred_probs[0]):.4f}\n"
                f"结论：gensim 向量可被 FastText pretrainedVectors 加载"
            )
    except Exception:                                  # noqa: BLE001
        return record("A0-5", "gensim→FastText 向量注入链路", False, traceback.format_exc())
    return record("A0-5", "gensim→FastText 向量注入链路", ok, detail)


# --------------------------------------------------------------------------
# A0-6 双编码读写正确性（GB18030 人民日报 / UTF-8 THUCNews）
# --------------------------------------------------------------------------
def check_a06() -> bool:
    detail_lines: list[str] = []
    ok = True

    def decode_with_candidates(path: Path, sample_bytes: int = 200_000):
        """严格解码文件头部样本，返回 (text, 实际编码, 错误信息)。

        编码判定采用 **BOM 感知** 策略，而非简单按候选顺序试解码：
        ``utf-8-sig`` 能成功解码无 BOM 的纯 UTF-8 文件（此时它只是不剥离 BOM），
        因此"逐个试解码"会把纯 UTF-8 误报为 utf-8-sig。
        正确做法是先看有无 BOM，再据此选定编码名，然后严格解码校验。
        """
        raw = path.read_bytes()
        head = raw[:sample_bytes]
        # 按换行边界回退截断，避免在多字节字符中间切断导致严格解码误报失败
        cut = head.rfind(b"\n")
        if cut > 0:
            head = head[:cut + 1]

        if head.startswith(b"\xef\xbb\xbf"):
            candidates = ("utf-8-sig",)
        else:
            candidates = ("utf-8", "gb18030")

        for enc in candidates:
            try:
                return head.decode(enc), enc, None
            except UnicodeDecodeError:
                continue
        return None, None, "候选编码均失败: " + ", ".join(candidates)

    # 人民日报 1998 语料（实测 UTF-8，见 config.ENCODING_PD1998）
    try:
        pd_path = config.DATA_PD1998 / "199801.txt"
        pd_first, pd_enc, err = decode_with_candidates(pd_path)
        if err:
            detail_lines.append(f"[PD1998] 解码失败：{err}")
            ok = False
        else:
            line1 = pd_first.splitlines()[0]
            has_expected = "迈向" in line1 and "世纪" in line1
            no_replacement = "\ufffd" not in line1
            enc_matches = (pd_enc == config.ENCODING_PD1998)
            detail_lines.append(
                f"[PD1998] {pd_path.name} 探测编码={pd_enc}"
                f"（配置声明={config.ENCODING_PD1998}，一致={enc_matches}）")
            detail_lines.append(f"          首行前 60 字符：{line1[:60]}")
            detail_lines.append(
                f"          含期望词={has_expected}  无替换符={no_replacement}")
            ok = ok and has_expected and no_replacement and enc_matches
    except Exception as exc:                           # noqa: BLE001
        detail_lines.append(f"[PD1998] 读取失败：{type(exc).__name__}: {exc}")
        ok = False

    # 全部 6 个月份文件均可用声明编码严格解码（决议 D2 已纳入 6 个月）
    bad_files = []
    for name in config.PD1998_FILES:
        p = config.DATA_PD1998 / name
        if not p.exists():
            bad_files.append(f"{name}(缺失)")
            continue
        try:
            p.read_bytes().decode(config.ENCODING_PD1998)
        except UnicodeDecodeError:
            bad_files.append(f"{name}(解码失败)")
    detail_lines.append(
        f"[PD1998] 6 个月份文件严格解码（{config.ENCODING_PD1998}）: "
        + ("全部通过" if not bad_files else f"异常 {bad_files}")
    )
    ok = ok and not bad_files

    # UTF-8：THUCNews
    try:
        tn_path = config.DATA_THUCNEWS / "体育" / "0.txt"
        tn_first, tn_enc, err = decode_with_candidates(tn_path)
        if err:
            detail_lines.append(f"[THUCNews] 解码失败：{err}")
            ok = False
        else:
            has_cn = any("\u4e00" <= ch <= "\u9fff" for ch in tn_first)
            no_replacement = "\ufffd" not in tn_first
            enc_matches = (tn_enc == config.ENCODING_THUCNEWS)
            detail_lines.append(
                f"[THUCNews] 探测编码={tn_enc}"
                f"（配置声明={config.ENCODING_THUCNEWS}，一致={enc_matches}）")
            detail_lines.append(f"           体育/0.txt 首行：{tn_first.splitlines()[0][:60]}")
            detail_lines.append(
                f"           含中文={has_cn}  无替换符={no_replacement}")
            ok = ok and has_cn and no_replacement and enc_matches
    except Exception as exc:                           # noqa: BLE001
        detail_lines.append(f"[THUCNews] 读取失败：{type(exc).__name__}: {exc}")
        ok = False

    # jieba 分词冒烟（跨数据集统一分词器，§2.3）
    try:
        import jieba
        jieba.setLogLevel(60)
        tokens = list(jieba.cut("中共中央总书记发表新年讲话"))
        detail_lines.append(f"[jieba]  切分结果 = {tokens}")
        ok = ok and len(tokens) > 1
    except Exception as exc:                           # noqa: BLE001
        detail_lines.append(f"[jieba]  分词失败：{type(exc).__name__}: {exc}")
        ok = False

    return record("A0-6", "语料编码探测与分词正确性", ok, "\n".join(detail_lines))


# --------------------------------------------------------------------------
# 数据可达性（非验收项，前置提示）
# --------------------------------------------------------------------------
def check_data_presence() -> None:
    lines = []
    for label, path in [
        ("THUCNews 目录", config.DATA_THUCNEWS),
        ("人民日报语料目录", config.DATA_PD1998),
    ]:
        exists = path.exists()
        lines.append(f"{label}: {'存在' if exists else '缺失'} -> {path}")
    missing = [f for f in config.PD1998_FILES
               if not (config.DATA_PD1998 / f).exists()]
    lines.append(
        "人民日报月份文件: " +
        ("6/6 齐全" if not missing else f"缺失 {missing}")
    )
    record("PRE", "原始数据可达性", not missing, "\n".join(lines))


def main() -> int:
    print("=" * 72)
    print("PA2 阶段 0 环境验收（PLAN.md §1.2）")
    print("=" * 72)

    config.ensure_dirs()
    check_data_presence()

    all_ok = True
    all_ok &= check_a01()
    all_ok &= check_a02_a03()
    all_ok &= check_a04()
    all_ok &= check_a05()
    all_ok &= check_a06()

    # 汇总落盘
    summary = {
        "stage": "0-environment",
        "passed": all(
            r["ok"] for r in RESULTS
            if not r["item"].startswith(("A0-3b",))
        ) and all_ok,
        "results": RESULTS,
    }
    os.makedirs(config.RESULT_METRICS, exist_ok=True)
    out = config.RESULT_METRICS / "stage0_env_check.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")

    print("=" * 72)
    passed = sum(1 for r in RESULTS if r["ok"])
    print(f"验收结果：{passed}/{len(RESULTS)} 通过   -> {out}")
    print("=" * 72)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
