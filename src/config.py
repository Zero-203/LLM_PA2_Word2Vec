"""PA2 全局配置：路径、种子、标签体系与验收阈值的单点定义。

所有阶段脚本统一从此处导入，禁止在脚本内散落硬编码常量，
以保证实验可复现、阈值可审计。
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# 运行环境兼容层：必须在任何第三方库（尤其 matplotlib）导入之前生效。
# 详见 src/compat.py 的问题背景说明。
# --------------------------------------------------------------------------
import compat  # noqa: E402  (副作用导入：安装 pathlib.mkdir 补丁)

# --------------------------------------------------------------------------
# 基础路径
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

# 绘图缓存目录收敛到项目内（须早于 import matplotlib 执行）
compat.configure_matplotlib(ROOT)

DATA_RAW = ROOT / "data"
DATA_THUCNEWS = DATA_RAW / "THUCNews"
DATA_PD1998 = DATA_RAW / "199801"          # 人民日报 1998 语料目录（含 199801~199806）

DATA_INTERIM = DATA_RAW / "interim"         # 清洗后 jsonl 与清洗报告
DATA_PROCESSED = DATA_RAW / "processed"     # U / L_* / T_* 与向量
DATA_VECTORS = DATA_PROCESSED / "vectors"

MODEL_DIR = ROOT / "model"
MODEL_W2V = MODEL_DIR / "word2vec"
MODEL_FT = MODEL_DIR / "fasttext"

RESULT_DIR = ROOT / "result"
RESULT_METRICS = RESULT_DIR / "metrics"
RESULT_FIGURES = RESULT_DIR / "figures"
RESULT_REPORTS = RESULT_DIR / "reports"

# --------------------------------------------------------------------------
# 复现性
# --------------------------------------------------------------------------
SEED = 20261012

# --------------------------------------------------------------------------
# 编码
#
# ⚠️ 实测更正（2026-10-12）：人民日报 1998 语料 6 个月份文件全部为 **UTF-8**，
#    而非计划初稿所记的 GB18030。验证方式：对每个文件做整文件严格解码，
#    utf-8 全部通过、gb18030 全部失败（首行含 0x84 等 GB18030 非法字节，
#    对应 UTF-8 中"满/望"等字的多字节序列）。
#    初稿误判源于 PowerShell 控制台以 GB18030 显示 UTF-8 字节产生的乱码。
# --------------------------------------------------------------------------
ENCODING_THUCNEWS = "utf-8"
ENCODING_PD1998 = "utf-8"

# 解码候选顺序：严格模式逐个尝试，全部失败才报错，禁止静默替换非法字节。
# 保留 gb18030 作为兜底，以应对后续可能混入的其他来源语料。
DECODING_CANDIDATES = ("utf-8-sig", "utf-8", "gb18030")

# --------------------------------------------------------------------------
# 语料文件
# --------------------------------------------------------------------------
# 人民日报 1998 语料：已确认纳入全部 6 个月（决议 D2）
PD1998_FILES = ["199801.txt", "199802.txt", "199803.txt",
                "199804.txt", "199805.txt", "199806.txt"]
PD1998_PRIMARY_MONTH = "199801"             # 人民日报主评估月

# DeepSeek API 人民日报逐篇标注配置（密钥仅从环境变量读取）
DEEPSEEK_API_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-flash"
DEEPSEEK_THINKING_MODE = "disabled"
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
PD1998_LABEL_PROMPT_VERSION = "deepseek-flash-thucnews14-v2-nonthinking"
PD1998_LABEL_CONFIDENCE_THRESHOLD = 0.75
PD1998_LABEL_BATCH_SIZE = 4
PD1998_LABEL_MAX_WORKERS = 2

# --------------------------------------------------------------------------
# THUCNews 类别体系（14 类，实测文档数见 LABEL_EXPECTED_COUNT）
# --------------------------------------------------------------------------
LABELS = [
    "体育", "娱乐", "家居", "彩票", "房产", "教育", "时尚",
    "时政", "星座", "游戏", "社会", "科技", "股票", "财经",
]

# 实测值（2026-10-12 统计），用于清洗阶段完整性断言 A1-1
LABEL_EXPECTED_COUNT = {
    "体育": 131604, "娱乐": 92632, "家居": 32586, "彩票": 7588,
    "房产": 20050, "教育": 41936, "时尚": 13368, "时政": 63086,
    "星座": 3578, "游戏": 24373, "社会": 50849, "科技": 162929,
    "股票": 154398, "财经": 37098,
}
THUCNEWS_TOTAL_EXPECTED = 836075

# 小类（验收 A4-9 要求逐类 recall 达标，防止被大类淹没）
SMALL_LABELS = ["星座", "彩票", "时尚"]

# 类别阶梯实验（作业要求"由少到多逐步展开"）
LADDER = {
    "S1": ["体育", "财经"],
    "S2": ["体育", "财经", "科技", "娱乐"],
    "S3": ["体育", "财经", "科技", "娱乐", "时政", "社会", "教育", "游戏"],
    "S4": LABELS,
}

# --------------------------------------------------------------------------
# 数据划分规模（决议 D3：先 E7 定饱和点）
# --------------------------------------------------------------------------
U_CAP_PER_LABEL = 30000        # 无标注嵌入集每类上限
L_CAP_PER_LABEL = 20000        # 标注集每类上限（分层降采样，控制 1:45 不平衡）
L_VAL_RATIO = 0.10
L_TEST_RATIO = 0.10
MIN_DOC_TOKENS = 20            # 文档级最小 token 数

# 语料量消融 E7 的采样档位（篇数）
E7_CORPUS_SIZES = [50000, 150000, None]     # None 表示全量

# 小类兜底：保证 U 中每类至少有 U_MIN_PER_LABEL 篇，使词向量覆盖该类领域词汇。
# 触发条件：某类总量 <= L_CAP（剩余全给了 L，U 拿不到或只拿到极少样本）。
# 让渡方式：**只补足到最小篇数**，不按比例多让 —— 因为 L 样本直接决定分类
# 性能（A4-6 macro-F1 阈值），而 U 已有 23 万+篇，边际收益远小于 L。
U_MIN_PER_LABEL = 500
# 让渡后 L 的下限，防止把小类的标注样本削得过少
L_FLOOR_AFTER_GIVE = 2000


def allocate_split(counts: "dict[str,int] | None" = None) -> "dict[str,dict[str,int]]":
    """按"L 优先 + 小类补足"策略为每类分配 U / L 篇数。

    为什么 L 优先（实测更正，2026-10-12）
    ------------------------------------
    若按"U 优先"（每类先取 min(n, 30000) 给 U，剩余给 L），则
    彩票(7588)/房产(20050)/时尚(13368)/星座(3578)/游戏(24373) 这 5 类
    总量均 <= 30000，会被 U 全部吃掉，导致 **L 中这 5 类样本数为 0**，
    14 类分类任务 S4 无法训练。故必须先满足 L，再分配 U。

    分配规则
    --------
    1. ``l = min(n, L_CAP_PER_LABEL)``            L 优先，上限 20000
    2. ``u = min(n - l, U_CAP_PER_LABEL)``        U 取剩余，上限 30000
    3. 小类补足：若 ``u < U_MIN_PER_LABEL`` 且补足后 ``l >= L_FLOOR_AFTER_GIVE``，
       则从 L 让渡 ``U_MIN_PER_LABEL - u`` 篇给 U。

    返回 ``{label: {"total": n, "u": u, "l": l}}``，且保证 ``u + l <= n``。
    """
    counts = counts or LABEL_EXPECTED_COUNT
    out: "dict[str,dict[str,int]]" = {}
    for lab, n in counts.items():
        l = min(n, L_CAP_PER_LABEL)
        u = min(n - l, U_CAP_PER_LABEL)
        if u < U_MIN_PER_LABEL:
            give = U_MIN_PER_LABEL - u
            if l - give >= L_FLOOR_AFTER_GIVE:
                l -= give
                u += give
        assert u + l <= n, f"{lab}: u+l={u+l} > n={n}"
        out[lab] = {"total": n, "u": u, "l": l}
    return out


def ladder_labels(stage: str) -> list[str]:
    """返回类别阶梯实验某一档的类别列表（S1/S2/S3/S4）。"""
    return list(LADDER[stage])

# --------------------------------------------------------------------------
# 人民日报 1998 语料实测基线（验收 A1-7）
# --------------------------------------------------------------------------
PD199801_EXPECTED = {
    "paragraph_count": 22564,
    "article_count": 3148,
    "banci_count": 12,
}
PD199801_BANCI_ARTICLES = {
    "01": 315, "02": 467, "03": 337, "04": 447, "05": 261, "06": 299,
    "07": 216, "08": 168, "09": 161, "10": 172, "11": 171, "12": 134,
}

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Word2Vec 参数（§3.1 基线与消融矩阵）
# --------------------------------------------------------------------------
W2V_BASELINE = {
    "vector_size": 200,
    "window": 5,
    "min_count": 5,
    "sg": 0,                # 0=CBOW, 1=Skip-gram
    "hs": 0,
    "negative": 5,
    "sample": 1e-3,
    "alpha": 0.025,
    "min_alpha": 1e-4,
    "epochs": 5,
    "seed": SEED,
}
W2V_WORKERS = max(1, (os.cpu_count() or 2) - 1)

W2V_GRID = {
    "vector_size": [100, 200, 300],
    "window": [3, 5, 10],
    "sg": [0, 1],
    "epochs": [5, 10, 15],
}

# --------------------------------------------------------------------------
# 文档向量化（§4.1）
# --------------------------------------------------------------------------
POOLING_METHODS = ["P1_mean", "P2_tfidf", "P3_nostop_mean"]

# --------------------------------------------------------------------------
# FastText 参数（§5.1）
# --------------------------------------------------------------------------
FT_BASELINE = {
    "dim": W2V_BASELINE["vector_size"],   # 必须与预训练向量维度一致
    "epoch": 5,
    "lr": 0.1,
    "wordNgrams": 1,
    "minCount": 1,
    "loss": "softmax",
    "bucket": 2000000,
    "thread": W2V_WORKERS,
    "verbose": 0,
}
FT_GRID = {
    "epoch": [5, 10, 25],
    "lr": [0.05, 0.1, 0.5, 1.0],
    "wordNgrams": [1, 2],
    "minCount": [1, 2],
    "loss": ["softmax", "hs", "ova"],
}

# --------------------------------------------------------------------------
# 验收阈值（决议 D4：保持自设）
# 编号对应 PLAN.md 中的验收项，便于自动断言与报告交叉引用
# --------------------------------------------------------------------------
THRESHOLDS = {
    # --- 阶段 2 词嵌入 ---
    "A2-4_spearman_delta_min": 0.20,      # w2v 相对随机基线的 Spearman 提升
    "A2-5_analogy_top1_delta_min": 20.0,  # 百分点
    "A2-6_intra_inter_cos_min": 0.05,     # 类内 − 类间余弦差
    "A2-10_repro_cosine_min": 0.999,      # 复现性余弦阈值（多线程不保证逐位一致）

    # --- 阶段 3 文档向量 ---
    "A3-3_norm_tolerance": 1e-5,
    "A3-6_pooling_pairwise_cos_max": 0.99,

    # --- 阶段 4 分类训练（L_val）---
    "A4-4_overfit_gap_max": 0.15,         # train_acc − val_acc
    "A4-5_S1_val_acc_min": 0.95,          # 二分类
    "A4-6_S4_val_macro_f1_min": 0.85,
    "A4-9_small_label_recall_min": 0.60,
    "A4-2_pretrain_vocab_overlap_min": 0.50,

    # --- 阶段 5 测试评估 ---
    "A5-1_Ltest_S4_macro_f1_min": 0.85,
    "A5-1_Ltest_S4_acc_min": 0.88,
    "A5-2_test_val_gap_max": 0.03,        # test 与 val 最优点差值
    "A5-5_per_class_recall_min": 0.60,
    "A5-6_chi2_p_max": 0.05,
    "cross_month_range_max": 0.10,        # 跨月极差（决议 D2）

    # --- 阶段 1 数据清洗 ---
    "A1-1_malformed_rate_max": 0.0,       # 畸形行率必须为 0
    "A1-5_label_ratio_deviation_max": 0.01,
    "A1-9_u_min_per_label": 400,
}

# --------------------------------------------------------------------------
# 输出文件名
# --------------------------------------------------------------------------
F_THUCNEWS_CLEAN = DATA_INTERIM / "thucnews_clean.jsonl"
F_THUCNEWS_REPORT = DATA_INTERIM / "thucnews_clean_report.json"
F_PD1998_CLEAN = DATA_INTERIM / "pd1998_clean.jsonl"
F_PD1998_REPORT = DATA_INTERIM / "pd1998_clean_report.json"
F_PD1998_FORMATTED = DATA_INTERIM / "pd1998_formatted.jsonl"
F_PD1998_FORMAT_REPORT = DATA_INTERIM / "pd1998_format_report.json"
F_PD1998_LLM_LABELS = DATA_INTERIM / "pd1998_llm_labels.jsonl"
F_PD1998_LLM_LABEL_REPORT = DATA_INTERIM / "pd1998_llm_label_report.json"
F_SPLIT_MANIFEST = DATA_PROCESSED / "split_manifest.json"
F_ENV_SNAPSHOT = RESULT_DIR / "env_snapshot.json"

# U 集与 L 集文件名
F_U_CORPUS = DATA_PROCESSED / "U_corpus.txt"
F_L_TRAIN = DATA_PROCESSED / "L_train.txt"
F_L_VAL = DATA_PROCESSED / "L_val.txt"
F_L_TEST = DATA_PROCESSED / "L_test.txt"
# 人民日报 FastText 格式数据：confirmed/high 与含有效复核候选的 all
F_T_PD1998 = DATA_PROCESSED / "T_pd199801_confirmed.txt"
F_T_PD1998_ALL = DATA_PROCESSED / "T_pd1998_all_confirmed.txt"
F_T_PD1998_HIGH = DATA_PROCESSED / "T_pd199801_high.txt"
F_T_PD1998_CANDIDATES = DATA_PROCESSED / "T_pd199801_all.txt"
F_T_PD1998_ALL_HIGH = DATA_PROCESSED / "T_pd1998_all_high.txt"
F_T_PD1998_ALL_CANDIDATES = DATA_PROCESSED / "T_pd1998_all.txt"


def ensure_dirs() -> None:
    """创建全部产出目录（幂等）。

    注意：此处刻意使用 os.makedirs 而非 pathlib.Path.mkdir。
    后者会向底层传入 mode 参数，在部分受限运行环境下会触发
    TypeError（mkdir 拦截器不接受 mode），故统一走 os.makedirs 以保证可移植性。
    """
    for path in (DATA_INTERIM, DATA_PROCESSED, DATA_VECTORS,
                 MODEL_W2V, MODEL_FT,
                 RESULT_METRICS, RESULT_FIGURES, RESULT_REPORTS):
        os.makedirs(path, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print("ROOT =", ROOT)
    print("labels =", len(LABELS))
    print("PD1998 labels = DeepSeek Flash article-level labels (pending)")
