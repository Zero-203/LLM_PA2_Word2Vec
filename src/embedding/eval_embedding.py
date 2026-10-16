# -*- coding: utf-8 -*-
"""阶段 2 · 词向量质量评估（PLAN.md §3.2，验收 A2-4 ~ A2-8）。

四项任务
--------
1. **相似度**（A2-4）：人工构造中文词对 + 相关度打分 → Spearman ρ
2. **类比**（A2-5）：``A:B :: C:?`` 四元组 → Top-1 准确率
3. **领域一致性**（A2-6）：14 类各取代表词，类内 − 类间余弦差
4. **随机基线**（A2-4/5 的对照下界）：同维度随机向量，用于证明
   学到的不是噪声

⚠️ 评估集的构建与用途边界
------------------------
相似度词对与类比四元组是**本项目自建**的（PLAN.md §3.2 已声明）。
它们是"通用语义常识"测试题（如 电脑—计算机、北京:中国::东京:日本），
**不含任何 THUCNews 或人民日报语料的统计信息**，因此：
- 不构成数据泄漏（未用测试语料构造题目）；
- 但也**不能**当作严格的外部基准，报告须如实说明其自建性质。

领域一致性任务用到 14 类代表词，这些词来自类别名与常识，
同样不依赖测试集文本内容。
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 本文件位于 <ROOT>/src/embedding/，故 parents[2] 才是项目根。
# 与 train_word2vec.py 同源缺陷：初版误写 parents[1]，拼出不存在的
# <ROOT>/src/src，导致 `python -m src.embedding.eval_embedding` 报
# ModuleNotFoundError: No module named 'compat'。
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import compat  # noqa: F401,E402
import config  # noqa: E402

# --------------------------------------------------------------------------
# 自建评估集 1：相似度词对（1–5 分，5 = 同义，1 = 无关）
# 覆盖：同义 / 上下位 / 领域相关 / 跨领域无关
# --------------------------------------------------------------------------
SIMILARITY_PAIRS: List[Tuple[str, str, float]] = [
    # 5 分：近义/同义
    ("电脑", "计算机", 5.0),
    ("手机", "电话", 4.5),
    ("医生", "大夫", 5.0),
    ("高兴", "快乐", 5.0),
    ("美丽", "漂亮", 5.0),
    ("购买", "买", 5.0),
    ("父亲", "爸爸", 5.0),
    ("快速", "迅速", 5.0),
    # 4 分：高度相关
    ("股票", "证券", 4.5),
    ("银行", "金融", 4.0),
    ("足球", "篮球", 4.0),
    ("教师", "学校", 4.0),
    ("电影", "演员", 4.0),
    ("医院", "护士", 4.0),
    ("汽车", "轮胎", 4.0),
    ("房价", "楼市", 4.0),
    ("芯片", "半导体", 4.5),
    ("软件", "程序", 4.0),
    # 3 分：中度相关
    ("比赛", "冠军", 3.0),
    ("学生", "考试", 3.0),
    ("公司", "员工", 3.0),
    ("天气", "雨伞", 2.5),
    ("音乐", "舞蹈", 3.0),
    ("飞机", "机场", 3.5),
    ("图书", "阅读", 3.0),
    ("网络", "安全", 3.0),
    ("投资", "风险", 3.0),
    ("政府", "政策", 3.5),
    # 2 分：弱相关
    ("体育", "健康", 2.5),
    ("科技", "未来", 2.0),
    ("城市", "乡村", 2.0),
    ("工作", "休息", 1.5),
    ("夏天", "冬天", 2.0),
    ("男性", "女性", 2.0),
    # 1 分：跨领域无关
    ("股票", "篮球", 1.0),
    ("电脑", "水稻", 1.0),
    ("律师", "蝴蝶", 1.0),
    ("银行", "月亮", 1.0),
    ("足球", "钢琴", 1.0),
    ("医院", "石油", 1.0),
    ("教师", "钢铁", 1.0),
    ("电影", "沙漠", 1.0),
    ("房价", "鲸鱼", 1.0),
    ("芯片", "麦田", 1.0),
]

# --------------------------------------------------------------------------
# 自建评估集 2：类比四元组 A:B :: C:D
# 覆盖：首都 / 性别 / 反义 / 职业场所 / 上下位
# --------------------------------------------------------------------------
ANALOGY_QUADS: List[Tuple[str, str, str, str]] = [
    # 首都关系
    ("北京", "中国", "东京", "日本"),
    ("巴黎", "法国", "柏林", "德国"),
    ("莫斯科", "俄罗斯", "伦敦", "英国"),
    ("首尔", "韩国", "曼谷", "泰国"),
    # 性别对应
    ("男人", "女人", "男孩", "女孩"),
    ("父亲", "母亲", "儿子", "女儿"),
    ("哥哥", "姐姐", "弟弟", "妹妹"),
    ("丈夫", "妻子", "爸爸", "妈妈"),
    # 反义
    ("大", "小", "高", "矮"),
    ("买", "卖", "进", "出"),
    ("上升", "下降", "增长", "减少"),
    # 原为 ("成功","失败","胜利","失败")：D 与 B 重复，校验器已拦截。
    # 该条类比语义本身也不成立（胜利的反义不是失败），改为独立反义对。
    ("成功", "失败", "富裕", "贫穷"),
    # 职业与场所
    ("医生", "医院", "教师", "学校"),
    ("农民", "农田", "工人", "工厂"),
    ("演员", "剧场", "球员", "球场"),
    ("记者", "报社", "厨师", "餐厅"),
    # 上下位/类别
    ("苹果", "水果", "玫瑰", "花"),
    ("汽车", "交通工具", "手机", "电子产品"),
    ("足球", "运动", "钢琴", "乐器"),
    ("股票", "金融", "算法", "计算机"),
    # 程度
    ("好", "更好", "坏", "更坏"),
    ("热", "炎热", "冷", "寒冷"),
    # 时间/序数
    ("一月", "二月", "周一", "周二"),
    ("春天", "夏天", "秋天", "冬天"),
]

# --------------------------------------------------------------------------
# 自建评估集 3：14 类领域代表词（领域一致性任务）
# --------------------------------------------------------------------------
# ⚠️ 词表约束（实测校验，2026-10-12）：14 类 × 15 词 = 210 个词，
#    必须**两两不重复**。曾出现 3 处重复（服务器 在 科技/游戏、
#    装修 在 房产/家居、主席 在 时政 类内重复），会同时抬高两个类别的
#    类内相似度并压低其类间距离，使领域一致性指标失真。
#    运行 ``python -m src.embedding.eval_embedding --check-words`` 可校验。
DOMAIN_WORDS: Dict[str, List[str]] = {
    "体育": ["足球", "篮球", "比赛", "运动员", "教练", "联赛", "进球", "冠军",
             "训练", "奥运", "球员", "裁判", "体育场", "赛事", "排球"],
    "娱乐": ["电影", "明星", "导演", "演出", "歌手", "票房", "电视剧", "娱乐",
             "艺人", "专辑", "演唱会", "剧组", "观众", "奖项", "综艺"],
    "科技": ["计算机", "网络", "软件", "芯片", "数据", "算法", "互联网", "技术",
             "研发", "系统", "人工智能", "服务器", "编程", "数码", "创新"],
    "财经": ["经济", "金融", "市场", "投资", "银行", "利率", "通胀", "财政",
             "增长", "企业", "消费", "贸易", "税收", "预算", "产业"],
    "股票": ["股票", "股价", "股市", "证券", "涨停", "跌停", "大盘", "基金",
             "上市公司", "股东", "市盈率", "成交量", "牛市", "熊市", "板块"],
    "时政": ["政府", "主席", "总理", "会议", "政策", "外交", "部长", "国家",
             "领导", "改革", "党委", "国务院", "官员", "议案", "议会"],
    "社会": ["市民", "社区", "事故", "案件", "居民", "志愿者", "老人", "儿童",
             "救助", "治安", "民生", "就业", "保障", "公益", "邻里"],
    "教育": ["学生", "老师", "学校", "考试", "高考", "课程", "教材", "招生",
             "毕业", "大学", "中学", "小学", "教师", "校园", "录取"],
    "房产": ["房价", "楼市", "买房", "户型", "开发商", "商品房", "二手房",
             "物业", "地产", "购房", "楼盘", "租金", "土地", "住宅", "售楼"],
    "家居": ["家具", "沙发", "装修", "厨房", "卧室", "灯具", "地板", "窗帘",
             "橱柜", "衣柜", "卫浴", "家装", "布艺", "瓷砖", "收纳"],
    "游戏": ["游戏", "玩家", "关卡", "网游", "手游", "副本", "装备", "角色",
             "竞技", "电竞", "任务", "攻略", "游戏机", "对战", "沙盒"],
    "时尚": ["时装", "潮流", "搭配", "服饰", "品牌", "模特", "设计", "秀场",
             "流行", "穿搭", "奢侈", "化妆品", "造型", "面料", "配饰"],
    "星座": ["星座", "运势", "塔罗", "占星", "生肖", "属相", "水星", "金星",
             "月亮", "上升", "命理", "八字", "紫微", "占卜", "命盘"],
    "彩票": ["彩票", "开奖", "中奖", "双色球", "福彩", "体彩", "号码", "投注",
             "奖金", "大奖", "彩民", "摇奖", "头奖", "选号", "购彩"],
}


# --------------------------------------------------------------------------
# 载入
# --------------------------------------------------------------------------
def load_keyed_vectors(exp_id: str):
    """从实验目录载入 KeyedVectors。"""
    from gensim.models import KeyedVectors

    path = config.MODEL_W2V / exp_id / "model.wv"
    if not path.exists():
        raise FileNotFoundError(f"模型不存在: {path}")
    return KeyedVectors.load(str(path))


def make_random_baseline(wv, seed: Optional[int] = None):
    """构造同维度、同词表的随机向量基线（A2-4/A2-5 的对照下界）。"""
    import numpy as np
    from gensim.models import KeyedVectors

    seed = seed if seed is not None else config.SEED
    rng = np.random.RandomState(seed)
    dim = wv.vector_size
    random_wv = KeyedVectors(vector_size=dim)
    vectors = rng.normal(size=(len(wv.index_to_key), dim)).astype("float32")
    random_wv.add_vectors(wv.index_to_key, vectors)
    return random_wv


# --------------------------------------------------------------------------
# 任务 1：相似度（Spearman）
# --------------------------------------------------------------------------
def eval_similarity(wv) -> Dict[str, Any]:
    """计算人工词对打分与模型余弦相似度的 Spearman 相关系数。"""
    from scipy.stats import spearmanr

    human, model, used, skipped = [], [], [], []
    for w1, w2, score in SIMILARITY_PAIRS:
        if w1 in wv and w2 in wv:
            human.append(score)
            model.append(float(wv.similarity(w1, w2)))
            used.append((w1, w2, score))
        else:
            missing = [w for w in (w1, w2) if w not in wv]
            skipped.append((w1, w2, missing))

    if len(human) < 10:
        return {
            "n_pairs": len(human), "n_skipped": len(skipped),
            "skipped_examples": skipped[:10],
            "error": f"可用词对仅 {len(human)} 对（<10），无法可靠计算相关系数",
        }

    rho, p_value = spearmanr(human, model)
    return {
        "n_pairs": len(human),
        "n_total": len(SIMILARITY_PAIRS),
        "n_skipped": len(skipped),
        "coverage": round(len(human) / len(SIMILARITY_PAIRS), 4),
        "skipped_examples": skipped[:10],
        "spearman_rho": round(float(rho), 4),
        "p_value": float(p_value),
    }


# --------------------------------------------------------------------------
# 任务 2：类比（Top-1 准确率）
# --------------------------------------------------------------------------
def eval_analogy(wv) -> Dict[str, Any]:
    """计算 A:B :: C:D 类比的 Top-1 准确率。"""
    correct, total, skipped, details = 0, 0, [], []
    for a, b, c, d in ANALOGY_QUADS:
        if not all(w in wv for w in (a, b, c, d)):
            missing = [w for w in (a, b, c, d) if w not in wv]
            skipped.append({"quad": [a, b, c, d], "missing": missing})
            continue
        total += 1
        try:
            result = wv.most_similar(positive=[b, c], negative=[a], topn=1)
            pred = result[0][0]
            hit = (pred == d)
            if hit:
                correct += 1
            details.append({"quad": [a, b, c, d], "predicted": pred, "hit": hit,
                            "score": round(float(result[0][1]), 4)})
        except KeyError:
            skipped.append({"quad": [a, b, c, d], "missing": ["KeyError"]})

    return {
        "n_total": len(ANALOGY_QUADS),
        "n_evaluated": total,
        "n_correct": correct,
        "top1_accuracy": round(correct / total * 100, 2) if total else 0.0,
        "n_skipped": len(skipped),
        "skipped_examples": skipped[:10],
        "details": details,
    }


# --------------------------------------------------------------------------
# 任务 3：领域一致性（类内 − 类间余弦）
# --------------------------------------------------------------------------
def eval_domain_coherence(wv) -> Dict[str, Any]:
    """计算各类别代表词的类内平均余弦与类间平均余弦之差。

    ⚠️ 度量口径（实测修正，2026-10-12）：类内与类间**必须用同一种口径**，
    否则二者不可比。初版实现"类内用两两余弦、类间用归一化质心余弦"，
    而质心平均会消掉随机噪声，使类间余弦被系统性抬高。
    实测在纯随机向量（无任何语义）下：

        n_labels=14, n_words=15:
            类内两两   = -0.0043
            类间(质心) = +0.0040  -> intra-inter = -0.0083
            类间(两两) = +0.0003  -> intra-inter = -0.0046
        n_labels= 4, n_words=15:
            质心法偏差 = -0.0027（方向相反）

    偏差量级约 ±0.004 且**符号不稳定**，而 A2-6 阈值仅 0.05，
    该噪声足以干扰判定。故统一为两两余弦，质心法仅作对照记录。
    """
    import numpy as np

    intra: Dict[str, float] = {}
    coverage: Dict[str, int] = {}
    normed_by_label: Dict[str, Any] = {}
    centroid_by_label: Dict[str, Any] = {}

    for label, words in DOMAIN_WORDS.items():
        vecs = [wv[w] for w in words if w in wv]
        coverage[label] = len(vecs)
        if len(vecs) < 2:
            continue
        arr = np.asarray(vecs, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        normed = arr / norms
        normed_by_label[label] = normed

        # 类内：两两余弦（排除自身对角线）
        sim = normed @ normed.T
        iu = np.triu_indices(len(arr), k=1)
        intra[label] = float(sim[iu].mean())

        c = normed.mean(axis=0)
        nc = np.linalg.norm(c)
        centroid_by_label[label] = c / nc if nc > 0 else None

    labels = sorted(normed_by_label)

    # 类间（两两余弦，主口径）：所有跨类词对的余弦均值
    inter_pair_vals: List[float] = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            cross = normed_by_label[labels[i]] @ normed_by_label[labels[j]].T
            inter_pair_vals.append(float(cross.mean()))
    inter_pair_mean = float(np.mean(inter_pair_vals)) if inter_pair_vals else 0.0

    # 类间（质心余弦，仅作对照记录，不参与判定）
    inter_centroid_vals: List[float] = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            v1 = centroid_by_label.get(labels[i])
            v2 = centroid_by_label.get(labels[j])
            if v1 is not None and v2 is not None:
                inter_centroid_vals.append(float(v1 @ v2))
    inter_centroid_mean = (float(np.mean(inter_centroid_vals))
                           if inter_centroid_vals else 0.0)

    intra_mean = float(np.mean(list(intra.values()))) if intra else 0.0

    return {
        "n_labels_evaluated": len(intra),
        "n_labels_total": len(DOMAIN_WORDS),
        "coverage": coverage,
        "metric_convention": "类内与类间均为两两余弦（apple-to-apple）",
        "intra_cosine_per_label": {k: round(v, 4) for k, v in intra.items()},
        "intra_cosine_mean": round(intra_mean, 4),
        "inter_cosine_mean": round(inter_pair_mean, 4),
        "intra_minus_inter": round(intra_mean - inter_pair_mean, 4),
        "reference_only": {
            "inter_centroid_mean": round(inter_centroid_mean, 4),
            "intra_minus_inter_centroid": round(
                intra_mean - inter_centroid_mean, 4),
            "note": "质心法仅作对照，不参与 A2-6 判定",
        },
    }


# --------------------------------------------------------------------------
# 定性抽样（A2-7 人工核验）
# --------------------------------------------------------------------------
def sample_most_similar(wv, probe_words: Optional[List[str]] = None,
                        topn: int = 10) -> Dict[str, List[Dict[str, Any]]]:
    """对各类别代表词查询 most_similar，供 A2-7 人工判定与报告截图。"""
    probes = probe_words or ["体育", "股票", "科技", "娱乐", "教育", "房产"]
    out = {}
    for w in probes:
        if w not in wv:
            out[w] = {"error": "不在词表中"}
            continue
        out[w] = [
            {"word": t, "similarity": round(float(s), 4)}
            for t, s in wv.most_similar(w, topn=topn)
        ]
    return out


# --------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# 自建评估集完整性校验
# --------------------------------------------------------------------------
def check_eval_sets() -> Dict[str, Any]:
    """校验三个自建评估集的结构约束。

    这些约束一旦被破坏，指标会静默失真而非报错，故必须显式校验：
    - 领域词表：14 类 × 15 词 = 210，且**两两不重复**
      （重复会同时抬高两类的类内相似度并压低其类间距离）
    - 全部为纯中文词（跨语料对齐的前提，避免 ASCII 词混入）
    - 相似度词对：词序无关的无序对不重复，打分在 [1, 5]
    - 类比四元组：四元组内无重复词，且 A≠D
    """
    from collections import Counter

    problems: List[str] = []

    # --- 领域词表 ---
    all_words = [w for ws in DOMAIN_WORDS.values() for w in ws]
    dup = [w for w, c in Counter(all_words).items() if c > 1]
    if dup:
        for w in dup:
            labs = [l for l, ws in DOMAIN_WORDS.items() if w in ws]
            problems.append(f"领域词重复: {w!r} 出现在 {labs}")
    sizes = {l: len(ws) for l, ws in DOMAIN_WORDS.items()}
    if len(sizes) != len(config.LABELS):
        problems.append(f"领域类别数 {len(sizes)} != THUCNews 类别数 {len(config.LABELS)}")
    missing_labels = set(config.LABELS) - set(DOMAIN_WORDS)
    if missing_labels:
        problems.append(f"领域词表缺少类别: {sorted(missing_labels)}")
    non_uniform = {l: n for l, n in sizes.items() if n != 15}
    if non_uniform:
        problems.append(f"各类词数不一致（应均为 15）: {non_uniform}")
    non_cn = [w for w in all_words if not all("\u4e00" <= c <= "\u9fff" for c in w)]
    if non_cn:
        problems.append(f"领域词含非纯中文字符: {non_cn}")

    # --- 相似度词对 ---
    seen_pairs = set()
    for w1, w2, score in SIMILARITY_PAIRS:
        if not (1.0 <= score <= 5.0):
            problems.append(f"相似度打分越界: ({w1},{w2})={score} 应在 [1,5]")
        key = tuple(sorted((w1, w2)))
        if key in seen_pairs:
            problems.append(f"相似度词对重复: {key}")
        seen_pairs.add(key)
        if w1 == w2:
            problems.append(f"相似度词对自身相同: {w1}")

    # --- 类比四元组 ---
    for quad in ANALOGY_QUADS:
        a, b, c, d = quad
        if len(set(quad)) != 4:
            problems.append(f"类比四元组内有重复词: {list(quad)}")
        if a == d:
            problems.append(f"类比四元组 A==D（答案即题面）: {list(quad)}")

    result = {
        "similarity_pairs": len(SIMILARITY_PAIRS),
        "analogy_quads": len(ANALOGY_QUADS),
        "domain_labels": len(DOMAIN_WORDS),
        "domain_words_total": len(all_words),
        "domain_words_unique": len(set(all_words)),
        "score_distribution": dict(Counter(
            int(s) for _, _, s in SIMILARITY_PAIRS)),
        "problems": problems,
        "ok": not problems,
    }
    return result


def evaluate(exp_id: str, with_random_baseline: bool = True) -> Dict[str, Any]:
    """对指定实验做全套词向量质量评估，并落盘。"""
    print(f"=== 评估 {exp_id} ===", flush=True)

    # 评估前先校验自建评估集，防止词表退化导致指标静默失真
    chk = check_eval_sets()
    if not chk["ok"]:
        print("  [FAIL] 自建评估集校验未通过：")
        for p in chk["problems"]:
            print(f"    - {p}")
        raise ValueError("自建评估集存在结构问题，请先修复再评估")
    print(f"  评估集校验通过：相似度 {chk['similarity_pairs']} 对 / "
          f"类比 {chk['analogy_quads']} 组 / 领域 {chk['domain_labels']} 类 "
          f"× {chk['domain_words_total'] // chk['domain_labels']} 词（无重复）")

    wv = load_keyed_vectors(exp_id)
    print(f"  词表 = {len(wv):,}  维度 = {wv.vector_size}", flush=True)

    sim = eval_similarity(wv)
    ana = eval_analogy(wv)
    dom = eval_domain_coherence(wv)
    samples = sample_most_similar(wv)

    result: Dict[str, Any] = {
        "exp_id": exp_id,
        "vocab_size": len(wv),
        "vector_size": wv.vector_size,
        "similarity": sim,
        "analogy": ana,
        "domain_coherence": dom,
        "most_similar_samples": samples,
    }

    if with_random_baseline:
        rwv = make_random_baseline(wv)
        r_sim = eval_similarity(rwv)
        r_ana = eval_analogy(rwv)
        r_dom = eval_domain_coherence(rwv)
        th = config.THRESHOLDS
        result["random_baseline"] = {
            "similarity": {k: v for k, v in r_sim.items() if k != "skipped_examples"},
            "analogy": {k: v for k, v in r_ana.items() if k != "details"},
            "domain_coherence": {k: v for k, v in r_dom.items()
                                 if k != "intra_cosine_per_label"},
        }
        result["deltas_vs_random"] = {
            "spearman_delta": round(
                sim.get("spearman_rho", 0.0) - r_sim.get("spearman_rho", 0.0), 4),
            "analogy_top1_delta_pp": round(
                ana["top1_accuracy"] - r_ana["top1_accuracy"], 2),
            "domain_intra_inter_delta": round(
                dom["intra_minus_inter"] - r_dom["intra_minus_inter"], 4),
        }
        d = result["deltas_vs_random"]
        result["thresholds"] = {
            "A2-4_spearman_delta_min": th["A2-4_spearman_delta_min"],
            "A2-5_analogy_top1_delta_min": th["A2-5_analogy_top1_delta_min"],
            "A2-6_intra_inter_cos_min": th["A2-6_intra_inter_cos_min"],
        }
        result["verdict"] = {
            "A2-4": d["spearman_delta"] > th["A2-4_spearman_delta_min"],
            "A2-5": d["analogy_top1_delta_pp"] > th["A2-5_analogy_top1_delta_min"],
            "A2-6": dom["intra_minus_inter"] > th["A2-6_intra_inter_cos_min"],
        }
        print(f"  Spearman ρ = {sim.get('spearman_rho')}  "
              f"(随机 {r_sim.get('spearman_rho')}, Δ={d['spearman_delta']})")
        print(f"  类比 Top-1 = {ana['top1_accuracy']}%  "
              f"(随机 {r_ana['top1_accuracy']}%, Δ={d['analogy_top1_delta_pp']}pp)")
        print(f"  类内−类间 = {dom['intra_minus_inter']}  "
              f"(随机 {r_dom['intra_minus_inter']})")
        print(f"  判定: {result['verdict']}")

    out = config.RESULT_METRICS / f"w2v_eval_{exp_id}.json"
    os.makedirs(out.parent, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"  -> {out}", flush=True)
    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="词向量质量评估")
    ap.add_argument("exp_id", nargs="?", default=None,
                    help="实验目录名，如 E1_baseline")
    ap.add_argument("--no-baseline", action="store_true",
                    help="跳过随机基线对照")
    ap.add_argument("--check-words", action="store_true",
                    help="仅校验自建评估集的结构约束后退出")
    args = ap.parse_args()

    if args.check_words:
        chk = check_eval_sets()
        print(json.dumps(chk, ensure_ascii=False, indent=2))
        raise SystemExit(0 if chk["ok"] else 1)

    if not args.exp_id:
        ap.error("请提供 exp_id，或使用 --check-words 仅校验评估集")
    evaluate(args.exp_id, with_random_baseline=not args.no_baseline)
