# -*- coding: utf-8 -*-
"""Generate and extract a one-pass human audit packet for Stage 5."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import compat  # noqa: F401,E402
import config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = ROOT / "result" / "manual_audit" / "stage5"
SAMPLES_PATH = WORK_DIR / "samples.jsonl"
TXT_PATH = WORK_DIR / "annotate.txt"
ANNOTATIONS_PATH = WORK_DIR / "annotations.jsonl"
SUMMARY_PATH = WORK_DIR / "summary.json"
EXAMPLE_PATH = WORK_DIR / "填写示例_虚构案例.txt"
SEED = config.SEED
BEGIN_PREFIX = "===== SAMPLE BEGIN: "
END_MARKER = "===== SAMPLE END ====="
MANUAL_MARKER = "【人工填写区】"
FIELDS = (
    "human_label", "label_clarity", "evidence", "error_pattern", "comment",
    "llm_label_supported", "model_prediction_supported", "attribution_causes",
    "attribution_confidence",
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def fasttext_model():
    import fasttext

    model_path = config.MODEL_FT / "S4_best_pretrained" / "model.bin"
    if not model_path.is_file():
        raise FileNotFoundError("缺少冻结模型: " + str(model_path))
    if os.name == "nt" and any(ord(char) > 127 for char in str(model_path)):
        ascii_path = Path(tempfile.gettempdir()) / "pa2_fasttext_ascii" / "S4_best_pretrained.bin"
        ascii_path.parent.mkdir(parents=True, exist_ok=True)
        if not ascii_path.exists() or ascii_path.stat().st_size != model_path.stat().st_size:
            shutil.copyfile(str(model_path), str(ascii_path))
        model_path = ascii_path
    return fasttext.load_model(str(model_path))


def predict(model: Any, texts: Sequence[str], batch_size: int = 512) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for start in range(0, len(texts), batch_size):
        batch = [text if text.strip() else "__empty__" for text in texts[start:start + batch_size]]
        names, probs = model.predict(batch, k=3)
        for row_names, row_probs in zip(names, probs):
            labels = [name.removeprefix("__label__") for name in row_names]
            values = [float(value) for value in row_probs]
            result.append({
                "prediction": labels[0] if labels else "",
                "confidence": values[0] if values else 0.0,
                "top3": [{"label": label, "probability": value}
                         for label, value in zip(labels, values)],
            })
    return result


def spread_sample(rows: Sequence[Dict[str, Any]], count: int,
                  group_key: str, rng: random.Random) -> List[Dict[str, Any]]:
    """Deterministically sample across months/classes by round-robin groups."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(group_key, "unknown"))].append(row)
    keys = sorted(groups)
    for key in keys:
        rng.shuffle(groups[key])
    rng.shuffle(keys)
    chosen = []
    while len(chosen) < min(count, len(rows)):
        advanced = False
        for key in keys:
            if groups[key] and len(chosen) < count:
                chosen.append(groups[key].pop())
                advanced = True
        if not advanced:
            break
    return chosen


def read_thucnews_test(model: Any) -> List[Dict[str, Any]]:
    id_path = config.DATA_PROCESSED / "doc_ids_test.txt"
    if not id_path.is_file():
        raise FileNotFoundError("缺少 doc_ids_test.txt；请先生成数据切分清单")
    test_ids = set(id_path.read_text(encoding="utf-8").splitlines())
    clean_path = config.F_THUCNEWS_CLEAN
    if not clean_path.is_file():
        raise FileNotFoundError("缺少清洗后的 THUCNews 文件: " + str(clean_path))
    records = [row for row in read_jsonl(clean_path) if row.get("doc_id") in test_ids]
    if len(records) != len(test_ids):
        raise ValueError("doc_ids_test 与清洗记录不匹配，无法安全构建错例抽查")
    predictions = predict(model, [" ".join(row.get("tokens") or []) for row in records])
    samples = []
    for row, predicted in zip(records, predictions):
        if row.get("split") != "L":
            raise ValueError("测试 ID 中出现非 L 划分记录: " + str(row.get("doc_id")))
        samples.append({
            "sample_id": "A54-" + row["doc_id"].replace("/", "-").replace("\\", "-"),
            "track": "A5-4",
            "source": {"dataset": "THUCNews", "split": "L_test", "doc_id": row["doc_id"]},
            "gold_label": row["label"],
            "model": predicted,
        })
    return samples


def select_a54(model: Any, rng: random.Random) -> List[Dict[str, Any]]:
    rows = read_thucnews_test(model)
    directions = [("财经", "股票"), ("教育", "社会"), ("股票", "财经")]
    selected: List[Dict[str, Any]] = []
    selected_ids = set()
    for true_label, predicted_label in directions:
        candidates = [row for row in rows
                      if row["gold_label"] == true_label
                      and row["model"]["prediction"] == predicted_label]
        sampled = spread_sample(candidates, 10, "gold_label", rng)
        for row in sampled:
            row["sample_group"] = "error:" + true_label + "->" + predicted_label
            selected.append(row)
            selected_ids.add(row["sample_id"])
    positive_classes = {"财经", "股票", "教育", "社会"}
    positive = [row for row in rows
                if row["gold_label"] in positive_classes
                and row["model"]["prediction"] == row["gold_label"]
                and row["sample_id"] not in selected_ids]
    positive_sample = spread_sample(positive, 10, "gold_label", rng)
    for row in positive_sample:
        row["sample_group"] = "positive:matched_classes"
        selected.append(row)
    for row in selected:
        raw_path = config.DATA_THUCNEWS / Path(row["source"]["doc_id"])
        try:
            original = raw_path.read_text(encoding=config.ENCODING_THUCNEWS)
        except (OSError, UnicodeError) as exc:
            raise ValueError("无法读取 THUCNews 原文: " + str(raw_path)) from exc
        row["text"] = {"original": original}
    return selected


def select_a57(model: Any, rng: random.Random) -> List[Dict[str, Any]]:
    formatted_path = config.F_PD1998_FORMATTED
    labels_path = config.F_PD1998_LLM_LABELS
    if not formatted_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError("缺少人民日报格式语料或 LLM 标签文件")
    formatted = read_jsonl(formatted_path)
    annotations = {row["article_id"]: row for row in read_jsonl(labels_path)}
    predictions = predict(model, [" ".join(row.get("tokens_jieba") or []) for row in formatted])
    groups: Dict[str, List[Dict[str, Any]]] = {
        "confirmed_disagreement": [], "confirmed_agreement": [], "needs_review": [],
    }
    for row, prediction in zip(formatted, predictions):
        annotation = annotations.get(row["article_id"], {})
        status = annotation.get("status", "missing")
        label = annotation.get("label") if status == "confirmed" else annotation.get("candidate_label")
        enriched = {
            "sample_id": "A57-" + row["article_id"],
            "track": "A5-7",
            "sample_group": "",
            "source": {"dataset": "PeopleDaily1998", "month": row.get("month"),
                       "article_id": row["article_id"], "banci": row.get("banci")},
            "text": {"title": row.get("title_text", ""), "body": row.get("body_text", "")},
            "llm": {"label": annotation.get("label"),
                    "candidate_label": annotation.get("candidate_label"),
                    "status": status, "confidence": annotation.get("confidence"),
                    "reason": annotation.get("reason"),
                    "model": annotation.get("model"),
                    "prompt_version": annotation.get("prompt_version")},
            "model": prediction,
        }
        if status == "confirmed":
            key = "confirmed_disagreement" if prediction["prediction"] != label else "confirmed_agreement"
        elif status == "needs_review":
            key = "needs_review"
        else:
            continue
        groups[key].append(enriched)

    plan = (("confirmed_disagreement", 20), ("confirmed_agreement", 10), ("needs_review", 20))
    selected = []
    for group, count in plan:
        chosen = spread_sample(groups[group], count, "month", rng)
        for row in chosen:
            row["sample_group"] = group
            selected.append(row)
    return selected


def txt_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).replace("\r", " ").replace("\n", " ")


def format_sample(sample: Dict[str, Any]) -> str:
    sid = sample["sample_id"]
    lines = [BEGIN_PREFIX + sid + " =====", "", "【抽样信息】",
             "抽样组：" + sample["sample_group"], "数据轨道：" + sample["track"]]
    if sample["track"] == "A5-4":
        lines.extend([
            "来源：THUCNews L_test / " + sample["source"]["doc_id"],
            "真值标签：" + sample["gold_label"],
            "本地模型预测：" + sample["model"]["prediction"],
            "本地模型置信度：{:.6f}".format(sample["model"]["confidence"]),
            "本地模型 Top-3：" + json.dumps(sample["model"]["top3"], ensure_ascii=False),
            "【原文】", sample["text"]["original"].rstrip(),
        ])
    else:
        lines.extend([
            "来源：人民日报 {month} / {article_id} / 版次 {banci}".format(**sample["source"]),
            "LLM 标签：" + txt_value(sample["llm"].get("label")),
            "LLM 候选标签：" + txt_value(sample["llm"].get("candidate_label")),
            "LLM 状态：" + txt_value(sample["llm"].get("status")),
            "LLM 置信度：" + txt_value(sample["llm"].get("confidence")),
            "LLM 判定理由：" + txt_value(sample["llm"].get("reason")),
            "LLM 模型/提示词：{}/{}".format(txt_value(sample["llm"].get("model")),
                                               txt_value(sample["llm"].get("prompt_version"))),
            "本地模型预测：" + sample["model"]["prediction"],
            "本地模型置信度：{:.6f}".format(sample["model"]["confidence"]),
            "本地模型 Top-3：" + json.dumps(sample["model"]["top3"], ensure_ascii=False),
            "【标题】", sample["text"]["title"], "【正文】", sample["text"]["body"].rstrip(),
        ])
    lines.extend([
        "", MANUAL_MARKER,
        "请按工作文件开头的填写指南填写；每个答案写在本行等号后，不要换行或修改字段名。",
        "人工主题标签 = 【填写】",
        "标签清晰度（清晰/有歧义/信息不足） = 【填写】",
        "原文证据（简述或摘录） = 【填写】",
        "错误模式（无错误/类别边界/上下文不足/标签或真值疑似错误/模型线索偏差/其他/不确定） = 【填写】",
        "人工意见 = 【填写】",
    ])
    if sample["track"] == "A5-7":
        lines.extend([
            "LLM 标签是否被原文支持（支持/不支持/有歧义/无法判断） = 【填写】",
            "本地模型预测是否被原文支持（支持/不支持/有歧义/无法判断） = 【填写】",
            "可能归因（无错误/词汇或时代漂移/分词错配假设/LLM标签噪声或口径/模型能力假设/证据不足/其他；可多选） = 【填写】",
            "归因把握度（1低-5高） = 【填写】",
        ])
    lines.extend(["", END_MARKER, ""])
    return "\n".join(lines)


def write_filled_example() -> None:
    EXAMPLE_PATH.write_text(
        "阶段5人工标注填写示例（完全虚构，不来自本次抽样或语料）\n"
        "此例仅演示格式；请勿将示例意见合并到 annotations.jsonl。\n\n"
        "===== SAMPLE BEGIN: DEMO-NOT-A-SAMPLE-001 =====\n\n"
        "【抽样信息】\n抽样组：虚构演示\n数据轨道：A5-7\n"
        "来源：虚构合成新闻 / DEMO-NOT-A-SAMPLE-001\n"
        "LLM 标签：体育\nLLM 候选标签：体育\nLLM 状态：confirmed（仅演示）\n"
        "LLM 置信度：0.91（仅演示）\nLLM 判定理由：报道校队足球友谊赛（虚构）\n"
        "LLM 模型/提示词：demo/demo\n本地模型预测：体育\n本地模型置信度：0.88（仅演示）\n"
        "本地模型 Top-3：体育、教育、社会（演示）\n【标题】虚构：校园足球队参加友谊赛\n"
        "【正文】以下内容为专门编写的虚构示例：某校足球队参加校际友谊赛，双方进行了足球比赛。\n\n"
        "【人工填写区】\n"
        "人工主题标签 = 体育\n"
        "标签清晰度（清晰/有歧义/信息不足） = 清晰\n"
        "原文证据（简述或摘录） = 原文明确说明足球队参加足球比赛\n"
        "错误模式（无错误/类别边界/上下文不足/标签或真值疑似错误/模型线索偏差/其他/不确定） = 无错误\n"
        "人工意见 = 标题和正文均以足球比赛为主题，人工判断与LLM标签及本地模型预测一致；本例未发现可见错误。\n"
        "LLM 标签是否被原文支持（支持/不支持/有歧义/无法判断） = 支持\n"
        "本地模型预测是否被原文支持（支持/不支持/有歧义/无法判断） = 支持\n"
        "可能归因（无错误/词汇或时代漂移/分词错配假设/LLM标签噪声或口径/模型能力假设/证据不足/其他；可多选） = 无错误\n"
        "归因把握度（1低-5高） = 4\n\n"
        "===== SAMPLE END =====\n\n"
        "示例说明：人工主题标签应按项目既定14类体系填写；若确实没有足够依据，应写‘信息不足’、‘无法判断’或‘证据不足’，不要根据模型预测反推标签。此虚构案例只说明格式，不构成真实标签或归因结论。\n",
        encoding="utf-8")


def generate(force: bool = False) -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    if TXT_PATH.is_file() and not force:
        existing = parse_txt_annotations(TXT_PATH.read_text(encoding="utf-8"))
        has_annotations = any(value for fields in existing.values() for value in fields.values())
        if has_annotations:
            raise FileExistsError("TXT 中检测到已填写内容，为避免覆盖请先备份；确认覆盖时使用 generate --force")
    rng = random.Random(SEED)
    model = fasttext_model()
    a54 = select_a54(model, rng)
    a57 = select_a57(model, rng)
    samples = a54 + a57
    SAMPLES_PATH.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in samples),
                            encoding="utf-8")
    TXT_PATH.write_text(
        "阶段5人工错例抽查表\n"
        "每条样本只需标注一次；请只填写人工填写区，不改样本 ID 或抽样信息。\n"
        "A5-4 真值为 THUCNews 原始标签；A5-7 的 LLM 标签不是人工真值。\n"
        "A5-7 人工结论只提供归因证据，单靠文本审阅不能证明分词错配或模型能力因果。\n\n"
        "【详细填写指南】\n"
        "1. 每条样本只需一位标注者填写一次；直接在‘ = 【填写】’后替换占位符，答案保持在同一行，字段名、样本ID、BEGIN/END标记不要改动。\n"
        "2. 人工主题标签：只按项目14类标签体系选一个最符合全文主旨的标签；若信息不足可填‘无法判断’，不要照抄本地模型或LLM标签。\n"
        "3. 标签清晰度：清晰=主题明确；有歧义=两个或多个主题均有原文依据；信息不足=文本残缺或证据不足。\n"
        "4. 原文证据：简述支持判断的事实/主题，可短引原文；避免只写‘模型错了’或只复述标签。\n"
        "5. 错误模式：无错误=预测/相关标注符合文本且A5-4正例无明显问题；类别边界=相邻主题难区分；上下文不足=标题/片段/正文不足；标签或真值疑似错误=记录可能问题，不自行改数据；模型线索偏差=模型被局部词汇误导；其他/不确定按实际填写。\n"
        "6. A5-7两项‘是否被原文支持’分别评价LLM标签和本地预测，选支持/不支持/有歧义/无法判断；不要把confirmed状态当成人工真值。\n"
        "7. A5-7可能归因可多选：无错误仅用于看不出错误证据；词汇或时代漂移需指出文本中的时代/词汇证据；分词错配是假设，人工只可标‘可能’，确认需配对实验；LLM标签噪声或口径需有原文与标签的具体矛盾/标准差异；模型能力是假设，不能仅从单篇文本证明；证据不足表示无法区分原因。\n"
        "8. 归因把握度：1=极低、2=偏低、3=中等、4=较高、5=很高；只评价本条文本支持程度，不代表总体因果置信度。A5-4样本无需填写A5-7专属字段。\n"
        "9. 不适用的非必填字段可填‘不适用’；必填项不确定时明确填‘不确定/无法判断’，不要留空。自动抽取会把未替换的【填写】判为未完成。\n\n"
        + "\n".join(format_sample(row) for row in samples), encoding="utf-8")
    write_filled_example()
    counts = Counter(row["sample_group"] for row in samples)
    print("人工工作文件夹:", WORK_DIR)
    print("TXT填写文件:", TXT_PATH)
    print("机器样本清单:", SAMPLES_PATH)
    print("实际抽样数:", len(samples), dict(counts))
    expected = {"error:财经->股票": 10, "error:教育->社会": 10,
                "error:股票->财经": 10, "positive:matched_classes": 10,
                "confirmed_disagreement": 20, "confirmed_agreement": 10,
                "needs_review": 20}
    for key, target in expected.items():
        actual = counts.get(key, 0)
        if actual != target:
            print("注意：{} 实际 {} 篇，目标 {} 篇（可用样本不足时按实际输出）".format(key, actual, target))


def parse_txt_annotations(text: str) -> Dict[str, Dict[str, str]]:
    annotations: Dict[str, Dict[str, str]] = {}
    current_id = None
    in_manual = False
    for line in text.splitlines():
        if line.startswith(BEGIN_PREFIX) and line.endswith(" ====="):
            current_id = line[len(BEGIN_PREFIX):-len(" =====")]
            annotations[current_id] = {}
            in_manual = False
            continue
        if line == MANUAL_MARKER:
            in_manual = True
            continue
        if line == END_MARKER:
            current_id = None
            in_manual = False
            continue
        if current_id is None or not in_manual or " = " not in line:
            continue
        key, value = line.split(" = ", 1)
        key_map = {
            "人工主题标签": "human_label",
            "标签清晰度（清晰/有歧义/信息不足）": "label_clarity",
            "原文证据（简述或摘录）": "evidence",
            "错误模式（无错误/类别边界/上下文不足/标签或真值疑似错误/模型线索偏差/其他/不确定）": "error_pattern",
            "人工意见": "comment",
            "LLM 标签是否被原文支持（支持/不支持/有歧义/无法判断）": "llm_label_supported",
            "本地模型预测是否被原文支持（支持/不支持/有歧义/无法判断）": "model_prediction_supported",
            "可能归因（无错误/词汇或时代漂移/分词错配假设/LLM标签噪声或口径/模型能力假设/证据不足/其他；可多选）": "attribution_causes",
            "归因把握度（1低-5高）": "attribution_confidence",
        }
        field = key_map.get(key)
        if field:
            cleaned = value.strip()
            annotations[current_id][field] = "" if cleaned in ("【填写】", "待填写", "（请填写）") else cleaned
    return annotations


def extract() -> None:
    if not SAMPLES_PATH.is_file() or not TXT_PATH.is_file():
        raise FileNotFoundError("请先运行 generate 生成工作包")
    samples = read_jsonl(SAMPLES_PATH)
    txt_annotations = parse_txt_annotations(TXT_PATH.read_text(encoding="utf-8"))
    expected_ids = {row["sample_id"] for row in samples}
    extra_ids = sorted(set(txt_annotations) - expected_ids)
    output = []
    missing_counts = Counter()
    for sample in samples:
        fields = txt_annotations.get(sample["sample_id"], {})
        required = ["human_label", "label_clarity", "evidence", "error_pattern", "comment"]
        if sample["track"] == "A5-7":
            required.extend(["llm_label_supported", "model_prediction_supported",
                             "attribution_causes", "attribution_confidence"])
        missing = [field for field in required if not fields.get(field)]
        missing_counts.update(missing)
        output.append({
            "sample_id": sample["sample_id"], "track": sample["track"],
            "sample_group": sample["sample_group"], "source": sample["source"],
            "gold_label": sample.get("gold_label"), "llm": sample.get("llm"),
            "model": sample["model"], "human_annotation": fields,
            "completed": not missing, "missing_fields": missing,
        })
    ANNOTATIONS_PATH.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output),
                                encoding="utf-8")
    summary = {
        "samples": len(output),
        "completed": sum(row["completed"] for row in output),
        "incomplete": sum(not row["completed"] for row in output),
        "completed_by_track": {
            track: sum(row["completed"] and row["track"] == track for row in output)
            for track in ("A5-4", "A5-7")
        },
        "missing_field_counts": dict(missing_counts),
        "extra_sample_ids_in_txt": extra_ids,
        "human_reviews_per_sample": 1,
        "note": "单人单次人工判断，不含双人一致性或仲裁；A5-7 人工归因是证据记录，不是因果证明。",
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("抽取完成:", ANNOTATIONS_PATH)
    print("汇总:", SUMMARY_PATH)
    print("完整填写: {}/{}；未完成: {}".format(summary["completed"], summary["samples"], summary["incomplete"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段5人工抽查TXT工作包生成与JSONL抽取")
    parser.add_argument("action", choices=("generate", "extract"))
    parser.add_argument("--force", action="store_true", help="允许覆盖已填写的 TXT（请先备份）")
    args = parser.parse_args()
    if args.action == "generate":
        generate(force=args.force)
    else:
        extract()


if __name__ == "__main__":
    main()
