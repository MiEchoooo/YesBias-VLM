import json
import re
from collections import defaultdict
import csv

# ======================
# 配置路径
# ======================

BASELINE_FILE = "baseline.jsonl"
SEMI_FILE = "semi.jsonl"
V2_FILE = "v2.jsonl"
GT_FILE = "pope.jsonl"

OUTPUT_CSV = "analysis_output.csv"  # 可选

# ======================
# 工具函数
# ======================

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def parse_yes_no(text):
    text = text.lower()
    if "yes" in text:
        return "yes"
    elif "no" in text:
        return "no"
    return "unknown"


def extract_target(question):
    """
    从 "Is there a <obj> in the image?" 提取 obj
    """
    question = question.lower()
    match = re.search(r"is there (a|an)?\s*(.*?)\s*(in|on|at)", question)
    if match:
        return match.group(2).strip()
    return ""


def parse_objects(obj_str):
    if not obj_str:
        return set()
    objs = re.split(r",|\n", obj_str.lower())
    return set(o.strip() for o in objs if o.strip())


# ======================
# 加载数据
# ======================

baseline = load_jsonl(BASELINE_FILE)
semi = load_jsonl(SEMI_FILE)
v2 = load_jsonl(V2_FILE)
gt = load_jsonl(GT_FILE)

# 对齐（假设顺序一致）
assert len(baseline) == len(semi) == len(v2) == len(gt)

# ======================
# 统计变量
# ======================

case_counter = defaultdict(int)

list_hit_stats = defaultdict(int)
ignore_list_count = 0
total_list_hit = 0

correction_stats = {
    "baseline_wrong_semi_correct": 0,
    "baseline_correct_semi_wrong": 0,
    "baseline_wrong_v2_correct": 0,
    "baseline_correct_v2_wrong": 0,
}

rows_for_csv = []

# ======================
# 主循环
# ======================

for i in range(len(gt)):
    q = gt[i]["text"]
    label = gt[i]["label"]

    base_ans = parse_yes_no(baseline[i]["answer"])
    semi_ans = parse_yes_no(semi[i]["answer"])
    v2_ans = parse_yes_no(v2[i]["answer"])

    target = extract_target(q)

    semi_objs = parse_objects(semi[i].get("objects", ""))
    v2_objs = parse_objects(v2[i].get("objects_consensus", ""))

    # 用 v2 list 作为分析主对象（你也可以换成 semi）
    obj_set = v2_objs if v2_objs else semi_objs

    list_hit = target in obj_set

    # ======================
    # Case 分类（以 semi 为例）
    # ======================

    if label == "no" and base_ans == "yes" and semi_ans == "no":
        case = "A_corrected"
    elif label == "yes" and base_ans == "yes" and semi_ans == "no":
        case = "B_over_correction"
    elif label == "yes" and list_hit and semi_ans == "no":
        case = "C_ignore_list"
    else:
        case = "D_consistent"

    case_counter[case] += 1

    # ======================
    # list 命中统计
    # ======================

    if list_hit:
        total_list_hit += 1
        if semi_ans == "no":
            ignore_list_count += 1

    list_hit_stats[f"list_hit_{list_hit}"] += 1

    # ======================
    # 纠错统计
    # ======================

    if base_ans != label and semi_ans == label:
        correction_stats["baseline_wrong_semi_correct"] += 1

    if base_ans == label and semi_ans != label:
        correction_stats["baseline_correct_semi_wrong"] += 1

    if base_ans != label and v2_ans == label:
        correction_stats["baseline_wrong_v2_correct"] += 1

    if base_ans == label and v2_ans != label:
        correction_stats["baseline_correct_v2_wrong"] += 1

    # ======================
    # CSV记录（可用于画图）
    # ======================

    rows_for_csv.append({
        "question": q,
        "target": target,
        "label": label,
        "baseline": base_ans,
        "semi": semi_ans,
        "v2": v2_ans,
        "list_hit": list_hit,
        "case": case
    })


# ======================
# 输出结果
# ======================

total = len(gt)

print("\n===== Case 分布 =====")
for k, v in case_counter.items():
    print(f"{k}: {v} ({v/total:.2%})")

print("\n===== List 命中统计 =====")
for k, v in list_hit_stats.items():
    print(f"{k}: {v} ({v/total:.2%})")

if total_list_hit > 0:
    print("\n===== 忽略 List 现象 =====")
    print(f"list_hit=True 但 answer=no: {ignore_list_count}")
    print(f"比例: {ignore_list_count / total_list_hit:.2%}")

print("\n===== 纠错能力 =====")
for k, v in correction_stats.items():
    print(f"{k}: {v}")

# ======================
# 保存 CSV
# ======================

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows_for_csv[0].keys())
    writer.writeheader()
    writer.writerows(rows_for_csv)

print("\nCSV saved to:", OUTPUT_CSV)