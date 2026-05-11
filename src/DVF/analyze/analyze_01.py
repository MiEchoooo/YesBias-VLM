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

OUTPUT_CSV = "analysis_output_8class.csv"

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
# 8类分类函数
# ======================

def classify_8(gt, base, v2):
    if gt == "yes":
        if base == "yes" and v2 == "yes":
            return "Y1_keep_correct"
        elif base == "yes" and v2 == "no":
            return "Y2_over_correction"
        elif base == "no" and v2 == "yes":
            return "Y3_corrected"
        elif base == "no" and v2 == "no":
            return "Y4_keep_wrong"

    elif gt == "no":
        if base == "no" and v2 == "no":
            return "N1_keep_correct"
        elif base == "no" and v2 == "yes":
            return "N2_new_error"
        elif base == "yes" and v2 == "no":
            return "N3_corrected"
        elif base == "yes" and v2 == "yes":
            return "N4_keep_wrong"

    return "UNKNOWN"


# ======================
# 加载数据
# ======================

baseline = load_jsonl(BASELINE_FILE)
semi = load_jsonl(SEMI_FILE)
v2 = load_jsonl(V2_FILE)
gt = load_jsonl(GT_FILE)

assert len(baseline) == len(semi) == len(v2) == len(gt)

# ======================
# 统计变量
# ======================

case_counter = defaultdict(int)

summary_stats = {
    "corrected": 0,        # Y3 + N3
    "over_correction": 0,  # Y2
    "new_error": 0         # N2
}

list_hit_stats = defaultdict(int)

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

    obj_set = v2_objs if v2_objs else semi_objs

    list_hit = target in obj_set

    # ===== 8类分类 =====
    case = classify_8(label, base_ans, v2_ans)
    case_counter[case] += 1

    # ===== 汇总统计 =====
    if case in ["Y3_corrected", "N3_corrected"]:
        summary_stats["corrected"] += 1
    elif case == "Y2_over_correction":
        summary_stats["over_correction"] += 1
    elif case == "N2_new_error":
        summary_stats["new_error"] += 1

    # ===== list统计 =====
    list_hit_stats[f"list_hit_{list_hit}"] += 1

    # ===== CSV =====
    rows_for_csv.append({
        "question": q,
        "target": target,
        "label": label,
        "baseline": base_ans,
        "semi": semi_ans,
        "v2": v2_ans,
        "list_hit": list_hit,
        "case_8": case
    })


# ======================
# 输出结果
# ======================

total = len(gt)

print("\n===== 8类分布 =====")
for k in sorted(case_counter.keys()):
    v = case_counter[k]
    print(f"{k}: {v} ({v/total:.2%})")


print("\n===== 汇总统计 =====")
for k, v in summary_stats.items():
    print(f"{k}: {v} ({v/total:.2%})")


print("\n===== List 命中统计 =====")
for k, v in list_hit_stats.items():
    print(f"{k}: {v} ({v/total:.2%})")


# ======================
# 保存 CSV
# ======================

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows_for_csv[0].keys())
    writer.writeheader()
    writer.writerows(rows_for_csv)

print("\nCSV saved to:", OUTPUT_CSV)