import json
import re
from collections import defaultdict

# ======================
# 配置路径
# ======================

BASELINE_FILE = "baseline.jsonl"
SEMI_FILE = "semi.jsonl"
V2_FILE = "v2.jsonl"
GT_FILE = "pope.jsonl"

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

assert len(baseline) == len(semi) == len(v2) == len(gt)

# ======================
# 统计变量
# ======================

stats = {
    "semi": defaultdict(int),
    "v2": defaultdict(int)
}

total = len(gt)

# ======================
# 主循环
# ======================

for i in range(total):

    q = gt[i]["text"]
    label = gt[i]["label"]

    base_ans = parse_yes_no(baseline[i]["answer"])
    semi_ans = parse_yes_no(semi[i]["answer"])
    v2_ans = parse_yes_no(v2[i]["answer"])

    target = extract_target(q)

    semi_objs = parse_objects(semi[i].get("objects", ""))
    v2_objs = parse_objects(v2[i].get("objects_consensus", ""))

    # ======================
    # Semi 分析
    # ======================

    semi_hit = target in semi_objs

    if semi_hit:
        stats["semi"]["hit"] += 1

        if semi_ans == "yes":
            stats["semi"]["trust"] += 1
        elif semi_ans == "no":
            stats["semi"]["ignore"] += 1

    else:
        stats["semi"]["miss"] += 1

        # list 没有，但模型答 yes → hallucination
        if semi_ans == "yes":
            stats["semi"]["false_positive_without_list"] += 1

    # list 驱动纠错（baseline错 → semi对）
    if base_ans != label and semi_ans == label:
        if semi_hit:
            stats["semi"]["correct_with_list"] += 1
        else:
            stats["semi"]["correct_without_list"] += 1

    # ======================
    # V2 分析
    # ======================

    v2_hit = target in v2_objs

    if v2_hit:
        stats["v2"]["hit"] += 1

        if v2_ans == "yes":
            stats["v2"]["trust"] += 1
        elif v2_ans == "no":
            stats["v2"]["ignore"] += 1

    else:
        stats["v2"]["miss"] += 1

        if v2_ans == "yes":
            stats["v2"]["false_positive_without_list"] += 1

    if base_ans != label and v2_ans == label:
        if v2_hit:
            stats["v2"]["correct_with_list"] += 1
        else:
            stats["v2"]["correct_without_list"] += 1


# ======================
# 输出函数
# ======================

def print_stats(name, s):

    print(f"\n===== {name.upper()} =====")

    hit = s["hit"]
    miss = s["miss"]

    trust = s["trust"]
    ignore = s["ignore"]

    correct_with = s["correct_with_list"]
    correct_without = s["correct_without_list"]

    fp_wo_list = s["false_positive_without_list"]

    hit_rate = hit / total if total else 0
    trust_rate = trust / hit if hit else 0
    ignore_rate = ignore / hit if hit else 0

    print(f"Total samples: {total}")

    print("\n--- List Coverage ---")
    print(f"Hit: {hit} ({hit_rate:.2%})")
    print(f"Miss: {miss} ({miss/total:.2%})")

    print("\n--- List Trust ---")
    print(f"Trust (hit & yes): {trust}")
    print(f"Ignore (hit & no): {ignore}")
    print(f"Trust Rate: {trust_rate:.2%}")
    print(f"Ignore Rate: {ignore_rate:.2%}")

    print("\n--- Correction ---")
    print(f"Correct WITH list: {correct_with}")
    print(f"Correct WITHOUT list: {correct_without}")

    if correct_with + correct_without > 0:
        print(f"List Contribution Ratio: {correct_with / (correct_with + correct_without):.2%}")

    print("\n--- Risk ---")
    print(f"False Positive WITHOUT list: {fp_wo_list}")


# ======================
# 打印结果
# ======================

print_stats("semi", stats["semi"])
print_stats("v2", stats["v2"])