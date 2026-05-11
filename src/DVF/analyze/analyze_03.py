import json
import os
import random
import base64
from collections import defaultdict

# ======================
# 路径配置
# ======================

BASELINE_FILE = "baseline.jsonl"
SEMI_FILE = "semi.jsonl"
V2_FILE = "v2.jsonl"
GT_FILE = "pope.jsonl"

IMAGE_FOLDER = r"D:\Pytorch_Projects\POPE\POPE-main\val2014"
OUTPUT_HTML = "analysis_view.html"

SAMPLE_PER_CASE = 30  # 每类抽样数量

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


# 图片转 base64
def image_to_base64(img_path):
    try:
        with open(img_path, "rb") as f:
            return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
    except:
        return ""


# ======================
# 8类划分
# ======================

def classify(label, base, v2):
    if label == "no":
        if base == "no" and v2 == "no":
            return "N1_keep_correct"
        elif base == "no" and v2 == "yes":
            return "N2_new_error"
        elif base == "yes" and v2 == "no":
            return "N3_corrected"
        else:
            return "N4_keep_wrong"
    else:
        if base == "yes" and v2 == "yes":
            return "Y1_keep_correct"
        elif base == "yes" and v2 == "no":
            return "Y2_over_correction"
        elif base == "no" and v2 == "yes":
            return "Y3_corrected"
        else:
            return "Y4_keep_wrong"


# ======================
# 加载数据
# ======================

baseline = load_jsonl(BASELINE_FILE)
semi = load_jsonl(SEMI_FILE)
v2 = load_jsonl(V2_FILE)
gt = load_jsonl(GT_FILE)

assert len(baseline) == len(semi) == len(v2) == len(gt)

# ======================
# 分组
# ======================

groups = defaultdict(list)

for i in range(len(gt)):
    q = gt[i]["text"]
    label = gt[i]["label"]
    question_id = gt[i]["question_id"]  # 读取问题ID
    image_name = gt[i]["image"]         # 读取图片文件名

    base_ans = parse_yes_no(baseline[i]["answer"])
    semi_ans = parse_yes_no(semi[i]["answer"])
    v2_ans = parse_yes_no(v2[i]["answer"])

    case = classify(label, base_ans, v2_ans)

    img_path = os.path.join(IMAGE_FOLDER, image_name)
    img_b64 = image_to_base64(img_path)

    groups[case].append({
        "question_id": question_id,    # 存入ID
        "image_name": image_name,      # 存入图片名称
        "image": img_b64,
        "question": q,
        "label": label,
        "baseline": base_ans,
        "semi": semi_ans,
        "v2": v2_ans,
        "semi_obj": semi[i].get("objects", ""),
        "v2_obj": v2[i].get("objects_consensus", "")
    })

# ======================
# HTML生成
# ======================

def html_escape(text):
    return text.replace("<", "&lt;").replace(">", "&gt;")


html = """
<html>
<head>
<meta charset="utf-8">
<title>Analysis View</title>
<style>
body { font-family: Arial; }
.case { margin-bottom: 50px; }
.card {
    display: inline-block;
    width: 300px;
    margin: 10px;
    border: 1px solid #ccc;
    padding: 10px;
    vertical-align: top;
}
img { width: 100%; height: auto; }
</style>
</head>
<body>
<h1>Qualitative Analysis</h1>
"""

# ======================
# 渲染每类（新增显示 ID 和 图片名）
# ======================

for case, items in groups.items():
    html += f"<div class='case'><h2>{case} ({len(items)})</h2>"

    sample = random.sample(items, min(SAMPLE_PER_CASE, len(items)))

    for item in sample:
        html += f"""
        <div class='card'>
            <img src="{item["image"]}" />
            <p><b>ID:</b> {item["question_id"]}</p>
            <p><b>Image:</b> {html_escape(item["image_name"])}</p>
            <p><b>Q:</b> {html_escape(item["question"])}</p>
            <p><b>GT:</b> {item["label"]}</p>
            <p><b>Baseline:</b> {item["baseline"]}</p>
            <p><b>Semi:</b> {item["semi"]}</p>
            <p><b>V2:</b> {item["v2"]}</p>
            <p><b>Semi Obj:</b> {html_escape(item["semi_obj"])}</p>
            <p><b>V2 Obj:</b> {html_escape(item["v2_obj"])}</p>
        </div>
        """

    html += "</div>"

html += "</body></html>"

# ======================
# 保存
# ======================

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"✅ HTML 已生成: {OUTPUT_HTML}")
print(f"✅ 已显示 question_id 和 图片名称！")