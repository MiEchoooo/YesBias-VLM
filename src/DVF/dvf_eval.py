import torch
from PIL import Image
import json
import os
import warnings
import argparse
from tqdm import tqdm

from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token

warnings.filterwarnings("ignore")

# ===================== 参数配置 =====================
parser = argparse.ArgumentParser()
parser.add_argument("--set", type=str, default="adversarial", choices=["random", "popular", "adversarial"])
args = parser.parse_args()

# 路径配置（自动转为绝对路径）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
model_path = os.path.join(parent_dir, "models/llava_quantized")
image_folder = os.path.join(parent_dir, "val2014/val2014")
pope_file = os.path.join(parent_dir, f"output/coco/coco_pope_{args.set}.json")
output_ans_file = f"./answer/llava_dvf_{args.set}.jsonl"

device = "cuda:0"
os.makedirs(os.path.dirname(output_ans_file), exist_ok=True)

# ===================== 加载模型 =====================
print(f"正在加载模型并准备评测数据集: {args.set}...")
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, None, model_name="llava-v1.5-7b"
)
model = model.to(device)
model.eval()


# ===================== 工具函数 =====================
def get_response(image_tensor, prompt_text, max_tokens=128):
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt_text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids, images=image_tensor, do_sample=False,
            temperature=0, max_new_tokens=max_tokens, use_cache=True
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def calculate_metrics(results):
    tp, tn, fp, fn = 0, 0, 0, 0
    for res in results:
        label = res['label'].lower()
        pred = res['dvf_answer'].lower()

        pred_yes = "yes" in pred[:10]  # 简易判断开头的yes/no
        label_yes = label == "yes"

        if label_yes and pred_yes:
            tp += 1
        elif not label_yes and not pred_yes:
            tn += 1
        elif not label_yes and pred_yes:
            fp += 1
        elif label_yes and not pred_yes:
            fn += 1

    acc = (tp + tn) / len(results) if len(results) > 0 else 0
    pre = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0
    return acc, pre, rec, f1


# ===================== 开始全量推理 =====================
with open(pope_file, "r", encoding="utf-8") as f:
    pope_data = [json.loads(line.strip()) for line in f if line.strip()]

all_results = []
print(f"开始 DVF 推理，总计 {len(pope_data)} 条数据...")

with open(output_ans_file, "w", encoding="utf-8") as ans_f:
    for item in tqdm(pope_data):
        image_name = item["image"]
        target_obj = item["text"].split()[-1].replace("?", "")  # 提取问题中的物体名
        image_path = os.path.join(image_folder, image_name)

        # 加载与处理图片
        image = Image.open(image_path).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config).to(device, dtype=torch.float16)

        # --- DVF Step 1: Perception ---
        p1 = "Please list all the objects you can clearly see in this image. Be concise and accurate."
        obj_list = get_response(image_tensor, p1, max_tokens=128)

        # --- DVF Step 2: Decoupled Verification ---
        # p2 = (
        #     f"I have a list of objects found in the image: {obj_list}.\n"
        #     f"Your task is to check if '{target_obj}' is in this list.\n"
        #     f"1. If '{target_obj}' is in the list, answer 'Yes'.\n"
        #     f"2. If '{target_obj}' is NOT in the list, look at the image again VERY carefully. "
        #     f"If you still cannot find it, you MUST answer 'No'.\n"
        #     f"Important: Do not imagine things. If it's not there, say 'No'.\n"
        #     f"Answer (Yes/No):"
        # )

        p2 = (
            f"Current Object List: {obj_list}\n"
            f"Question: Based strictly on this list and the image, is there a {target_obj}? \n"
            f"Rules:\n"
            f"1. Answer 'No' if the '{obj_list}' is not in the list.\n"
            f"2. Answer 'Yes' only if you are absolutely sure it is in the list.\n"
        )
        # p2 = f"You just identified these objects: {obj_list}. Based strictly on this list and the image, is there a {target_obj}? "

        # p2 = (
        #     f"Task: Cross-check the existence of '{target_obj}'.\n"
        #     f"Evidence provided by you: {obj_list}\n"
        #     f"Rule 1: If '{target_obj}' is NOT explicitly mentioned in your evidence list, you must answer NO.\n"
        #     f"Rule 2: Do not use common sense to guess. Only rely on the list and the image.\n"
        #     f"Question: Is there a {target_obj}? Answer with 'No' or 'Yes'."
        # )
        dvf_res = get_response(image_tensor, p2, max_tokens=20)

        # 保存结果
        res_item = {
            "question": item["text"],
            "label": item["label"],
            "perception_list": obj_list,
            "dvf_answer": dvf_res
        }
        ans_f.write(json.dumps(res_item) + "\n")
        all_results.append(res_item)

# ===================== 输出最终指标 =====================
acc, pre, rec, f1 = calculate_metrics(all_results)
print("\n" + "=" * 30)
print(f"DVF Evaluation Results ({args.set}):")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {pre:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("=" * 30)