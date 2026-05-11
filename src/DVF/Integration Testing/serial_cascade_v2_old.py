import torch
from PIL import Image
import json
import os
import warnings
import argparse
import re
from collections import defaultdict
from tqdm import tqdm

from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token

warnings.filterwarnings("ignore")


# ===================== 1. 逻辑组件：清洗与融合 =====================

def parse_objects(objects_str):
    if not objects_str:
        return []

    s = objects_str.lower().strip()

    # 1️⃣ 去掉换行（但不只取第一行，避免丢信息）
    s = s.replace("\n", " ")

    # 2️⃣ 去掉结构标记（兼容 prompt 3）
    for noise in ["big:", "small:", "[", "]", ";"]:
        s = s.replace(noise, " ")

    # 3️⃣ 去掉编号（1. 2.）
    s = re.sub(r'\d+\.', ' ', s)

    # ⚠️ 关键：不要删除字母之间的结构
    # 只去掉明显干扰符号（保留空格和逗号）
    s = re.sub(r'[^a-z,\s]', ' ', s)

    # 4️⃣ 分割（逗号 + and）
    parts = re.split(r',|\band\b', s)

    objects = []
    for obj in parts:
        obj = obj.strip()

        # 5️⃣ 基础过滤
        if len(obj) < 3:
            continue

        # 6️⃣ 过滤句子碎片（关键）
        # 如果包含多个词且像句子，就丢弃
        words = obj.split()
        if len(words) > 3:
            continue

        # 7️⃣ 过滤无意义词
        if obj in [
            "the", "there", "image", "objects", "object",
            "something", "anything", "everything"
        ]:
            continue

        objects.append(obj)

    # 8️⃣ 去重
    return list(set(objects))


def fuse_object_lists(object_lists, strategy='vote'):
    """
    融合多个列表
    vote: 出现 >= 2 次才保留（最稳健）
    """
    count_dict = defaultdict(int)
    all_objects = set()
    for obj_list in object_lists:
        for obj in set(obj_list):
            count_dict[obj] += 1
            all_objects.add(obj)

    if strategy == 'vote':
        return [obj for obj in all_objects if count_dict[obj] >= 2]
    return list(all_objects)


# ===================== 2. 基础推理函数 =====================

def get_response(model, tokenizer, image_tensor, prompt_text, device, max_tokens=128):
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


# ===================== 3. 主程序 =====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", type=str, default="adversarial", choices=["random", "popular", "adversarial"])
    args = parser.parse_args()

    # 路径配置（请根据你的服务器实际情况检查）
    model_path = "./models/llava_quantized"
    image_folder = "./val2014/val2014"
    pope_file = f"./output/coco/coco_pope_{args.set}.json"
    output_ans_file = f"./answer/llava_serial_v2_{args.set}.jsonl"
    device = "cuda:0"

    # 加载模型
    print("Loading model...")
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, "llava-v1.5-7b")
    model = model.to(device).eval()

    # 读取 POPE 原始数据
    with open(pope_file, "r") as f:
        pope_data = [json.loads(line) for line in f if line.strip()]

    print(f"Starting Stage 1 Consensus + Stage 2 Decoupling on {args.set}...")

    # Stage 1 的三个不同视角 Prompt
    PROMPTS_S1 = [
        "Please list all the objects you can clearly see in this image. Format: object1, object2, object3...",
        "Look at the image very carefully. List every single object you can see, do not miss anything.Format: object1, object2, object3...",
        "First list the large main objects, then list the small background objects. big: [obj1, obj2]; small: [obj3, obj4]."
    ]

    os.makedirs(os.path.dirname(output_ans_file), exist_ok=True)

    with open(output_ans_file, "w") as ans_f:
        for item in tqdm(pope_data):
            image_path = os.path.join(image_folder, item["image"])
            try:
                image = Image.open(image_path).convert("RGB")
            except:
                continue

            image_tensor = process_images([image], image_processor, model.config).to(device, dtype=torch.float16)

            # --- [阶段 1a] 多视感知 ---
            raw_lists = []
            for p_s1 in PROMPTS_S1:
                res = get_response(model, tokenizer, image_tensor, p_s1, device, max_tokens=100)
                raw_lists.append(parse_objects(res))

            # --- [阶段 1b] 逻辑融合 (投票制) ---
            consensus_objects = fuse_object_lists(raw_lists, strategy='vote')
            consensus_str = ", ".join(consensus_objects)

            # --- [阶段 2] 半解耦判定 ---
            # 提取目标物体名
            question = item["text"]
            target_obj = question.replace("Is there a ", "").replace("Is there an ", "").replace(" in the image?",
                                                                                                 "").replace("?",
                                                                                                             "").strip()

            p2 = (
                f"Objects I clearly detected in the image: {consensus_str}\n"
                f"Question: {question}\n"
                f"Based on the detected objects and the image, answer yes or no:"
            )

            final_res_raw = get_response(model, tokenizer, image_tensor, p2, device, max_tokens=10).lower()
            final_ans = 'yes' if 'yes' in final_res_raw else 'no'

            # 保存完整记录
            res_item = {
                "question": question,
                "label": item["label"],
                "final_answer": final_ans,
                "image": item["image"],
                "consensus_objects": consensus_objects,
                "raw_perception_lists": raw_lists
            }
            ans_f.write(json.dumps(res_item) + "\n")


if __name__ == "__main__":
    main()