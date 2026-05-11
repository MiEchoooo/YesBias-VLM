import torch
from PIL import Image
import json
import os
import warnings
import argparse
import re
from collections import Counter

from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token

warnings.filterwarnings("ignore")

# ===================== 参数 =====================

parser = argparse.ArgumentParser()
parser.add_argument("--set", type=str, default="random",
                    choices=["random", "popular", "adversarial"])

args = parser.parse_args()
pope_set = args.set

# ===================== 基础配置 =====================

model_path = "./models/llava_quantized"
image_folder = "./val2014/val2014"

pope_file = f"./output/coco/coco_pope_{pope_set}.json"
output_ans_file = f"./answer/llava_answer_{pope_set}_semi_consistency.json"

device = "cuda:0"
torch.cuda.empty_cache()

# ===================== Stage Prompts =====================

STAGE1_PROMPTS = [
    "List all visible objects in this image. Output only objects separated by commas.",
    "Carefully list every visible object in the image. Only output objects separated by commas.",
    "List all objects (both large and small) in a single comma-separated list."
]

STAGE2_TEMPLATE = """{image_token}
Detected objects (from multiple observations): {objects}
Question: {question}
Refer to the object list and the image, then answer yes or no:"""

# ===================== 加载模型 =====================

print("Loading LLaVA model...")

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path,
    None,
    model_name="llava-v1.5-7b"
)

model = model.to(device)
model.eval()

print("Model loaded.")

os.makedirs(os.path.dirname(output_ans_file), exist_ok=True)

with open(pope_file, "r", encoding="utf-8") as f:
    pope_data = [json.loads(line.strip()) for line in f if line.strip()]

print("Dataset:", pope_set)
print("Total samples:", len(pope_data))

# ===================== Stage1 =====================

def generate_object_list(image_tensor, prompt_text):
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text

    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    formatted_prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        formatted_prompt, tokenizer, return_tensors="pt"
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            temperature=0,
            max_new_tokens=30,
            use_cache=True
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip().lower()


def stage1_multi_lists(image_tensor):
    lists = []
    for prompt in STAGE1_PROMPTS:
        try:
            obj_str = generate_object_list(image_tensor, prompt)
            lists.append(obj_str)
        except Exception as e:
            print(f"[Stage1 Prompt Failed] {e}")
            lists.append("")
    return lists


def parse_objects(obj_str):
    objs = re.split(r",|\n|\d+\.", obj_str)

    clean_objs = []
    for o in objs:
        o = o.strip().lower()
        o = re.sub(r"^(a|an|the)\s+", "", o)
        o = re.sub(r"\b(small|large|big|little)\b\s*", "", o)

        if len(o) > 0:
            clean_objs.append(o)

    return set(clean_objs)


def fuse_lists(lists):
    counter = Counter()

    for l in lists:
        objs = parse_objects(l)
        for obj in objs:
            counter[obj] += 1

    consensus = [obj for obj, cnt in counter.items() if cnt >= 2]

    if len(consensus) == 0:
        consensus = list(counter.keys())

    return ", ".join(consensus)

# ===================== Stage2 =====================

def stage2_verify(objects, question, image_tensor):
    conv = conv_templates["llava_v1"].copy()

    prompt = STAGE2_TEMPLATE.format(
        image_token=DEFAULT_IMAGE_TOKEN,
        objects=objects,
        question=question
    )

    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    formatted_prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        formatted_prompt, tokenizer, return_tensors="pt"
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            temperature=0,
            max_new_tokens=10,
            use_cache=True
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip().lower()

# ===================== 主循环 =====================

print("\nStarting semi-consistency inference...")

with open(output_ans_file, "w", encoding="utf-8") as ans_f:
    with torch.no_grad():

        for idx, item in enumerate(pope_data):

            image_name = item["image"]
            image_path = os.path.join(image_folder, image_name)

            if not os.path.exists(image_path):
                print(f"[Missing] {image_path}")
                continue

            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"[Error] {image_path}: {e}")
                continue

            image_tensor = process_images(
                [image], image_processor, model.config
            ).to(device, dtype=torch.float16)

            question = item["text"]

            # Stage 1
            lists = []
            try:
                lists = stage1_multi_lists(image_tensor)
                objects = fuse_lists(lists)
            except Exception as e:
                print(f"[Stage1 Failed] {image_name}: {e}")
                objects = "unknown"

            # Stage 2
            try:
                final_answer = stage2_verify(objects, question, image_tensor)
            except Exception as e:
                print(f"[Stage2 Failed] {image_name}: {e}")
                final_answer = "unknown"

            ans_item = {
                "question": question,
                "answer": final_answer,
                "image": image_name,
                "lists": lists,
                "objects_consensus": objects
            }

            ans_f.write(json.dumps(ans_item) + "\n")

            if (idx + 1) % 100 == 0:
                print(f"[{idx + 1}/{len(pope_data)}] {final_answer}")

print("\nInference finished.")
print("Answer file:", output_ans_file)