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
synonym_file = "./synonyms.txt"

pope_file = f"./output/coco/coco_repope_{pope_set}.json"
output_ans_file = f"./answer/llava_answer_{pope_set}_semantic_align_re.json"

device = "cuda:0"
torch.cuda.empty_cache()

# ===================== Stage Prompts =====================

STAGE1_PROMPTS = [
    "List all visible objects in this image. Output only objects separated by commas.",
    "Carefully list every visible object in the image. Only output objects separated by commas.",
    "List all objects (both large and small) in a single comma-separated list."
]

STAGE2_TEMPLATE = """{image_token}
Detected objects (normalized): {objects}
Question: {question}
Refer to the object list and the image, then answer yes or no:"""

# ===================== 语义对齐 =====================

def load_synonyms(path):
    synonym_map = {}
    coco_classes = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            words = [w.strip().lower() for w in line.split(",")]
            canonical = words[0]

            coco_classes.add(canonical)

            for w in words:
                synonym_map[w] = canonical

    return synonym_map, coco_classes


def normalize_phrase(text):
    text = text.lower()
    text = re.sub(r"^(a|an|the)\s+", "", text)
    text = re.sub(r"\b(small|large|big|little)\b\s*", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = text.strip()

    return text


def extract_head_noun(phrase):
    words = phrase.split()
    if len(words) == 0:
        return ""
    return words[-1]


def semantic_align(obj_str, synonym_map, coco_classes):
    objs = [x.strip() for x in obj_str.split(",") if x.strip()]
    aligned = set()

    for obj in objs:
        obj = normalize_phrase(obj)
        head = extract_head_noun(obj)

        if not head:
            continue

        if head.endswith("s"):
            head = head[:-1]

        if head in synonym_map:
            canonical = synonym_map[head]

            if canonical in coco_classes:
                aligned.add(canonical)

    return ", ".join(sorted(aligned))


synonym_map, coco_classes = load_synonyms(synonym_file)

print("Loaded synonyms:", len(synonym_map))
print("Loaded COCO classes:", len(coco_classes))

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
        formatted_prompt,
        tokenizer,
        return_tensors="pt"
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

    return tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    ).strip().lower()


def stage1_multi_lists(image_tensor):
    lists = []

    for prompt in STAGE1_PROMPTS:
        try:
            result = generate_object_list(image_tensor, prompt)
            lists.append(result)
        except Exception as e:
            print(f"[Stage1 Prompt Failed] {e}")
            lists.append("")

    return lists


def parse_objects(obj_str):
    objs = re.split(r",|\n|\d+\.", obj_str)

    clean_objs = []

    for o in objs:
        o = normalize_phrase(o)

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
        objects=objects if objects else "none",
        question=question
    )

    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)

    formatted_prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        formatted_prompt,
        tokenizer,
        return_tensors="pt"
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

    return tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    ).strip().lower()

# ===================== 主循环 =====================

print("\nStarting semantic-aligned inference...")

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
                print(f"[Image Error] {image_path}: {e}")
                continue

            image_tensor = process_images(
                [image],
                image_processor,
                model.config
            ).to(device, dtype=torch.float16)

            question = item["text"]

            # ---------- Stage1 ----------
            try:
                lists = stage1_multi_lists(image_tensor)
                objects_raw = fuse_lists(lists)
                objects_aligned = semantic_align(
                    objects_raw,
                    synonym_map,
                    coco_classes
                )
            except Exception as e:
                print(f"[Stage1 Failed] {image_name}: {e}")
                lists = []
                objects_raw = ""
                objects_aligned = ""

            # ---------- Stage2 ----------
            try:
                final_answer = stage2_verify(
                    objects_aligned,
                    question,
                    image_tensor
                )
            except Exception as e:
                print(f"[Stage2 Failed] {image_name}: {e}")
                final_answer = "unknown"

            ans_item = {
                "question": question,
                "answer": final_answer,
                "image": image_name,
                "lists": lists,
                "objects_raw": objects_raw,
                "objects_aligned": objects_aligned
            }

            ans_f.write(json.dumps(ans_item) + "\n")

            if (idx + 1) % 100 == 0:
                print(f"[{idx + 1}/{len(pope_data)}] {final_answer}")

print("\nInference finished.")
print("Answer file:", output_ans_file)