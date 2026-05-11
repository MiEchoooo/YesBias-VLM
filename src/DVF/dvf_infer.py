import torch
from PIL import Image
import json
import os
import warnings
import argparse

from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token

warnings.filterwarnings("ignore")

# ===================== 参数 =====================

parser = argparse.ArgumentParser()
parser.add_argument("--set", type=str, default="random",
                    choices=["random", "popular", "adversarial"])
parser.add_argument("--mode", type=str, default="full",
                    choices=["full", "semi"],  # full=完全解耦, semi=半解耦
                    help="decoupling mode: full (no image in stage2) or semi (with image)")

args = parser.parse_args()

pope_set = args.set

# ===================== 基础配置 =====================

model_path = "./models/llava_quantized"
image_folder = "./val2014/val2014"

pope_file = f"./output/coco/coco_pope_{pope_set}.json"
output_ans_file = f"./answer/llava_answer_{pope_set}_decoupled_{args.mode}.json"

device = "cuda:0"

torch.cuda.empty_cache()

# ===================== Stage Prompts =====================

# Stage 1: 纯视觉感知
STAGE1_PROMPT = "List all visible objects in this image. Format: object1, object2, object3."

# Stage 2: 结合Object List进行判断（关键：用图像token强制模型关注视觉）
STAGE2_TEMPLATE_FULL = """Objects I detected: {objects}
Question: {question}
Based only on these detected objects, answer yes or no:"""

STAGE2_TEMPLATE_SEMI = """{image_token}
Objects I detected in the image: {objects}
Question: {question}
First verify the detected objects against the image, then answer yes or no:"""

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
print("Mode:", args.mode)

os.makedirs(os.path.dirname(output_ans_file), exist_ok=True)

with open(pope_file, "r", encoding="utf-8") as f:
    pope_data = [json.loads(line.strip()) for line in f if line.strip()]

print("Dataset:", pope_set)
print("Total samples:", len(pope_data))


# ===================== Stage 1: 提取Object List =====================

def stage1_extract_objects(image_tensor):
    """纯视觉编码，生成Object List"""
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + STAGE1_PROMPT

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

    objects = tokenizer.decode(
        output_ids[0], skip_special_tokens=True
    ).strip().lower()

    return objects


# ===================== Stage 2: 验证（关键修正）=====================

def stage2_verify(objects, question, image_tensor, mode):
    """
    Stage 2: 根据模式选择输入

    full:  仅文本（Object List）→ 完全解耦
    semi:  图像 + Object List  → 半解耦（用列表引导视觉注意力）
    """
    conv = conv_templates["llava_v1"].copy()

    if mode == "full":
        # 完全解耦：纯文本，无图像token
        prompt = STAGE2_TEMPLATE_FULL.format(objects=objects, question=question)
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        formatted_prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            formatted_prompt, tokenizer, return_tensors="pt"
        ).unsqueeze(0).to(device)

        stage2_images = None  # 关键：不提供图像

    else:  # semi
        # 半解耦：图像token + Object List
        prompt = STAGE2_TEMPLATE_SEMI.format(
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

        stage2_images = image_tensor  # 关键：再次提供图像

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=stage2_images,  # 根据模式决定是否输入图像
            do_sample=False,
            temperature=0,
            max_new_tokens=10,
            use_cache=True
        )

    answer = tokenizer.decode(
        output_ids[0], skip_special_tokens=True
    ).strip().lower()

    return answer


# ===================== 主循环 =====================

print(f"\nStarting {args.mode} decoupled inference...")

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
            try:
                objects = stage1_extract_objects(image_tensor)
            except Exception as e:
                print(f"[Stage1 Failed] {image_name}: {e}")
                objects = "unknown"

            # Stage 2（传入mode参数）
            try:
                final_answer = stage2_verify(objects, question, image_tensor, args.mode)
            except Exception as e:
                print(f"[Stage2 Failed] {image_name}: {e}")
                final_answer = "unknown"

            ans_item = {
                "question": question,
                "answer": final_answer,
                "image": image_name,
                "objects": objects,
                "mode": args.mode
            }

            ans_f.write(json.dumps(ans_item) + "\n")

            if (idx + 1) % 100 == 0:
                print(f"[{idx + 1}/{len(pope_data)}] {final_answer}")

print("\nInference finished.")
print("Answer file:", output_ans_file)