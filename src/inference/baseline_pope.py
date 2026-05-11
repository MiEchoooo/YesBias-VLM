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

args = parser.parse_args()

pope_set = args.set

# ===================== 基础配置 =====================

model_path = "./models/llava_quantized"
image_folder = "./val2014/val2014"

pope_file = f"./output/coco/coco_pope1_{pope_set}.json"
output_ans_file = f"./answer/llava_answer1_{pope_set}.json"

device = "cuda:0"

torch.cuda.empty_cache()

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

# 创建输出目录

os.makedirs(os.path.dirname(output_ans_file), exist_ok=True)

# 读取 POPE 数据

with open(pope_file, "r", encoding="utf-8") as f:
    pope_data = [json.loads(line.strip()) for line in f if line.strip()]

print("Dataset:", pope_set)
print("Total samples:", len(pope_data))

# 推理

with open(output_ans_file, "w", encoding="utf-8") as ans_f:
    with torch.no_grad():

        for idx, item in enumerate(pope_data):

            image_name = item["image"]
            image_path = os.path.join(image_folder, image_name)

            if not os.path.exists(image_path):
                print("Image missing:", image_path)
                continue

            try:
                image = Image.open(image_path).convert("RGB")
            except:
                print("Image error:", image_path)
                continue

            image_tensor = process_images(
                [image],
                image_processor,
                model.config
            ).to(device, dtype=torch.float16)

            question = item["text"]

            prompt = DEFAULT_IMAGE_TOKEN + "\n" + question

            conv = conv_templates["llava_v1"].copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)

            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt,
                tokenizer,
                return_tensors="pt"
            ).unsqueeze(0).to(device)

            try:

                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=10,
                    use_cache=True
                )

            except:

                ans_item = {"question": question, "answer": "unknown"}
                ans_f.write(json.dumps(ans_item) + "\n")
                continue

            outputs = tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True
            ).strip().lower()

            final_answer = outputs

            ans_item = {
                "question": question,
                "answer": final_answer
            }

            ans_f.write(json.dumps(ans_item) + "\n")

            if (idx + 1) % 100 == 0:
                print(
                    f"[{idx + 1}/{len(pope_data)}] "
                    f"{final_answer}"
                )

print("\nInference finished.")
print("Answer file:", output_ans_file)