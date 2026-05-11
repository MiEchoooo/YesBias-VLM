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

# =========================
# 参数
# =========================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--sample_idx",
    type=int,
    default=0,
    help="index in pope file"
)

parser.add_argument(
    "--set",
    type=str,
    default="adversarial",
    choices=["random", "popular", "adversarial"]
)

args = parser.parse_args()

sample_idx = args.sample_idx
pope_set = args.set

# =========================
# 路径
# =========================

device = "cuda:0"

model_path = "./models/llava_quantized"
image_folder = "./val2014/val2014"
pope_file = f"./output/coco/coco_repope_adversarial.json"
save_root = "./attention_output"

os.makedirs(save_root, exist_ok=True)

# =========================
# 读取POPE数据
# =========================

with open(pope_file, "r", encoding="utf-8") as f:
    pope_data = [
        json.loads(line.strip())
        for line in f
        if line.strip()
    ]

print("Dataset:", pope_set)
print("Total samples:", len(pope_data))

if sample_idx >= len(pope_data):
    raise ValueError("sample_idx out of range")

item = pope_data[sample_idx]

image_name = item["image"]
question = item["text"]
label = item["label"]
question_id = item["question_id"]

print("\n===== SAMPLE =====")
print("question_id:", question_id)
print("image:", image_name)
print("question:", question)
print("label:", label)

# =========================
# 保存目录
# =========================

sample_save_dir = os.path.join(
    save_root,
    f"{pope_set}_{question_id}"
)

os.makedirs(sample_save_dir, exist_ok=True)

# =========================
# 加载模型
# =========================

print("\nLoading model...")

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path,
    None,
    model_name="llava-v1.5-7b"
)

model = model.to(device)
model.eval()

print("Model loaded.")

# =========================
# 读取图像
# =========================

image_path = os.path.join(image_folder, image_name)

if not os.path.exists(image_path):
    raise FileNotFoundError(image_path)

image = Image.open(image_path).convert("RGB")

image_tensor = process_images(
    [image],
    image_processor,
    model.config
).to(device, dtype=torch.float16)

# =========================
# 构造prompt
# =========================

prompt = DEFAULT_IMAGE_TOKEN + "\n" + question

conv = conv_templates["llava_v1"].copy()

conv.append_message(conv.roles[0], prompt)
conv.append_message(conv.roles[1], None)

prompt = conv.get_prompt()

print("\n===== PROMPT =====")
print(prompt)

# =========================
# tokenizer
# =========================

input_ids = tokenizer_image_token(
    prompt,
    tokenizer,
    return_tensors="pt"
).unsqueeze(0).to(device)

tokens = []
for token_id in input_ids[0]:
    token_id = token_id.item()
    # LLaVA image token
    if token_id < 0:
        tokens.append("<image_token>")
    else:
        try:
            tok = tokenizer.convert_ids_to_tokens(token_id)
            tokens.append(tok)
        except:
            tokens.append("<unk>")

# =========================
# 打印tokens
# =========================

print("\n===== TOKENS =====")

for idx, tok in enumerate(tokens):
    print(idx, tok)

# 保存 token

with open(
    os.path.join(sample_save_dir, "tokens.txt"),
    "w",
    encoding="utf-8"
) as f:

    for idx, tok in enumerate(tokens):
        f.write(f"{idx}\t{tok}\n")

# =========================
# Forward attention
# =========================

print("\nRunning forward pass...")

with torch.no_grad():

    outputs = model(
        input_ids=input_ids,
        images=image_tensor,
        output_attentions=True,
        return_dict=True
    )

# =========================
# attentions
# =========================

attentions = outputs.attentions

print("\n===== ATTENTION INFO =====")

print("Number of layers:", len(attentions))

print("Attention shape:", attentions[0].shape)

"""
shape:
(batch, heads, seq_len, seq_len)
"""

# =========================
# 保存attention
# =========================

attention_cpu = []

for layer_attn in attentions:
    attention_cpu.append(layer_attn.cpu())

torch.save(
    attention_cpu,
    os.path.join(sample_save_dir, "attentions.pt")
)

print("\nAttention saved.")

# =========================
# generate answer
# =========================

print("\nGenerating answer...")

with torch.no_grad():

    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        do_sample=False,
        temperature=0,
        max_new_tokens=10,
        use_cache=True
    )

generated_text = tokenizer.decode(
    output_ids[0],
    skip_special_tokens=True
).strip()

print("\n===== GENERATED =====")
print(generated_text)

# =========================
# metadata
# =========================

metadata = {
    "question_id": question_id,
    "image": image_name,
    "question": question,
    "label": label,
    "generated_text": generated_text,
    "num_layers": len(attentions),
    "attention_shape": list(attentions[0].shape)
}

with open(
    os.path.join(sample_save_dir, "metadata.json"),
    "w",
    encoding="utf-8"
) as f:

    json.dump(metadata, f, indent=2)

print("\nSaved to:", sample_save_dir)