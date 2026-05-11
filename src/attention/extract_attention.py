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

# =====================================================
# Args
# =====================================================

parser = argparse.ArgumentParser()

parser.add_argument("--sample_idx", type=int, default=0)
parser.add_argument("--set", type=str, default="adversarial")

args = parser.parse_args()

sample_idx = args.sample_idx
pope_set = args.set

# =====================================================
# Paths
# =====================================================

device = "cuda:0"

model_path = "./models/llava_quantized"

image_folder = "./val2014/val2014"

pope_file = f"./output/coco/coco_pope1_{pope_set}.json"

save_dir = "./attention_results"

os.makedirs(save_dir, exist_ok=True)

# =====================================================
# Load dataset
# =====================================================

with open(pope_file, "r", encoding="utf-8") as f:
    data = [json.loads(l.strip()) for l in f if l.strip()]

item = data[sample_idx]

image_name = item["image"]
question = item["text"]
label = item["label"]
qid = item["question_id"]

print("\n===== SAMPLE =====")
print("qid:", qid)
print("image:", image_name)
print("question:", question)
print("label:", label)

# =====================================================
# Load model
# =====================================================

print("\nLoading model...")

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path,
    None,
    model_name="llava-v1.5-7b"
)

model = model.to(device)
model.eval()

print("Model loaded.")

# =====================================================
# Image
# =====================================================

image_path = os.path.join(image_folder, image_name)

image = Image.open(image_path).convert("RGB")

image_tensor = process_images(
    [image],
    image_processor,
    model.config
).to(device, dtype=torch.float16)

# =====================================================
# Prompt
# =====================================================

prompt = DEFAULT_IMAGE_TOKEN + "\n" + question

conv = conv_templates["llava_v1"].copy()

conv.append_message(conv.roles[0], prompt)
conv.append_message(conv.roles[1], None)

prompt = conv.get_prompt()

print("\n===== PROMPT =====")
print(prompt)

# =====================================================
# Tokenize
# =====================================================

input_ids = tokenizer_image_token(
    prompt,
    tokenizer,
    return_tensors="pt"
).unsqueeze(0).to(device)

# =====================================================
# Safe tokenization
# =====================================================

tokens = []

for tid in input_ids[0]:

    tid = tid.item()

    if tid < 0:
        tokens.append("<image_token>")
    else:
        try:
            tokens.append(tokenizer.convert_ids_to_tokens(tid))
        except:
            tokens.append("<unk>")

print("\n===== TOKENS =====")

for i, t in enumerate(tokens):
    print(i, t)

# =====================================================
# Forward ONLY (NO generate)
# =====================================================

print("\nRunning forward pass (next-token analysis)...")

with torch.no_grad():

    outputs = model(
        input_ids=input_ids,
        images=image_tensor,
        output_attentions=True,
        return_dict=True
    )

# =====================================================
# Logits (next token prediction)
# =====================================================

logits = outputs.logits

next_token_id = torch.argmax(logits[:, -1, :], dim=-1)

pred_token = tokenizer.decode(next_token_id)

print("\n===== NEXT TOKEN =====")
print(pred_token)

# =====================================================
# Attentions
# =====================================================

attentions = outputs.attentions

# last 4 layers only
last_layers = attentions[-4:]

layer_attn_list = []

for layer in last_layers:

    layer = layer[0]               # remove batch
    layer = layer.mean(dim=0)     # average heads

    layer_attn_list.append(layer)

# average last 4 layers
final_attn = torch.stack(layer_attn_list).mean(dim=0)

"""
shape:
(seq_len, seq_len)
"""

# =====================================================
# Focus: LAST TOKEN (prediction state)
# =====================================================

query_attn = final_attn[-1]

"""
shape:
(seq_len,)
"""

# =====================================================
# Locate image token span
# =====================================================

image_pos = None

for i, t in enumerate(tokens):
    if t == "<image_token>":
        image_pos = i
        break

if image_pos is None:
    raise ValueError("No image token found")

# LLaVA visual tokens (~576)
image_start = image_pos
image_end = image_pos + 576

# question tokens
question_start = image_end + 1
question_end = len(query_attn)

# =====================================================
# Compute attention
# =====================================================

image_attn = query_attn[image_start:image_end].sum().item()

question_attn = query_attn[question_start:question_end].sum().item()

other_attn = 1.0 - (image_attn + question_attn)

# density normalization
image_density = image_attn / (image_end - image_start)
question_density = question_attn / (question_end - question_start)

# =====================================================
# Output
# =====================================================

print("\n===== ATTENTION RESULT =====")

print(f"Image attention: {image_attn:.6f}")
print(f"Question attention: {question_attn:.6f}")
print(f"Other attention: {other_attn:.6f}")

print()

print(f"Image density: {image_density:.8f}")
print(f"Question density: {question_density:.8f}")

# =====================================================
# Save result only (NO tensor saving)
# =====================================================

result = {
    "qid": qid,
    "image": image_name,
    "question": question,
    "label": label,
    "pred_token": pred_token,
    "image_attn": image_attn,
    "question_attn": question_attn,
    "other_attn": other_attn,
    "image_density": image_density,
    "question_density": question_density
}

out_path = os.path.join(save_dir, f"{pope_set}_{qid}.json")

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print("\nSaved:", out_path)