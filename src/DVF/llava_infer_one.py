import torch
from PIL import Image
import json
import os
import warnings

from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token

warnings.filterwarnings("ignore")

# ===================== 你只需要改这里 =====================
model_path = "./models/llava_quantized"  # 你的模型文件夹路径
image_path = "./val2014/COCO_val2014_000000240434.jpg"                # 你的图片路径
question = "Is there a clock in the image?"           # 你的问题
save_file = "./llava_qa_log.json"        # 保存问答的文件
# =========================================================

device = "cuda:0"
torch.cuda.empty_cache()

# ===================== 加载模型 =====================
print("正在加载 LLaVA 模型...")

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path,
    None,
    model_name="llava-v1.5-7b"
)

model = model.to(device)
model.eval()
print("模型加载完成！")

# ===================== 加载图片 =====================
try:
    image = Image.open(image_path).convert("RGB")
except Exception as e:
    print(f"图片打开失败：{e}")
    exit()

# ===================== 构造提问 =====================
prompt = DEFAULT_IMAGE_TOKEN + "\n" + question

conv = conv_templates["llava_v1"].copy()
conv.append_message(conv.roles[0], prompt)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

# ===================== 模型推理 =====================
with torch.no_grad():
    # 处理图片
    image_tensor = process_images(
        [image],
        image_processor,
        model.config
    ).to(device, dtype=torch.float16)

    # 处理输入
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        return_tensors="pt"
    ).unsqueeze(0).to(device)

    # 生成回答
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        do_sample=False,
        temperature=0,
        max_new_tokens=512,  # 回答长度，可改大
        use_cache=True
    )

    # 解码回答
    answer = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    ).strip()

# ===================== 打印结果 =====================
print("\n" + "="*50)
print(f"图片：{image_path}")
print(f"问题：{question}")
print(f"回答：{answer}")
print("="*50 + "\n")

# ===================== 保存到文件 =====================
# 如果文件不存在，创建；存在则追加一行
qa_record = {
    "image": image_path,
    "question": question,
    "answer": answer
}

# 写入（每次运行追加一条）
with open(save_file, "a", encoding="utf-8") as f:
    f.write(json.dumps(qa_record, ensure_ascii=False) + "\n")

print(f"✅ 问答已保存到：{save_file}")