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

# ===================== 配置路径 =====================
# ===================== 配置路径 =====================
# 1. 使用 os.path.abspath 将路径转为绝对路径，避免被误判为 Repo ID
current_dir = os.path.dirname(os.path.abspath(__file__)) # DVF 目录
parent_dir = os.path.dirname(current_dir) # POPE-main 目录

# 确保指向你真正的模型存放位置
model_path = os.path.join(parent_dir, "models/llava_quantized")
# 确保指向你真正的图片存放位置
image_folder = os.path.join(parent_dir, "val2014/val2014")

print(f"Checking Model Path: {model_path}")
print(f"Checking Image Folder: {image_folder}")

# 检查路径是否存在，避免再次报错
if not os.path.exists(model_path):
    raise FileNotFoundError(f"找不到模型目录: {model_path}")

device = "cuda:0"

# 请根据你之前 find_hallucination.py 找到的样本填入此处
# 格式: {"img": "图片名", "obj": "被模型误认的物体", "original_ans": "原本错误的回答"}
test_cases = [
    {"img": "COCO_val2014_000000210789.jpg", "obj": "car", "original_ans": "yes"},
    {"img": "COCO_val2014_000000265719.jpg", "obj": "dining table", "original_ans": "yes"},
    {"img": "COCO_val2014_000000429109.jpg", "obj": "motorcycle", "original_ans": "yes"},
]

# ===================== 加载模型 =====================
print("Loading LLaVA model for DVF validation...")
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, None, model_name="llava-v1.5-7b"
)
model = model.to(device)
model.eval()


# ===================== 定义推理函数 =====================
def get_llava_response(image_tensor, prompt_text):
    """通用的 LLaVA 推理函数"""
    conv = conv_templates["llava_v1"].copy()
    # 注意：LLaVA 1.5 习惯在第一轮对话带上 IMAGE TOKEN
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt_text)
    conv.append_message(conv.roles[1], None)

    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            temperature=0,
            max_new_tokens=128,  # 第一步列表可能较长，给多一点 token
            use_cache=True
        )

    outputs = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return outputs


# ===================== 开始验证 Idea A =====================
print("\n" + "=" * 60)
print("DVF (Decoupled Visual Verification) Experiment Start")
print("=" * 60)

for case in test_cases:
    image_path = os.path.join(image_folder, case["img"])
    if not os.path.exists(image_path):
        print(f"Skipping: {case['img']} not found.")
        continue

    # 处理图片
    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config).to(device, dtype=torch.float16)

    print(f"\n[Target]: {case['obj']} | [Image]: {case['img']}")
    print(f"Original Answer (Vanilla): {case['original_ans']}")

    # --- Step 1: Perception Phase ---
    p1 = "Please list all the objects you can clearly see in this image. Be concise and accurate."
    perception_list = get_llava_response(image_tensor, p1)
    print(f"--- Step 1 (Object List): {perception_list}")

    # --- Step 2: Verification Phase ---
    # 我们把 Step 1 的结果喂给模型，强迫它基于自己看到的列表做判断
    # 修改后的强约束 Prompt
    # p2 = f"You just identified these objects: {perception_list}. Based strictly on this list and the image, is there a {case['obj']}? "
    # --- DVF Step 2: 结构化验证 ---
    # 强制要求模型只看列表，不许废话，不许复读
    p2 = (
        f"Current Object List: {perception_list}\n"
        f"Question: Based strictly on this list and the image, is there a {case['obj']}? \n"
        f"Rules:\n"
        f"1. Answer 'No' if the '{perception_list}' is not in the list.\n"
        f"2. Answer 'Yes' only if you are absolutely sure it is in the list.\n"
    )
    final_verification = get_llava_response(image_tensor, p2)
    print(f"--- Step 2 (DVF Decision): {final_verification}")

    # 简单的逻辑判断是否修正成功
    is_corrected = "no" in final_verification.lower()[:5]
    print(f"Result: {'✅ Corrected!' if is_corrected else '❌ Still Hallucinating'}")
    print("-" * 60)

print("\nValidation Finished.")