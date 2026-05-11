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


# ===================== 1. 质疑验证核心组件 =====================

def generate_challenging_questions(target_object):
    """针对Yes回答生成的质疑问题集"""
    return {
        'Q5': f"Is it true that there is NO {target_object} in the image?",
        'Q6': f"Are you sure there is a {target_object} in the image? Please check again very carefully.",
        'Q7': f"Let's think step by step:\n1. Describe what you see in the image.\n2. Does it contain {target_object}?\n3. Final answer (yes/no):"
    }


def parse_challenge_answer(qid, text):
    """解析质疑环节的回答"""
    text = text.lower().strip()
    if qid == 'Q5':
        # Q5问的是"没有...对吗？"，如果模型答yes/correct，代表确认"没有"
        if 'yes' in text[:10] or 'correct' in text[:10] or 'true' in text[:10]:
            return 'no'  # 物体不存在
        return 'yes'  # 物体存在
    else:
        # Q6, Q7 是常规问法
        if 'yes' in text or 'there is' in text:
            return 'yes'
        return 'no'


def consistency_check(ans0, challenge_results):
    """
    一致性检验逻辑：
    - 如果 Q5 产生了强矛盾，直接判 No。
    - 统计 Q5/Q6/Q7 中支持 'yes' 的比例。
    """
    parsed = {}
    for qid, text in challenge_results.items():
        parsed[qid] = parse_challenge_answer(qid, text)

    # 强矛盾检测：原判定有，但Q5确认说没有
    if ans0 == 'yes' and parsed['Q5'] == 'no':
        return 'no', "Strong Contradiction (Q5 Confirmed No)"

    # 投票统计
    yes_count = sum(1 for v in parsed.values() if v == 'yes')
    # 只要支持率低于 2/3 (即只有0或1个yes)，就判定为幻觉
    if yes_count >= 2:
        return 'yes', f"Consistency Passed ({yes_count}/3)"
    else:
        return 'no', f"Consistency Failed ({yes_count}/3)"

def extract_target(question):
    # 移除句尾的 "?" 和 " in the image"
    q = question.replace("?", "").replace(" in the image", "")
    # 移除开头的 "Is there a " 或 "Is there an "
    q = q.replace("Is there a ", "").replace("Is there an ", "")
    return q.strip()

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
    parser.add_argument("--set", type=str, default="adversarial")
    args = parser.parse_args()

    # 路径配置
    model_path = "./models/llava_quantized"
    image_folder = "./val2014/val2014"
    pope_file = f"./output/coco/coco_pope_{args.set}.json"
    output_ans_file = f"./answer/llava_self_consistency_v1_{args.set}.jsonl"
    device = "cuda:0"

    # 加载模型
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, "llava-v1.5-7b")
    model = model.to(device).eval()

    semi_decoupled_file = "./answer/llava_answer_adversarial_decoupled_semi.json"

    with open(semi_decoupled_file, "r") as f:
        # 假设之前的格式是每行一个 JSON
        previous_results = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(previous_results)} previous semi-decoupled results.")

    with open(output_ans_file, "w") as ans_f:
        for item in tqdm(previous_results):
            # 获取基本信息
            image_name = item["image"]
            ans0 = item["answer"].lower()  # 之前脚本保存的键名是 'answer'
            question = item["question"]
            target_obj = extract_target(item["question"])

            final_ans = 'no' if 'no' in ans0 else 'yes'
            reason = "Initial No - Accepted"
            challenge_details = {}

            # 【串行级联核心】只审问说 Yes 的样本
            if final_ans == 'yes':
                image_path = os.path.join(image_folder, image_name)
                image = Image.open(image_path).convert("RGB")
                image_tensor = process_images([image], image_processor, model.config).to(device, dtype=torch.float16)

                # 发起挑战
                challenges = generate_challenging_questions(target_obj)
                challenge_res = {}
                for qid, q_text in challenges.items():
                    res = get_response(model, tokenizer, image_tensor, q_text, device, max_tokens=64)
                    challenge_res[qid] = res

                # 重新判定
                final_ans, reason = consistency_check('yes', challenge_res)
                challenge_details = challenge_res

            # 保存最终结果（保留原有信息，增加验证信息）
            item.update({
                "final_answer": final_ans,
                "reason": reason,
                "challenges": challenge_details
            })
            ans_f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()