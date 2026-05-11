import json

# 配置你的文件名
model_answer_file = "/root/POPE-main/answer/llava_answer_adversarial.json"
ground_truth_file = "/root/POPE-main/output/coco/coco_pope_adversarial.json"


def find_cases():
    hallucinations = []

    # 使用 with 同时打开两个文件，按行并行读取
    with open(model_answer_file, 'r') as f_ans, open(ground_truth_file, 'r') as f_gt:
        for idx, (line_ans, line_gt) in enumerate(zip(f_ans, f_gt)):
            ans_data = json.loads(line_ans)
            gt_data = json.loads(line_gt)

            # 1. 提取预测结果（转小写，判断是否以yes开头）
            raw_ans = ans_data['answer'].lower().strip()
            pred_yes = raw_ans.startswith("yes")

            # 2. 提取标准答案
            label_yes = gt_data['label'].lower() == "yes"

            # 3. 寻找幻觉：标签是 no，但模型预测是 yes
            if not label_yes and pred_yes:
                hallucinations.append({
                    "line_number": idx + 1,
                    "image": gt_data.get('image', 'Unknown'),
                    "question": gt_data['text'],
                    "model_raw_answer": ans_data['answer']
                })

    # 打印结果
    print(f"找到幻觉样本总数: {len(hallucinations)}")
    print("=" * 50)
    for case in hallucinations[:10]:  # 只展示前10个典型的
        print(f"行号: {case['line_number']} | 图片: {case['image']}")
        print(f"问题: {case['question']}")
        print(f"模型原始回答: {case['model_raw_answer']}")
        print("-" * 30)


if __name__ == "__main__":
    find_cases()