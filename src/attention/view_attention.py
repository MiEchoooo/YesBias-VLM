import torch
import os

# ======================
# 你只需要改这里
# ======================
attn_path = "attention_output/adversarial_1/attentions.pt"
token_path = "attention_output/adversarial_1/tokens.txt"

# ======================
# 加载注意力
# ======================
print("Loading attention...")
attentions = torch.load(attn_path)  # list of 32 layers
print(f"Loaded layers: {len(attentions)}")
print(f"Layer shape: {attentions[0].shape}")

# ======================
# 加载 tokens
# ======================
tokens = []
with open(token_path, "r", encoding="utf-8") as f:
    for line in f:
        idx, tok = line.strip().split("\t", 1)
        tokens.append(tok)

seq_len = len(tokens)
print(f"Seq len: {seq_len}")

# ======================
# 取最后 4 层平均
# ======================
print("\nComputing mean of last 4 layers...")
attn_stack = torch.stack([a.float() for a in attentions[-4:]])  # [4,1,32,627,627]
attn_mean = attn_stack.mean(dim=(0,1))  # [627,627]

# ======================
# 取答案位置（最后一个 token）
# ======================
answer_pos = -1
attn_answer = attn_mean[answer_pos]  # [627]

# ======================
# 找到 <image_token>
# ======================
image_pos = [i for i, t in enumerate(tokens) if "<image_token>" in t][0]
print(f"\nImage token at: {image_pos}")

# ======================
# 找到问题范围（你可以自己看 tokens.txt 调整）
# ======================
q_start = image_pos + 2
q_end = len(tokens) - 4
print(f"Question range: {q_start} ~ {q_end}")

# ======================
# 计算 VAR / QAR
# ======================
vis = attn_answer[image_pos].item()
q = attn_answer[q_start:q_end].sum().item()
total = vis + q

VAR = vis / total
QAR = q / total

# ======================
# 输出结果
# ======================
print("\n" + "="*50)
print("FINAL RESULT (你的核心指标)")
print("="*50)
print(f"Visual Attention (VAR): {VAR:.4f}")
print(f"Question Attention (QAR): {QAR:.4f}")

if QAR > 0.6:
    print("\n❌ POPE 范式：注意力偏向文本 → YES BIAS")
else:
    print("\n✅ 注意力偏向图像 → 低偏差")