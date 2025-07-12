import os


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com/'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'
import pandas as pd
from evaluate import load
from tqdm import tqdm

# 加载 BLEU 指标
# bleu = load("bleu")

# def rank_by_bleu_scores(references, predictions):
#     """
#     根据 BLEU 分数对预测句子排序，并返回排序后的句子、对应 BLEU 分数和原始索引。
#
#     Args:
#         references (list[str]): 原始参考句子。
#         predictions (list[str]): 预测生成的句子。
#
#     Returns:
#         list[tuple[str, float, int]]: (预测句子, bleu分数, 原始索引)，按分数降序排列
#     """
#     results = []
#     for idx, (ref, pred) in tqdm(enumerate(zip(references, predictions))):
#         score = bleu.compute(predictions=[pred], references=[[ref]])['bleu']
#         results.append((pred, score, idx))
#
#     # 按 BLEU 分数排序（从高到低）
#     results.sort(key=lambda x: x[1], reverse=True)
#     return results
#
# # 示例用法
# if __name__ == "__main__":
#     df = pd.read_csv("/data/hugang/GlhCode/VulAdvisor/data/data/test.csv")
#     references = list(df["suggestion"])
#
#     file = "/data/hugang/GlhCode/VulAdvisor/prefix-tuning/predictions/qwen-max/final_old/checkpoint-26336.out"
#     with open(file) as f:
#         predictions = [line.strip() for line in f.readlines()]
#
#
#     original_file = "/data/hugang/GlhCode/VulAdvisor/prefix-tuning/predictions/qwen-max/final_old/checkpoint-26336.out"
#     with open(file) as f:
#         predictions = [line.strip() for line in f.readlines()]
#
#     ranked = rank_by_bleu_scores(references, predictions)
#     for pred, score, idx in ranked:
#         print(f"Original Index: {idx}, BLEU Score: {score:.4f}, Prediction: '{pred}'")


# 读取参考建议
path_ref = "/data/hugang/GlhCode/VulAdvisor/data/data/test.csv"
df = pd.read_csv(path_ref)
references = list(df["suggestion"])

# 读取两个模型的预测建议
path_model1 = "/data/hugang/GlhCode/VulAdvisor/prefix-tuning/predictions/qwen-max/final_old/checkpoint-26336.out"
path_model2 = "/data/hugang/GlhCode/VulAdvisor/core/vuladvisor/checkpoint-27780.out"

with open(path_model1) as f1:
    predictions_1 = [line.strip() for line in f1.readlines()]

with open(path_model2) as f2:
    predictions_2 = [line.strip() for line in f2.readlines()]

# 检查长度一致性
assert len(references) == len(predictions_1) == len(predictions_2), "长度不一致"

# def compute_bleu(reference, hypothesis):
#     score =  bleu.compute(predictions=[hypothesis], references=[[reference]])
#     return score["bleu"]

# results = []
# for idx, (ref, pred1, pred2) in tqdm(enumerate(zip(references, predictions_1, predictions_2)), total=len(references)):
#     bleu1 = compute_bleu(ref, pred1)
#     bleu2 = compute_bleu(ref, pred2)
#     delta_bleu = bleu1 - bleu2
#     mean_bleu = (bleu1 + bleu2) / 2
#     if bleu1 > 0.25 and bleu2 > 0.25:  # 可调整的BLEU阈值
#         results.append((idx + 2, bleu1, bleu2, delta_bleu))
#
# # 按BLEU差值排序，取前100项
# results_sorted = sorted(results, key=lambda x: x[3], reverse=True)[:100]
#
# # 保存或打印结果
# output_path = "top100_bleu_diff.csv"
# pd.DataFrame(results_sorted, columns=["index", "bleu_model1", "bleu_model2", "delta_bleu"]).to_csv(output_path, index=False)
# print(f"保存结果到 {output_path}")

print("This i reference====================================")
print(references[386])
print("This i out prediction=====================================")
print(predictions_1[386])

print("This i vul prediction======================================")
print(predictions_2[386])

