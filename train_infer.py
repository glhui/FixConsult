import os
import pandas as pd
import torch
# from peft import PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer
from evaluate import load as load_metric
from VulCodeT5Model import CodeT5ForConditionalGeneration
from utils import build_input_text, parse_args

def load_model_and_tokenizer(model_path):
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
    # additional_tokens = list(SPECIAL_TOKENS.values())
    # if tokenizer.add_tokens(additional_tokens) > 0:
    #     print("Added additional tokens:", additional_tokens)
    model = CodeT5ForConditionalGeneration.from_pretrained_with_encoder(model_path)
    # model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def generate_predictions(texts, batch_size=64):
    # print(f"Batch size: {batch_size}")
    predictions = []

    # 按批次处理文本
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        # 使用 tokenizer 批量处理输入文本
        model_inputs = tokenizer(batch_texts, return_tensors="pt", padding="longest", truncation=True)
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

        # 调用 generate 方法（可根据需要调整参数）
        batch_outputs = model.generate(**model_inputs, max_new_tokens=100, num_beams=5)
        batch_preds = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
        predictions.extend(batch_preds)

    return predictions


def compute_metrics(predictions, references):

    # 计算 BLEU
    bleu_score = bleu_metric.compute(predictions=predictions, references=[[ref] for ref in references])["score"]
    # 计算 ROUGE，这里传入字符串列表
    # rouge_score = rouge_metric.compute(predictions=predictions, references=references)

    metrics = {
        "BLEU": bleu_score,
        # "ROUGE-1": rouge_score["rouge1"],
        # "ROUGE-2": rouge_score["rouge2"],
        # "ROUGE-L": rouge_score["rougeL"]
    }
    return metrics


if __name__ == "__main__":
    args = parse_args()
    # SPECIAL_TOKENS = {
    #     "added": "<ADDED>",
    #     "removed": "<REMOVED>"
    # }
    bleu_metric = load_metric("sacrebleu")
    # rouge_metric = load_metric("rouge")
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # test_csv_path = "/data/hugang/GlhCode/VulAdvisor/data/qwen-max/label_test.csv"
    test_csv_path = f"/data/hugang/GlhCode/VulAdvisor/data/{args.llm_type}/label_test.csv"
    test_df = pd.read_csv(test_csv_path)
    input_texts = []
    # reference_texts = []
    for idx, row in test_df.iterrows():
        func = row.get("function", "")
        patch = row.get("patch", "")
        defect_type = row.get("defect_type", "")
        input_text = build_input_text(func, patch, defect_type, args.strategy)
        input_texts.append(input_text)
    reference_texts = test_df['suggestion'].tolist()

    # model_dir = args.model_dir
    model_dir = f"./saved/{args.llm_type}/{args.strategy}"
    print(f"Testing with models from {model_dir}")
    model_files = [f for f in os.listdir(model_dir) if f.startswith("checkpoint-")]
    model_files.sort(key=lambda x: int(x.split("-")[-1]), reverse=True)

    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        model, tokenizer = load_model_and_tokenizer(model_path)
        model = model.to(device)
        model.eval()

        predictions = generate_predictions(input_texts, batch_size=32)
        metrics = compute_metrics(predictions, reference_texts)

        # 保存预测结果到文件
        # output_file_path = os.path.join(args.output_dir, f"{model_file}.out")
        output_file_path = os.path.join(f"./predictions/{args.llm_type}/{args.strategy}", f"{model_file}.out")
        output_dir = f"./predictions/{args.llm_type}/{args.strategy}"
        os.makedirs(output_dir, exist_ok=True)
        with open(output_file_path, 'w', encoding="utf-8") as f:
            for prediction in predictions:
                f.write(f"{prediction}\n")

        print(f"Metrics for model {model_file}: {metrics}")
        # 清理 GPU 内存
        del model
        torch.cuda.empty_cache()

