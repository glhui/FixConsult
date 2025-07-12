import argparse
import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com/'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'
import numpy as np
import torch
from datasets import load_dataset
from torch.backends.opt_einsum import strategy
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer

from utils import process_patch, create_ra_masks, postprocess_text
from VulCodeT5Model import CodeT5ForConditionalGeneration
from evaluate import load as load_metric


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenize_function_and_suggestion(batch):
    """
    针对 batched 数据进行处理：
    - 输入：拼接源代码（function）和补丁（patch），格式化为任务提示文本；
    - 标签：直接使用 CSV 中的 suggestion 列（自然语言描述漏洞和修复建议）。
    """
    functions = batch["function"]
    patches = batch.get("patch", [""] * len(functions))
    suggestions = batch.get("suggestion", [""] * len(functions))
    defect_types = batch.get("defect_type", [""] * len(functions))
    input_texts = []
    label_texts = []

    for func_text, patch_text, suggestion, defect_type in zip(functions, patches, suggestions, defect_types):
        defect_type = defect_type.strip() if isinstance(defect_type, str) else ""
        func_text = func_text.strip() if isinstance(func_text, str) else ""
        patch_text = patch_text.strip() if isinstance(patch_text, str) else ""
        processed_patch = process_patch(patch_text) if patch_text else ""

        # 拼接源代码和补丁，可以根据需要调整格式，例如添加任务提示
        # 例如：<CODE> ... </CODE> <PATCH> ... </PATCH>
        if strategy == "dfp":
            input_text = f"<DEFECT_TYPE>{defect_type}</DEFECT_TYPE><CODE> {func_text} </CODE> <PATCH> {processed_patch} </PATCH>"
        elif strategy == "dfp_no_patch":
            input_text = f"<DEFECT_TYPE>{defect_type}</DEFECT_TYPE><CODE> {func_text} </CODE>"
        elif strategy == "dfp_no_mask":
            input_text = f"<DEFECT_TYPE>{defect_type}</DEFECT_TYPE><CODE> {func_text} </CODE> <PATCH> {processed_patch} </PATCH>"
        elif strategy == "dfp_no_label":
            input_text = f"<CODE> {func_text} </CODE> <PATCH> {processed_patch} </PATCH>"
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        # input_text = f"<DEFECT_TYPE>{defect_type}</DEFECT_TYPE><CODE> {func_text} </CODE> <PATCH> {processed_patch} </PATCH>"  # 最终方案
        # input_text = f"<DEFECT_TYPE>{defect_type}</DEFECT_TYPE><CODE> {func_text} </CODE>" # 缺少补丁
        # input_text = f"<CODE> {func_text} </CODE> <PATCH> {processed_patch} </PATCH>"  # 基本方案 补丁 + 源代码，此方案不需要ra—mask,也不需要梯度
        input_texts.append(input_text)
        label_texts.append(suggestion.strip())

    # Tokenize 输入和标签
    model_inputs = tokenizer(
        input_texts,
        padding="longest",
        truncation=True,
    )
    # model_inputs["dummy_input_ids"] = model_inputs["input_ids"]

    labels, ra_masks = create_ra_masks(suggestions, tokenizer)
    if strategy == "dfp_no_mask":
        pass
    else:
        model_inputs["ra_mask"] = ra_masks
    model_inputs["labels"] = labels

    return model_inputs

def compute_metrics_infer(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    # decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

def main():
    raw_datasets = load_dataset('csv', data_files=data_files)
    # raw_datasets["train"] = raw_datasets["train"].select(range(32))
    # raw_datasets["valid"] = raw_datasets["valid"].select(range(32))
    tokenized_datasets = raw_datasets.map(tokenize_function_and_suggestion, batched=True)
    if strategy == "dfp_no_mask":
        tokenized_datasets.set_format(
            type='torch',
            columns=['input_ids', 'attention_mask', 'labels'],
        )
    else:
        tokenized_datasets.set_format(
            type='torch',
            columns=['input_ids', 'attention_mask', 'labels', 'ra_mask'],
        )

    model = CodeT5ForConditionalGeneration.from_pretrained(model_checkpoint)
    model = model.to(device)
    model.resize_token_embeddings(len(tokenizer))

    training_args = Seq2SeqTrainingArguments(
        output_dir=save_path,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        predict_with_generate=True,
        num_train_epochs=50,
        weight_decay=0.01,
        save_total_limit=20,
        generation_max_length=100,
        remove_unused_columns=False
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_infer,
    )

    trainer.train()

def parse_args():
    parser = argparse.ArgumentParser(
        description="config"
    )
    # 数据文件路径
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to saved model directory"
    )

    parser.add_argument(
        "--strategy", type=str, required=True,
        help="The strategy to use"
    )

    parser.add_argument(
        "--save_path", type=str, required=True,
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    strategy = args.strategy
    print(f"[info]train strategy is {strategy}!===================================================================================================================")
    tokenizer_path = "/data/hugang/GlhCode/PromptCS/codet5-base"
    model_name = "/data/hugang/GlhCode/PromptCS/codet5-base"

    model_checkpoint = "/data/hugang/GlhCode/PromptCS/codet5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
    if not strategy == "dfp_no_patch":
        SPECIAL_TOKENS = {
            "added": "<ADDED>",
            "removed": "<REMOVED>"
        }
        additional_tokens = list(SPECIAL_TOKENS.values())
        if tokenizer.add_tokens(additional_tokens) > 0:
            print("Added additional tokens:", additional_tokens)

    data_files = {
        "train": f"/data/hugang/GlhCode/VulAdvisor/data/{args.data_dir}/label_train.csv",
        "valid": f"/data/hugang/GlhCode/VulAdvisor/data/{args.data_dir}/label_valid.csv",
    }

    print(f"[info] data_files: {data_files}=-==================================================================================================================")

    save_path = f"./saved/{args.save_path}"
    print(f"[info] save_path: {save_path}=-==================================================================================================================")

    metric = load_metric("sacrebleu")
    main()

