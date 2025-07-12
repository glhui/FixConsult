#!/bin/bash

# 设置 Hugging Face 镜像源
export HF_ENDPOINT="https://hf-mirror.com/"
export WANDB_DISABLED=true


# 设置使用的 GPU 设备编号
export CUDA_VISIBLE_DEVICES=2

# 设置 PyTorch CUDA 内存分配策略
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:1024"

# 禁用 Hugging Face Tokenizers 的并行化，防止多进程死锁
export TOKENIZERS_PARALLELISM=false

# 打印环境变量，便于调试
echo "HF_ENDPOINT: $HF_ENDPOINT"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo "TOKENIZERS_PARALLELISM: $TOKENIZERS_PARALLELISM"

echo "====================train start===================="

# 启动训练-需要去修改save——path
python3 train_main.py | tee log/train_qwenmax-lable_data_no_patch_ra_6_3_lab_1.txt

echo "====================infer start===================="

# 启动推理 model_dir = save——path
python  train_infer.py \
        --model_dir "./saved/results_qwen-max_label_no_patch_data_ra_6_3_lab_1" \
        --output_dir "./predictions/predict_with_qwen-max_label_no_patch_data_ra_6_3_lab_1" \
        | tee log/predict_with_qwen-max_label_no_patch_data_ra_6_3_lab_1.txt


# 启动评估 output_dir和推理的output_dir一致
echo "====================eval start===================="
python  evaluation.py \
        --model_dir "./saved/results_qwen-max_label_no_patch_data_ra_6_3_lab_1" \
        --output_dir "./predictions/predict_with_qwen-max_label_no_patch_data_ra_6_3_lab_1" \
        | tee log/eval_with_qwen-max_no_patch_data_ra_6_3_lab_1.txt