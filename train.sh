#!/bin/bash

# 设置 Hugging Face 镜像源
export HF_ENDPOINT="https://hf-mirror.com/"
export WANDB_DISABLED=true


# 设置使用的 GPU 设备编号
export CUDA_VISIBLE_DEVICES=0

# 设置 PyTorch CUDA 内存分配策略
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:1024"

# 禁用 Hugging Face Tokenizers 的并行化，防止多进程死锁
export TOKENIZERS_PARALLELISM=false

# 打印环境变量，便于调试
echo "HF_ENDPOINT: $HF_ENDPOINT"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo "TOKENIZERS_PARALLELISM: $TOKENIZERS_PARALLELISM"

## 启动 Python 程序
#python3 train_main.py | tee log/train_gpt-4o-mini_no_patch_data_ra_6_6_lab_1.txt

#pyth3on  train_main.py \
#        --data_dir "gpt-4o-mini" \
#        --save_path "gpt-4o-mini/no_patch" \
#        --strategy "dfp_no_patch" \
#        | tee log/gpt-4o-mini/train_no_patch.txt

python  train_main.py \
        --data_dir "gpt-3.5-turbo" \
        --save_path "gpt-3.5-turbo/no_patch" \
        --strategy "dfp_no_patch" \
        | tee log/gpt-3.5-turbo/train_no_patch_new.txt

#dfp_no_mask
#python  train_main.py \
#        --data_dir "qwen-max" \
#        --save_path "qwen-max/dfp_no_mask" \
#        --strategy "dfp_no_mask" \
#        | tee log/qwen-max/train_no_mask.txt

python  train_main.py \
        --data_dir "qwen-max" \
        --save_path "qwen-max/final" \
        --strategy "dfp" \
        | tee log/qwen-max/train_final.txt
