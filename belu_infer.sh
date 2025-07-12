
# 设置 Hugging Face 镜像源
export HF_ENDPOINT="https://hf-mirror.com/"

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




# strategy = "no_mask" | "no_patch" | "no_label" | "final"
# llm_type = "gpt-3.5-turbo" | "gpt-4o-mini" | "qwen-max"
#python  train_infer.py \
#        --llm_type "qwen-max" \
#        --strategy "no_patch" \
#        | tee log/qwen-max/infer_no_patch.txt
#
#python  train_infer.py \
#        --llm_type "qwen-max" \
#        --strategy "no_mask" \
#        | tee log/qwen-max/infer_no_mask.txt

python  train_infer.py \
        --llm_type "gpt-3.5-turbo" \
        --strategy "no_patch" \
        | tee log/qwen-max/infer_final.txt

python  evaluation.py \
        --llm_type "qwen-max" \
        --strategy "final" \
        | tee log/qwen-max/eval/final.txt