
# 设置 Hugging Face 镜像源
export HF_ENDPOINT="https://hf-mirror.com/"

# 设置使用的 GPU 设备编号
export CUDA_VISIBLE_DEVICES=1

# 设置 PyTorch CUDA 内存分配策略
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:1024"

# 禁用 Hugging Face Tokenizers 的并行化，防止多进程死锁
export TOKENIZERS_PARALLELISM=false

# 打印环境变量，便于调试
echo "HF_ENDPOINT: $HF_ENDPOINT"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo "TOKENIZERS_PARALLELISM: $TOKENIZERS_PARALLELISM"

python  evaluation.py \
        --llm_type "qwen-max" \
        --strategy "no_patch" \
        | tee log/qwen-max/eval/no_patch.txt

python  evaluation.py \
        --llm_type "qwen-max" \
        --strategy "no_mask" \
        | tee log/qwen-max/eval/no_mask.txt

#python  evaluation.py \
#        --llm_type "gpt-3.5-turbo" \
#        --strategy "no_patch" \
#        | tee log/gpt-3.5-turbo/eval/no_patch.txt

#python  evaluation.py \
#        --llm_type "gpt-4o-mini" \
#        --strategy "final" \
#        | tee log/gpt-4o-mini/eval/final.txt

#
## 缺少补丁
#python  evaluation.py \
#        --model_dir "./saved/predict_with_base_data_ra_5_27_lab_1" \
#        --output_dir "./predictions/predict_with_qwen-max_no_patch_data_ra_6_3_lab_1" \
#        | tee log/qwen-max/eval_no_patch.txt
#
##缺少标签
#python  evaluation.py \
#        --model_dir "./saved/predict_with_base_data_ra_5_27_lab_1" \
#        --output_dir "./predictions/predict_with_qwen-max_no_label_data_ra_6_4_lab_1" \
#        | tee log/qwen-max/eval_no_label.txt