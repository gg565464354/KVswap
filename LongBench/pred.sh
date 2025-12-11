#!/bin/bash
# 定义任务列表（已手动去掉 narrativeqa）
tasks=("multifieldqa_en" "hotpotqa" "musique" "dureader" "gov_report" "samsum" "passage_retrieval_en" "lcc")

# 循环执行
for task in "${tasks[@]}"
do
    echo "=================================================="
    echo "Start Running Task: $task"
    echo "=================================================="
    
    python pred.py \
        --model_name Qwen3-8B \
        --method kvswap \
        --kv_group_size 4 \
        --kv_top_k_groups 250 \
        --task "$task"
        
    echo "Finished Task: $task"
done