# !/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <model_name> $1 <attn_type> $2 "
    exit 1
fi

NUM_EXAMPLES=-1
MODEL=${1}
# TASK=${2}
ATTN_TYPE=${2}
# DTYPE=${4}
# BUDGET_RATIO=${5}
# ESTIMATE_RATIO=${6}

RESULT_DIR="./results/pred/${MODEL}/${ATTN_TYPE}"
RESULT_DIR_E="./results/pred_e/${MODEL}/${ATTN_TYPE}"

echo "remove previous result file..."
rm -f "${RESULT_DIR}/${TASK}.jsonl"
rm -f "${RESULT_DIR_E}/${TASK}.jsonl"

echo "Start to predict..."
python -u pred.py --model ${MODEL} --method ${ATTN_TYPE} 