#!/bin/bash
set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <model>"
    exit 1
fi

MODEL=${1}

for ATTN_TYPE in "full" "kvdrive" "shadow" "quest"; do
    RESULT_DIR="./results/pred/${MODEL}/${ATTN_TYPE}"
    echo "Parameters: ${MODEL} ${ATTN_TYPE}"

    bash pred.sh ${MODEL} ${ATTN_TYPE}

    echo "Start to evaluate..."
    python -u eval.py \
        --attn_type ${ATTN_TYPE} \
        --model ${MODEL}

    echo "Results:"
    cat "${RESULT_DIR}/result.json"
done

