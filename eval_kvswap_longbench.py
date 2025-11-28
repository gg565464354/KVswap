import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from rouge_score import rouge_scorer
import numpy as np
# ================= 配置 =================
MODEL_PATH = "/root/autodl-tmp/Qwen3-8B"
DATASET_NAME = "THUDM/LongBench"
TASK_NAME = "qmsum" # 会议总结任务，适合测试长文本能力
DEVICE = "cuda"

# KVSwap 设置
GROUP_SIZE = 4
TOP_K_GROUPS = 100 # 选中 400 个 Token (加上最新的)

def build_prompt(context, question):
    # Qwen 的标准 Chat 模板
    prompt = f"<|im_start|>user\nRead the following meeting transcript and answer the question.\n\nContext:\n{context}\n\nQuestion: {question}<|im_end|>\n<|im_start|>assistant\n"
    return prompt

def generate_response(model, tokenizer, input_ids, max_new_tokens=100):
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False, # 为了对比公平，使用 Greedy Search
        temperature=None,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

def reset_kv_state(model):
    """强制清空模型中所有 KVSwap 层的压缩缓存"""
    for layer in model.model.layers:
        if hasattr(layer.self_attn, "compressed_k_cache"):
            layer.self_attn.compressed_k_cache = None

def main():
    print(f"Loading Model: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        device_map="auto", 
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores_baseline_vs_gold = []
    scores_vs_gold = []     # KVSwap vs 标准答案
    scores_vs_baseline = [] # KVSwap vs Full Cache (还原度)
    # 加载 LongBench 数据
    print(f"Loading LongBench task: {TASK_NAME}...")
    local_data_path = "/root/KVswap/data/qmsum.jsonl"
    # dataset = load_dataset(DATASET_NAME, TASK_NAME, split="test", streaming=True, trust_remote_code=True)
    dataset = load_dataset("json", data_files=local_data_path, split="train")
    # data_iter = iter(dataset)

    # sample = next(data_iter)
    from tqdm import tqdm # 建议加上进度条

    # ... 加载 dataset 代码不变 ...

    # === 修改处开始 ===
    # 直接遍历 dataset 即可跑完全部样本
    for i, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        context = sample["context"]
        ground_truth = sample["answers"][0]
        input_len = len(context)
        print(f"Sample loaded. Context Length (chars): {input_len}")
        
        # 截断以防止爆显存 (根据你的显存情况调整，比如 8K 或 16K)
        # Qwen2.5 支持 32K，但在单卡上做全量对比可能OOM，这里限制一下输入长度
        # MAX_INPUT_LEN = 8000 
        # if len(context) > MAX_INPUT_LEN * 4: # 粗略估计字符到token
        #     context = context[:MAX_INPUT_LEN * 4]
            
        prompt = build_prompt(context, sample["input"])
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        seq_len = inputs.input_ids.shape[1]
        print(f"Tokenized Input Length: {seq_len} tokens")

        # ================= 1. 运行 Full Cache Baseline =================
        print("\n--- Running Baseline (Full Cache) ---")
        # 关闭 KVSwap
        for layer in model.model.layers:
            if hasattr(layer.self_attn, "kvswap_enabled"):
                layer.self_attn.kvswap_enabled = False
        reset_kv_state(model)
        torch.cuda.empty_cache()
        res_base = generate_response(model, tokenizer, inputs.input_ids)
        print(f"[baseline Output]:\n{res_base}")
        # ================= 2. 运行 KVSwap =================
        print(f"\n--- Running KVSwap (Top-{TOP_K_GROUPS} Groups) ---")
        # 开启 KVSwap
        for layer in model.model.layers:
            if hasattr(layer.self_attn, "kvswap_enabled"):
                layer.self_attn.kvswap_enabled = True
                layer.self_attn.kv_group_size = GROUP_SIZE
                layer.self_attn.kv_top_k_groups = TOP_K_GROUPS
        reset_kv_state(model)
        torch.cuda.empty_cache()        
        res_kv = generate_response(model, tokenizer, inputs.input_ids)
        print(f"[KVSwap Output]:\n{res_kv}")
        
        # ================= 3. 统计分析 =================
        score_gold = scorer.score(ground_truth, res_kv)
        scores_vs_gold.append(score_gold['rougeL'].fmeasure)
        
        # Metric 2: 还原度 (Rouge-L vs Baseline)
        # 衡量 KVSwap 相比不压缩的模型损失了多少信息
        score_base = scorer.score(res_base, res_kv)
        scores_vs_baseline.append(score_base['rougeL'].fmeasure)

        s_base_gold = scorer.score(ground_truth, res_base)['rougeL'].fmeasure
        scores_baseline_vs_gold.append(s_base_gold)
        # 打印当前样本结果
        print(f"\n[Sample {i}]")
        print(f"  Rouge-L (vs Gold): {scores_vs_gold[-1]:.4f}")
        print(f"  Rouge-L (vs Full): {scores_vs_baseline[-1]:.4f}")
    print("\n" + "="*30)
    print("Final Average Accuracy:")
    print(f"  Avg Rouge-L (KVswap Accuracy): {np.mean(scores_vs_gold):.4f}")
    print(f"  Avg Rouge-L (Baseline Accuracy): {np.mean(scores_baseline_vs_gold):.4f}")
    print(f"  Avg Rouge-L (Fidelity):      {np.mean(scores_vs_baseline):.4f}")
    print("="*30)
if __name__ == "__main__":
    main()