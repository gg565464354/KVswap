import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ================= 配置 =================
MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
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

def main():
    print(f"Loading Model: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        device_map="auto", 
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # 加载 LongBench 数据
    print(f"Loading LongBench task: {TASK_NAME}...")
    dataset = load_dataset(DATASET_NAME, TASK_NAME, split="test", streaming=True)
    data_iter = iter(dataset)
    
    # 取第一个样本
    sample = next(data_iter)
    context = sample["context"]
    input_len = len(context)
    print(f"Sample loaded. Context Length (chars): {input_len}")
    
    # 截断以防止爆显存 (根据你的显存情况调整，比如 8K 或 16K)
    # Qwen2.5 支持 32K，但在单卡上做全量对比可能OOM，这里限制一下输入长度
    MAX_INPUT_LEN = 8000 
    if len(context) > MAX_INPUT_LEN * 4: # 粗略估计字符到token
        context = context[:MAX_INPUT_LEN * 4]
        
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
            
    try:
        res_base = generate_response(model, tokenizer, inputs.input_ids)
        print(f"[Baseline Output]:\n{res_base}")
    except torch.cuda.OutOfMemoryError:
        print("[Baseline] OOM! Skipping baseline.")
        res_base = "OOM"

    # ================= 2. 运行 KVSwap =================
    print(f"\n--- Running KVSwap (Top-{TOP_K_GROUPS} Groups) ---")
    # 开启 KVSwap
    for layer in model.model.layers:
        if hasattr(layer.self_attn, "kvswap_enabled"):
            layer.self_attn.kvswap_enabled = True
            layer.self_attn.kv_group_size = GROUP_SIZE
            layer.self_attn.kv_top_k_groups = TOP_K_GROUPS
            
    res_kv = generate_response(model, tokenizer, inputs.input_ids)
    print(f"[KVSwap Output]:\n{res_kv}")
    
    # ================= 3. 统计分析 =================
    used_tokens = TOP_K_GROUPS * GROUP_SIZE
    compression_rate = used_tokens / seq_len
    print("\n" + "="*30)
    print("Summary:")
    print(f"Original Sequence Length: {seq_len}")
    print(f"KVSwap Active KV Tokens: ~{used_tokens} (Plus rolling buffer)")
    print(f"Compression Ratio: {compression_rate:.2%} (Only used {compression_rate*100:.1f}% of cache)")
    print("="*30)

if __name__ == "__main__":
    main()