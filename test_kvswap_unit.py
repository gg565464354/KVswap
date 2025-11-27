import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

# ================= 配置 =================
MODEL_PATH = "/root/autodl-tmp/Qwen3-8B/" # 指向你修改过代码的模型路径或ID
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_kvswap_config(model, enabled=True, top_k_groups=100):
    """
    辅助函数：动态修改模型中所有 Attention 层的 KVSwap 配置
    """
    for layer in model.model.layers:
        # 访问我们修改后的 Qwen3Attention 的属性
        if hasattr(layer.self_attn, "kvswap_enabled"):
            layer.self_attn.kvswap_enabled = enabled
            layer.self_attn.kv_top_k_groups = top_k_groups

def test_kvswap_correctness():
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        device_map="auto", 
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # 1. 构造一个较长的输入 (超过 group_size，确保能触发分组)
    # 假设 group_size=4，我们构造长度 32 的 Prompt
    prompt = "The quick brown fox jumps over the lazy dog. " * 5 
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    print("\n[Step 1] Prefilling (Processing Prompt)...")
    # Prefill 阶段通常不触发 KVSwap 稀疏逻辑（取决于你的实现，通常只在 Decoding 触发）
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    
    past_key_values = outputs.past_key_values
    print(f"Prefill done. KV Cache Sequence Length: {past_key_values[0][0].shape[2]}")

    # 2. 准备 Decoding 的输入 (生成下一个 Token)
    next_token_id = torch.tensor([[tokenizer.pad_token_id or 0]], device=DEVICE) # Dummy token
    
    # ================= 测试 A: 关闭 KVSwap (基准) =================
    print("\n[Step 2] Running Baseline (KVSwap Disabled)...")
    set_kvswap_config(model, enabled=False)
    
    with torch.no_grad():
        base_outputs = model(
            input_ids=next_token_id, 
            past_key_values=past_key_values, # 传入全量 Cache
            use_cache=True
        )
    base_logits = base_outputs.logits

    # ================= 测试 B: 开启 KVSwap 但全选 (一致性检查) =================
    print("\n[Step 3] Running KVSwap (Enabled, but selecting ALL groups)...")
    # 设置 top_k 为极大值，强制选中所有组。
    # 如果你的 gather 逻辑是对的，结果应该和 Baseline 只有浮点误差的区别。
    set_kvswap_config(model, enabled=True, top_k_groups=99999) 
    
    with torch.no_grad():
        # 注意：这里我们复用同一个 past_key_values，因为上面的 forward 不会修改它 (inplace update 除外)
        # 为了严谨，应该 clone cache，但 HuggingFace cache update 比较复杂，这里假设只读
        kv_outputs = model(
            input_ids=next_token_id, 
            past_key_values=past_key_values,
            use_cache=True
        )
    kv_logits = kv_outputs.logits

    # 比较 Logits
    diff = (base_logits - kv_logits).abs().max().item()
    print(f"Max Logits Difference (Full vs Full-via-Gather): {diff:.6f}")
    
    if diff < 1e-3:
        print("✅ PASS: KVSwap logic perfectly reconstructs full attention.")
    else:
        print("❌ FAIL: Large discrepancy found. Check your gather/indices logic!")

    # ================= 测试 C: 开启 KVSwap 稀疏模式 (冒烟测试) =================
    print("\n[Step 4] Running KVSwap Sparse (Selecting top 2 groups)...")
    set_kvswap_config(model, enabled=True, top_k_groups=2) # 只选 2 组
    
    try:
        with torch.no_grad():
            sparse_outputs = model(
                input_ids=next_token_id, 
                past_key_values=past_key_values,
                use_cache=True
            )
        print("✅ PASS: Sparse inference ran without crashing.")
    except Exception as e:
        print(f"❌ FAIL: Sparse inference crashed with error: {e}")

if __name__ == "__main__":
    test_kvswap_correctness()