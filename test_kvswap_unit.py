import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= 配置 =================
MODEL_PATH = "/root/autodl-tmp/Qwen3-8B/" 
PROJECTION_DIR = "/root/autodl-tmp/kvswap_projections/"  # 必须指向 calc_A.py 输出的目录
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_projections(model, projection_dir):
    """
    加载离线计算好的 A 矩阵，并赋值给模型的每一层
    """
    print(f"Loading projection matrices from {projection_dir}...")
    
    loaded_count = 0
    for i, layer in enumerate(model.model.layers):
        # 对应 calc_A.py 保存的文件名
        file_path = os.path.join(projection_dir, f"projection_layer_{i}.pt")
        
        if os.path.exists(file_path):
            # 加载矩阵
            A = torch.load(file_path, map_location="cpu")
            
            # 转换设备和精度，必须与模型一致
            A = A.to(device=model.device, dtype=model.dtype)
            
            ref_device = layer.self_attn.q_proj.weight.device
            ref_dtype = layer.self_attn.q_proj.weight.dtype
            
            # 3. 转换 A 的设备和精度
            A = A.to(device=ref_device, dtype=ref_dtype)
            
            # 4. 【核心修改】直接赋值，而不是 register
            # 假设你在 modeling_qwen2.py 里定义的变量名叫 projection_matrix
            layer.self_attn.projection_matrix = A
            
            # 如果你的变量名叫 kvswap_projection_matrix，请用下面这行：
            # layer.self_attn.kvswap_projection_matrix = A

            loaded_count += 1
        else:
            print(f"Warning: No projection matrix found for layer {i}")

    print(f"Successfully loaded projection matrices for {loaded_count}/{len(model.model.layers)} layers.")

def set_kvswap_config(model, enabled=True, top_k_groups=100):
    for layer in model.model.layers:
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
    
    # 加载矩阵 (确保这里路径正确)
    load_projections(model, "/root/autodl-tmp/kvswap_projections/")

    # 构造输入
    prompt = "The quick brown fox jumps over the lazy dog. " * 5 
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    next_token_id = torch.tensor([[tokenizer.pad_token_id or 0]], device=DEVICE)

    # ================= 辅助函数：获取干净的 Prefill Cache =================
    def get_fresh_cache():
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)
        return outputs.past_key_values
    
    print("\n[Step 1] Initial Prefill check...")
    cache_check = get_fresh_cache()
    # 兼容不同版本的 Cache 长度获取
    if hasattr(cache_check, "get_seq_length"):
        slen = cache_check.get_seq_length()
    else:
        slen = cache_check[0][0].shape[2]
    print(f"Prefill done. KV Cache Sequence Length: {slen}")

    # ================= 测试 A: 关闭 KVSwap (基准) =================
    print("\n[Step 2] Running Baseline (KVSwap Disabled)...")
    set_kvswap_config(model, enabled=False)
    
    # 关键修改：每次测试前重新获取干净的 Cache
    cache_base = get_fresh_cache() 
    
    with torch.no_grad():
        base_outputs = model(
            input_ids=next_token_id, 
            past_key_values=cache_base, 
            use_cache=True
        )
    base_logits = base_outputs.logits

    # ================= 测试 B: 开启 KVSwap 但全选 =================
    print("\n[Step 3] Running KVSwap (Enabled, but selecting ALL groups)...")
    set_kvswap_config(model, enabled=True, top_k_groups=99999) 
    
    # 关键修改：再次获取干净的 Cache！不要复用 cache_base
    cache_kv = get_fresh_cache()
    
    with torch.no_grad():
        kv_outputs = model(
            input_ids=next_token_id, 
            past_key_values=cache_kv,
            use_cache=True
        )
    kv_logits = kv_outputs.logits

    # 比较
    diff = (base_logits - kv_logits).abs().max().item()
    print(f"Max Logits Difference (Full vs Full-via-Gather): {diff:.6f}")
    
    if diff < 1e-3:
        print("✅ PASS: KVSwap logic perfectly reconstructs full attention.")
    else:
        print("❌ FAIL: Discrepancy is still high.")

    # ================= 测试 C: 稀疏模式 =================
    print("\n[Step 4] Running KVSwap Sparse (Selecting top 2 groups)...")
    set_kvswap_config(model, enabled=True, top_k_groups=2)
    cache_sparse = get_fresh_cache()
    
    try:
        with torch.no_grad():
            sparse_outputs = model(
                input_ids=next_token_id, 
                past_key_values=cache_sparse,
                use_cache=True
            )
        print("✅ PASS: Sparse inference ran without crashing.")
    except Exception as e:
        print(f"❌ FAIL: Sparse inference crashed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_kvswap_correctness()