import torch
import os
import glob
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ================= 配置部分 =================
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"  # 替换为你的模型路径
OUTPUT_DIR = "./kvswap_projections"                # 矩阵保存路径
CALIBRATION_DATASET = "allenai/c4"                 # 论文指定数据集 
NUM_BATCHES = 20                                   # 论文指定校准 Batch 数 
# SEQ_LEN = 2048                                     # 校准序列长度，根据显存调整
COMPRESSION_SIGMA = 32                             # 论文中的最大压缩比 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_calibration_data(tokenizer, num_batches, seq_len):
    """
    加载 C4 数据集并预处理 
    """
    print(f"Loading {num_batches} batches from {CALIBRATION_DATASET}...")
    dataset = load_dataset(CALIBRATION_DATASET, "en", split="train", streaming=True)
    
    batch_data = []
    iterator = iter(dataset)
    
    pbar = tqdm(total=num_batches, desc="Tokenizing")
    while len(batch_data) < num_batches:
        try:
            sample = next(iterator)
            # 简单的截断处理，保证每个 sample 都是 seq_len 长度
            tokenized = tokenizer(
                sample["text"], 
                truncation=True, 
                # max_length=seq_len, 
                return_tensors="pt"
            )
            # 过滤掉太短的文本
            if tokenized.input_ids.shape[1] >= seq_len // 2:
                batch_data.append(tokenized.input_ids[:, :seq_len])
                pbar.update(1)
        except StopIteration:
            break
            
    return batch_data

def collect_and_compute_svd(model, dataloader):
    """
    核心逻辑：Hook 收集 Key -> Flatten -> SVD -> 保存矩阵 A
    """
    # 存储每一层的 Key States
    # Key: layer_index, Value: List of tensors
    collected_keys = {} 
    
    # 定义 Hook 函数：拦截 Attention 层的输出
    def get_key_hook(layer_idx):
        def hook(module, args, output):
            # LlamaAttention 的输出通常是 (attn_output, past_key_value, attn_weights)
            # past_key_value 是一个 tuple (key, value)
            # key 的形状通常是 (Batch, Num_KV_Heads, Seq_Len, Head_Dim)
            
            # 兼容性处理：不同 transformers 版本输出结构略有不同
            if isinstance(output, tuple):
                # 寻找形状符合 Key 预期的 tensor
                # 这里的 output[1] 应该是 past_key_values (tuple)
                # output[1][0] 是 Key
                try:
                    past_kv = output[1]
                    key_state = past_kv[0] 
                except:
                    # 如果 output 结构复杂，这里需要根据具体 transformers 版本调整
                    # 也可以尝试在 forward 内部直接 hook key_states 变量，但更麻烦
                    print(f"Warning: Could not extract key from layer {layer_idx} output.")
                    return
            else:
                return

            # 将 key_state 转移到 CPU 暂存，防止显存爆炸
            if layer_idx not in collected_keys:
                collected_keys[layer_idx] = []
            collected_keys[layer_idx].append(key_state.detach().cpu().to(torch.float32))
            
        return hook

    # 1. 注册 Hook
    print("Registering hooks...")
    hooks = []
    for i, layer in enumerate(model.model.layers):
        # 针对 Llama 的 self_attn 模块注册
        h = layer.self_attn.register_forward_hook(get_key_hook(i))
        hooks.append(h)

    # 2. 运行推理 (收集数据)
    print("Running inference for calibration...")
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Forward Pass"):
            batch = batch.to(DEVICE)
            model(batch)

    # 3. 移除 Hook
    for h in hooks:
        h.remove()

    # 4. 计算 SVD 并生成矩阵 A
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    num_layers = len(model.model.layers)
    
    print("Computing SVD and generating Projection Matrices...")
    for i in range(num_layers):
        if i not in collected_keys or len(collected_keys[i]) == 0:
            continue
            
        # 拼接所有 Batch 的 Keys
        # Shape: [Total_Batch, KV_Heads, Seq_Len, Head_Dim]
        all_keys = torch.cat(collected_keys[i], dim=0)
        
        # 获取维度信息
        B, H_k, S, D = all_keys.shape
        
        # 论文步骤 1: Reshape into matrix N x (H_k * d) 
        # 需要先转置把 Head 放到后面: (B, H_k, S, D) -> (B, S, H_k, D)
        all_keys = all_keys.permute(0, 2, 1, 3).contiguous()
        # Flatten: (N, H_k * D), 其中 N = B * S
        K_ftn = all_keys.view(-1, H_k * D)
        
        # 计算目标 Rank r [cite: 184, 199]
        # r = (H_k * d) / sigma
        full_dim = H_k * D
        target_rank = int(full_dim / COMPRESSION_SIGMA)
        
        print(f"Layer {i}: Flattened Shape {K_ftn.shape}. "
              f"Full Dim: {full_dim}, Compression Sigma: {COMPRESSION_SIGMA}, Target Rank: {target_rank}")
        
        # 论文步骤 2: SVD 分解 
        # SVD(K_ftn) = U @ diag(S) @ V.T
        # torch.linalg.svd 返回的 Vh 就是 V.T
        # 注意：为了精度，建议在 float32 下进行，如果 OOM 可以切块或用低精度
        try:
            # 这里的 K_ftn 可能很大，如果内存不足，可以使用 torch.svd_lowrank
            # 但论文使用的是标准 SVD [cite: 181]
            U, S, Vh = torch.linalg.svd(K_ftn.to(DEVICE), full_matrices=False)
        except torch.cuda.OutOfMemoryError:
            print(f"Layer {i} OOM on GPU, switching to CPU for SVD (might be slow)...")
            K_ftn = K_ftn.cpu()
            U, S, Vh = torch.linalg.svd(K_ftn, full_matrices=False)

        # 论文步骤 3: 提取前 r 个右奇异向量作为 A 
        # Vh shape is (min(N, D_full), D_full) if full_matrices=False? No, typically (D_full, D_full)
        # We need the top-r rows of Vh (which correspond to columns of V)
        # Matrix A should be shape (D_full, r) so that K_ftn @ A -> (N, r)
        
        # Vh[:target_rank, :] 取出了前 r 行
        # 转置后变成 (D_full, r)
        A = Vh[:target_rank, :].T 
        
        # 保存矩阵
        save_path = os.path.join(OUTPUT_DIR, f"projection_layer_{i}.pt")
        torch.save(A.cpu(), save_path)
        print(f"Saved projection matrix to {save_path}")
        
        # 清理内存
        del collected_keys[i]
        del all_keys, K_ftn, U, S, Vh, A
        torch.cuda.empty_cache()

def main():
    print(f"Initializing model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16, 
        device_map="auto",
        use_cache=True,  # 必须开启 Cache 才能 Hook 到 Key
        trust_remote_code=True
    )
    
    # 准备数据
    dataloader = get_calibration_data(tokenizer, NUM_BATCHES, SEQ_LEN)
    
    # 执行主逻辑
    collect_and_compute_svd(model, dataloader)
    print("Done! Offline preparation complete.")

if __name__ == "__main__":
    main()