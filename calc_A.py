import torch
import os
import glob
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ================= 配置部分 =================
MODEL_ID = "/root/autodl-tmp/Qwen3-8B/"  # 替换为你的模型路径
OUTPUT_DIR = "/root/autodl-tmp/kvswap_projections"                # 矩阵保存路径
CALIBRATION_DATASET = "allenai/c4"                 # 论文指定数据集 
NUM_BATCHES = 20                                   # 论文指定校准 Batch 数 
SEQ_LEN = 2048                                     # 校准序列长度，根据显存调整
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
    Final Fixed Version: 
    1. 适配 DynamicCache (.keys / .key_cache / .layers)
    2. 适配变长序列 (Flatten then Concat)
    3. 修复 Cleanup 变量名错误 (UnboundLocalError)
    """
    collected_keys = {} 
    
    print("Running inference for calibration...")
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Forward Pass"):
            batch = batch.to(DEVICE)
            
            # 执行推理
            outputs = model(input_ids=batch, use_cache=True)
            past_kv = outputs.past_key_values
            
            iterator = []
            
            # ================= 数据收集适配逻辑 =================
            # Case 1: Qwen3/DynamicCache with .layers
            if hasattr(past_kv, "layers"):
                for idx, layer_obj in enumerate(past_kv.layers):
                    # 尝试直接访问 .keys 属性 (你调试发现的结构)
                    if hasattr(layer_obj, "keys"):
                        iterator.append((idx, layer_obj.keys))
                    elif hasattr(layer_obj, "key_cache"):
                        iterator.append((idx, layer_obj.key_cache))
                    elif isinstance(layer_obj, tuple):
                        iterator.append((idx, layer_obj[0]))

            # Case 2: 标准 DynamicCache (.key_cache)
            elif hasattr(past_kv, "key_cache"):
                iterator = enumerate(past_kv.key_cache)
                
            # Case 3: 旧版 Tuple
            elif isinstance(past_kv, tuple):
                for idx in range(len(past_kv)):
                    iterator.append((idx, past_kv[idx][0]))
            
            # Case 4: 无法识别
            else:
                print(f"\n[Error] Unknown Cache Structure. Type: {type(past_kv)}")
                continue
            # ==============================================
            
            # 统一遍历提取 Key
            for layer_idx, key_state in iterator:
                if layer_idx not in collected_keys:
                    collected_keys[layer_idx] = []
                
                if isinstance(key_state, torch.Tensor) and key_state.numel() > 0:
                    collected_keys[layer_idx].append(key_state.detach().cpu().to(torch.float32))
            
            del outputs, past_kv
            torch.cuda.empty_cache()

    # 2. 计算 SVD
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    num_layers = len(model.model.layers)
    
    print("Computing SVD and generating Projection Matrices...")
    for i in range(num_layers):
        if i not in collected_keys or len(collected_keys[i]) == 0:
            print(f"Skipping layer {i} (No keys collected)")
            continue
            
        # ================= 变长序列处理逻辑 =================
        flat_layer_keys = []
        H_k, D = 0, 0
        
        for key_tensor in collected_keys[i]:
            # 获取维度信息
            if H_k == 0:
                _, H_k, _, D = key_tensor.shape
            
            # [Batch, H_k, Seq, D] -> [Batch, Seq, H_k, D] -> Flatten
            key_perm = key_tensor.permute(0, 2, 1, 3)
            key_flat = key_perm.reshape(-1, H_k * D)
            flat_layer_keys.append(key_flat)
            
        # 拼接
        if len(flat_layer_keys) == 0:
             continue
        K_ftn = torch.cat(flat_layer_keys, dim=0)
        # ===================================================
        
        full_dim = H_k * D
        target_rank = int(full_dim / COMPRESSION_SIGMA)
        
        print(f"Layer {i}: Flattened Shape {K_ftn.shape}. Rank: {target_rank}")
        
        # SVD
        try:
            U, S_vals, Vh = torch.linalg.svd(K_ftn.to(DEVICE), full_matrices=False)
        except torch.cuda.OutOfMemoryError:
            print(f"Layer {i} OOM on GPU, switching to CPU...")
            K_ftn = K_ftn.cpu()
            U, S_vals, Vh = torch.linalg.svd(K_ftn, full_matrices=False)

        # 提取并保存 A
        A = Vh[:target_rank, :].T 
        save_path = os.path.join(OUTPUT_DIR, f"projection_layer_{i}.pt")
        torch.save(A.cpu(), save_path)
        print(f"Saved to {save_path}")
        
        # ================= 修复后的清理逻辑 =================
        # 必须先判断变量是否存在，防止报错
        del collected_keys[i]
        
        if 'flat_layer_keys' in locals(): del flat_layer_keys
        if 'K_ftn' in locals(): del K_ftn
        if 'U' in locals(): del U
        if 'S_vals' in locals(): del S_vals
        if 'Vh' in locals(): del Vh
        if 'A' in locals(): del A
        
        # 注意：这里千万不要 del all_keys，因为这个变量已经不存在了
        
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