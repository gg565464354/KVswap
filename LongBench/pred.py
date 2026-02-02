import os
import sys
import json
import torch
import numpy as np
import random
import argparse
import time
import importlib
from tqdm import tqdm
# from datasets import load_dataset # 不再需要，直接用原生 json 读取
os.environ["HF_HUB_OFFLINE"] = "1" 
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# ==================== 路径配置 ====================
# 添加项目根目录到 path，以便导入 models
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 尝试导入 Quest (仅当环境中有 Quest 时)
try:
    import quest.utils
except ImportError:
    pass

# 读取配置
model2path = json.load(open("config/model2path.json", "r"))
model2maxlen = json.load(open("config/model2maxlen.json", "r"))
dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

os.environ["ENABLE_METRICS"] = "0"
os.environ["ENABLE_MONITOR"] = "0"


# ==================== 核心功能 1: 动态模型文件替换 (Symlink) ====================
def set_symlink(model_type, fname):
    """
    将 transformers 库中的 modeling_{model_type}.py 软链接到 source 目录下的指定文件
    """
    try:
        import transformers
        tf_path = os.path.dirname(transformers.__file__)
    except ImportError:
        print("Error: Transformers library not found.")
        sys.exit(1)

    # 目标路径 (transformers 安装目录)
    model_path = os.path.join(tf_path, "models", model_type)
    # 源文件路径 (你的魔改代码目录)
    source_dir = "/root/KVswap/LongBench/source"
    linker_path = os.path.join(source_dir, fname)

    # 安全检查
    if not os.path.exists(linker_path):
        print(f"Error: Source file not found at {linker_path}")
        print(f"Please ensure {fname} exists in {source_dir}")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"Error: Transformers model directory not found at {model_path}")
        sys.exit(1)

    curr_dir = os.getcwd()
    os.chdir(model_path)
    
    target_file = f'modeling_{model_type}.py'
    
    # 删除旧文件/链接
    if os.path.exists(target_file) or os.path.islink(target_file):
        os.system(f"rm {target_file}")
    
    # 创建新链接
    cmd = f"ln -s {linker_path} {target_file}"
    os.system(cmd)
    
    print(f"✅ Symlink created: transformers/.../modeling_{model_type}.py -> {linker_path}")
    os.chdir(curr_dir)

def reload_modules(model_type):
    """
    强制从 sys.modules 中移除 transformers 模型相关的模块，
    确保 Python 重新加载被 Symlink 替换后的代码。
    """
    modules_to_remove = [
        k for k in sys.modules.keys() 
        if f'transformers.models.{model_type}' in k or 'models' in k
    ]
    for module_name in modules_to_remove:
        del sys.modules[module_name]
    print(f"🔄 Reloaded {len(modules_to_remove)} modules for {model_type}")


# ==================== 核心功能 2: 参数注入 (Infinigen/KVSwap) ====================
def configure_model_params(llm_wrapper, args):
    """
    根据 args.method 将特定参数注入到模型层中
    """
    # 尝试获取底层的 HuggingFace 模型对象
    model = getattr(llm_wrapper, "model", None)
    if model is None: 
        print("Error: Could not find underlying HF model in llm_wrapper.")
        return

    # 兼容 Llama/Qwen/Mistral 的层结构路径
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        print("Error: Could not find layers in model structure.")
        return

    device = model.device
    dtype = model.dtype

    # ---------- KVSwap 参数注入 ----------
    if args.method == "kvswap":
        print(f"🔧 Configuring KVSwap: Group={args.kv_group_size}, TopK={args.kv_top_k_groups}")
        enabled_cnt = 0
        for layer in layers:
            # 检查投影矩阵是否已加载 (通常在 modeling_xxx_kvswap.py 的 init 中完成)
            has_matrix = hasattr(layer.self_attn, "projection_matrix") and layer.self_attn.projection_matrix is not None
            
            if has_matrix:
                layer.self_attn.kvswap_enabled = True
                layer.self_attn.kv_group_size = args.kv_group_size
                
                # 自动计算 TopK (参考源码逻辑 MG=400)
                if args.kv_top_k_groups == -1:
                    layer.self_attn.kv_top_k_groups = 400 // args.kv_group_size
                else:
                    layer.self_attn.kv_top_k_groups = args.kv_top_k_groups
                enabled_cnt += 1
            else:
                # 显式关闭，防止残留
                if hasattr(layer.self_attn, "kvswap_enabled"):
                    layer.self_attn.kvswap_enabled = False
        
        if enabled_cnt == 0:
            print("⚠️ Warning: KVSwap selected but NO layers have projection_matrix loaded!")

    # ---------- Infinigen 参数注入 ----------
    elif args.method == "infinigen":
        print(f"🔧 Configuring InfiniGen: Alpha={args.alpha}, Ratio={args.partial_weight_ratio}")
        
        # 1. 加载 Skewing Matrix (如果指定且存在)
        A = None
        if args.skewing_matrix_path is not None and os.path.exists(args.skewing_matrix_path):
            A = torch.load(args.skewing_matrix_path)
        for layer in range(len(model.model.layers)):
            la = model.model.layers[layer].self_attn
            la.partial_weight_ratio = args.partial_weight_ratio
            if args.partial_weight_path is not None and os.path.isdir(args.partial_weight_path):
                pwq_file = os.path.join(args.partial_weight_path, f"partial_weight_q_{layer}.pt")
                if os.path.exists(pwq_file):
                    la.partial_weight_q = torch.load(pwq_file, map_location=device).to(model.dtype)
                    # la.partial_weight_q = torch.load(pwq_file)
                else:
                    la.partial_weight_q = None
            else:
                la.partial_weight_q = None
            la.alpha = args.alpha
            la.capacity = args.capacity
            la.budget = args.budget
            if A is not None:
                # la.skewing_matrix = A[layer]
                la.skewing_matrix = A[layer].to(device).to(model.dtype)
            if hasattr(la, "cache_pool_enabled"):
                la.cache_pool_enabled = bool(args.infinigen_cache_pool_enabled)
                la.cache_pool_strategy = args.infinigen_cache_pool_strategy
                la.cache_pool_k = args.infinigen_cache_pool_k
                la.cache_pool_cap_ratio = args.infinigen_cache_pool_cap_ratio
                la.local_window_size = args.infinigen_local_window
            if hasattr(la, "fixed_topk"):
                la.fixed_topk = args.infinigen_fixed_topk
    elif args.method == "quest":
        # 计算 Top-K: 预算 tokens / page_size
        # 例如: 预算 4096 / page 64 = 64 个 pages
        budget = args.sparse_budget if args.sparse_budget > 0 else 4096
        page_size = args.quest_page_size
        top_k = max(1, budget // page_size)
        
        print(f"🔧 Configuring Quest: Budget={budget}, PageSize={page_size} -> Selecting Top-{top_k} Pages")
        
        for layer in layers:
            # 1. 开启我们修改后的逻辑开关 (我们在代码里复用了 kvswap_enabled 变量名，或者你可以改成 quest_enabled)
            # 确保这与你在 modeling_qwen3.py 里写的判断条件一致
            layer.self_attn.kvswap_enabled = True 
            
            # 2. 注入 Page Size
            layer.self_attn.page_size = page_size
            
            # 3. 注入 Top-K 数量
            layer.self_attn.kv_top_k_groups = top_k
def reset_kv_state(llm_wrapper):
    """
    强制清空模型中每一层的自定义 KV 缓存状态 (KVSwap/Quest/Infinigen)
    """
    model = getattr(llm_wrapper, "model", None)
    if model is None: return

    # 兼容获取层
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        return

    for layer in layers:
        # KVSwap 缓存
        if hasattr(layer.self_attn, "compressed_k_cache"):
            layer.self_attn.compressed_k_cache = None
        # Quest 缓存 (部分实现可能用这个属性)
        if hasattr(layer.self_attn, "quest_cache"):
             layer.self_attn.quest_cache = None


# ==================== 核心功能 3: 推理循环 (断点续传 + Quest Controller) ====================
def get_pred(llm, data, max_new_tokens, prompt_format, model_name, out_path, args):
    # 1. 注入参数 (KVSwap / Infinigen)
    configure_model_params(llm, args)
    
    # 2. 断点续传检查
    start = 0
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            start = len(list(f))
        if start > 0:
            print(f"⏩ Resuming from index {start}, skipping processed examples.")
    
    if start >= len(data):
        print("All data processed.")
        return

    data_ = data[start:] # 仅处理剩余数据
    
    print(f"🚀 Starting inference with method: [{args.method}]")
    
    for json_obj in tqdm(data_):
        # 清理状态
        reset_kv_state(llm)

        prompt = prompt_format.format(**json_obj)
        inputs = llm.tokenizer([prompt], return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(llm.device)
    
        
        # ---------- 生成配置 ----------
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,   # 确定性生成
            "temperature": 1.0,   # Greedy decoding (with do_sample=False)
            "top_p": 1.0,
            "pad_token_id": llm.tokenizer.pad_token_id,
            "use_cache": True,
        }
        if args.method == "kvswap":
            gen_kwargs.update({
                "kvswap_enable_prefetch": bool(args.kvswap_enable_prefetch),
                "kvswap_cache_pool_enabled": bool(args.kvswap_cache_pool_enabled),
                "kvswap_cache_pool_strategy": args.kvswap_cache_pool_strategy,
            })
        if args.method == "infinigen":
            gen_kwargs.update({
                "infinigen_cache_pool_enabled": bool(args.infinigen_cache_pool_enabled),
                "infinigen_cache_pool_strategy": args.infinigen_cache_pool_strategy,
                "infinigen_fixed_topk": args.infinigen_fixed_topk,
            })

        # ---------- 执行生成 ---------
        try:
            # 关键：尝试直接调用底层 HF model.generate，绕过 Wrapper 的参数过滤
            if hasattr(llm, "model") and hasattr(llm.model, "generate"):
                output_ids = llm.model.generate(input_ids, **gen_kwargs)
            else:
                # 回退方案 (如果 LLM Wrapper 实现了透传)
                output_ids = llm.generate(input_ids, **gen_kwargs)
            
            # 解码
            input_len = input_ids.shape[1]
            pred = llm.tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
            
        except Exception as e:
            print(f"❌ Generation Failed: {e}")
            pred = ""


        # ---------- 保存结果 (追加模式) ----------
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump(
                {
                    "pred": pred, 
                    "answers": json_obj["answers"], 
                    "all_classes": json_obj["all_classes"], 
                    "length": json_obj["length"]
                }, 
                f, 
                ensure_ascii=False
            )
            f.write('\n')


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    
    # 基础配置
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument("--num_examples", type=int, default=-1, help="Num examples to run (-1 for all)")
    parser.add_argument("--model_name", type=str, default=None, choices=["llama-3-8b-262k", "Phi-4-mini-instruct", "Qwen3-8B"])
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--datalen", type=int, default=128*1024)
    parser.add_argument("--cache_dir", type=str, default="/root/.cache/datasets/THUDM___long_bench")

    # 方法选择
    parser.add_argument("--method", type=str, default="full", 
                        choices=["full", "kvdrive", "shadow", "quest", "kvswap", "infinigen"], 
                        help="Attention method to use")

    # KVSwap 参数
    parser.add_argument("--kv_group_size", type=int, default=4, help="KVSwap Group Size")
    parser.add_argument("--kv_top_k_groups", type=int, default=100, help="KVSwap TopK. -1 for auto (400//G)")
    parser.add_argument("--kvswap_enable_prefetch", type=int, default=1, choices=[0, 1],
                    help="0=disable, 1=enable KVSwap look-ahead prefetch")
    parser.add_argument("--kvswap_cache_pool_enabled", type=int, default=1, choices=[0, 1],
                        help="0=disable, 1=enable Quest-style cache pool")
    parser.add_argument("--kvswap_cache_pool_strategy", type=str, default="fixed_k",
                        choices=["fixed_k", "threshold"], help="Cache pool strategy")
    # Infinigen 参数
    parser.add_argument("--partial_weight_ratio", type=float, default=0.1)
    parser.add_argument("--partial_weight_path", type=str, default=None, help="Directory for partial_weight_q_*.pt")
    parser.add_argument("--skewing_matrix_path", type=str, default=None, help="Path to skewing_matrix.pt")
    parser.add_argument("--alpha", type=float, default=5)
    parser.add_argument("--capacity", type=float, default=1.0)
    parser.add_argument("--budget", type=float, default=0.2)
    parser.add_argument("--infinigen_cache_pool_enabled", type=int, default=0, choices=[0, 1],
                        help="0=disable, 1=enable Infinigen cache pool")
    parser.add_argument("--infinigen_cache_pool_strategy", type=str, default="fixed_k",
                        choices=["fixed_k", "threshold"], help="Infinigen cache pool strategy")
    parser.add_argument("--infinigen_cache_pool_k", type=int, default=4)
    parser.add_argument("--infinigen_cache_pool_cap_ratio", type=float, default=0.75)
    parser.add_argument("--infinigen_local_window", type=int, default=0)
    parser.add_argument("--infinigen_fixed_topk", type=int, default=-1,
                        help="Force fixed Top-K tokens for Infinigen (<=0 to disable)")

    # Quest / Common 参数
    parser.add_argument("--sparse_budget", type=int, default=2048, help="Token budget for Quest/Sparse methods")
    parser.add_argument("--quest_page_size", type=int, default=64, help="Page size for Quest (block size)")
    parser.add_argument("--rank", type=int, default=160)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--minference", action='store_true', default=False)
    parser.add_argument("--name", type=str, default=None, 
                        help="自定义保存结果的文件夹名。如果不填，默认使用 method 名字 (如 infinigen)。")
    return parser.parse_args(args)


if __name__ == '__main__':
    # seed_everything(42)
    args = parse_args()

    model_name = args.model_name
    
    # -------------------------------------------------------------------------
    # 1. 动态替换模型文件 (Symlink) & 重载模块
    # -------------------------------------------------------------------------
    # 简单判断模型类型以确定文件名 (需要根据实际情况调整)
    if "llama" in model_name.lower():
        model_type = "llama"
    elif "qwen" in model_name.lower():
        model_type = "qwen3" # 注意库名可能是 qwen2
    else:
        model_type = "llama" # 默认
    
    print(f"\n⚡ Switching Method to [{args.method}] for model type [{model_type}]...")
    
    target_source_file = None
    if args.method == "kvswap":
        target_source_file = f"modeling_{model_type}_kvswap.py"
    elif args.method == "infinigen":
        target_source_file = f"modeling_{model_type}_ours.py"
    elif args.method == "quest":
        target_source_file = f"modeling_{model_type}_quest.py"
    else:
        # Full Attention / Default
        # 假设有个 orig 备份，或者你可以选择不替换，保持当前状态
        # 这里为了安全，尝试链接回 orig
        if os.path.exists(f"/root/KVswap/LongBench/source/modeling_{model_type}_orig.py"):
            target_source_file = f"modeling_{model_type}_orig.py"
        else:
            print("Notice: No specific file for 'full', using current transformers state.")

    if target_source_file:
        set_symlink(model_type, target_source_file)
        # 必须重载，否则 Python 不会重新读取被 symlink 更改的文件
        reload_modules(model_type)

    # -------------------------------------------------------------------------
    # 2. 延迟导入模型类 (确保加载的是新代码)
    # -------------------------------------------------------------------------
    from models import choose_model_class
    
    max_length = model2maxlen[model_name]
    model_path = model2path[model_name]
    
    # 加载模型
    print(f"Loading Model: {model_name}...")
    LLM = choose_model_class(model_name)
    llm = LLM(model_name=model_path, device='cuda:0')
    llm.tokenizer.pad_token = llm.tokenizer.eos_token
    llm.tokenizer.padding_side = "left"

    # -------------------------------------------------------------------------
    # 3. 准备数据集
    # -------------------------------------------------------------------------
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", 
                    "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        if args.task:
            datasets = [args.task]
        else:
            datasets = ["hotpotqa","narrativeqa","multifieldqa_en","musique","dureader",
                        "gov_report","samsum","passage_retrieval_en","lcc"]
    
    # 结果路径
    if not os.path.exists("results/pred"): os.makedirs("results/pred", exist_ok=True)
    if not os.path.exists("results/pred_e"): os.makedirs("results/pred_e", exist_ok=True)
    
    # 【修改点 1】：更改为你的本地路径
    LOCAL_DATA_ROOT = "/root/autodl-tmp/data" 

    # -------------------------------------------------------------------------
    # 4. 遍历数据集并评测
    # -------------------------------------------------------------------------
    for dataset in datasets:
        file_name = f"{dataset}_e.jsonl" if args.e else f"{dataset}.jsonl"
        local_file_path = os.path.join(LOCAL_DATA_ROOT, file_name)
        
        print(f"\nProcessing Dataset: {dataset}")
        
        # 【修改点 2】：放弃 load_dataset，改用原生 json 读取
        # 这种方式对于 JSONL 文件速度极快，且没有缓存开销
        data_all = []
        try:
            if not os.path.exists(local_file_path):
                print(f"⚠️ File not found: {local_file_path}, skipping...")
                continue
                
            with open(local_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip(): # 避免空行报错
                        data_all.append(json.loads(line))
            print(f"✅ Loaded {len(data_all)} examples from {local_file_path}")
            
        except Exception as e:
            print(f"⚠️ Failed to load dataset {local_file_path}: {e}")
            continue

        folder_name = args.name if args.name else args.method
        # 构造输出路径
        if args.e:
            prefix = f"results/pred_e/{model_name}/{folder_name}"
        else:
            prefix = f"results/pred/{model_name}/{folder_name}"
            
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        out_path = f"{prefix}/{dataset}.jsonl"

        prompt_format = dataset2prompt[dataset]
        max_new_tokens = dataset2maxlen[dataset]
        
        # 取部分样本 (如果指定了 num_examples)
        if args.num_examples > 0:
            data_all = data_all[:args.num_examples]

        # 执行预测
        get_pred(
            llm,
            data_all,
            max_new_tokens,
            prompt_format,
            model_name,
            out_path,
            args,
        )
