import os
import sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# ------------------- ⬇️ 保持你原有的路径添加逻辑 ⬇️ -------------------
# 告诉 Python 去哪里寻找 'infinigen' 或 'quest' 包
# 假设它们在 'SparseCache' 或 'quest' 目录下
sys.path.insert(0, "/root/sparse-load/quest") 
# ------------------- ⬆️ 保持你原有的路径添加逻辑 ⬆️ -------------------

os.environ.setdefault("HF_DATASETS_ALLOW_LOCAL_SCRIPT", "1")
os.environ["HF_DATASETS_OFFLINE"] = "1"
from datasets import load_dataset
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig, 
    LlamaConfig
)
from tqdm import tqdm
import numpy as np
import random
import argparse
import importlib

# 尝试导入 quest，如果在路径中
try:
    import quest.utils
except ImportError:
    pass

def reset_kv_state(model):
    """
    强制清空模型中每一层的自定义 KV 缓存状态。
    适配 HuggingFace 原生模型对象。
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        return

    for layer in layers:
        if hasattr(layer.self_attn, "compressed_k_cache"):
            layer.self_attn.compressed_k_cache = None

def set_symlink(model_type, fname):
    model_path = "/root/KVswap/transformers/src/transformers/models/" + model_type
    linker_path = os.path.realpath("/root/KVswap/LongBench/source/" + fname)
    if not os.path.exists(linker_path):
        print(f"No file exists at {linker_path}")
        exit(0)
    if not os.path.exists(model_path):
        print(f"No file exists at {model_path}")
        exit(0)
    curr_dir = os.getcwd()
    os.chdir(model_path)
    if os.path.exists(f'modeling_{model_type}.py'):
        cmd = f"rm modeling_{model_type}.py"
        os.system(cmd)
    cmd = f"ln -s {linker_path} modeling_{model_type}.py"
    os.system(cmd)
    os.chdir(curr_dir)

def parse_args(cmd_args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        type=str,
        default="longchat-v1.5-7b-32k",
        choices=[
            "llama-3-8b-inst-262k",
            "llama2-7b-chat-4k",
            "longchat-v1.5-7b-32k",
            "vicuna-v1.5-7b-16k",
            "mistral-7b-inst",
            "llama-3-8b-inst-64k",
            "llama-3-8b-inst-1048k",
            "llama-3.1-8b",
            "mistral-7b-instruct-v0.2-local",
            "llama-2-7b-inst-32k",
            "Qwen3-8B"
        ],
    )
    ap.add_argument("--name", type=str, default="default")
    ap.add_argument('--model-type', type=str, default='llama')
    ap.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    ap.add_argument("--arkvale", action="store_true", help="Enable ArkVale")
    ap.add_argument("--page-size", type=int, default=32)
    ap.add_argument("--page-budgets", type=int, default=128)
    ap.add_argument("--n-unlimited-layers", type=int, default=2)
    ap.add_argument("--n-max-bytes", type=int, default=40 * (1 << 30))
    ap.add_argument("--n-max-cpu-bytes", type=int, default=80 * (1 << 30))
    ap.add_argument("--page-topks", type=int, default=32)
    ap.add_argument("--n-win-pages", type=int, default=2)
    ap.add_argument("--n-sink-pages", type=int, default=1)
    ap.add_argument("--use-3-stages-gen", action="store_true")
    # InfiniGen
    ap.add_argument('--infinigen', action='store_true')
    ap.add_argument('--quest', action='store_true', help="Enable Quest Sparse Attention")
    ap.add_argument('--kvswap', action='store_true', help="Enable KVSwap Sparse Attention")
    ap.add_argument('--kvswap-group-size', type=int, default=4, help="KVSwap Group Size (G)")
    ap.add_argument('--kvswap-topk', type=int, default=100, help="KVSwap Top-K Groups (M)")
    ap.add_argument("--partial_weight_ratio", type=float, default=0.1)
    ap.add_argument("--partial_weight_path", type=str)
    ap.add_argument("--skewing_matrix_path", type=str)
    ap.add_argument("--alpha",type=float, default=5)
    ap.add_argument("--capacity",type=float, default=1.0)
    ap.add_argument("--budget",type=float, default=0.2)
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"],
    )
    args = ap.parse_args(cmd_args)
    args.e = False
    if args.page_budgets is not None and args.page_budgets < 0:
        args.page_budgets = None
    return args


def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif ("llama2" in model_name) or ("llama-2" in model_name):
        sys_prompt = "You are a helpful, respectful, and honest assistant. Always answer as concisely as possible."
        bos = getattr(tokenizer, "bos_token", None) or ""
        prompt = f"{bos}[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
    elif "xgen" in model_name:
        header = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def get_pred(
    model,
    tokenizer,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    device,
    model_name,
    out_path,
    use_3_stages_gen=False,
    args=None, # [修改] 传入 args 以获取 quest 配置
):
    start = 0
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            start = len(list(f))
    if start >= len(data):
        return
    print(out_path, start)

    if use_3_stages_gen:
        from arkvale.adapter import generate
        generate.enable_3_stages_gen()
        delim = "1145141919810"
        prompt_format, q_prompt_format = prompt_format.split("{context}")
        prompt_format = prompt_format + "{context}" + delim + q_prompt_format

    data_ = data[start:]
    for json_obj in tqdm(data_):
        reset_kv_state(model)
        torch.cuda.empty_cache()
        
        prompt = prompt_format.format(**json_obj)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
            
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat(tokenizer, prompt, model_name)

        if use_3_stages_gen:
            prompt, q_prompt = prompt.split(delim)
            q_input_ids = tokenizer(q_prompt, truncation=False, return_tensors="pt").to(device)["input_ids"]
            generate.reset_q_input_ids(q_input_ids)

        if "chatglm3" in model_name:
            assert False
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)

        context_length = input.input_ids.shape[-1]
        cur_max_gen = max_gen
        if use_3_stages_gen:
            context_length += q_input_ids.shape[-1]
            cur_max_gen += q_input_ids.shape[-1]

        # ========== [新增] Quest Controller 初始化 ==========
        iController = None
        if args is not None and args.quest:
            # 确保 token_budget 至少够 current context + generation
            # Quest 通常需要 page_budget * page_size
            token_budget = args.page_budgets * args.page_size if args.page_budgets else 4096
            
            iController = quest.utils.InferenceController(
                page_size=args.page_size,
                token_budget=token_budget,
            )
        # ==================================================

        generation_kwargs = {
            **input,
            "max_new_tokens": cur_max_gen,
            "num_beams": 1,
            "do_sample": False,
            "temperature": 1.0,
            "pad_token_id": tokenizer.eos_token_id,
        }

        if dataset == "samsum":
            generation_kwargs["min_length"] = context_length + 1
            generation_kwargs["eos_token_id"] = [
                tokenizer.eos_token_id,
                tokenizer.encode("\n", add_special_tokens=False)[-1],
            ]
        else:
            generation_kwargs["repetition_penalty"] = 1.1
            generation_kwargs["no_repeat_ngram_size"] = 3

        # ========== [修改] 传入 iController 到 generate ==========
        # 如果 args.quest 为 True，modeling_llama_quest.py 会接收这个参数
        if iController is not None:
            generation_kwargs["iController"] = iController
        # =======================================================

        output = model.generate(**generation_kwargs)[0]

        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump(
                {
                    "length": json_obj["length"],
                    "pred": pred,
                    "answers": json_obj["answers"],
                    "all_classes": json_obj["all_classes"],
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")

        if use_3_stages_gen:
            generate.reset_q_input_ids()
            
        # Quest 清理 (如果需要)
        if iController is not None:
            del iController

    if use_3_stages_gen:
        generate.disable_3_stages_gen()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path, model_name, device, args):
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    config_path = os.path.join(path, 'config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)

    # 如果是 Llama 模型，直接使用 AutoConfig 或 LlamaConfig
    config = AutoConfig.from_pretrained(path)

    if args.arkvale:
        from arkvale import adapter
        model = AutoModelForCausalLM.from_pretrained(
            path, config=config, device_map=device, torch_dtype=torch.float16, attn_implementation="flash_attention_2"
        )
        adapter.enable_arkvale(
            model, dtype=torch.float16, device=device, page_size=32,
            page_budgets=4096 // 32, page_topks=32,
            n_max_bytes=5 * (1 << 30), n_max_cpu_bytes=60 * (1 << 30),
        )
    else:
        print(f"Loading {args.model_type}...")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        # 注意：modeling_llama_quest.py 的 LlamaAttention 使用了 transformers.integrations.use_kernel_forward_from_hub
        # 通常这不需要 attn_implementation="flash_attention_2"，除非你的 quest 代码是基于 FA2 修改的
        # 为了兼容性，这里保持原样，但 Quest 往往有自己的 CUDA kernel
        model = AutoModelForCausalLM.from_pretrained(
            path,
            config=config,
            torch_dtype=dtype,
            device_map=device,
            # attn_implementation="flash_attention_2" # 视 Quest 实现而定，可能需要改为 eager 或 sdpa
            attn_implementation="sdpa"
        )
    model = model.eval()

    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    seed_everything(42)
    args = parse_args()

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model
    
    # ========== [修改] Symlink 逻辑 ==========
    if args.quest:
        print("Using Quest model implementation...")
        # 假设你上面的 modeling_llama_quest.py 文件名就在 benchmark/source 目录下
        set_symlink(args.model_type, "modeling_llama_quest.py")
    elif args.infinigen:
        print("Using InfiniGen model implementation...")
        set_symlink(args.model_type, f"modeling_{args.model_type}_ours.py")
    elif args.kvswap:  # <--- [新增]
        print("Using KVSwap model implementation...")
        # 注意：这里假设你把上一轮修改后的代码保存为了 modeling_llama_kvswap.py
        # 并且放在了 /root/sparse-load/SparseCache/accuracy/benchmark/source/ 目录下
        set_symlink(args.model_type, f"modeling_{args.model_type}_kvswap.py")
    else:
        print("Using Original model implementation...")
        set_symlink(args.model_type, f"modeling_{args.model_type}_orig.py")
    # =======================================
    
    # Force reload modules
    if f'transformers.models.{args.model_type}' in sys.modules:
        print(f"Reloading transformers.models.{args.model_type} module...")
        modules_to_remove = [k for k in sys.modules.keys() if f'transformers.models.{args.model_type}' in k]
        for module_name in modules_to_remove:
            del sys.modules[module_name]
    
    max_length = model2maxlen[model_name]
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device, args)
    
    # InfiniGen 参数注入 (保持原样)
    if args.infinigen:
        if args.model_type == "opt":
            for layer in range(len(model.model.decoder.layers)):
                model.model.decoder.layers[layer].self_attn.partial_weight_ratio = args.partial_weight_ratio
                model.model.decoder.layers[layer].self_attn.partial_weight_q = torch.load(args.partial_weight_path + "/partial_weight_q_" + str(layer) + ".pt")
                model.model.decoder.layers[layer].self_attn.alpha = args.alpha
                model.model.decoder.layers[layer].self_attn.capacity = args.capacity
                model.model.decoder.layers[layer].self_attn.budget = args.budget
        elif args.model_type in ["llama", "qwen3"]:
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
    if args.kvswap:
        print(f"Injecting KVSwap params: GroupSize={args.kvswap_group_size}, TopK={args.kvswap_topk}")
        
        # 遍历所有层，修改 Attention 模块的属性
        # 注意：这里假设是 Llama 结构，layers 位于 model.model.layers
        layers = model.model.layers 
        
        for i, layer in enumerate(layers):
            # 获取 attention 模块 (Self Attention)
            attn = layer.self_attn
            
            # 只有当该层成功加载了 KVSwap (kvswap_enabled=True) 时才修改
            # 或者你可以强制修改，取决于你的需求
            if hasattr(attn, "kvswap_enabled") and attn.kvswap_enabled:
                attn.kv_group_size = args.kvswap_group_size
                attn.kv_top_k_groups = args.kvswap_topk
            else:
                # 这种情况可能是该层没有对应的投影矩阵文件
                print(f"Warning: Layer {i} does not have KVSwap enabled (maybe missing projection file).")
    # ==========================================
    datasets = args.datasets
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))


    # 设置你的本地数据根目录
    base_data_dir = "/root/autodl-tmp/data"

    for dataset in datasets:
        if args.e:
            # 1. 构造 LongBench-E 的具体文件路径 (例如 hotpotqa_e.jsonl)
            file_path = os.path.join(base_data_dir, f"{dataset}_e.jsonl")
            
            # 2. 检查文件是否存在，防止路径错误导致崩溃
            if not os.path.exists(file_path):
                print(f"Warning: File not found {file_path}, skipping...")
                continue

            # 3. 使用 "json" 加载器直接读取文件
            # data_files={"test": ...} 强制将其作为 test 集加载，配合 split="test"
            data = load_dataset("json", data_files={"test": file_path}, split="test")
            
            os.makedirs(f"pred_e/{model_name}/{args.name}", exist_ok=True)
            out_path = f"pred_e/{model_name}/{args.name}/{dataset}.jsonl"
        else:
            # 1. 构造标准版 LongBench 的具体文件路径
            file_path = os.path.join(base_data_dir, f"{dataset}.jsonl")
            
            if not os.path.exists(file_path):
                print(f"Warning: File not found {file_path}, skipping...")
                continue
                
            # 3. 使用 "json" 加载器直接读取文件
            data = load_dataset("json", data_files={"test": file_path}, split="test")
            
            os.makedirs(f"pred/{model_name}/{args.name}", exist_ok=True)
            out_path = f"pred/{model_name}/{args.name}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        # ... (后续调用 get_pred 的代码保持不变) ...
        get_pred(
            model,
            tokenizer,
            list(data), # 这里 data 已经是加载好的 Dataset 对象
            max_length,
            max_gen,
            prompt_format,
            dataset,
            device,
            model_name,
            out_path,
            use_3_stages_gen=args.use_3_stages_gen,
            args=args
        )