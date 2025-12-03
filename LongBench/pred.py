import os
import sys
from datasets import load_dataset
import torch
import json
from tqdm import tqdm
import numpy as np
import random
import argparse
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJECT_ROOT)
from models import choose_model_class
# from model_hub import LlamaModel, QwenModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from config import generate_config, parse_attn_args

model2path = json.load(open("config/model2path.json", "r"))
model2maxlen = json.load(open("config/model2maxlen.json", "r"))
# we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
os.environ["ENABLE_METRICS"] = "0"
os.environ["ENABLE_MONITOR"] = "0"

def reset_kv_state(llm_wrapper):
    """
    强制清空模型中所有 KVSwap 层的压缩缓存状态。
    假设 llm_wrapper.model 是 HuggingFace 的模型实例。
    """
    # 尝试获取底层的 HF 模型
    model = getattr(llm_wrapper, "model", None)
    if model is None:
        return # 如果找不到模型实例，跳过

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        return

    for layer in layers:
        if hasattr(layer.self_attn, "compressed_k_cache"):
            layer.self_attn.compressed_k_cache = None

def configure_kvswap(llm_wrapper, args):
    """
    根据参数配置每一层的 KVSwap 开关和超参数
    """
    model = getattr(llm_wrapper, "model", None)
    if model is None: return

    # 获取层列表 (兼容 Qwen/Llama 结构)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        return

    for layer in layers:
        # 仅当该层成功加载了矩阵 (projection_matrix is not None) 时才开启
        has_matrix = hasattr(layer.self_attn, "projection_matrix") and layer.self_attn.projection_matrix is not None
        
        if args.method == "kvswap" and has_matrix:
            layer.self_attn.kvswap_enabled = True
            # 设置超参数
            layer.self_attn.kv_group_size = args.kv_group_size
            # 如果没指定 top_k，根据 MG=400 自动计算
            if args.kv_top_k_groups == -1:
                layer.self_attn.kv_top_k_groups = 400 // args.kv_group_size
            else:
                layer.self_attn.kv_top_k_groups = args.kv_top_k_groups
            # print("Configured KVSwap: Group Size =", layer.self_attn.kv_group_size, 
            #       ", Top K Groups =", layer.self_attn.kv_top_k_groups)
        else:
            # 如果不是 kvswap 方法，或者这一层没矩阵，强制关闭
            if hasattr(layer.self_attn, "kvswap_enabled"):
                layer.self_attn.kvswap_enabled = False

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    # parser.add_argument("--attn_type", type=str, default="Full_Flash_Attn",                                                     \
    #                     choices=["Full_Flash_Attn", "RetroInfer"],                          \
    #                     help="Attention method")
    # parser.add_argument('--model', type=str, default=None, choices=
    #                     ["llama-3-8b-1048k", "qwen2.5-7b", "llama-3.1-8b", "qwen2.5-72b"])
    # parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"], help="Dtype")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    # parser.add_argument('--task', type=str, required=True, help="task name. work when --e is false")
    # parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--num_examples", type=int, default=-1, help="num of example to evaluate. -1 for all.")

    parser.add_argument("--model_name", type=str, default=None,choices=["llama-3-8b-1048k","Phi-4-mini-instruct","Qwen3-8B"])
    # parser.add_argument("--dataset_name", type=str_to_list, default=["ruler/niah_single_1"])
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--cache_dir", type=str, default="/root/.cache/datasets/THUDM___long_bench",
                        help="HF datasets 缓存路径")
    parser.add_argument("--datalen", type=int, default=128*1024, help="The length of the context.")
    parser.add_argument("--method", type=str, default="full",choices=["full", "kvdrive","shadow","quest", "kvswap"],                          \
                        help="Attention method")
    parser.add_argument("--kv_group_size", type=int, default=4, help="KVSwap Group Size (G)")
    parser.add_argument("--kv_top_k_groups", type=int, default=100, help="KVSwap Top K Groups (M). Set -1 to auto-calc from MG=400")
    parser.add_argument("--sparse_budget", type=int, default=2048)
    parser.add_argument("--rank", type=int, default=160)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--minference", action='store_true', default=False)
    # parser = parse_attn_args(parser)

    return parser.parse_args(args)


def get_pred(llm, data, max_new_tokens, prompt_format, model_name, out_path, args):
    configure_kvswap(llm, args)
    print(f"Method: {args.method}, KVSwap Configured.")
    for json_obj in tqdm(data):
        reset_kv_state(llm)
        prompt = prompt_format.format(**json_obj)

        inputs = llm.tokenizer([prompt], return_tensors="pt", padding=True)
        input_ids = inputs.input_ids
        attention_masks = inputs.attention_mask

        # attn_config = generate_config(
        #     model2path[model_name], 
        #     input_ids.shape[1], 
        #     attn_type,
        #     budget_ratio=args.budget_ratio,
        #     estimate_ratio=args.estimate_ratio,
        # )

        output = llm.generate(input_ids.to(llm.device), gen_len=max_new_tokens, verbose=False, top_p=1.0, temperature=0.0)

        # output = llm.tokenizer.batch_decode(out, skip_special_tokens=True)

        torch.cuda.empty_cache()
                
        print("Chunked generation:", output[0][:50])

        pred = output[0]

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


# def load_model(model_path, max_len, dtype, device):
#     if 'Llama' in model_path:
#         llm = LlamaModel(model_path,
#             max_length=max_len,
#             dtype=dtype,
#             device_map=device)
#     elif 'Qwen' in model_path:
#         llm = QwenModel(model_path,
#             max_length=max_len,
#             dtype=dtype,
#             device_map=device)
#     else:
#         raise ValueError(f"Unsupported model: {model_path}")

#     llm.tokenizer.pad_token = llm.tokenizer.eos_token
#     llm.tokenizer.padding_side = "left"
    
#     return llm


if __name__ == '__main__':
    # seed_everything(42)
    args = parse_args()

    # num_examples = args.num_examples
    # attn_type = args.attn_type
    # model_name = args.model # not hf model path
    # device = args.device
    # dtype = torch.float16 if args.dtype == 'fp16' else torch.bfloat16
    model_name = args.model_name
    max_length = model2maxlen[model_name]
    model_path = model2path[model_name]
    
    batch_size = args.batch_size
    # dataset_names = args.dataset_name
    num_samples = args.num_samples
    datalen = args.datalen
    sparse_budget = args.sparse_budget
    dtype = torch.bfloat16
    rank = args.rank
    chunk_size = args.chunk_size
    minference = args.minference
    only_gpu = True
    attn_mode = args.method
    num_examples = args.num_examples

    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)

    from models import choose_model_class
    LLM = choose_model_class(model_name)
    llm = LLM(model_name=model_path, device='cuda:0')
    llm.tokenizer.pad_token = llm.tokenizer.eos_token
    llm.tokenizer.padding_side = "left"
    # llm = load_model(model_path, max_length, dtype, device)

    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        if args.task:
            datasets = [args.task]
        else:
            datasets = ["narrativeqa","multifieldqa_en","hotpotqa","musique","dureader","gov_report","samsum","passage_retrieval_en","lcc"]
        # datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
        #             "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
        #             "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    
    # predict on each dataset
    if not os.path.exists("results/pred"):
        os.makedirs("results/pred")
    if not os.path.exists("results/pred_e"):
        os.makedirs("results/pred_e")

    LOCAL_DATA_ROOT = "/root/KVswap/data"
    for dataset in datasets:
        file_name = f"{dataset}_e.jsonl" if args.e else f"{dataset}.jsonl"
        local_file_path = os.path.join(LOCAL_DATA_ROOT, file_name)
        print(f"Loading local dataset: {local_file_path}")

        # 2. 加载本地数据
        # 注意：本地加载 JSONL 时，默认 split 通常是 "train"，即使它是测试数据
        data = load_dataset("json", data_files=local_file_path, split="train")

        # 3. 设置结果保存路径 (保持原有逻辑)
        if args.e:
            prefix = f"results/pred_e/{model_name}/{attn_mode}"
        else:
            prefix = f"results/pred/{model_name}/{attn_mode}"

        if not os.path.exists(prefix):
            os.makedirs(prefix)
        out_path = f"{prefix}/{dataset}.jsonl"

        prompt_format = dataset2prompt[dataset]
        max_new_tokens = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        data_all = data_all[:num_examples] if num_examples > 0 else data_all

        get_pred(
            llm,
            data_all,
            max_new_tokens,
            prompt_format,
            model_name,
            out_path,
            args,
        )