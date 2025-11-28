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
    parser.add_argument("--task", type=str, default="hotpotqa")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--cache_dir", type=str, default="/root/.cache/datasets/THUDM___long_bench",
                        help="HF datasets 缓存路径")
    parser.add_argument("--datalen", type=int, default=128*1024, help="The length of the context.")
    parser.add_argument("--method", type=str, default="full",choices=["full", "kvdrive","shadow","quest"],                          \
                        help="Attention method")
    parser.add_argument("--sparse_budget", type=int, default=2048)
    parser.add_argument("--rank", type=int, default=160)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--minference", action='store_true', default=False)
    # parser = parse_attn_args(parser)

    return parser.parse_args(args)


def get_pred(llm, data, max_new_tokens, prompt_format, model_name, out_path, args):
    for json_obj in tqdm(data):
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

    LLM = choose_model_class(model_name)
    llm = LLM(model_name=model_path, batch_size=batch_size, device='cuda:0', max_length=datalen+2048,
              attn_mode=attn_mode, dtype=dtype, sparse_budget=sparse_budget, rank=rank, chunk_size=chunk_size,
              minference=minference, only_gpu=only_gpu)
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

    for dataset in datasets:
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test',cache_dir=args.cache_dir)

            prefix = f"results/pred_e/{model_name}/{attn_mode}"
            if not os.path.exists(prefix):
                os.makedirs(prefix)
            out_path = f"{prefix}/{dataset}.jsonl"
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test', cache_dir=args.cache_dir)

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