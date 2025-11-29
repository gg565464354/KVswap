# 文件名: /root/KVswap/LongBench/models.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class GenericHFModel:
    def __init__(self, model_name, device='cuda', **kwargs):
        print(f"Loading Generic HF Model from: {model_name}")
        self.device = device
        
        # 1. 加载 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            padding_side="left" # 生成任务通常左填充
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. 加载 Model (这会自动调用你修改过的 transformers)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            # device_map=device,
            use_cache=True # KVSwap 必须开启 cache
        )
        print(f"Moving model to {device}...")
        self.model.to(device)
        self.model.eval()

    def generate(self, input_ids, gen_len, **kwargs):
        """
        封装 generate 函数，适配 pred.py 的调用格式
        """
        kwargs.pop('verbose', None)      # HF generate 不支持 verbose
        kwargs.pop('top_p', None)        # 我们强制 greedy，剔除外部传入的 top_p
        kwargs.pop('temperature', None)  # 我们强制 greedy，剔除外部传入的 temperature
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=gen_len,
                do_sample=False,        # Greedy Search
                temperature=None,       
                top_p=None,             
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )
            
            # 截取生成的对应部分（去掉 Input Prompt）
            generated_tokens = output_ids[:, input_ids.shape[1]:]
            
            # 解码为字符串列表
            outputs = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
        return outputs

def choose_model_class(model_name):
    """
    工厂函数：为了兼容 pred.py 的调用方式
    直接返回我们定义的 GenericHFModel 类
    """
    return GenericHFModel