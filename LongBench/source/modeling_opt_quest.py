# coding=utf-8
# Copyright 2022 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OPT model."""

from math import inf
from typing import Callable, Optional, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import json
import numpy as np

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, is_torch_flex_attn_available, logging
from ...utils.deprecation import deprecate_kwarg
from .configuration_opt import OPTConfig


if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask

    from ...integrations.flex_attention import make_flex_block_causal_mask


logger = logging.get_logger(__name__)


class OPTLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(
        self,
        attention_mask: torch.LongTensor,
        past_key_values_length: int = 0,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""

        if position_ids is None:
            position_ids = torch.cumsum(attention_mask, dim=1)
            position_ids = (position_ids * attention_mask - 1).long()
            # cut positions if `past_key_values_length` is > 0
            position_ids = position_ids[:, past_key_values_length:]

        return super().forward(position_ids + self.offset)


# Copied from transformers.models.siglip.modeling_siglip.eager_attention_forward
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    layer_idx: Optional[int] = None,
    **kwargs,
):
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
    
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    # if layer_idx == 30 and query.shape[2] != 1:
    #     print(f"attn_weights: {attn_weights[0, 0]}")
    #     # 保存第0层第0头的attn_weights到txt文件，方便直接用文本方式查看
    #     try:
    #         aw_np = attn_weights[0, 0].detach().cpu().numpy()
    #         with open("/opt/conda/envs/trans/lib/python3.12/site-packages/transformers/attn_weights_layer30_head0.txt", "w") as f:
    #             for row in aw_np:
    #                 f.write(" ".join(str(float(x)) for x in row) + "\n")
    #         print("已保存第0层第0头的attn_weights到 attn_weights_layer0_head0.txt")
    #     except Exception as _e:
    #         print(f"保存attn_weights失败: {_e}")


    # if layer_idx == 0:
    #     # 保存第一个batch第一个head的attn_weights到txt文件，方便直接用文本方式查看
    #     print(f"attn_weights: {attn_weights[0, 0]}")
    #     first_head_weights = attn_weights[0, 0].detach().cpu().numpy()
    #     # 保存为txt格式
    #     with open("attn_weights_layer0_head0.txt", "w") as f:
    #         for row in first_head_weights:
    #             f.write(" ".join([str(x) for x in row]) + "\n")
    #     print("已保存第0层第0头的attn_weights到 attn_weights_layer0_head0.txt")
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class OPTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: OPTConfig,
        layer_idx: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.dropout = config.attention_dropout
        self.enable_bias = config.enable_bias
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.head_dim = self.embed_dim // self.num_heads
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Optional[tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, _ = hidden_states.size()

        # Scaling is susceptible to floating point arithmetics' inprecisions
        # which can lead to different results (this is dependent from model
        # to model, e.g. whisper is one such case). We therefore keep the
        # original order of scaling to follow the original implementation
        # and enforce no scaling (1.0) in the attention call below.
        query_states = self.q_proj(hidden_states) * self.scaling
        query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if past_key_values is not None:
            # save all key/value_states to cache to be re-used for fast auto-regressive generation
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )

        attention_interface: Callable = eager_attention_forward

        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and output_attentions:
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        
        block_size = 4
        ratio = 0.2      # 计算能整除的部分和余数部分（使用key_states的长度）
        key_seq_len = key_states.shape[2]  # key_states: (bsz, num_heads, src_len, head_dim)
        num_complete_blocks = key_seq_len // block_size
        remainder = key_seq_len % block_size
        
        key_max_list = []
        key_min_list = []
        
        if num_complete_blocks > 0:
            # 处理能完整分block的部分
            complete_len = num_complete_blocks * block_size
            key_complete = key_states[:, :, :complete_len, :]  # (bsz, num_heads, complete_len, head_dim)
            
            # reshape为block格式
            key_blocked = key_complete.reshape(bsz, self.num_heads, num_complete_blocks, block_size, self.head_dim)
            
            # 计算每个block的最大值和最小值
            key_max_blocks, _ = torch.max(key_blocked, dim=3)  # (bsz, num_heads, num_complete_blocks, head_dim)
            key_min_blocks, _ = torch.min(key_blocked, dim=3)  # (bsz, num_heads, num_complete_blocks, head_dim)
            
            key_max_list.append(key_max_blocks)
            key_min_list.append(key_min_blocks)
        
        if remainder > 0:
            # 处理剩余部分（不足一个block的部分）
            key_remainder = key_states[:, :, -remainder:, :]  # (bsz, num_heads, remainder, head_dim)
            
            # 对剩余部分计算最大值和最小值
            key_max_remainder, _ = torch.max(key_remainder, dim=2, keepdim=True)  # (bsz, num_heads, 1, head_dim)
            key_min_remainder, _ = torch.min(key_remainder, dim=2, keepdim=True)  # (bsz, num_heads, 1, head_dim)
            
            key_max_list.append(key_max_remainder)
            key_min_list.append(key_min_remainder)
        
        # 拼接所有部分
        key_max = torch.cat(key_max_list, dim=2)  # (bsz, num_heads, total_segments, head_dim)
        key_min = torch.cat(key_min_list, dim=2)  # (bsz, num_heads, total_segments, head_dim)

        # 将query的head_dim维度与key的head_dim维度逐元素相乘
        query_expanded = query_states.unsqueeze(3)  # (bsz, num_heads, tgt_len, 1, head_dim)
        key_max_expanded = key_max.unsqueeze(2)  # (bsz, num_heads, 1, total_segments, head_dim)
        key_min_expanded = key_min.unsqueeze(2)  # (bsz, num_heads, 1, total_segments, head_dim)
        
        # 逐元素相乘
        query_key_max = query_expanded * key_max_expanded  # (bsz, num_heads, tgt_len, total_segments, head_dim)
        query_key_min = query_expanded * key_min_expanded  # (bsz, num_heads, tgt_len, total_segments, head_dim)
        
        # 先取最大值，再在head_dim维度上求和
        query_key_combined = torch.max(query_key_max, query_key_min)  # (bsz, num_heads, tgt_len, total_segments, head_dim)
        attnweightmax = torch.sum(query_key_combined, dim=-1)  # (bsz, num_heads, tgt_len, total_segments)
        
        
        attnweightmax_last = attnweightmax[:, :, -1, :]   # (bsz, num_heads, total_segments)
        
        # 额外取倒数第二个块的最后一个token对应的attnweightmax
        # 计算倒数第二个块的最后一个token位置
        if tgt_len > block_size:
            # 倒数第二个块的起始位置
            second_last_block_start = ((tgt_len - 1) // block_size - 1) * block_size
            # 倒数第二个块的最后一个token位置
            second_last_block_end = min(second_last_block_start + block_size - 1, tgt_len - 1)
            attnweightmax_second_last = attnweightmax[:, :, second_last_block_end, :]  # (bsz, num_heads, total_segments)
        else:
            # 如果tgt_len <= block_size，则使用最后一个query
            attnweightmax_second_last = attnweightmax[:, :, -1, :]
        
        # 将两个attention权重相加 # (bsz, num_heads, total_segments)  # (bsz, num_heads, total_segments)
       
        # 计算要选择的block数量
        total_segments = attnweightmax_last.shape[2]
        num_selected = max(1, int(total_segments * ratio))  # 至少选择1个
        
        # 选择attention权重最大的前num_selected个blocks
        top_values1, top_indices1 = torch.topk(attnweightmax_last , num_selected, dim=2)
        top_values2, top_indices2 = torch.topk(attnweightmax_second_last , num_selected, dim=2)
        # top_values: (bsz,num_heads, num_selected) - 最大的权重值
        # top_indices: (bsz,num_heads, num_selected) - 对应的block索引
        
        # 创建mask来标识被选中的positions
        selected_mask = torch.zeros(bsz,self.num_heads, key_seq_len, dtype=torch.bool, device=key_states.device)  # (bsz, key_seq_len)

        for batch_idx in range(bsz):
            for head_idx in range(self.num_heads):
                indices1 = top_indices1[batch_idx, head_idx].tolist()
                indices2 = top_indices2[batch_idx, head_idx].tolist()
                set1 = set(int(i) for i in indices1)
                set2 = set(int(i) for i in indices2)
                union_indices = sorted(list({0} | set1  ))
                
                # 计算set1中有多少在set2中
                overlap_count = len(set1 )  # 交集大小
                overlap_ratio = overlap_count / max(1, len(set1))  # set1中在set2中的比例

                # 新建一个文件夹，命名为rte-static-ratio{ratio}
                # import os
                # import json
                # folder_name = f"rte-static-ratio{ratio}"
                # if not os.path.exists(folder_name):
                #     os.makedirs(folder_name)
                
                # # 保存overlap_ratio到JSONL文件，head_id = 32*layer_id + head_idx
                # try:
                #     layer_idx = int(self.layer_idx) if hasattr(self, 'layer_idx') and self.layer_idx is not None else -1
                #     head_id = 32 * layer_idx + head_idx
                #     filename = f"{folder_name}/head{head_id}.jsonl"
                #     with open(filename, "a", encoding="utf-8") as jf:
                #         jf.write(json.dumps({
                #             "overlap_ratio": float(overlap_ratio)
                #         }, ensure_ascii=False) + "\n")
                # except Exception as e:
                #     pass
                for block_idx in union_indices:
                    if block_idx < num_complete_blocks:
                        # 处理完整的blocks
                        start_pos = block_idx * block_size
                        end_pos = start_pos + block_size
                        selected_mask[batch_idx, head_idx, start_pos:end_pos] = True
                
                # 无论是否被选中，强制选中最后一个块（完整或不完整）
                if remainder > 0:
                    # 存在不完整的最后块
                    start_pos = num_complete_blocks * block_size
                    end_pos = start_pos + remainder
                    if start_pos < key_seq_len:
                        selected_mask[batch_idx, head_idx, start_pos:end_pos] = True
                else:
                    # 最后一个块为完整块
                    if num_complete_blocks > 0:
                        start_pos = (num_complete_blocks - 1) * block_size
                        end_pos = start_pos + block_size
                        if start_pos < key_seq_len:
                            selected_mask[batch_idx, head_idx, start_pos:end_pos] = True
        
        # if self.layer_idx == 16 and tgt_len != 1:
        #     print(f"selected_mask: {selected_mask}")
        #     # 保存selected_mask到txt文件
        #     selected_mask_np = selected_mask.detach().cpu().numpy()
        #     with open("/opt/conda/envs/trans/lib/python3.12/site-packages/transformers/selected_mask_layer16.txt", "w") as f:
        #         f.write(f"selected_mask shape: {selected_mask_np.shape}\n")
        #         f.write(f"batch_size: {bsz}, num_heads: {self.num_heads}, key_seq_len: {key_seq_len}\n")
        #         f.write("="*50 + "\n")
        #         for batch_idx in range(bsz):
        #             f.write(f"Batch {batch_idx}:\n")
        #             for head_idx in range(self.num_heads):
        #                 f.write(f"  Head {head_idx}: ")
        #                 head_mask = selected_mask_np[batch_idx, head_idx]
        #                 # 将True/False转换为1/0，并用空格分隔
        #                 mask_str = " ".join(["1" if x else "0" for x in head_mask])
        #                 f.write(mask_str + "\n")
        #             f.write("\n")
        #     print("已保存selected_mask到 selected_mask_layer16.txt")
        
        # 强制选中remainder部分的所有tokens（如果存在）
        if remainder > 0:
            remainder_start = num_complete_blocks * block_size
            selected_mask[:, :, remainder_start:] = True  # (bsz, num_heads, remainder_tokens)
        # 根据selected_mask设置attention_mask，适用于prefill阶段
        # attention_mask形状需要是(bsz, num_heads, seq_len, seq_len)
        if attention_mask is not None and tgt_len > 1 and self.layer_idx>1:  # prefill阶段
            # 创建新的attention_mask: (bsz, num_heads, tgt_len, key_seq_len)
            large_negative = torch.tensor(-65504, dtype=query_states.dtype, device=query_states.device)
            
            if attention_mask.dim() == 4:  # (bsz, 1, tgt_len, key_seq_len)
                # 扩展到每个头: (bsz, num_heads, tgt_len, key_seq_len)
                expanded_attention_mask = attention_mask.expand(bsz, self.num_heads, tgt_len, key_seq_len).clone()
                
                # 对于每个头，根据selected_mask设置mask
                for head_idx in range(self.num_heads):
                    # 获取当前头的selected_mask: (bsz, key_seq_len)
                    head_selected_mask = selected_mask[:, head_idx, :]  # (bsz, key_seq_len)
                    
                    # 对未被选中的key位置设置为large_negative
                    # 扩展head_selected_mask到(bsz, tgt_len, key_seq_len)
                    head_mask_expanded = head_selected_mask.unsqueeze(1).expand(bsz, tgt_len, key_seq_len)
                    
                    # 应用mask：未选中的位置设为large_negative
                    expanded_attention_mask[:, head_idx, :, :] = torch.where(
                        head_mask_expanded, 
                        expanded_attention_mask[:, head_idx, :, :], 
                        large_negative
                    )
                
                attention_mask = expanded_attention_mask
                
                # 保存不同head的attention_mask到txt文件（仅在第16层且prefill阶段）
                # if self.layer_idx == 16 and tgt_len > 1:
                #     am_np = attention_mask.detach().cpu().numpy()
                #     with open("/opt/conda/envs/trans/lib/python3.12/site-packages/transformers/attention_mask_layer16.txt", "w") as f:
                #         f.write(f"attention_mask shape: {am_np.shape}\n")
                #         f.write(f"batch_size: {bsz}, num_heads: {self.num_heads}, tgt_len: {tgt_len}, key_seq_len: {key_seq_len}\n")
                #         f.write("="*60 + "\n")
                #         for batch_idx in range(bsz):
                #             f.write(f"Batch {batch_idx}:\n")
                #             for head_idx in range(self.num_heads):
                #                 f.write(f"  Head {head_idx}:\n")
                #                 head_attn = am_np[batch_idx, head_idx]  # (tgt_len, key_seq_len)
                #                 for tgt_idx in range(tgt_len):
                #                     f.write(f"    Query {tgt_idx}: ")
                #                     row_vals = head_attn[tgt_idx]
                #                     line = []
                #                     for val in row_vals:
                #                         if not np.isfinite(val):
                #                             if np.isneginf(val):
                #                                 line.append("-inf")
                #                             elif np.isposinf(val):
                #                                 line.append("inf")
                #                             else:
                #                                 line.append("nan")
                #                         else:
                #                             line.append(f"{float(val):.3f}")
                #                     f.write(" ".join(line) + "\n")
                #                 f.write("\n")
                #             f.write("\n")
                #     print("已保存attention_mask到 attention_mask_layer16.txt")
                
        
        
        # 修改attention_mask：对未被选中的位置进行掩码
        # key的token数量为1
        # 只在decode阶段（生成时）应用稀疏attention
        # if attention_mask is not None and tgt_len == 1:
        #     # attention_mask使用数值标识：0表示允许，-inf表示掩码
        #     # 对未被选中的位置设置为负无穷
        #     large_negative = torch.tensor(-1e9, dtype=query_states.dtype, device=query_states.device)
        #     zero_value = torch.tensor(0.0, dtype=query_states.dtype, device=query_states.device)
            
        #     if attention_mask.dim() == 2:  # (bsz, seq_len)
        #         # 对于2D mask，我们需要将selected_mask从(bsz, num_heads, key_seq_len)转换为(bsz, key_seq_len)
        #         # 使用any操作：如果任何一个头选择了某个位置，则该位置被选中
        #         selected_mask_2d = torch.any(selected_mask, dim=1)  # (bsz, key_seq_len)
        #         attention_mask = torch.where(selected_mask_2d, attention_mask, large_negative)
        #         # 强制确保remainder部分设置为0（允许关注）
        #         if remainder > 0:
        #             remainder_start = num_complete_blocks * block_size
        #             attention_mask[:, remainder_start:] = torch.tensor(-0.0, dtype=torch.float32, device=query_states.device)
        #     elif attention_mask.dim() == 4:  # (bsz, 1, query_len, key_len)
        #         # 对key维度（最后一维）应用mask
        #         # 将selected_mask从(bsz, num_heads, key_seq_len)转换为(bsz, 1, 1, key_seq_len)
        #         selected_mask_key = torch.any(selected_mask, dim=1).unsqueeze(1).unsqueeze(1)  # (bsz, 1, 1, key_seq_len)
        #         attention_mask = torch.where(selected_mask_key, attention_mask, large_negative)
        #         # 强制确保remainder部分设置为0（允许关注）
        #         if remainder > 0:
        #             remainder_start = num_complete_blocks * block_size
        #             attention_mask[:, :, :, remainder_start:] = torch.tensor(-0.0, dtype=torch.float32, device=query_states.device)
        # if self.layer_idx == 16:
        #     print(f"attention_mask: {attention_mask}")
        # 调试：打印第一个头的attention mask
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=1.0,
            layer_idx=self.layer_idx,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


class OPTDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: OPTConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.embed_dim = config.hidden_size

        self.self_attn = OPTAttention(config=config, layer_idx=layer_idx)

        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
        )
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=config.enable_bias)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=config.enable_bias)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`, *optional*): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_values (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence..
        """

        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            position_ids=position_ids,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


@auto_docstring
class OPTPreTrainedModel(PreTrainedModel):
    config: OPTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OPTDecoderLayer"]
    _supports_attention_backend = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()


class OPTDecoder(OPTPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`OPTDecoderLayer`]

    Args:
        config: OPTConfig
    """

    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
        else:
            self.project_in = None

        # Note that the only purpose of `config._remove_final_layer_norm` is to keep backward compatibility
        # with checkpoints that have been fine-tuned before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size, elementwise_affine=config.layer_norm_elementwise_affine
            )
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList([OPTDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.gptj.modeling_gptj.GPTJModel._update_causal_mask
    def _update_causal_mask(
        self,
        attention_mask: Union[torch.Tensor, "BlockMask"],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_compilable_cache = past_key_values.is_compileable if past_key_values is not None else False

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_compilable_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype = input_tensor.dtype
        sequence_length = input_tensor.shape[1]
        if using_compilable_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    # Copied from transformers.models.gptj.modeling_gptj.GPTJModel._prepare_4d_causal_attention_mask_with_cache_position
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.n_positions - 1]`. for padding use -1.

                [What are position IDs?](../glossary#position-ids)
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
                this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
                the complete sequence length.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if input_ids is not None:
            input_ids = input_ids.view(-1, input_ids.shape[-1])

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if attention_mask is None:
            seq_length = past_seen_tokens + inputs_embeds.shape[1]
            attention_mask = torch.ones(inputs_embeds.shape[0], seq_length, device=inputs_embeds.device)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # embed positions
        if position_ids is None:
            # position_ids = cache_position.unsqueeze(0)
            position_ids = torch.cumsum(attention_mask, dim=1)
            position_ids = (position_ids * attention_mask - 1).long()
            # cut positions if `past_seen_tokens` is > 0
            position_ids = position_ids[:, past_seen_tokens:]

        pos_embeds = self.embed_positions(attention_mask, past_seen_tokens, position_ids=position_ids)

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds.to(inputs_embeds.device)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


@auto_docstring
class OPTModel(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.decoder = OPTDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[list[torch.FloatTensor], Cache]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, past_key_values, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        return BaseModelOutputWithPast(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )


class OPTForCausalLM(OPTPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = OPTModel(config)

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[list[torch.FloatTensor], Cache]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForCausalLM

        >>> model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious. I'm just a little bit of a weirdo."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        logits = self.lm_head(outputs[0]).contiguous()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            loss = self.loss_function(
                logits,
                labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    The OPT Model transformer with a sequence classification head on top (linear layer).

    [`OPTForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """
)
class OPTForSequenceClassification(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = OPTModel(config)
        self.score = nn.Linear(config.word_embed_proj_dim, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Union[list[torch.FloatTensor], Cache]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value


@auto_docstring
class OPTForQuestionAnswering(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.model = OPTModel(config)
        self.qa_outputs = nn.Linear(config.word_embed_proj_dim, 2)

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Union[list[torch.FloatTensor], Cache]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[tuple, QuestionAnsweringModelOutput]:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForQuestionAnswering
        >>> import torch

        >>> torch.manual_seed(4)  # doctest: +IGNORE_RESULT
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> # note: we are loading a OPTForQuestionAnswering from the hub here,
        >>> # so the head will be randomly initialized, hence the predictions will be random
        >>> model = OPTForQuestionAnswering.from_pretrained("facebook/opt-350m")

        >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

        >>> inputs = tokenizer(question, text, return_tensors="pt")
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> answer_start_index = outputs.start_logits.argmax()
        >>> answer_end_index = outputs.end_logits.argmax()

        >>> answer_offset = len(tokenizer(question)[0])

        >>> predict_answer_tokens = inputs.input_ids[
        ...     0, answer_offset + answer_start_index : answer_offset + answer_end_index + 1
        ... ]
        >>> predicted = tokenizer.decode(predict_answer_tokens)
        >>> predicted
        ' a nice puppet'
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index).to(logits.device)
            end_positions = end_positions.clamp(0, ignored_index).to(logits.device)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + transformer_outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value


__all__ = [
    "OPTForCausalLM",
    "OPTModel",
    "OPTPreTrainedModel",
    "OPTForSequenceClassification",
    "OPTForQuestionAnswering",
]
