# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights
# reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
from typing import Callable, Optional, Union
import math
import torch.nn.functional as F
import torch
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...integrations import use_kernel_forward_from_hub
from ...masking_utils import create_causal_mask
from ...modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import check_model_inputs
from .configuration_llama import LlamaConfig


logger = logging.get_logger(__name__)


@use_kernel_forward_from_hub("RMSNorm")
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        #### InfiniGen Hyperparams ####
        self.cache_ratio = None
        self.partial_weight_ratio = None
        self.previous_hidden_states = None
        self.current_hidden_states = None
        self.partial_weight_q = None
        self.skewing_matrix = None
        self.skewing_matrx = None  # alias for potential external usage
        self.alpha = 5
        self.capacity = 1.0
        self.budget = 0.2
        self.eviction_policy = "counter"  # "fifo" | "lru" | "counter"
        self.density = None
        ###############################

    def kv_cache_mask(self, attn):
        # Hyperparameters
        # budget: maximum kv cache percentage to prefetch per layer
        # capacity: maximum kv cache percentage to store in cpu
        assert self.budget < self.capacity

        b, h, tgt_len, src_len = attn.shape
        attn = attn.view(b * h, tgt_len, src_len)
        heads = b * h

        attn_mask = torch.full(attn.shape, -10000, dtype=attn.dtype, device=attn.device)
        attn_mask = torch.triu(attn_mask, diagonal=1)
        fetch_mask = torch.zeros_like(attn)
        m_inf = torch.tensor([[-10000]], dtype=attn.dtype, device=attn.device)
        attn = attn + attn_mask
        del attn_mask

        max_val = torch.max(attn, dim=-1, keepdim=True)[0][0]
        threshold = max_val - self.alpha
        fetch_num = (attn >= threshold).sum(dim=-1)  # heads, tgt_len
        del threshold

        fetch_num = torch.mean(fetch_num.to(attn.dtype), dim=0).to(torch.int32)  # fetch same amount for each head
        fetch_max = int(src_len * self.budget)
        fetch_num = torch.where(fetch_num >= fetch_max, torch.tensor(fetch_max, device=attn.device), fetch_num)  # tgt_len

        store_max = int(src_len * self.capacity)

        # always fetch lower triangle for the first fetch_max steps
        fetch_mask[:, :fetch_max] = torch.tril(
            torch.ones((fetch_max, src_len), dtype=attn.dtype, device=attn.device)
        ).unsqueeze(0)

        for i in range(fetch_max, store_max):
            k = int(fetch_num[i].item()) if isinstance(fetch_num[i], torch.Tensor) else int(fetch_num[i])
            if k > 0:
                _, ind = torch.topk(attn[:, i, : i + 1], k=k, dim=-1)
                fetch_mask[:, i, : i + 1] = fetch_mask[:, i, : i + 1].scatter(-1, ind, 1)

        for i in range(store_max, tgt_len):
            k = int(fetch_num[i].item()) if isinstance(fetch_num[i], torch.Tensor) else int(fetch_num[i])
            if k > 0:
                _, ind = torch.topk(attn[:, i, : i + 1], k=k, dim=-1)
                fetch_mask[:, i, : i + 1] = fetch_mask[:, i, : i + 1].scatter(-1, ind, 1)

            if i == (tgt_len - 1):
                continue

            # Before adding KV cache, evict one
            if self.eviction_policy == "fifo":
                evict_idx = i - store_max
                attn[:, (i + 1) :, evict_idx] = -10000

            elif self.eviction_policy == "lru":
                idx = torch.arange(i + 1, device=attn.device).unsqueeze(0).unsqueeze(-1)
                idx = idx * fetch_mask[:, : i + 1, : int(i / 2)]  # avoid evicting recently added ones
                # Most recently fetched idx per each KV cache
                _, idx = torch.max(idx, dim=1, keepdim=True)  # heads, 1, i/2
                _, ind = torch.min(idx, dim=-1, keepdim=True)  # heads, 1, 1
                ind = ind.repeat(1, tgt_len - (i + 1), 1)
                attn[:, (i + 1) :] = attn[:, (i + 1) :].scatter(-1, ind, -10000)

            elif self.eviction_policy == "counter":
                counter = torch.sum(fetch_mask[:, : i + 1, : int(i / 2)], dim=1, keepdim=True)  # heads, 1, i/2
                _, ind = torch.min(counter, dim=-1, keepdim=True)  # heads, 1, 1
                ind = ind.repeat(1, tgt_len - (i + 1), 1)
                attn[:, (i + 1) :] = attn[:, (i + 1) :].scatter(-1, ind, -10000)

            else:
                raise NotImplementedError

        density = fetch_mask.float().sum().item() / heads / (tgt_len * (tgt_len + 1) / 2)
        fetch_mask = torch.where(fetch_mask == 1, 0, m_inf)
        fetch_mask = fetch_mask.view(b, h, tgt_len, src_len)
        return fetch_mask, density

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        if hidden_states.shape[1] == 1:
            self.current_hidden_states = hidden_states.clone()
        else:
            self.current_hidden_states = None

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # 1. 计算 Q, K, V
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        key_states_r = repeat_kv(key_states, self.num_key_value_groups)
        value_states_r = repeat_kv(value_states, self.num_key_value_groups)

        # ==================== 核心修复 ====================
        # 判断是 Prefill (长序列) 还是 Decoding (单 Token)
        is_prefill = query_states.shape[2] > 1

        if is_prefill:
            # === Prefill 阶段: 必须使用 Flash Attention (SDPA) 避免 OOM ===
            # 使用 PyTorch 内置的 SDPA，它会自动调用 Flash Attention 2
            
            # SDPA 期望的 mask 逻辑比较特殊，通常如果是 causal 的话，如果不传 mask 且设置 is_causal=True 会最快
            # 但 transformers 传进来的 attention_mask 通常是 4D 的 [B, 1, Q, K]
            # print("DEBUG: Running Flash Attention for Prefill...")
            # 尝试使用 SDPA
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states_r,
                value_states_r,
                attn_mask=attention_mask if attention_mask is not None else None,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=True if attention_mask is None else False, # 如果有 mask 就传 mask，没有就设为 causal
                scale=self.scaling
            )
            
            # 这里的 attn_weights 返回 None，因为 FA2 不返回权重矩阵
            attn_weights = None
        # 强制走 Eager 模式
        else:

            # === InfiniGen: Gather 模式实现 ===
            # 只有当有上一层的信息，且不是第一层时才进行 Top-K 稀疏计算
            if (self.previous_hidden_states is not None) and (self.partial_weight_q is not None):
                # A. 低成本预测 (Speculate)
                # -----------------------------------------------------------
                query_prev = self.q_proj(self.previous_hidden_states).view(hidden_shape).transpose(1, 2)
                query_prev, _ = apply_rotary_pos_emb(query_prev, key_states, cos, sin)
                
                # 应用 Partial Weight Mask (降低维度)
                mask = (
                    self.partial_weight_q[0]
                    .view(-1, 128)
                    .unsqueeze(0)
                    .unsqueeze(2)
                    .repeat(1, 1, query_states.shape[2], 1)
                )
                query_prev = torch.where(mask.to(torch.bool), query_prev, torch.zeros_like(query_prev))
                
                # 计算粗糙分数 [Batch, Heads, Q_Len, KV_Len]
                # 注意：在 Prefill 阶段这步依然费显存，但在 Decoding 阶段非常小
                attn_spec = torch.matmul(query_prev, key_states_r.transpose(2, 3)) * self.scaling
                
                # 如果有 mask (如 causal mask)，先加上去，防止选到未来的 token
                if attention_mask is not None:
                    # 确保维度匹配
                    causal_mask = attention_mask[:, :, :, : key_states_r.shape[-2]]
                    attn_spec = attn_spec + causal_mask

                # B. Top-K 选址 (Selection)
                # -----------------------------------------------------------
                # 确定 K 的大小 (Budget)
                total_tokens = key_states_r.shape[-2]
                target_k = int(total_tokens * self.budget)
                # 至少保留几个 token 防止报错
                target_k = max(target_k, 16) 
                
                # 选取分数最高的 Top-K 个索引
                # indices shape: [Batch, Heads, Q_Len, K]
                topk_values, topk_indices = torch.topk(attn_spec, k=target_k, dim=-1)

                # C. 提取数据 (Gather)
                # -----------------------------------------------------------
                # 我们需要根据 topk_indices 从 key_states_r 和 value_states_r 中抓取数据
                # 为了使用 torch.gather，我们需要展平 Batch 和 Heads 维度
                
                bsz, num_heads, q_len, _ = topk_indices.shape
                head_dim = key_states_r.shape[-1]

                # 展平: [Batch*Heads, Total_Len, Head_Dim]
                k_flat = key_states_r.contiguous().view(bsz * num_heads, total_tokens, head_dim)
                v_flat = value_states_r.contiguous().view(bsz * num_heads, total_tokens, head_dim)
                
                # 索引展平: [Batch*Heads, Q_Len * K]
                # 我们把 (Q_Len, K) 视为一个长序列，一次性抓出来，后面再 reshape 回去
                flat_indices = topk_indices.view(bsz * num_heads, -1)
                
                # 扩展索引维度以匹配 Head_Dim: [Batch*Heads, Q_Len * K, Head_Dim]
                gather_indices = flat_indices.unsqueeze(-1).expand(-1, -1, head_dim)

                # 执行 Gather
                selected_keys = torch.gather(k_flat, 1, gather_indices)
                selected_values = torch.gather(v_flat, 1, gather_indices)

                # 恢复形状: [Batch, Heads, Q_Len, K, Head_Dim]
                selected_keys = selected_keys.view(bsz, num_heads, q_len, target_k, head_dim)
                selected_values = selected_values.view(bsz, num_heads, q_len, target_k, head_dim)

                # D. 小矩阵计算 (Sparse Attention)
                # -----------------------------------------------------------
                # Query: [Batch, Heads, Q_Len, Head_Dim]
                # Selected Keys Transposed: [Batch, Heads, Q_Len, Head_Dim, K]
                # 这里的 matmul 有点 tricky，因为 Q 和 K 现在的维度是对齐的 (Q_Len 对 Q_Len)
                # 我们需要在最后两个维度做运算: (1, Head_Dim) @ (Head_Dim, K) -> (1, K)
                
                # [Batch, Heads, Q_Len, 1, Head_Dim]
                q_unsqueezed = query_states.unsqueeze(3) 
                # [Batch, Heads, Q_Len, Head_Dim, K]
                k_transposed = selected_keys.transpose(-1, -2)
                
                # 结果: [Batch, Heads, Q_Len, 1, K]
                attn_output_weights = torch.matmul(q_unsqueezed, k_transposed) * self.scaling
                attn_output_weights = attn_output_weights.squeeze(3) # [Batch, Heads, Q_Len, K]

                # 这里的 Softmax 只对这 Top-K 个元素进行归一化
                attn_output_weights = nn.functional.softmax(attn_output_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                
                # Value 聚合
                # Weights: [Batch, Heads, Q_Len, 1, K] (unsqueeze 方便广播)
                # Values:  [Batch, Heads, Q_Len, K, Head_Dim]
                attn_output = torch.matmul(attn_output_weights.unsqueeze(3), selected_values)
                # 结果: [Batch, Heads, Q_Len, 1, Head_Dim] -> squeeze -> [Batch, Heads, Q_Len, Head_Dim]
                attn_output = attn_output.squeeze(3)
                
                # 记录密度 (用于统计)
                self.density = target_k / total_tokens
            
            # === 标准全量 Attention (第一层或没有上一层信息时) ===
            else:
                attn_weights = torch.matmul(query_states, key_states_r.transpose(2, 3)) * self.scaling
                if attention_mask is not None:
                    causal_mask = attention_mask[:, :, :, : key_states_r.shape[-2]]
                    attn_weights = attn_weights + causal_mask
                
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_output = torch.matmul(attn_weights, value_states_r)

            # 最终输出 Projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None # 注意：这里为了简化，不返回 attn_weights


class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class LlamaPreTrainedModel(PreTrainedModel):
    config: LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": LlamaDecoderLayer,
        "attentions": LlamaAttention,
    }


@auto_docstring
class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            # === InfiniGen: pass previous hidden states to next layer ===
            if (idx > 0) and (idx < (self.config.num_hidden_layers - 1)):
                self.layers[idx + 1].self_attn.previous_hidden_states = self.layers[idx].self_attn.current_hidden_states

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def get_density(self):
        density = []
        for l in self.model.layers:
            if hasattr(l.self_attn, "density") and l.self_attn.density is not None:
                density.append(l.self_attn.density)
        return (sum(density) / len(density)) if len(density) > 0 else None

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LlamaForSequenceClassification(GenericForSequenceClassification, LlamaPreTrainedModel): ...


class LlamaForQuestionAnswering(GenericForQuestionAnswering, LlamaPreTrainedModel):
    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`


class LlamaForTokenClassification(GenericForTokenClassification, LlamaPreTrainedModel): ...


__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
    "LlamaForSequenceClassification",
    "LlamaForQuestionAnswering",
    "LlamaForTokenClassification",
]
