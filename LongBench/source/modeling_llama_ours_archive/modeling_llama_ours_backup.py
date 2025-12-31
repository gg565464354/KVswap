# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# =================================================================
# <<< ⬇️ 粘贴这个新的路径修复代码块 ⬇️ >>>
# =================================================================
import sys
import os

# 找到此文件 (modeling_llama_ours.py) 的真实源文件路径
# 即使它是 symlink, realpath 也会找到源文件
# 即 /root/sparse-load/SparseCache/accuracy/benchmark/source/modeling_llama_ours.py
try:
    _CURRENT_FILE_PATH = os.path.realpath(__file__)
except NameError:
    # 在某些环境中 __file__ 可能未定义，使用 cwd 作为备选
    _CURRENT_FILE_PATH = os.path.abspath("modeling_llama_ours.py") 

# 从这个文件往上找四级目录，找到 SparseCache 的根目录
# .../source -> .../benchmark -> .../accuracy -> /root/sparse-load/SparseCache
_SPARSE_CACHE_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(_CURRENT_FILE_PATH)))
)

# 将根目录添加到 sys.path
if _SPARSE_CACHE_ROOT not in sys.path:
    sys.path.insert(0, _SPARSE_CACHE_ROOT) 

print(f"[Infinigen Import Fix]: Added '{_SPARSE_CACHE_ROOT}' to sys.path")
# =================================================================
# <<< ⬆️ 粘贴完毕 ⬆️ >>>
# =================================================================
from typing import Callable, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F
import math

# =================================================================
# <<< 1. 导入 reform_hidden_states >>>
# 假设这个包在你的 Python 环境中
# =================================================================
try:
    # 你的 flexgen.tch 文件显示此函数来自这里
    from infinigen.skewing_controller import reform_hidden_states
except ImportError:
    # Fallback: identity mapping to keep input dim == config.hidden_size
    # (the concatenation strategy requires concatenated weights, which we don't apply here)
    def reform_hidden_states(hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states
# =================================================================


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
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
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
        self.head_dim = self.hidden_size // self.num_heads

        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )
        
        #### InfiniGen Hyperparams (从 longbench_pred.py 注入) ####
        self.partial_weight_ratio = None
        self.previous_hidden_states = None
        self.current_hidden_states = None
        self.partial_weight_q = None
        self.skewing_matrix = None
        self.alpha = 5
        self.capacity = 1.0
        self.budget = 0.2
        self.eviction_policy = "counter"
        self.density = None
        ###############################

    # kv_cache_mask 函数保持不变 (它只在 Eager 路径中被调用)
    def kv_cache_mask(self, attn):
        assert self.budget < self.capacity
        b, h, tgt_len, src_len = attn.shape
        attn = attn.view(b*h, tgt_len, src_len)
        heads = b * h
        attn_mask = torch.full(attn.shape, -10000, dtype=attn.dtype, device=attn.device)
        attn_mask = torch.triu(attn_mask, diagonal = 1)
        fetch_mask = torch.zeros_like(attn)
        m_inf = torch.tensor([[-10000]], dtype=attn.dtype, device=attn.device)
        attn = attn + attn_mask
        del attn_mask
        max_val = torch.max(attn, dim = -1, keepdim = True)[0] 
        threshold = max_val - self.alpha
        fetch_num  = (attn >= threshold).sum(dim = -1)
        del threshold
        fetch_num = torch.mean(fetch_num.to(attn.dtype), dim = 0).to(torch.int32)
        fetch_max = int(src_len * self.budget)
        fetch_num = torch.where(fetch_num >= fetch_max, fetch_max, fetch_num)
        store_max = int(src_len * self.capacity)
        if fetch_max > 0:
            fetch_mask[:, :fetch_max] = torch.tril(torch.ones((fetch_max, src_len), dtype = attn.dtype, device = attn.device)).unsqueeze(0)
        for i in range(fetch_max, store_max):
            if fetch_num[i] > 0:
                _, ind = torch.topk(attn[:,i, :i+1], k = fetch_num[i].item(), dim = -1)
                fetch_mask[:, i, :i+1] = fetch_mask[:, i, :i + 1].scatter(-1, ind, 1)
        for i in range(store_max, tgt_len):
            if fetch_num[i] > 0:
                _, ind = torch.topk(attn[:,i, :i+1], k = fetch_num[i].item(), dim = -1)
                fetch_mask[:, i, :i + 1] = fetch_mask[:, i, :i + 1].scatter(-1, ind, 1)
            if i == (tgt_len - 1):
                continue
            if self.eviction_policy == "fifo":
                evict_idx = i - store_max
                attn[:, (i + 1):, evict_idx] = -10000
            elif self.eviction_policy == "lru":
                idx_range = torch.arange(i + 1, device = attn.device).unsqueeze(0).unsqueeze(-1)
                idx = idx_range * fetch_mask[:, :i + 1, :int(i / 2) + 1]
                _, idx = torch.max(idx, dim = 1, keepdim = True)
                _, ind = torch.min(idx, dim = -1, keepdim = True)
                ind = ind.repeat(1, tgt_len - (i + 1), 1)
                attn[:, (i + 1):] = attn[:, (i + 1):].scatter(-1, ind, -10000)
            elif self.eviction_policy == "counter":
                counter = torch.sum(fetch_mask[:, :i + 1, :int(i / 2) + 1], dim = 1, keepdim = True)
                _, ind = torch.min(counter, dim = -1, keepdim = True)
                ind = ind.repeat(1,tgt_len-(i+1),1)
                attn[:, (i + 1):] = attn[:, (i + 1):].scatter(-1, ind, -10000)
            else:
                raise NotImplementedError
        density = fetch_mask.float().sum().item() / heads / (tgt_len * (tgt_len + 1) / 2)
        fetch_mask = torch.where(fetch_mask == 1, 0, m_inf)
        fetch_mask = fetch_mask.view(b, h, tgt_len, src_len)
        return fetch_mask, density
    
    # ================================================================
    # <<< 2. 这是最终的 forward 方法 >>>
    # ================================================================
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:

        bsz, q_len, _ = hidden_states.size()
        self.current_hidden_states = hidden_states.clone()
        
        # ================================================================
        # <<< 3. 最终修复：应用 reform_hidden_states (来自 flexgen) >>>
        # Q 和 K 投影使用 "reformed" (重整) 的 hidden_states
        # V 投影使用 *原始* 的 hidden_states
        # ================================================================
        hidden_states_reformed = reform_hidden_states(hidden_states)
        
        query_states = self.q_proj(hidden_states_reformed).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states_reformed).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len = past_key_value.get_seq_length() + key_states.shape[-2]
        else:
            kv_seq_len = key_states.shape[-2]
            
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        # ================================================================
        # <<< 4. GQA 修复：只对 key_states (8头) 应用 skewing_matrix >>>
        # ================================================================
        if hasattr(self, 'skewing_matrix') and self.skewing_matrix is not None:
            if self.skewing_matrix.dim() == 2:
                skewing_matrix_expanded = self.skewing_matrix.unsqueeze(0).unsqueeze(0).to(key_states.device, dtype=key_states.dtype)
            elif self.skewing_matrix.dim() == 3:
                 skewing_matrix_expanded = self.skewing_matrix.unsqueeze(0).to(key_states.device, dtype=key_states.dtype)
            else:
                raise ValueError(f"Unexpected skewing_matrix shape: {self.skewing_matrix.shape}")
            
            # 只对 key_states (8头) 应用
            key_states = torch.matmul(key_states, skewing_matrix_expanded)
        
        # GQA: 手动重复 K/V 头 (从 8 -> 32)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # ==================================================================
        # <<< 5. OOM 修复：SDPA 调度逻辑 (现在是主要路径) >>>
        # ==================================================================
        
        if self.config._attn_implementation in ["flash_attention_2", "sdpa"]:
            if output_attentions:
                logger.warning_once(
                    "LlamaAttention: output_attentions=True is not supported with Flash Attention or SDPA. "
                    "Setting output_attentions=False."
                )
                output_attentions = False
            
            attn_output = F.scaled_dot_product_attention(
                query_states, # (B, 32, S, H)
                key_states,   # (B, 32, S, H)
                value_states, # (B, 32, S, H)
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=False, 
            )
            
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)
            
            self.previous_hidden_states = self.current_hidden_states.detach() 
            return attn_output, None 

        # ================================================================
        # <<< 6. Eager 路径 (OOM 路径，作为回退) >>>
        # ================================================================
        
        ### Speculate attention ###
        if (self.previous_hidden_states is not None) and (self.partial_weight_q is not None):
            
            query = (torch.matmul(self.previous_hidden_states, self.q_proj.weight.data.transpose(-1,-2))).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            cos_unsqueezed = cos.unsqueeze(1)
            sin_unsqueezed = sin.unsqueeze(1)
            query = (query * cos_unsqueezed) + (rotate_half(query) * sin_unsqueezed)

            spec_key_states = key_states
            if hasattr(self, 'skewing_matrix') and self.skewing_matrix is not None:
                logger.warning_once("Eager path does not support skewing_matrix after GQA repeat. Skipping.")
                pass

            # !!!!! OOM 点 !!!!!
            attn = torch.matmul(query, spec_key_states.transpose(2, 3)) * self.scaling
            attn_mask, density = self.kv_cache_mask(attn)
            self.density = density
        ###########################

        # !!!!! OOM 点 !!!!!
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        
        ### Apply mask ###
        if (self.previous_hidden_states is not None) and (self.partial_weight_q is not None):
            attn_weights = attn_weights + attn_mask
        ###########################

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        self.attn = attn_weights.clone()
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)
        self.previous_hidden_states = self.current_hidden_states.detach()
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights

# ================================================================
# 以下所有代码 (LlamaDecoderLayer, LlamaModel, LlamaForCausalLM 等) 保持不变
# ================================================================

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
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
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

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

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
        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

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
    base_model_prefix = "transformer"
class LlamaForTokenClassification(GenericForTokenClassification, LlamaPreTrainedModel): ...


__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
    "LlamaForSequenceClassification",
    "LlamaForQuestionAnswering",
    "LlamaForTokenClassification",
]