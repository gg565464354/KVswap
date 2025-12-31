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
import os
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

        
        # [修改] 路径指向 kvswap_projections_llama
        proj_dir = "/root/autodl-tmp/kvswap_projections_llama"
        proj_path = f"{proj_dir}/projection_layer_{layer_idx}.pt"
        
        if os.path.exists(proj_path):
            try:
                # 加载并注册为 buffer
                A = torch.load(proj_path, map_location="cpu") 
                self.register_buffer("projection_matrix", A.to(dtype=torch.float16))
                self.kvswap_enabled = True
                
                # KVSwap 超参数
                self.kv_group_size = 4      # G: 组大小
                self.kv_top_k_groups = 100  # M: 选中多少组
                self.target_rank = A.shape[1]
                
                # print(f"Layer {layer_idx}: KVSwap enabled. Projection shape: {A.shape}")
            except Exception as e:
                print(f"Layer {layer_idx}: Failed to load KVSwap matrix from {proj_path}: {e}")
        else:
            # print(f"Layer {layer_idx}: KVSwap projection not found at {proj_path}")
            pass
        ###############################
    def _kvswap_predict_indices(self, query_states, full_key_states):
        """
        KVSwap 核心逻辑：预测哪些 KV 块是重要的
        移植自 Qwen3, 适配 Llama 的变量命名
        """
        bsz, q_len, _, _ = query_states.shape
        _, self.num_kv_heads, total_seq_len, self.head_dim = full_key_states.shape
        
        # 1. 现场全量压缩 (On-the-fly Compression)
        # Llama Keys: [B, H_k, S, D] -> transpose -> [B, S, H_k, D] -> Flatten [B, S, Feature]
        k_flat = full_key_states.transpose(1, 2).reshape(bsz, total_seq_len, -1)
        
        # 确保类型匹配
        if self.projection_matrix.dtype != k_flat.dtype:
            self.projection_matrix = self.projection_matrix.to(k_flat.dtype)
        if self.projection_matrix.device != k_flat.device:
            self.projection_matrix = self.projection_matrix.to(k_flat.device)
            
        # Project: [B, S, H_k * d] @ [H_k * d, r] -> [B, S, r]
        # 注意: Qwen 代码里是 [B, 1, r] 因为它是 chunk 处理，这里我们对整个历史压缩
        compressed_k = torch.matmul(k_flat, self.projection_matrix)
        
        # 2. 计算低秩 Query
        # A: [H_k * d, r] -> [H_k, d, r]
        A_reshaped = self.projection_matrix.view(self.num_kv_heads, self.head_dim, -1)
        
        # 扩展 A 以匹配 Query Heads (GQA)
        # [H_k, d, r] -> [H_q, d, r]
        A_expanded = repeat_kv(A_reshaped.unsqueeze(0), self.num_key_value_groups).squeeze(0)
        
        # 计算 Q_lr: Q [B, H_q, 1, d] -> permute -> [B, 1, H_q, d]
        # Q_lr = Q * A
        q_permuted = query_states.transpose(1, 2) 
        q_lr = torch.einsum("bshd,hdr->bshr", q_permuted, A_expanded)
        
        # 3. 计算分数: Q_lr @ Compressed_K.T
        # [B, 1, H_q, r] @ [B, r, S] -> [B, 1, H_q, S]
        # 注意: compressed_k 是 [B, S, r], 需要 transpose(1, 2) -> [B, r, S]
        approx_scores = torch.matmul(q_lr, compressed_k.transpose(1, 2))
        
        # 4. Head-wise 聚合 (Sum Pooling within GQA groups)
        num_q_heads = approx_scores.shape[2]
        gqa_group = num_q_heads // self.num_kv_heads
        
        # Reshape: [B, 1, H_k, GQA_Group, S]
        scores_view = approx_scores.view(bsz, 1, self.num_kv_heads, gqa_group, -1)
        # Sum: [B, 1, H_k, S]
        agg_scores = scores_view.sum(dim=3)
        
        # 5. Group Max Pooling (Block-wise)
        seq_len = agg_scores.shape[-1]
        pad_len = (self.kv_group_size - (seq_len % self.kv_group_size)) % self.kv_group_size
        if pad_len > 0:
            agg_scores = torch.nn.functional.pad(agg_scores, (0, pad_len), value=-float('inf'))
            
        num_groups = agg_scores.shape[-1] // self.kv_group_size
        
        # View: [B, 1, H_k, Num_Groups, Group_Size]
        grouped_scores = agg_scores.view(bsz, 1, self.num_kv_heads, num_groups, self.kv_group_size)
        # Max: [B, 1, H_k, Num_Groups]
        group_max_scores = grouped_scores.max(dim=-1).values
        
        # 6. TopK
        k = min(self.kv_top_k_groups, num_groups)
        topk_indices = torch.topk(group_max_scores, k, dim=-1).indices # [B, 1, H_k, k]
        
        return topk_indices, seq_len
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
        
        is_decoding = query_states.shape[2] == 1 and past_key_value.get_seq_length(self.layer_idx) > 1

        if self.kvswap_enabled and is_decoding:
            # 1. 预测重要块索引
            top_group_indices, total_seq_len = self._kvswap_predict_indices(query_states, key_states)
            
            # 2. 索引展开 (Block -> Token Indices)
            bsz, _, num_kv_heads, k = top_group_indices.shape
            offsets = torch.arange(self.kv_group_size, device=query_states.device)
            # [B, 1, H_k, k, 1] + [Group] -> [B, 1, H_k, k, Group]
            token_indices = (top_group_indices.unsqueeze(-1) * self.kv_group_size) + offsets.view(1, 1, 1, 1, -1)
            token_indices = token_indices.view(bsz, num_kv_heads, -1)
            token_indices = token_indices.clamp(max=total_seq_len - 1)
            
            # 3. 强制包含最近窗口 (Rolling Window)
            WINDOW_SIZE = 4
            start_idx = max(0, total_seq_len - WINDOW_SIZE)
            window_indices = torch.arange(start_idx, total_seq_len, device=query_states.device)
            window_indices = window_indices.view(1, 1, -1).expand(bsz, num_kv_heads, -1)
            
            # 拼接索引
            token_indices = torch.cat([token_indices, window_indices], dim=-1)
            
            # 4. Gather 稀疏 KV
            # key_states: [B, H_k, S, D]
            # token_indices: [B, H_k, Sparse_S] -> expand dim -> [B, H_k, Sparse_S, D]
            gather_indices = token_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
            
            sparse_key_states = torch.gather(key_states, 2, gather_indices)
            sparse_value_states = torch.gather(value_states, 2, gather_indices)
            
            # 5. 重复 KV 以匹配 Query Heads (GQA)
            sparse_key_states = repeat_kv(sparse_key_states, self.num_key_value_groups)
            sparse_value_states = repeat_kv(sparse_value_states, self.num_key_value_groups)
            
            # 6. 计算 Attention (SDPA)
            attn_output = F.scaled_dot_product_attention(
                query_states,
                sparse_key_states,
                sparse_value_states,
                attn_mask=None, # 稀疏attn不需要mask，因为gather出来的都是合法的
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=False 
            )
            
        else:
            # ==================== 原有的标准/Prefill 逻辑 ====================
            key_states_r = repeat_kv(key_states, self.num_key_value_groups)
            value_states_r = repeat_kv(value_states, self.num_key_value_groups)

            # 使用 SDPA (Prefill 阶段或 KVSwap 未启用时)
            use_causal_mask = (query_states.shape[2] > 1) and (attention_mask is None)
            
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states_r,
                value_states_r,
                attn_mask=attention_mask, 
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=use_causal_mask,
                scale=self.scaling
            )
        
        # ===============================================================

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None


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
