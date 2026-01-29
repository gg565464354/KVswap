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
    inv_freq: torch.Tensor

    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        rope_type = None
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        elif hasattr(config, "rope_parameters") and isinstance(getattr(config, "rope_parameters"), dict):
            rope_type = config.rope_parameters.get("rope_type", config.rope_parameters.get("type"))
        elif hasattr(config, "rope_type"):
            rope_type = getattr(config, "rope_type")

        self.rope_type = rope_type or "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        rope_init_fn = ROPE_INIT_FUNCTIONS.get(self.rope_type)
        if rope_init_fn is None or self.rope_type == "default":
            rope_init_fn = self.compute_default_rope_parameters
        self.rope_init_fn = rope_init_fn

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[LlamaConfig] = None,
        device: Optional["torch.device"] = None,
        seq_len: Optional[int] = None,
    ) -> tuple["torch.Tensor", float]:
        base = getattr(config, "rope_theta", 283461213.0)
        dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
        attention_factor = 1.0
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

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

        # Infinigen cache pool (fixed_k / threshold)
        self.cache_pool_enabled = bool(getattr(config, "infinigen_cache_pool_enabled", False))
        self.cache_pool_strategy = getattr(config, "infinigen_cache_pool_strategy", "fixed_k")
        if self.cache_pool_strategy not in ("fixed_k", "threshold"):
            self.cache_pool_strategy = "fixed_k"
        self.cache_pool_k = int(getattr(config, "infinigen_cache_pool_k", 4))
        self.cache_pool_cap_ratio = float(getattr(config, "infinigen_cache_pool_cap_ratio", 0.75))
        self.local_window_size = int(getattr(config, "infinigen_local_window", 0))
        self.fixed_topk = int(getattr(config, "infinigen_fixed_topk", -1))
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

    def _get_skewing_matrix(self):
        return self.skewing_matrix if self.skewing_matrix is not None else self.skewing_matrx

    def _apply_skewing(self, x: torch.Tensor, skew: Optional[torch.Tensor] = None) -> torch.Tensor:
        sm = skew if skew is not None else self._get_skewing_matrix()
        if sm is None:
            return x
        sm = sm.to(device=x.device, dtype=x.dtype)
        if sm.dim() == 2:
            sm_exp = sm.unsqueeze(0).unsqueeze(0)
        elif sm.dim() == 3:
            if sm.shape[0] == x.shape[1]:
                sm_exp = sm.unsqueeze(0)
            elif x.shape[1] % sm.shape[0] == 0:
                sm_exp = sm.repeat_interleave(x.shape[1] // sm.shape[0], dim=0).unsqueeze(0)
            else:
                sm_exp = sm[:1].unsqueeze(0).expand(1, x.shape[1], -1, -1)
        else:
            return x
        return torch.matmul(x, sm_exp)

    # -------------------- InfiniGen cache pool utils --------------------
    def _infinigen_get_layer_state(self, past_key_values: Optional[Cache]):
        if past_key_values is None:
            return None
        try:
            if not hasattr(past_key_values, "_infinigen_state"):
                past_key_values._infinigen_state = {}
        except Exception:
            return None

        st = past_key_values._infinigen_state.get(self.layer_idx)
        if st is None:
            st = {
                "pool": None,        # [B,H,N] long (pad=-1)
                "pool_valid": None,  # [B,H,N] bool
                "pool_step": 0,
                "shape": None,       # (B,H)
            }
            past_key_values._infinigen_state[self.layer_idx] = st
        return st

    @staticmethod
    def _clamp_keep_neg1(x: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
        return torch.where(x < 0, x, x.clamp(lo, hi))

    @torch.no_grad()
    def _unique_pad_bh(self, x: torch.Tensor, pad_value: int = -1):
        B, H, N = x.shape
        device = x.device

        xs, _ = torch.sort(x, dim=-1)
        valid = xs != pad_value
        prev = torch.cat(
            [torch.full((B, H, 1), pad_value - 1, device=device, dtype=xs.dtype), xs[..., :-1]],
            dim=-1,
        )
        is_new = valid & (xs != prev)

        pos_raw = torch.cumsum(is_new.to(torch.int32), dim=-1) - 1
        pos = torch.where(is_new, pos_raw.to(torch.long), torch.full_like(pos_raw, -1, dtype=torch.long))

        cnt = is_new.sum(dim=-1)
        M = int(cnt.max().item()) if cnt.numel() > 0 else 0

        padded = torch.full((B, H, M), pad_value, device=device, dtype=torch.long)
        valid_out = torch.zeros((B, H, M), device=device, dtype=torch.bool)

        if M == 0:
            return padded, valid_out

        bh = torch.arange(B * H, device=device).unsqueeze(1).expand(B * H, N).reshape(-1)
        pos_f = pos.reshape(-1)
        val_f = xs.reshape(-1)
        m = pos_f >= 0

        out_f = padded.view(B * H, M)
        v_f = valid_out.view(B * H, M)
        out_f[bh[m], pos_f[m]] = val_f[m]
        v_f[bh[m], pos_f[m]] = True

        return padded, valid_out

    def _pad_last_dim(self, x: torch.Tensor, target: int, pad_value):
        n = x.shape[-1]
        if n == target:
            return x
        return torch.nn.functional.pad(x, (0, target - n), value=pad_value)

    @torch.no_grad()
    def _update_cache_pool(
        self,
        base_indices: torch.Tensor,
        total_seq_len: int,
        past_key_values: Optional[Cache],
        cache_pool_enabled: Optional[bool] = None,
        cache_pool_strategy: Optional[str] = None,
    ):
        st = self._infinigen_get_layer_state(past_key_values)
        base = self._clamp_keep_neg1(base_indices, 0, total_seq_len - 1)

        base_u, base_valid = self._unique_pad_bh(base, pad_value=-1)
        base_cnt = base_valid.sum(dim=-1)
        critical_cnt = base_cnt + self.local_window_size + 1

        if st is None:
            return base_u, base_valid

        if cache_pool_enabled is None:
            cache_pool_enabled = self.cache_pool_enabled
        if cache_pool_strategy is None:
            cache_pool_strategy = self.cache_pool_strategy
        if cache_pool_strategy not in ("fixed_k", "threshold"):
            cache_pool_strategy = "fixed_k"

        if not cache_pool_enabled:
            return base_u, base_valid

        cur_shape = (base.shape[0], base.shape[1])
        if st["shape"] is None or st["shape"] != cur_shape:
            st["pool"], st["pool_valid"], st["pool_step"] = None, None, 0
            st["shape"] = cur_shape

        if cache_pool_strategy == "fixed_k":
            step = int(st.get("pool_step", 0))
            do_reset = (self.cache_pool_k > 0) and ((step + 1) % self.cache_pool_k == 0)

            if do_reset or st["pool"] is None:
                pool_u, pool_valid = base_u, base_valid
            else:
                merged = torch.cat([st["pool"], base], dim=-1)
                merged = self._clamp_keep_neg1(merged, 0, total_seq_len - 1)
                pool_u, pool_valid = self._unique_pad_bh(merged, pad_value=-1)

            st["pool_step"] = step + 1
            st["pool"], st["pool_valid"] = pool_u, pool_valid
            return pool_u, pool_valid

        if st["pool"] is None:
            merged = base
        else:
            merged = torch.cat([st["pool"], base], dim=-1)
            merged = self._clamp_keep_neg1(merged, 0, total_seq_len - 1)

        merged_u, merged_valid = self._unique_pad_bh(merged, pad_value=-1)

        cap = int(self.cache_pool_cap_ratio * float(total_seq_len))
        thr = torch.minimum(2 * critical_cnt, torch.full_like(critical_cnt, cap))

        merged_cnt = merged_valid.sum(dim=-1)
        need_rebuild = merged_cnt > thr

        N = max(base_u.shape[-1], merged_u.shape[-1])
        base_u_pad = self._pad_last_dim(base_u, N, pad_value=-1)
        base_valid_pad = self._pad_last_dim(base_valid, N, pad_value=False)
        merged_u_pad = self._pad_last_dim(merged_u, N, pad_value=-1)
        merged_valid_pad = self._pad_last_dim(merged_valid, N, pad_value=False)

        mask = need_rebuild.unsqueeze(-1)
        pool_u = torch.where(mask, base_u_pad, merged_u_pad)
        pool_valid = torch.where(mask, base_valid_pad, merged_valid_pad)
        pool_u = torch.where(pool_valid, pool_u, torch.full_like(pool_u, -1))

        st["pool"], st["pool_valid"] = pool_u, pool_valid
        return pool_u, pool_valid

    @torch.no_grad()
    def _append_local_and_cur_and_dedup(self, indices: torch.Tensor, total_seq_len: int):
        B, H, _ = indices.shape
        device = indices.device
        cur = total_seq_len - 1

        start = max(0, total_seq_len - self.local_window_size)
        window = torch.arange(start, total_seq_len, device=device, dtype=torch.long)
        window = window.view(1, 1, -1).expand(B, H, -1)

        cur_idx = torch.full((B, H, 1), cur, device=device, dtype=torch.long)

        merged = torch.cat([indices, window, cur_idx], dim=-1)
        merged = self._clamp_keep_neg1(merged, 0, cur)
        return self._unique_pad_bh(merged, pad_value=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        infinigen_cache_pool_enabled: Optional[bool] = None,
        infinigen_cache_pool_strategy: Optional[str] = None,
        infinigen_fixed_topk: Optional[int] = None,
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
                query_prev = self.q_proj(self.previous_hidden_states).view(hidden_shape).transpose(1, 2)
                query_prev, _ = apply_rotary_pos_emb(query_prev, key_states, cos, sin)

                query_prev = self._apply_skewing(query_prev)
                key_for_spec = self._apply_skewing(key_states_r)

                mask = (
                    self.partial_weight_q[0]
                    .view(-1, self.head_dim)
                    .unsqueeze(0)
                    .unsqueeze(2)
                    .repeat(1, 1, query_states.shape[2], 1)
                )
                query_prev = torch.where(mask.to(torch.bool), query_prev, torch.zeros_like(query_prev))

                attn_spec = torch.matmul(query_prev, key_for_spec.transpose(2, 3)) * self.scaling
                if attention_mask is not None:
                    causal_mask = attention_mask[:, :, :, : key_for_spec.shape[-2]]
                    attn_spec = attn_spec + causal_mask

                total_tokens = key_states_r.shape[-2]
                fixed_topk = infinigen_fixed_topk if infinigen_fixed_topk is not None else self.fixed_topk
                if fixed_topk is not None and fixed_topk > 0:
                    target_k = min(int(fixed_topk), total_tokens)
                else:
                    target_k = int(total_tokens * self.budget)
                target_k = max(target_k, 1)

                topk_indices = torch.topk(attn_spec, k=target_k, dim=-1).indices  # [B,H,Q,K]

                base_indices = topk_indices[:, :, 0, :] if topk_indices.shape[2] == 1 else topk_indices.squeeze(2)
                pooled_indices, _ = self._update_cache_pool(
                    base_indices,
                    total_tokens,
                    past_key_value,
                    cache_pool_enabled=infinigen_cache_pool_enabled,
                    cache_pool_strategy=infinigen_cache_pool_strategy,
                )
                token_indices, valid_mask = self._append_local_and_cur_and_dedup(pooled_indices, total_tokens)

                safe_token_indices = token_indices.clamp(min=0)
                gather_indices = safe_token_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
                selected_keys = torch.gather(key_states_r, 2, gather_indices)
                selected_values = torch.gather(value_states_r, 2, gather_indices)

                attn_output_weights = torch.matmul(query_states, selected_keys.transpose(2, 3)) * self.scaling
                if attention_mask is not None and attention_mask.dim() == 4:
                    mask_expanded = attention_mask.expand(-1, selected_keys.shape[1], -1, -1)
                    mask_indices = safe_token_indices.unsqueeze(2)
                    sparse_mask = torch.gather(mask_expanded, 3, mask_indices)
                    attn_output_weights = attn_output_weights + sparse_mask

                invalid = (~valid_mask).unsqueeze(2)
                attn_output_weights = attn_output_weights.masked_fill(
                    invalid, torch.finfo(attn_output_weights.dtype).min
                )

                attn_output_weights = nn.functional.softmax(
                    attn_output_weights, dim=-1, dtype=torch.float32
                ).to(query_states.dtype)
                attn_output = torch.matmul(attn_output_weights, selected_values)

                sel_cnt = valid_mask.sum(dim=-1).float()
                self.density = (sel_cnt.mean() / float(total_tokens)).item()
            
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

    @check_model_inputs()
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
        infinigen_cache_pool_enabled: Optional[bool] = None,
        infinigen_cache_pool_strategy: Optional[str] = None,
        infinigen_fixed_topk: Optional[int] = None,
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
            infinigen_cache_pool_enabled=infinigen_cache_pool_enabled,
            infinigen_cache_pool_strategy=infinigen_cache_pool_strategy,
            infinigen_fixed_topk=infinigen_fixed_topk,
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
