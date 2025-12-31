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

from typing import Callable, Optional, Union

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
    inv_freq: torch.Tensor  # for register_buffer

    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()

        # 兼容读取 rope_type（尽量和不同版本 config 字段兼容）
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

        # ✅ 关键修复：default / unknown 时走本地默认实现，避免 KeyError
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
        # Llama 默认 rope_theta 通常是 10000；新模型也可能不同，优先从 config 读
        base = getattr(config, "rope_theta", 283461213.0)
        dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
        attention_factor = 1.0

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

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
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k.
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
    Equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    (batch, kv_heads, seqlen, head_dim) -> (batch, attn_heads, seqlen, head_dim)
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
    """
    LlamaAttention + Quest(Paper) Cache + Prefetch + CachePool:
    - Page-level reduced keys (K_min/K_max) cached incrementally on past_key_values
    - Decoding-only sparse token indices predicted by Quest, optionally prefetched by previous layer
    - Token-index cache pool with 2 strategies:
        (1) fixed_k: every k steps reset pool (default k=4)
        (2) threshold: rebuild when pool size > min(2*critical_kv, cap_ratio*T) with cap_ratio default 0.75
    """

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

        # ---------------- Quest / CachePool config (all via getattr for BC) ----------------
        self.kvswap_enabled = bool(getattr(config, "quest_enabled", True))

        self.page_size = int(getattr(config, "quest_page_size", 64))
        self.kv_top_k_groups = int(getattr(config, "quest_top_k_pages", 100))
        self.local_window_size = int(getattr(config, "quest_local_window", 32))

        self.cache_pool_strategy = str(getattr(config, "quest_cache_pool_strategy", "fixed_k"))  # "fixed_k" | "threshold"
        self.cache_pool_k = int(getattr(config, "quest_cache_pool_k", 4))
        self.cache_pool_cap_ratio = float(getattr(config, "quest_cache_pool_cap_ratio", 0.75))

        self.num_kv_heads = config.num_key_value_heads
        # ----------------------------------------------------------------------------------

    # ====================== Quest state stored on past_key_values ======================

    def _quest_get_layer_state(self, past_key_values: Optional[Cache]):
        """
        Attach per-layer state to past_key_values (safe for DynamicCache/Cache).
        """
        if past_key_values is None:
            return None
        if not hasattr(past_key_values, "_quest_state"):
            past_key_values._quest_state = {}

        st = past_key_values._quest_state.get(self.layer_idx)
        if st is None:
            st = {
                "seq_len": 0,
                "k_min": None,       # [B, H_kv, Pages, D] fp32
                "k_max": None,       # [B, H_kv, Pages, D] fp32
                "pool": None,        # [B, H_kv, N] long, pad=-1
                "pool_valid": None,  # [B, H_kv, N] bool
                "step": 0,           # for fixed_k strategy (decode steps)
            }
            past_key_values._quest_state[self.layer_idx] = st
        return st

    @staticmethod
    def _clamp_keep_neg1(x: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
        # clamp only non-negative indices; keep -1 padding as-is
        return torch.where(x < 0, x, x.clamp(lo, hi))

    # ====================== Page stats cache (K_min/K_max) ======================

    @torch.no_grad()
    def _quest_compute_page_stats_full(self, full_key_states: torch.Tensor):
        """
        Full recompute reduced keys:
          full_key_states: [B, H_kv, T, D]
          returns k_min/k_max: [B, H_kv, Pages, D] float32, and T
        """
        B, H, T, D = full_key_states.shape
        pages = (T + self.page_size - 1) // self.page_size
        pad_len = pages * self.page_size - T

        if pad_len > 0:
            k_padded = torch.nn.functional.pad(full_key_states, (0, 0, 0, pad_len))
        else:
            k_padded = full_key_states

        k_pages = k_padded.view(B, H, pages, self.page_size, D).to(torch.float32)

        # mask padding so min/max not polluted
        idx = torch.arange(pages * self.page_size, device=full_key_states.device)
        valid = (idx < T).view(1, 1, pages, self.page_size, 1)

        k_min = k_pages.masked_fill(~valid, float("inf")).min(dim=3).values
        k_max = k_pages.masked_fill(~valid, float("-inf")).max(dim=3).values
        return k_min, k_max, T

    @torch.no_grad()
    def _quest_get_page_stats_cached(self, full_key_states: torch.Tensor, past_key_values: Optional[Cache]):
        """
        Incrementally update K_min/K_max if +1 token; otherwise full recompute.
        """
        T = full_key_states.shape[2]
        st = self._quest_get_layer_state(past_key_values)

        if st is None:
            k_min, k_max, _ = self._quest_compute_page_stats_full(full_key_states)
            return k_min, k_max

        if (
            st["k_min"] is None
            or st["seq_len"] <= 0
            or T <= 0
            or T < st["seq_len"]
            or (T - st["seq_len"]) != 1
        ):
            k_min, k_max, seq = self._quest_compute_page_stats_full(full_key_states)
            st["k_min"], st["k_max"], st["seq_len"] = k_min, k_max, seq
            return k_min, k_max

        # +1 token update
        new_pos = T - 1
        page_id = new_pos // self.page_size

        token = full_key_states[:, :, new_pos, :].to(torch.float32)  # [B,H_kv,D]
        k_min = st["k_min"]
        k_max = st["k_max"]

        if page_id == k_min.shape[2]:
            k_min = torch.cat([k_min, token.unsqueeze(2)], dim=2)
            k_max = torch.cat([k_max, token.unsqueeze(2)], dim=2)
        else:
            k_min[:, :, page_id, :] = torch.minimum(k_min[:, :, page_id, :], token)
            k_max[:, :, page_id, :] = torch.maximum(k_max[:, :, page_id, :], token)

        st["k_min"], st["k_max"], st["seq_len"] = k_min, k_max, T
        return k_min, k_max

    # ====================== Token-index cache pool helpers ======================

    @torch.no_grad()
    def _unique_pad_bh(self, x: torch.Tensor, pad_value: int = -1):
        """
        Vectorized unique per (B,H) with padding.
        x: [B,H,N] long, pad_value marks padding (excluded from unique)
        returns:
          padded: [B,H,M] long (pad=pad_value)
          valid:  [B,H,M] bool
        """
        B, H, N = x.shape
        device = x.device
        if N == 0:
            padded = torch.full((B, H, 0), pad_value, device=device, dtype=torch.long)
            valid_out = torch.zeros((B, H, 0), device=device, dtype=torch.bool)
            return padded, valid_out

        xs, _ = torch.sort(x, dim=-1)  # pad_value=-1 will come first
        valid = xs != pad_value
        prev = torch.cat(
            [torch.full((B, H, 1), pad_value - 1, device=device, dtype=xs.dtype), xs[..., :-1]], dim=-1
        )
        is_new = valid & (xs != prev)  # [B,H,N]

        pos_raw = torch.cumsum(is_new.to(torch.int32), dim=-1) - 1
        pos = torch.where(is_new, pos_raw.to(torch.long), torch.full_like(pos_raw, -1, dtype=torch.long))

        cnt = is_new.sum(dim=-1)  # [B,H]
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

    @staticmethod
    def _pad_last_dim(x: torch.Tensor, target: int, pad_value):
        n = x.shape[-1]
        if n == target:
            return x
        return torch.nn.functional.pad(x, (0, target - n), value=pad_value)

    @torch.no_grad()
    def _update_cache_pool(self, base_indices: torch.Tensor, total_seq_len: int, past_key_values: Optional[Cache]):
        """
        base_indices: [B,H_kv,N] (no local/cur)
        returns pool_u, pool_valid (pad=-1)
        """
        st = self._quest_get_layer_state(past_key_values)
        base = self._clamp_keep_neg1(base_indices, 0, total_seq_len - 1)

        if st is None:
            return self._unique_pad_bh(base, pad_value=-1)

        # base unique (needed for both strategies)
        base_u, base_valid = self._unique_pad_bh(base, pad_value=-1)
        base_cnt = base_valid.sum(dim=-1)  # [B,H_kv]
        critical_cnt = base_cnt + self.local_window_size + 1

        # fixed_k: early return on reset (avoid merged unique cost)
        if self.cache_pool_strategy == "fixed_k":
            step = int(st.get("step", 0))
            do_reset = (self.cache_pool_k > 0) and ((step + 1) % self.cache_pool_k == 0)

            if do_reset or st["pool"] is None:
                pool_u, pool_valid = base_u, base_valid
            else:
                merged = torch.cat([st["pool"], base], dim=-1)
                merged = self._clamp_keep_neg1(merged, 0, total_seq_len - 1)
                pool_u, pool_valid = self._unique_pad_bh(merged, pad_value=-1)

            st["step"] = step + 1
            st["pool"], st["pool_valid"] = pool_u, pool_valid
            return pool_u, pool_valid

        # threshold: need merged stats
        if st["pool"] is None:
            merged = base
        else:
            merged = torch.cat([st["pool"], base], dim=-1)
            merged = self._clamp_keep_neg1(merged, 0, total_seq_len - 1)

        merged_u, merged_valid = self._unique_pad_bh(merged, pad_value=-1)

        cap = int(self.cache_pool_cap_ratio * float(total_seq_len))
        thr = torch.minimum(2 * critical_cnt, torch.full_like(critical_cnt, cap))

        merged_cnt = merged_valid.sum(dim=-1)  # [B,H_kv]
        need_rebuild = merged_cnt > thr        # [B,H_kv] bool

        N = max(base_u.shape[-1], merged_u.shape[-1])
        base_u_pad = self._pad_last_dim(base_u, N, pad_value=-1)
        base_valid_pad = self._pad_last_dim(base_valid, N, pad_value=False)

        merged_u_pad = self._pad_last_dim(merged_u, N, pad_value=-1)
        merged_valid_pad = self._pad_last_dim(merged_valid, N, pad_value=False)

        mask = need_rebuild.unsqueeze(-1)  # [B,H_kv,1]
        pool_u = torch.where(mask, base_u_pad, merged_u_pad)
        pool_valid = torch.where(mask, base_valid_pad, merged_valid_pad)
        pool_u = torch.where(pool_valid, pool_u, torch.full_like(pool_u, -1))

        st["pool"], st["pool_valid"] = pool_u, pool_valid
        return pool_u, pool_valid

    @torch.no_grad()
    def _append_local_and_cur_and_dedup(self, indices: torch.Tensor, total_seq_len: int):
        """
        indices: [B,H_kv,N] with pad=-1 allowed
        returns token_indices [B,H_kv,M] (pad=-1), valid_mask [B,H_kv,M]
        """
        B, H, _ = indices.shape
        device = indices.device
        cur = total_seq_len - 1

        start = max(0, total_seq_len - self.local_window_size)
        window = torch.arange(start, total_seq_len, device=device, dtype=torch.long).view(1, 1, -1).expand(B, H, -1)
        cur_idx = torch.full((B, H, 1), cur, device=device, dtype=torch.long)

        merged = torch.cat([indices, window, cur_idx], dim=-1)
        merged = self._clamp_keep_neg1(merged, 0, cur)
        return self._unique_pad_bh(merged, pad_value=-1)

    # ====================== Quest index prediction ======================

    def _compute_indices_from_groups(self, top_group_indices: torch.Tensor, total_seq_len: int, device):
        """
        top_group_indices: [B,1,H_kv,k]
        returns token indices [B,H_kv,k*page_size] (clamped)
        """
        bsz, _, num_kv_heads, k = top_group_indices.shape
        offsets = torch.arange(self.page_size, device=device, dtype=torch.long)
        token_indices = (top_group_indices.unsqueeze(-1) * self.page_size) + offsets.view(1, 1, 1, 1, -1)
        token_indices = token_indices.view(bsz, num_kv_heads, -1)
        token_indices = token_indices.clamp(max=total_seq_len - 1)
        return token_indices

    @torch.no_grad()
    def predict_indices(
        self,
        query_states: torch.Tensor,          # [B,H_q,1,D]
        full_key_states: torch.Tensor,       # [B,H_kv,T,D]
        past_key_values: Optional[Cache] = None,
    ):
        if not self.kvswap_enabled:
            return None

        bsz, num_q_heads, q_len, head_dim = query_states.shape
        _, num_kv_heads, total_seq_len, _ = full_key_states.shape

        if q_len != 1:
            return None
        if total_seq_len < self.page_size * 2:
            return None
        if num_kv_heads != self.num_kv_heads:
            # should not happen, but keep safe
            self.num_kv_heads = num_kv_heads

        device = query_states.device

        # Stage 1: cached page stats
        k_min, k_max = self._quest_get_page_stats_cached(full_key_states, past_key_values)  # [B,H_kv,Pages,D]
        num_pages = k_min.shape[2]

        # Stage 2: Quest score
        gqa_group_size = num_q_heads // self.num_kv_heads
        q_view = query_states.view(bsz, self.num_kv_heads, gqa_group_size, q_len, head_dim)  # [B,H_kv,G,1,D]

        k_min_exp = k_min.unsqueeze(2)  # [B,H_kv,1,Pages,D]
        k_max_exp = k_max.unsqueeze(2)

        score_min = q_view * k_min_exp
        score_max = q_view * k_max_exp
        score_merged = torch.max(score_min, score_max)
        page_scores = score_merged.sum(dim=-1)      # [B,H_kv,G,Pages]
        agg_page_scores = page_scores.sum(dim=2)    # [B,H_kv,Pages]

        k = min(self.kv_top_k_groups, num_pages)
        topk_indices = torch.topk(agg_page_scores, k, dim=-1).indices  # [B,H_kv,k]
        topk_indices = topk_indices.unsqueeze(1)  # [B,1,H_kv,k]
        return self._compute_indices_from_groups(topk_indices, total_seq_len, device)

    # ====================== Forward ======================

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        prefetched_token_indices: Optional[torch.Tensor] = None,  # [B,H_kv,N] from prefetch
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [B,H_q,S,D]
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)    # [B,H_kv,S,D]
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [B,H_kv,S,D]

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # optional: keep for debugging parity
        self.rope_query = query_states
        self.rope_key = key_states

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        is_decoding = (query_states.shape[2] == 1) and (key_states.shape[2] > 1)

        # ---------------- Decoding sparse path ----------------
        if self.kvswap_enabled and is_decoding:
            B, H_kv, S_full, D = key_states.shape
            device = key_states.device

            # base indices (no local/cur)
            if prefetched_token_indices is not None:
                base_indices = prefetched_token_indices
            else:
                base_indices = self.predict_indices(query_states, key_states, past_key_values=past_key_value)

            if base_indices is None:
                base_indices = torch.arange(S_full, device=device, dtype=torch.long).view(1, 1, -1).expand(B, H_kv, -1)
            else:
                base_indices = base_indices.to(device=device, dtype=torch.long)

            # update pool (fixed_k / threshold)
            pooled_indices, _ = self._update_cache_pool(base_indices, S_full, past_key_value)

            # add local + cur + dedup
            token_indices, valid_mask = self._append_local_and_cur_and_dedup(pooled_indices, S_full)

            # gather needs non-negative indices
            token_indices_safe = token_indices.clamp(min=0)

            gather_idx = token_indices_safe.unsqueeze(-1).expand(-1, -1, -1, D)
            sparse_key_states = torch.gather(key_states, 2, gather_idx)
            sparse_value_states = torch.gather(value_states, 2, gather_idx)

            # gather causal mask -> sparse mask
            sparse_mask = None
            if attention_mask is not None and attention_mask.dim() == 4 and attention_mask.shape[-1] == key_states.shape[2]:
                # attention_mask: [B,1,1,S_full] (decode)
                mask_expanded = attention_mask.expand(-1, H_kv, -1, -1)           # [B,H_kv,1,S_full]
                mask_indices = token_indices_safe.unsqueeze(2)                      # [B,H_kv,1,N]
                sparse_mask = torch.gather(mask_expanded, 3, mask_indices)          # [B,H_kv,1,N]
                sparse_mask = repeat_kv(sparse_mask, self.num_key_value_groups)     # [B,H_q,1,N]

            # add -inf for invalid (pad) positions
            invalid = (~valid_mask).unsqueeze(2)  # [B,H_kv,1,N]
            add = torch.zeros((B, H_kv, 1, valid_mask.shape[-1]), device=device, dtype=query_states.dtype)
            add.masked_fill_(invalid, torch.finfo(query_states.dtype).min)
            add = repeat_kv(add, self.num_key_value_groups)  # [B,H_q,1,N]
            sparse_mask = add if sparse_mask is None else (sparse_mask + add)

            # repeat kv to match query heads
            sparse_key_states = repeat_kv(sparse_key_states, self.num_key_value_groups)     # [B,H_q,N,D]
            sparse_value_states = repeat_kv(sparse_value_states, self.num_key_value_groups)

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                sparse_key_states,
                sparse_value_states,
                attn_mask=sparse_mask,
                dropout_p=0.0 if not self.training else self.attention_dropout,
                is_causal=False,
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_weights = None

        # ---------------- Prefill / dense path ----------------
        else:
            attention_interface: Callable = eager_attention_forward
            if self.config._attn_implementation != "eager":
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )
            attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @torch.no_grad()
    def prefetch_indices(self, hidden_states, past_key_values, position_embeddings):
        """
        Prefetch indices for THIS layer using approximate query computed from its input.
        Reads full key_states for this layer from past_key_values (DynamicCache variants supported).
        """
        if not self.self_attn.kvswap_enabled:
            return None
        if past_key_values is None:
            return None

        # approximate query: use layernorm + q_proj on the layer input (same as actual path)
        x = self.input_layernorm(hidden_states)
        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, -1, self.self_attn.head_dim)

        approx_q = self.self_attn.q_proj(x).view(hidden_shape).transpose(1, 2)  # [B,H_q,1,D] in decoding

        cos, sin = position_embeddings
        approx_q, _ = apply_rotary_pos_emb(approx_q, approx_q, cos, sin)

        # fetch full key cache for this layer
        full_key_states = None
        target_layer = self.self_attn.layer_idx

        try:
            # Case A: DynamicCache.layers[i].keys (some HF variants)
            if hasattr(past_key_values, "layers"):
                if len(past_key_values.layers) > target_layer:
                    layer_obj = past_key_values.layers[target_layer]
                    if hasattr(layer_obj, "keys"):
                        full_key_states = layer_obj.keys
                    else:
                        return None
                else:
                    return None

            # Case B: DynamicCache.key_cache list (common)
            elif hasattr(past_key_values, "key_cache"):
                if len(past_key_values.key_cache) > target_layer:
                    full_key_states = past_key_values.key_cache[target_layer]
                else:
                    return None

            # Case C: legacy tuple list (k,v)
            else:
                if len(past_key_values) > target_layer:
                    item = past_key_values[target_layer]
                    if isinstance(item, (tuple, list)):
                        full_key_states = item[0]
                    else:
                        return None
                else:
                    return None
        except Exception:
            return None

        if full_key_states is None:
            return None

        return self.self_attn.predict_indices(approx_q, full_key_states, past_key_values=past_key_values)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        prefetched_token_indices: Optional[torch.Tensor] = None,  # [B,H_kv,N]
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            prefetched_token_indices=prefetched_token_indices,
            **kwargs,
        )
        hidden_states = residual + hidden_states

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
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.post_init()

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

        # ---- Prefetch pipeline (decoding only) ----
        is_decoding = (
            hidden_states.shape[1] == 1
            and past_key_values is not None
            and past_key_values.get_seq_length() > 0
        )

        prefetch_buffer_indices = None  # holds indices for current layer, predicted by previous layer

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            indices_for_current_layer = prefetch_buffer_indices

            # prefetch for next layer
            if is_decoding and i < (self.config.num_hidden_layers - 1):
                next_layer_obj = self.layers[i + 1]
                prefetch_buffer_indices = next_layer_obj.prefetch_indices(
                    hidden_states=hidden_states,
                    past_key_values=past_key_values,
                    position_embeddings=position_embeddings,
                )
            else:
                prefetch_buffer_indices = None

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                prefetched_token_indices=indices_for_current_layer,
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
