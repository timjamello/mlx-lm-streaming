# Copyright © 2025 Apple Inc.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import KVCache, MambaCache
from .gated_delta import gated_delta_update
from .rope_utils import initialize_rope
from .streaming_cache import StreamingCache, StreamingCacheList
from .switch_layers import SwitchGLU


def create_streaming_attention_mask(
    query_len: int,
    cache: StreamingCache,
    is_source_query: bool = False,
    device: Optional[mx.Device] = None,
) -> mx.array:
    """
    Create attention mask for streaming mode with full attention layers.

    Args:
        query_len: Number of query tokens (will be added to cache)
        cache: StreamingCache instance with state information
        is_source_query: Whether queries are source (input) tokens
        device: Device to create mask on

    Returns:
        Attention mask of shape [1, 1, query_len, total_kv_len]
        where True/1 = can attend, False/-inf = cannot attend
    """
    # Handle zero query length edge case
    if query_len == 0:
        # Return empty mask with correct shape
        return mx.zeros((1, 1, 0, 0), dtype=mx.bfloat16)

    # Calculate sizes based on actual cache state
    # The mask size must match what the keys/values will be AFTER update_and_fetch
    if cache.stream_state.merged:
        # Merged: all tokens in one cache
        current_total = cache.merged_cache.offset
        total_kv_len = current_total + query_len
        # For the mask logic, we need to know the source/target split
        source_len = cache.source_length
        target_len = cache.target_length
        if not is_source_query:
            # Adding to target in merged mode
            target_len += query_len
    else:
        # Separated: source and target are in separate caches
        source_len = cache.source_cache.offset
        target_len = cache.target_cache.offset
        if is_source_query:
            # Processing source tokens - they only see source, not target
            source_len += query_len
            total_kv_len = source_len  # Only source tokens!
        else:
            # Processing target tokens - they see all source + target
            target_len += query_len
            total_kv_len = source_len + target_len

    if total_kv_len == 0:
        # First tokens, use standard causal
        return create_attention_mask(mx.zeros((1, query_len, 1)), None)

    # Create mask (use bfloat16 to match model dtype)
    mask = mx.full((query_len, total_kv_len), -mx.inf, dtype=mx.bfloat16)

    if is_source_query:
        # Source tokens processing new input
        # They can see all previous source tokens (causal) but NO target tokens
        query_offset = source_len - query_len

        # Build mask row by row
        rows = []
        for i in range(query_len):
            actual_pos = query_offset + i
            row_mask = mx.full((total_kv_len,), -mx.inf, dtype=mx.bfloat16)
            # Can attend to source tokens up to current position (causal)
            if actual_pos >= 0:
                row_mask = mx.where(
                    mx.arange(total_kv_len) <= actual_pos,
                    mx.array(0.0, dtype=mx.bfloat16),
                    row_mask,
                )
            rows.append(row_mask)
        mask = mx.stack(rows)
    else:
        # Target tokens (generation mode)
        # They can see all source tokens and previous target tokens
        query_offset = target_len - query_len

        # Build mask row by row
        rows = []
        for i in range(query_len):
            row_mask = mx.full((total_kv_len,), -mx.inf, dtype=mx.bfloat16)

            # Can see all source tokens (they've all "arrived")
            row_mask = mx.where(
                mx.arange(total_kv_len) < source_len,
                mx.array(0.0, dtype=mx.bfloat16),
                row_mask,
            )

            # Can see target tokens up to current position (autoregressive)
            target_pos = query_offset + i
            if target_pos >= 0:
                row_mask = mx.where(
                    (mx.arange(total_kv_len) >= source_len)
                    & (mx.arange(total_kv_len) < source_len + target_pos + 1),
                    mx.array(0.0, dtype=mx.bfloat16),
                    row_mask,
                )

            rows.append(row_mask)

        mask = mx.stack(rows)

    # Reshape to [1, 1, query_len, kv_len] for broadcasting
    mask = mask[None, None, :, :]

    # Validate mask shape
    expected_shape = (1, 1, query_len, total_kv_len)
    assert mask.shape == expected_shape, (
        f"Attention mask shape mismatch: got {mask.shape}, "
        f"expected {expected_shape} "
        f"(query_len={query_len}, total_kv_len={total_kv_len}, "
        f"source_len={source_len}, target_len={target_len}, "
        f"is_source_query={is_source_query}, merged={cache.stream_state.merged})"
    )

    return mask


def create_streaming_ssm_mask(
    query_len: int,
    cache: StreamingCache,
    is_source_query: bool = False,
) -> Optional[mx.array]:
    """
    Create mask for linear attention (GatedDeltaNet) layers.

    SSM layers use state-based processing where the mask indicates
    valid token positions vs padding.

    Args:
        query_len: Number of query tokens being processed
        cache: StreamingCache instance with state information
        is_source_query: Whether processing source (input) tokens

    Returns:
        Mask of shape [1, query_len] where True = valid, False = masked
        Or None if no masking needed
    """
    if not is_source_query:
        # Target tokens: State-based causality handles masking automatically
        return None

    # Source tokens: Mark all positions as valid (no padding within chunks)
    # Shape [1, query_len] to broadcast with [B, S, ...] tensors in conv1d
    mask = mx.ones((1, query_len), dtype=mx.bool_)

    return mask


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    linear_num_value_heads: int
    linear_num_key_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    num_experts: int
    num_experts_per_tok: int
    decoder_sparse_step: int
    shared_expert_intermediate_size: int
    mlp_only_layers: List[int]
    moe_intermediate_size: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    rope_theta: float
    partial_rotary_factor: float
    max_position_embeddings: int
    norm_topk_prob: bool = False
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    head_dim: Optional[int] = None
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    full_attention_interval: int = 4
    # Streaming-specific parameters
    streaming_mode: bool = False
    target_position_offset: int = 0  # Changed from position_offset


class Qwen3NextRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)

    def __call__(
        self, hidden_states: mx.array, gate: mx.array | None = None
    ) -> mx.array:
        x = mx.fast.rms_norm(hidden_states, self.weight, self.eps)
        if gate is not None:
            x = x * nn.silu(gate)
        return x


class Qwen3NextAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_key_value_heads = args.num_key_value_heads
        self.num_attention_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size,
            self.num_attention_heads * self.head_dim * 2,
            bias=args.attention_bias,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )

        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.rope = initialize_rope(
            int(self.head_dim * args.partial_rotary_factor),
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        q_proj_output = self.q_proj(x)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)

        keys, values = self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        if isinstance(cache, StreamingCache):
            # Streaming mode: source and target use separate position spaces
            if cache.stream_state.read_mode:
                # Processing source tokens: continuous positions from 0
                offset = cache.source_length
            else:
                # Generating target tokens: separate position space
                # Starts at target_position_offset (can be 0)
                offset = cache.target_position_offset + cache.target_length

            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output * mx.sigmoid(gate))


class Qwen3NextMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3NextGatedDeltaNet(nn.Module):
    """
    Linear attention layer that uses state-based processing.
    For streaming, this needs different handling than attention layers.
    """

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_norm_epsilon = config.rms_norm_eps

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )

        self.in_proj_qkvz = nn.Linear(
            self.hidden_size, self.key_dim * 2 + self.value_dim * 2, bias=False
        )
        self.in_proj_ba = nn.Linear(self.hidden_size, self.num_v_heads * 2, bias=False)

        self.dt_bias = mx.ones(self.num_v_heads)

        A = mx.random.uniform(low=0, high=16, shape=(self.num_v_heads,))
        self.A_log = mx.log(A)

        self.norm = Qwen3NextRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        # For streaming: track whether we're processing source or target
        self.streaming_mode = False

    def fix_query_key_value_ordering(
        self, mixed_qkvz: mx.array, mixed_ba: mx.array
    ) -> mx.array:
        nk, dn, nv, dv = (
            self.num_k_heads,
            self.head_k_dim,
            self.num_v_heads,
            self.head_v_dim,
        )
        mixed_qkvz = mixed_qkvz.reshape(*mixed_qkvz.shape[:-1], nk, -1)
        mixed_ba = mixed_ba.reshape(*mixed_ba.shape[:-1], nk, -1)
        q, k, v, z = mx.split(mixed_qkvz, [dn, 2 * dn, 2 * dn + nv // nk * dv], axis=-1)
        b, a = mx.split(mixed_ba, [nv // nk], axis=-1)
        return (
            q,
            k,
            v.reshape(*v.shape[:2], -1, dv),
            z.reshape(*z.shape[:2], -1, dv),
            b.reshape(*b.shape[:2], nv),
            a.reshape(*a.shape[:2], nv),
        )

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, S, _ = inputs.shape

        # For streaming with linear attention, we need to handle state differently
        # The MambaCache handles conv_state and ssm_state
        if isinstance(cache, StreamingCache):
            # In streaming mode, linear attention maintains separate states
            # for source and target processing
            self.streaming_mode = True
            is_source = cache.stream_state.read_mode

            # Process normally but with awareness of mode
            # The state updates will be handled by the MambaCache

        q, k, v, z, b, a = self.fix_query_key_value_ordering(
            self.in_proj_qkvz(inputs), self.in_proj_ba(inputs)
        )

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )

        mixed_qkv = mx.concatenate(
            [q.reshape(B, S, -1), k.reshape(B, S, -1), v.reshape(B, S, -1)], axis=-1
        )
        if mask is not None:
            mixed_qkv = mx.where(mask[..., None], mixed_qkv, 0)
        conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
        if cache is not None:
            cache[0] = conv_input[:, -(self.conv_kernel_size - 1) :]
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        state = cache[1] if cache else None
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
        )

        if cache is not None:
            cache[1] = state

        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1))


class Qwen3NextSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        intermediate_size = args.moe_intermediate_size
        shared_expert_intermediate_size = args.shared_expert_intermediate_size

        self.norm_topk_prob = args.norm_topk_prob
        self.num_experts = num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok

        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.switch_mlp = SwitchGLU(dim, intermediate_size, num_experts)

        self.shared_expert = Qwen3NextMLP(dim, shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(dim, 1, bias=False)

    def __call__(
        self,
        x: mx.array,
    ) -> mx.array:
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)

        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y

        return y + shared_y


class Qwen3NextDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_linear = (layer_idx + 1) % args.full_attention_interval != 0

        if self.is_linear:
            self.linear_attn = Qwen3NextGatedDeltaNet(args)
        else:
            self.self_attn = Qwen3NextAttention(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

        if (layer_idx not in args.mlp_only_layers) and (
            args.num_experts > 0 and (layer_idx + 1) % args.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3NextSparseMoeBlock(args)
        else:
            self.mlp = Qwen3NextMLP(args.hidden_size, args.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class Qwen3NextModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Qwen3NextDecoderLayer(args=args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ssm_idx = 0
        self.fa_idx = args.full_attention_interval - 1

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        hidden_states = self.embed_tokens(inputs)

        # Handle different cache types
        if isinstance(cache, StreamingCacheList):
            # Streaming mode with unified cache handling
            cache_list = cache
            # Find the first StreamingCache to check read mode
            streaming_cache = next(
                (c for c in cache_list if isinstance(c, StreamingCache)), None
            )
            is_source_input = (
                streaming_cache.stream_state.read_mode if streaming_cache else True
            )

            for layer, layer_cache in zip(self.layers, cache_list):
                if layer.is_linear:
                    # Linear layers use SSM-style masking
                    mask = create_streaming_ssm_mask(
                        hidden_states.shape[1],
                        layer_cache,
                        is_source_query=is_source_input,
                    )
                else:
                    # Full attention layers use attention masking
                    mask = create_streaming_attention_mask(
                        hidden_states.shape[1],
                        layer_cache,
                        is_source_query=is_source_input,
                    )

                hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)

        elif cache is None:
            # No cache, standard forward pass
            cache = [None] * len(self.layers)
            for layer, c in zip(self.layers, cache):
                mask = None  # Let layers handle default masking
                hidden_states = layer(hidden_states, mask=mask, cache=c)

        else:
            # Standard cache mode (non-streaming)
            for layer, c in zip(self.layers, cache):
                if layer.is_linear:
                    mask = create_ssm_mask(hidden_states, c)
                else:
                    mask = create_attention_mask(hidden_states, c)
                hidden_states = layer(hidden_states, mask=mask, cache=c)

        return self.norm(hidden_states)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3NextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self, streaming: bool = False, target_position_offset: int = 0):
        """
        Create cache appropriate for the model configuration.

        Args:
            streaming: If True, create StreamingCache for simultaneous input/output
            target_position_offset: Starting position for target tokens in streaming.
                0 (default) = separate position space from source
                Large value (e.g. 10000) = avoid position conflicts with source
        """
        if streaming:
            from .streaming_cache import StreamingCache, StreamingCacheList

            caches = []
            for layer in self.layers:
                if layer.is_linear:
                    # Linear layers use MambaCache (no streaming support yet)
                    cache = MambaCache()
                else:
                    # Full attention layers use StreamingCache
                    cache = StreamingCache(
                        cache_type="kv",
                        target_position_offset=target_position_offset,
                    )
                caches.append(cache)

            return StreamingCacheList(caches)
        else:
            # Standard cache creation
            return [MambaCache() if l.is_linear else KVCache() for l in self.layers]

    def sanitize(self, weights):
        if "model.layers.0.mlp.experts.0.up_proj.weight" not in weights:
            return weights
        weights = {key: value for key, value in weights.items() if "mtp." not in key}

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}.mlp"
            for n in ["up_proj", "down_proj", "gate_proj"]:
                to_join = [
                    weights.pop(f"{prefix}.experts.{e}.{n}.weight")
                    for e in range(self.args.num_experts)
                ]
                weights[f"{prefix}.switch_mlp.{n}.weight"] = mx.stack(to_join)

        norm_keys = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        )
        for k, v in weights.items():
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
            if any(k.endswith(sfx) for sfx in norm_keys):
                if v.ndim == 1:
                    weights[k] = v + 1.0
        return weights

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate
