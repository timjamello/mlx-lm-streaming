# Copyright © 2025 Apple Inc.
# Streaming variant of Qwen3MoE for StreamingLLM implementation

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope
from .streaming_cache import DualStreamingCache
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_experts: int
    num_experts_per_tok: int
    decoder_sparse_step: int
    mlp_only_layers: List[int]
    moe_intermediate_size: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float
    tie_word_embeddings: bool
    max_position_embeddings: int
    norm_topk_prob: bool
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None


class StreamingAttention(nn.Module):
    """
    Streaming variant of Qwen3MoE Attention with dual cache support.

    Key differences from standard Attention:
    1. Accepts custom position_ids instead of using cache.offset
    2. Routes to source_cache or target_cache based on is_reading flag
    3. Applies separate position encodings for source and target streams
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        assert args.num_key_value_heads is not None
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        head_dim = getattr(
            args, "head_dim", args.hidden_size // args.num_attention_heads
        )
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)

        self.rope = initialize_rope(
            head_dim,
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
        position_ids: Optional[mx.array] = None,
        is_reading: bool = True,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(
            0, 2, 1, 3
        )
        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            # Determine offset/position for RoPE
            if position_ids is not None:
                offset = int(position_ids.flatten()[0])
            else:
                if isinstance(cache, DualStreamingCache):
                    offset = cache.source_offset if is_reading else cache.target_offset
                else:
                    offset = cache.offset

            # Apply RoPE with computed offset
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)

            # Route to appropriate cache based on mode
            if isinstance(cache, DualStreamingCache):
                if is_reading:
                    # Reading phase: update source cache ONLY
                    keys, values = cache.update_source(keys, values)
                else:
                    # Writing phase: update target cache
                    cache.update_target(keys, values)

                    # Merge caches for attention
                    cache.merge_source_target()
                    keys, values = cache.get_merged()
            else:
                # Standard cache (non-streaming mode)
                keys, values = cache.update_and_fetch(keys, values)
        else:
            # No cache - standard attention
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        # Separate caches after attention (writing phase only)
        if (
            cache is not None
            and isinstance(cache, DualStreamingCache)
            and not is_reading
        ):
            cache.separate_source_target()

        return self.o_proj(output)


class MLP(nn.Module):
    """Standard MLP - same as original Qwen3MoE"""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3MoeSparseMoeBlock(nn.Module):
    """MoE block - no streaming changes needed as it operates on hidden states only"""

    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        intermediate_size = args.moe_intermediate_size

        self.num_experts = num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob

        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.switch_mlp = SwitchGLU(dim, intermediate_size, num_experts)

    def __call__(
        self,
        x: mx.array,
    ):
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)

        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores /= mx.sum(scores, axis=-1, keepdims=True)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        return y


class StreamingDecoderLayer(nn.Module):
    """
    Streaming variant of Qwen3MoeDecoderLayer that passes through streaming parameters.
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.self_attn = StreamingAttention(args, layer_idx)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args

        # Conditionally use MoE or standard MLP
        if (layer_idx not in args.mlp_only_layers) and (
            args.num_experts > 0 and (layer_idx + 1) % args.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3MoeSparseMoeBlock(args)
        else:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_ids: Optional[mx.array] = None,
        is_reading: bool = True,
    ) -> mx.array:
        """
        Forward pass for streaming decoder layer.

        Args:
            x: Input tensor
            mask: Optional attention mask
            cache: DualStreamingCache or standard cache
            position_ids: Custom position IDs
            is_reading: Reading (True) or writing (False) mode

        Returns:
            Output tensor
        """
        r = self.self_attn(
            self.input_layernorm(x), mask, cache, position_ids, is_reading
        )
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class Qwen3MoeModelStreaming(nn.Module):
    """
    Streaming variant of Qwen3MoeModel that supports dual cache and streaming generation.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            StreamingDecoderLayer(args=args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        is_reading: bool = True,
    ):
        """
        Forward pass for streaming model.

        Args:
            inputs: Input token IDs
            cache: List of caches (can be DualStreamingCache or standard)
            input_embeddings: Optional pre-computed embeddings
            position_ids: Custom position IDs for streaming
            is_reading: Reading (True) or writing (False) mode

        Returns:
            Hidden states after all layers
        """
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        # Only use causal mask during READ mode
        # During WRITE mode, mask = None to allow full attention to source tokens
        # Matches StreamingLLM's qwen_streaming.py:879 where causal_mask = None
        if is_reading:
            mask = create_attention_mask(h, cache[0])
        else:
            mask = None  # No mask during generation = attend to all tokens

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c, position_ids, is_reading)

        return self.norm(h)


class Model(nn.Module):
    """
    Top-level streaming Qwen3MoE model with language modeling head.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3MoeModelStreaming(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        is_reading: bool = True,
    ):
        """
        Forward pass with language modeling head.

        Args:
            inputs: Input token IDs
            cache: List of caches
            input_embeddings: Optional pre-computed embeddings
            position_ids: Custom position IDs for streaming
            is_reading: Reading (True) or writing (False) mode

        Returns:
            Logits for next token prediction
        """
        out = self.model(inputs, cache, input_embeddings, position_ids, is_reading)
        return self.lm_head(out)

    def make_cache(self):
        """
        Create streaming caches for all layers.

        Returns:
            List of DualStreamingCache instances
        """
        return [DualStreamingCache() for _ in range(self.args.num_hidden_layers)]

    def sanitize(self, weights):
        """Remove unnecessary weights and reorganize MoE expert weights for loading."""
        if "model.layers.0.mlp.experts.0.up_proj.weight" not in weights:
            return weights

        # Reorganize MoE expert weights
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            for n in ["up_proj", "down_proj", "gate_proj"]:
                if f"{prefix}.mlp.experts.0.{n}.weight" in weights:
                    to_join = [
                        weights.pop(f"{prefix}.mlp.experts.{e}.{n}.weight")
                        for e in range(self.args.num_experts)
                    ]
                    weights[f"{prefix}.mlp.switch_mlp.{n}.weight"] = mx.stack(to_join)

        return weights

    @property
    def quant_predicate(self):
        """Quantization predicate for the model."""

        def predicate(path, _):
            if path.endswith("mlp.gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def layers(self):
        return self.model.layers
