
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from . import qwen3_streaming
from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict

    @classmethod
    def from_dict(cls, params):
        if "text_config" not in params:
            return cls(model_type=params["model_type"], text_config=params)
        return super().from_dict(params)


class Model(nn.Module):
    """
    Streaming Qwen3-VL model wrapper.

    This wraps the streaming language model and passes through streaming parameters.
    Vision tower weights are handled during sanitize.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = qwen3_streaming.Model(
            qwen3_streaming.ModelArgs.from_dict(args.text_config)
        )

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        is_reading: bool = True,
    ):
        """
        Forward pass with streaming support.

        Args:
            inputs: Input token IDs
            cache: List of caches
            input_embeddings: Optional pre-computed embeddings
            position_ids: Custom position IDs for streaming
            is_reading: Reading (True) or writing (False) mode

        Returns:
            Logits for next token prediction
        """
        return self.language_model(
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            position_ids=position_ids,
            is_reading=is_reading,
        )

    def make_cache(self):
        """
        Create streaming caches for all layers.

        Returns:
            List of DualStreamingCache instances
        """
        return self.language_model.make_cache()

    def sanitize(self, weights):
        """Remove vision tower weights and reorganize for loading."""
        weights = tree_unflatten(list(weights.items()))
        weights.pop("vision_tower", None)
        weights = dict(tree_flatten(weights))

        sanitized = {}
        for key, value in weights.items():
            if not key.startswith("language_model."):
                key = "language_model." + key
            sanitized[key] = value
        return sanitized

    @property
    def layers(self):
        return self.language_model.model.layers
