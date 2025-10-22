# mlx_lm/models/streaming_cache.py
# Copyright © 2025 Apple Inc.

"""
Streaming cache implementation for simultaneous input processing and output generation.
Enables models to process continuous input streams while generating responses.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .cache import (
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
    _BaseCache,
)


class StreamingCacheError(Exception):
    """Base exception for streaming cache operations."""

    pass


class CacheMergeError(StreamingCacheError):
    """Raised when cache merge operations fail."""

    pass


class CacheSeparationError(StreamingCacheError):
    """Raised when cache separation operations fail."""

    pass


@dataclass
class StreamingCacheState:
    """State tracking for streaming cache operations."""

    source_length: int = 0
    target_length: int = 0
    merged: bool = False
    read_mode: bool = True
    position_offset: int = 0


class StreamingCache(_BaseCache):
    """
    A cache implementation that maintains separate source and target caches
    for streaming language model generation.

    This cache enables models to process input tokens while simultaneously
    generating output tokens, supporting true streaming and interruptible
    generation.
    """

    def __init__(
        self,
        cache_type: str = "kv",
        position_offset: int = 10000,
        max_size: Optional[int] = None,
        keep: int = 4,
        quantization_config: Optional[dict] = None,
    ):
        """
        Initialize a streaming cache.

        Args:
            cache_type: Type of underlying cache ("kv", "rotating", "quantized")
            position_offset: Offset for target token positions to avoid conflicts
            max_size: Maximum cache size for rotating caches
            keep: Number of tokens to keep for rotating caches
            quantization_config: Configuration for quantized caches
        """
        super().__init__()

        self.cache_type = cache_type
        self.position_offset = position_offset
        self.max_size = max_size
        self.keep = keep
        self.quantization_config = quantization_config or {}

        # Initialize the three cache components
        self.source_cache = self._create_cache()
        self.target_cache = self._create_cache()
        self.merged_cache = self._create_cache()

        # State tracking
        self.stream_state = StreamingCacheState(position_offset=position_offset)

    def _create_cache(self) -> _BaseCache:
        """Create a cache instance based on the specified type."""
        if self.cache_type == "kv":
            return KVCache()
        elif self.cache_type == "rotating":
            if self.max_size is None:
                raise ValueError("max_size required for rotating cache")
            return RotatingKVCache(max_size=self.max_size, keep=self.keep)
        elif self.cache_type == "quantized":
            return QuantizedKVCache(
                group_size=self.quantization_config.get("group_size", 64),
                bits=self.quantization_config.get("bits", 8),
            )
        else:
            raise ValueError(f"Unknown cache type: {self.cache_type}")

    @property
    def is_merged(self) -> bool:
        """Check if caches are currently merged."""
        return self.stream_state.merged

    @property
    def source_length(self) -> int:
        """Get the current source sequence length."""
        return self.stream_state.source_length

    @property
    def target_length(self) -> int:
        """Get the current target sequence length."""
        return self.stream_state.target_length

    def update_source(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """
        Update the source cache with new input tokens.

        Args:
            keys: New key states
            values: New value states

        Returns:
            Updated keys and values
        """
        if self.stream_state.merged:
            raise CacheMergeError("Cannot update source cache while merged")

        keys, values = self.source_cache.update_and_fetch(keys, values)
        self.stream_state.source_length = self.source_cache.offset
        return keys, values

    def update_target(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """
        Update the target cache with generated tokens.

        Args:
            keys: New key states
            values: New value states

        Returns:
            Updated keys and values
        """
        if self.stream_state.merged:
            # Update the merged cache directly
            num_tokens = keys.shape[2]
            keys, values = self.merged_cache.update_and_fetch(keys, values)
            self.stream_state.target_length += num_tokens
        else:
            keys, values = self.target_cache.update_and_fetch(keys, values)
            self.stream_state.target_length = self.target_cache.offset

        return keys, values

    def merge(self) -> None:
        """
        Merge source and target caches for full attention during generation.
        This allows target tokens to attend to all cached tokens.
        """
        if self.stream_state.merged:
            return

        # Get the cached states (slice to used portion only)
        source_keys = self.source_cache.keys
        source_values = self.source_cache.values
        target_keys = self.target_cache.keys
        target_values = self.target_cache.values

        if source_keys is None and target_keys is None:
            self.stream_state.merged = True
            return

        # Slice to only the used portions based on offset
        if source_keys is not None:
            source_keys = source_keys[:, :, : self.source_cache.offset, :]
            source_values = source_values[:, :, : self.source_cache.offset, :]
        if target_keys is not None:
            target_keys = target_keys[:, :, : self.target_cache.offset, :]
            target_values = target_values[:, :, : self.target_cache.offset, :]

        # Reset merged cache
        self.merged_cache = self._create_cache()

        # Merge the caches
        if source_keys is not None and target_keys is not None:
            merged_keys = mx.concatenate([source_keys, target_keys], axis=2)
            merged_values = mx.concatenate([source_values, target_values], axis=2)
        elif source_keys is not None:
            merged_keys = source_keys
            merged_values = source_values
        else:
            merged_keys = target_keys
            merged_values = target_values

        # Update merged cache state
        self.merged_cache.keys = merged_keys
        self.merged_cache.values = merged_values
        self.merged_cache.offset = (
            self.stream_state.source_length + self.stream_state.target_length
        )

        self.stream_state.merged = True

    def separate(self) -> None:
        """
        Separate the merged cache back into source and target components.
        This is needed when switching from generation back to input processing.
        """
        if not self.stream_state.merged:
            return

        if self.merged_cache.keys is None:
            self.stream_state.merged = False
            return

        keys = self.merged_cache.keys
        values = self.merged_cache.values

        # Split at the source boundary
        source_len = self.stream_state.source_length

        if source_len > 0 and self.stream_state.target_length > 0:
            # Restore source cache
            self.source_cache.keys = keys[:, :, :source_len, :]
            self.source_cache.values = values[:, :, :source_len, :]
            self.source_cache.offset = source_len

            # Restore target cache
            self.target_cache.keys = keys[:, :, source_len:, :]
            self.target_cache.values = values[:, :, source_len:, :]
            self.target_cache.offset = self.stream_state.target_length
        elif source_len > 0:
            self.source_cache.keys = keys
            self.source_cache.values = values
            self.source_cache.offset = source_len
        else:
            self.target_cache.keys = keys
            self.target_cache.values = values
            self.target_cache.offset = self.stream_state.target_length

        self.stream_state.merged = False

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """
        Update cache based on current mode (read/write).

        Args:
            keys: New key states
            values: New value states

        Returns:
            Updated keys and values
        """
        if self.stream_state.read_mode:
            return self.update_source(keys, values)
        else:
            return self.update_target(keys, values)

    def set_mode(self, read_mode: bool) -> None:
        """
        Switch between read mode (processing input) and write mode (generation).

        Args:
            read_mode: True for input processing, False for generation
        """
        self.stream_state.read_mode = read_mode

        if not read_mode and not self.stream_state.merged:
            # Automatically merge when switching to write mode
            self.merge()
        elif read_mode and self.stream_state.merged:
            # Automatically separate when switching back to read mode
            self.separate()

    def clear(self) -> None:
        """Clear all caches and reset state."""
        self.source_cache = self._create_cache()
        self.target_cache = self._create_cache()
        self.merged_cache = self._create_cache()
        self.stream_state = StreamingCacheState(position_offset=self.position_offset)

    def trim_source(self, num_tokens: int) -> None:
        """
        Trim tokens from the source cache.

        Args:
            num_tokens: Number of tokens to remove from the end
        """
        if self.stream_state.merged:
            raise CacheMergeError("Cannot trim source while merged")

        if hasattr(self.source_cache, "trim"):
            self.source_cache.trim(num_tokens)
            self.stream_state.source_length = max(
                0, self.stream_state.source_length - num_tokens
            )

    def trim_target(self, num_tokens: int) -> None:
        """
        Trim tokens from the target cache.

        Args:
            num_tokens: Number of tokens to remove from the end
        """
        if self.stream_state.merged:
            raise CacheMergeError("Cannot trim target while merged")

        if hasattr(self.target_cache, "trim"):
            self.target_cache.trim(num_tokens)
            self.stream_state.target_length = max(
                0, self.stream_state.target_length - num_tokens
            )

    @property
    def keys(self) -> Optional[mx.array]:
        """Get current keys based on mode and merge state."""
        if self.stream_state.merged:
            return self.merged_cache.keys
        elif self.stream_state.read_mode:
            return self.source_cache.keys
        else:
            return self.target_cache.keys

    @property
    def values(self) -> Optional[mx.array]:
        """Get current values based on mode and merge state."""
        if self.stream_state.merged:
            return self.merged_cache.values
        elif self.stream_state.read_mode:
            return self.source_cache.values
        else:
            return self.target_cache.values

    @property
    def offset(self) -> int:
        """Get current offset based on mode and merge state."""
        if self.stream_state.merged:
            return self.merged_cache.offset
        elif self.stream_state.read_mode:
            return self.source_cache.offset
        else:
            return self.target_cache.offset

    def state_dict(self) -> dict:
        """Export cache state for checkpointing."""
        return {
            "source_keys": self.source_cache.keys,
            "source_values": self.source_cache.values,
            "source_offset": self.source_cache.offset,
            "target_keys": self.target_cache.keys,
            "target_values": self.target_cache.values,
            "target_offset": self.target_cache.offset,
            "state": {
                "source_length": self.stream_state.source_length,
                "target_length": self.stream_state.target_length,
                "merged": self.stream_state.merged,
                "read_mode": self.stream_state.read_mode,
                "position_offset": self.stream_state.position_offset,
            },
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load cache state from checkpoint."""
        self.source_cache.keys = state_dict["source_keys"]
        self.source_cache.values = state_dict["source_values"]
        self.source_cache.offset = state_dict["source_offset"]

        self.target_cache.keys = state_dict["target_keys"]
        self.target_cache.values = state_dict["target_values"]
        self.target_cache.offset = state_dict["target_offset"]

        state = state_dict["state"]
        self.stream_state = StreamingCacheState(
            source_length=state["source_length"],
            target_length=state["target_length"],
            merged=state["merged"],
            read_mode=state["read_mode"],
            position_offset=state["position_offset"],
        )

        # Rebuild merged cache if needed
        if self.stream_state.merged:
            self.merge()


class StreamingCacheList(list):
    """
    A list of caches for multi-layer models.
    Provides unified operations across all layers. Can contain both
    StreamingCache instances and other cache types (e.g., MambaCache).
    """

    def __init__(self, caches: List):
        """
        Initialize with a list of cache instances.

        Args:
            caches: List of cache instances, one per layer.
                   Can be StreamingCache or other cache types.
        """
        super().__init__(caches)

    def set_mode(self, read_mode: bool) -> None:
        """Set mode for all StreamingCache instances."""
        for cache in self:
            if isinstance(cache, StreamingCache):
                cache.set_mode(read_mode)

    def merge_all(self) -> None:
        """Merge all StreamingCache instances."""
        for cache in self:
            if isinstance(cache, StreamingCache):
                cache.merge()

    def separate_all(self) -> None:
        """Separate all StreamingCache instances."""
        for cache in self:
            if isinstance(cache, StreamingCache):
                cache.separate()

    def clear_all(self) -> None:
        """Clear all caches."""
        for cache in self:
            if isinstance(cache, StreamingCache):
                cache.clear()
            elif hasattr(cache, "keys"):
                # Clear regular KV-style caches
                cache.keys = None
                cache.values = None
                cache.offset = 0

    @property
    def is_merged(self) -> bool:
        """Check if all StreamingCache instances are merged."""
        streaming_caches = [c for c in self if isinstance(c, StreamingCache)]
        return (
            all(cache.is_merged for cache in streaming_caches)
            if streaming_caches
            else False
        )

    @property
    def total_source_length(self) -> int:
        """Get maximum source length across all StreamingCache instances."""
        streaming_caches = [c for c in self if isinstance(c, StreamingCache)]
        return (
            max(cache.source_length for cache in streaming_caches)
            if streaming_caches
            else 0
        )

    @property
    def total_target_length(self) -> int:
        """Get maximum target length across all StreamingCache instances."""
        streaming_caches = [c for c in self if isinstance(c, StreamingCache)]
        return (
            max(cache.target_length for cache in streaming_caches)
            if streaming_caches
            else 0
        )
