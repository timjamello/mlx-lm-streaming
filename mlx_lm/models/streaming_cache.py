# Copyright © 2023-2024 Apple Inc.
# Streaming cache implementation for StreamingLLM

from typing import Optional, Tuple

import mlx.core as mx

from .cache import KVCache, _BaseCache


class DualStreamingCache(_BaseCache):
    """
    Dual cache system for streaming LLM generation.

    Maintains separate caches for source (input) and target (output) tokens,
    enabling independent position encodings for each. This is the foundation
    of the StreamingLLM approach.

    The cache can be in two modes:
    - Separated: source_cache and target_cache are independent
    - Merged: combined cache for attention computation

    Example usage:
        cache = DualStreamingCache()

        # During source reading phase
        cache.update_source(source_keys, source_values)

        # During target generation phase
        cache.update_target(target_keys, target_values)

        # For attention computation, merge caches
        cache.merge_source_target()
        merged_keys, merged_values = cache.get_merged()

        # After attention, separate back
        cache.separate_source_target()
    """

    def __init__(self):
        self.source_cache = KVCache()
        self.target_cache = KVCache()
        self._merged_keys = None
        self._merged_values = None
        self._is_merged = False

    @property
    def offset(self) -> int:
        """
        Total offset is the sum of source and target offsets.
        """
        return self.source_cache.offset + self.target_cache.offset

    @property
    def source_offset(self) -> int:
        """Offset in source cache only."""
        return self.source_cache.offset

    @property
    def target_offset(self) -> int:
        """Offset in target cache only."""
        return self.target_cache.offset

    def update_source(self, keys: mx.array, values: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Update the source cache with new keys/values.

        Args:
            keys: Source keys of shape (B, n_kv_heads, seq_len, head_dim)
            values: Source values of shape (B, n_kv_heads, seq_len, head_dim)

        Returns:
            Tuple of (all_source_keys, all_source_values)
        """
        if self._is_merged:
            raise RuntimeError(
                "Cannot update source cache while in merged state. "
                "Call separate_source_target() first."
            )
        return self.source_cache.update_and_fetch(keys, values)

    def update_target(self, keys: mx.array, values: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Update the target cache with new keys/values.

        Args:
            keys: Target keys of shape (B, n_kv_heads, seq_len, head_dim)
            values: Target values of shape (B, n_kv_heads, seq_len, head_dim)

        Returns:
            Tuple of (all_target_keys, all_target_values)
        """
        if self._is_merged:
            raise RuntimeError(
                "Cannot update target cache while in merged state. "
                "Call separate_source_target() first."
            )
        return self.target_cache.update_and_fetch(keys, values)

    def merge_source_target(self):
        """
        Merge source and target caches for attention computation.

        Concatenates source and target KV caches along the sequence dimension,
        allowing the model to attend to both source and previously generated
        target tokens.
        """
        if self._is_merged:
            return  # Already merged

        source_len = self.source_cache.offset
        target_len = self.target_cache.offset

        # Get the actual used portions of the caches
        if self.source_cache.keys is not None and source_len > 0:
            source_keys = self.source_cache.keys[..., :source_len, :]
            source_values = self.source_cache.values[..., :source_len, :]
        else:
            source_keys = None
            source_values = None

        if self.target_cache.keys is not None and target_len > 0:
            target_keys = self.target_cache.keys[..., :target_len, :]
            target_values = self.target_cache.values[..., :target_len, :]
        else:
            target_keys = None
            target_values = None

        # Handle different cases
        if source_keys is None and target_keys is None:
            # Both empty - shouldn't happen but handle gracefully
            self._merged_keys = None
            self._merged_values = None
        elif target_keys is None or target_len == 0:
            # Only source tokens
            self._merged_keys = source_keys
            self._merged_values = source_values
        elif source_keys is None or source_len == 0:
            # Only target tokens (unusual but possible)
            self._merged_keys = target_keys
            self._merged_values = target_values
        else:
            # Both source and target present - concatenate
            self._merged_keys = mx.concatenate(
                [source_keys, target_keys],
                axis=-2
            )
            self._merged_values = mx.concatenate(
                [source_values, target_values],
                axis=-2
            )

        self._is_merged = True

    def separate_source_target(self):
        """
        Separate the merged cache back into source and target caches.

        This extracts the source and target portions from the merged cache
        and updates the individual caches accordingly.

        Note: This method preserves the existing cache states. If the merged
        cache hasn't been modified, separation is a no-op beyond updating the
        merged state flags.
        """
        if not self._is_merged:
            return  # Already separated

        # Simply clear the merged cache state
        # The source and target caches maintain their own state
        self._merged_keys = None
        self._merged_values = None
        self._is_merged = False

    def get_merged(self) -> Tuple[mx.array, mx.array]:
        """
        Get the merged keys and values.

        Returns:
            Tuple of (merged_keys, merged_values)

        Raises:
            RuntimeError: If cache is not in merged state
        """
        if not self._is_merged:
            raise RuntimeError(
                "Cache is not in merged state. Call merge_source_target() first."
            )
        return self._merged_keys, self._merged_values

    def get_source(self) -> Tuple[Optional[mx.array], Optional[mx.array]]:
        """
        Get source keys and values.

        Returns:
            Tuple of (source_keys, source_values)
        """
        if self.source_cache.keys is None:
            return None, None
        source_len = self.source_cache.offset
        return (
            self.source_cache.keys[..., :source_len, :],
            self.source_cache.values[..., :source_len, :]
        )

    def get_target(self) -> Tuple[Optional[mx.array], Optional[mx.array]]:
        """
        Get target keys and values.

        Returns:
            Tuple of (target_keys, target_values)
        """
        if self.target_cache.keys is None:
            return None, None
        target_len = self.target_cache.offset
        return (
            self.target_cache.keys[..., :target_len, :],
            self.target_cache.values[..., :target_len, :]
        )

    def reset_target(self):
        """
        Reset the target cache while keeping source cache intact.
        Useful for generating multiple outputs from the same source.
        """
        self.target_cache = KVCache()
        if self._is_merged:
            self._merged_keys = None
            self._merged_values = None
            self._is_merged = False

    def reset(self):
        """Reset both source and target caches."""
        self.source_cache = KVCache()
        self.target_cache = KVCache()
        self._merged_keys = None
        self._merged_values = None
        self._is_merged = False

    @property
    def state(self):
        """Get state for serialization."""
        return [
            self.source_cache.state,
            self.target_cache.state,
            self._is_merged,
        ]

    @state.setter
    def state(self, v):
        """Set state from serialization."""
        source_state, target_state, is_merged = v
        self.source_cache.state = source_state
        self.target_cache.state = target_state
        self._is_merged = is_merged
        if is_merged and source_state and target_state:
            self.merge_source_target()

    @property
    def meta_state(self):
        """Get metadata for serialization."""
        return f"{self.source_cache.offset},{self.target_cache.offset},{int(self._is_merged)}"

    @meta_state.setter
    def meta_state(self, v):
        """Set metadata from serialization."""
        parts = v.split(',')
        # Metadata will be restored through state setter
        pass

    def is_trimmable(self):
        """Both caches must be trimmable."""
        return self.source_cache.is_trimmable() and self.target_cache.is_trimmable()

    def trim(self, n):
        """
        Trim tokens from the beginning.
        Currently trims from source cache only.
        """
        return self.source_cache.trim(n)

    def __repr__(self):
        return (
            f"DualStreamingCache("
            f"source_offset={self.source_offset}, "
            f"target_offset={self.target_offset}, "
            f"is_merged={self._is_merged})"
        )
