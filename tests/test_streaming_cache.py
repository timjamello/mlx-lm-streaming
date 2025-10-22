# tests/test_streaming_cache.py
# Copyright © 2025 Apple Inc.

"""
Unit tests for streaming cache implementation.
"""

import unittest
from typing import Tuple

import mlx.core as mx
import numpy as np

from mlx_lm.models.streaming_cache import (
    CacheMergeError,
    CacheSeparationError,
    StreamingCache,
    StreamingCacheList,
    StreamingCacheState,
)


class TestStreamingCache(unittest.TestCase):
    """Test cases for StreamingCache functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.num_heads = 8
        self.seq_len = 10
        self.head_dim = 64

        # Create sample key/value tensors
        self.keys = mx.random.normal(
            (self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        )
        self.values = mx.random.normal(
            (self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        )

    def _create_test_tensors(
        self,
        seq_len: int,
    ) -> Tuple[mx.array, mx.array]:
        """Helper to create test key/value tensors."""
        keys = mx.random.normal(
            (self.batch_size, self.num_heads, seq_len, self.head_dim)
        )
        values = mx.random.normal(
            (self.batch_size, self.num_heads, seq_len, self.head_dim)
        )
        return keys, values

    def test_initialization(self):
        """Test cache initialization with different configurations."""
        # Test KV cache initialization
        cache = StreamingCache(cache_type="kv")
        self.assertIsNotNone(cache.source_cache)
        self.assertIsNotNone(cache.target_cache)
        self.assertIsNotNone(cache.merged_cache)
        self.assertFalse(cache.is_merged)
        self.assertEqual(cache.source_length, 0)
        self.assertEqual(cache.target_length, 0)

        # Test rotating cache initialization
        cache = StreamingCache(cache_type="rotating", max_size=100)
        self.assertEqual(cache.max_size, 100)

        # Test quantized cache initialization
        cache = StreamingCache(
            cache_type="quantized", quantization_config={"group_size": 32, "bits": 4}
        )
        self.assertEqual(cache.quantization_config["group_size"], 32)
        self.assertEqual(cache.quantization_config["bits"], 4)

        # Test invalid cache type
        with self.assertRaises(ValueError):
            StreamingCache(cache_type="invalid")

        # Test rotating cache without max_size
        with self.assertRaises(ValueError):
            StreamingCache(cache_type="rotating")

    def test_source_cache_update(self):
        """Test updating source cache with input tokens."""
        cache = StreamingCache(cache_type="kv")

        # Update source cache
        source_keys, source_values = self._create_test_tensors(5)
        updated_keys, updated_values = cache.update_source(source_keys, source_values)

        self.assertEqual(cache.source_length, 5)
        self.assertEqual(cache.target_length, 0)
        self.assertFalse(cache.is_merged)

        # Verify shapes
        self.assertEqual(updated_keys.shape[2], 5)
        self.assertEqual(updated_values.shape[2], 5)

        # Update with more tokens
        more_keys, more_values = self._create_test_tensors(3)
        updated_keys, updated_values = cache.update_source(more_keys, more_values)

        self.assertEqual(cache.source_length, 8)
        self.assertEqual(updated_keys.shape[2], 8)

    def test_target_cache_update(self):
        """Test updating target cache with generated tokens."""
        cache = StreamingCache(cache_type="kv")

        # First add some source tokens
        source_keys, source_values = self._create_test_tensors(5)
        cache.update_source(source_keys, source_values)

        # Update target cache
        target_keys, target_values = self._create_test_tensors(3)
        updated_keys, updated_values = cache.update_target(target_keys, target_values)

        self.assertEqual(cache.source_length, 5)
        self.assertEqual(cache.target_length, 3)
        self.assertFalse(cache.is_merged)

        # Verify shapes
        self.assertEqual(updated_keys.shape[2], 3)
        self.assertEqual(updated_values.shape[2], 3)

    def test_cache_merge(self):
        """Test merging source and target caches."""
        cache = StreamingCache(cache_type="kv")

        # Add source tokens
        source_keys, source_values = self._create_test_tensors(5)
        cache.update_source(source_keys, source_values)

        # Add target tokens
        target_keys, target_values = self._create_test_tensors(3)
        cache.update_target(target_keys, target_values)

        # Merge caches
        cache.merge()

        self.assertTrue(cache.is_merged)
        self.assertEqual(cache.merged_cache.offset, 8)

        # Verify merged cache contains both source and target
        merged_keys = cache.keys
        self.assertEqual(merged_keys.shape[2], 8)

        # Test that merging again is a no-op
        cache.merge()
        self.assertTrue(cache.is_merged)

    def test_cache_separation(self):
        """Test separating merged cache back to components."""
        cache = StreamingCache(cache_type="kv")

        # Add tokens and merge
        source_keys, source_values = self._create_test_tensors(5)
        cache.update_source(source_keys, source_values)

        target_keys, target_values = self._create_test_tensors(3)
        cache.update_target(target_keys, target_values)

        cache.merge()
        self.assertTrue(cache.is_merged)

        # Separate caches
        cache.separate()

        self.assertFalse(cache.is_merged)
        self.assertEqual(cache.source_length, 5)
        self.assertEqual(cache.target_length, 3)

        # Verify separated caches have correct shapes
        self.assertEqual(cache.source_cache.keys.shape[2], 5)
        self.assertEqual(cache.target_cache.keys.shape[2], 3)

        # Test that separating again is a no-op
        cache.separate()
        self.assertFalse(cache.is_merged)

    def test_mode_switching(self):
        """Test switching between read and write modes."""
        cache = StreamingCache(cache_type="kv")

        # Start in read mode
        self.assertTrue(cache.stream_state.read_mode)

        # Add source tokens
        source_keys, source_values = self._create_test_tensors(5)
        cache.update_source(source_keys, source_values)

        # Switch to write mode
        cache.set_mode(read_mode=False)
        self.assertFalse(cache.stream_state.read_mode)
        self.assertTrue(cache.is_merged)  # Should auto-merge

        # Add target tokens
        target_keys, target_values = self._create_test_tensors(3)
        cache.update_target(target_keys, target_values)

        # Switch back to read mode
        cache.set_mode(read_mode=True)
        self.assertTrue(cache.stream_state.read_mode)
        self.assertFalse(cache.is_merged)  # Should auto-separate

    def test_update_and_fetch(self):
        """Test the unified update_and_fetch interface."""
        cache = StreamingCache(cache_type="kv")

        # In read mode, should update source
        keys1, values1 = self._create_test_tensors(5)
        cache.update_and_fetch(keys1, values1)
        self.assertEqual(cache.source_length, 5)
        self.assertEqual(cache.target_length, 0)

        # Switch to write mode
        cache.set_mode(read_mode=False)

        # Should update target
        keys2, values2 = self._create_test_tensors(3)
        cache.update_and_fetch(keys2, values2)
        self.assertEqual(cache.source_length, 5)
        self.assertEqual(cache.target_length, 3)

    def test_cache_trimming(self):
        """Test trimming tokens from caches."""
        cache = StreamingCache(cache_type="kv")

        # Add tokens
        source_keys, source_values = self._create_test_tensors(10)
        cache.update_source(source_keys, source_values)

        target_keys, target_values = self._create_test_tensors(5)
        cache.update_target(target_keys, target_values)

        # Trim source
        cache.trim_source(3)
        self.assertEqual(cache.source_length, 7)

        # Trim target
        cache.trim_target(2)
        self.assertEqual(cache.target_length, 3)

        # Test error when trying to trim while merged
        cache.merge()
        with self.assertRaises(CacheMergeError):
            cache.trim_source(1)
        with self.assertRaises(CacheMergeError):
            cache.trim_target(1)

    def test_cache_clear(self):
        """Test clearing all caches."""
        cache = StreamingCache(cache_type="kv")

        # Add tokens
        keys, values = self._create_test_tensors(5)
        cache.update_source(keys, values)
        cache.update_target(keys, values)

        self.assertEqual(cache.source_length, 5)
        self.assertEqual(cache.target_length, 5)

        # Clear
        cache.clear()

        self.assertEqual(cache.source_length, 0)
        self.assertEqual(cache.target_length, 0)
        self.assertFalse(cache.is_merged)

    def test_state_dict(self):
        """Test state serialization and loading."""
        cache1 = StreamingCache(cache_type="kv", target_position_offset=5000)

        # Add tokens
        source_keys, source_values = self._create_test_tensors(5)
        cache1.update_source(source_keys, source_values)

        target_keys, target_values = self._create_test_tensors(3)
        cache1.update_target(target_keys, target_values)

        # Export state
        state_dict = cache1.state_dict()

        # Create new cache and load state
        cache2 = StreamingCache(cache_type="kv")
        cache2.load_state_dict(state_dict)

        # Verify state is restored
        self.assertEqual(cache2.source_length, 5)
        self.assertEqual(cache2.target_length, 3)
        self.assertEqual(cache2.stream_state.target_position_offset, 5000)

        # Verify tensors are restored
        np.testing.assert_array_equal(
            cache1.source_cache.keys, cache2.source_cache.keys
        )
        np.testing.assert_array_equal(
            cache1.target_cache.values, cache2.target_cache.values
        )

    def test_error_conditions(self):
        """Test various error conditions."""
        cache = StreamingCache(cache_type="kv")

        # Add source tokens
        keys, values = self._create_test_tensors(5)
        cache.update_source(keys, values)

        # Merge cache
        cache.merge()

        # Try to update source while merged
        with self.assertRaises(CacheMergeError):
            cache.update_source(keys, values)

    def test_empty_cache_operations(self):
        """Test operations on empty caches."""
        cache = StreamingCache(cache_type="kv")

        # Test merge on empty cache
        cache.merge()
        self.assertTrue(cache.is_merged)

        # Test separate on empty cache
        cache.separate()
        self.assertFalse(cache.is_merged)

        # Test properties on empty cache
        self.assertIsNone(cache.keys)
        self.assertIsNone(cache.values)
        self.assertEqual(cache.offset, 0)


class TestStreamingCacheList(unittest.TestCase):
    """Test cases for StreamingCacheList functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.num_layers = 4
        self.caches = StreamingCacheList(
            [StreamingCache(cache_type="kv") for _ in range(self.num_layers)]
        )

    def test_initialization(self):
        """Test list initialization."""
        self.assertEqual(len(self.caches), self.num_layers)
        for cache in self.caches:
            self.assertIsInstance(cache, StreamingCache)

        # Test with mixed cache types (now supported for models with both attention and linear layers)
        from mlx_lm.models.cache import MambaCache

        mixed_caches = StreamingCacheList([StreamingCache(), MambaCache()])
        self.assertEqual(len(mixed_caches), 2)

    def test_unified_operations(self):
        """Test unified operations across all caches."""
        # Add tokens to all caches
        for cache in self.caches:
            keys = mx.random.normal((1, 4, 5, 32))
            values = mx.random.normal((1, 4, 5, 32))
            cache.update_source(keys, values)

        # Test merge all
        self.caches.merge_all()
        self.assertTrue(self.caches.is_merged)
        for cache in self.caches:
            self.assertTrue(cache.is_merged)

        # Test separate all
        self.caches.separate_all()
        self.assertFalse(self.caches.is_merged)
        for cache in self.caches:
            self.assertFalse(cache.is_merged)

        # Test set mode
        self.caches.set_mode(read_mode=False)
        for cache in self.caches:
            self.assertFalse(cache.stream_state.read_mode)

        # Test clear all
        self.caches.clear_all()
        for cache in self.caches:
            self.assertEqual(cache.source_length, 0)
            self.assertEqual(cache.target_length, 0)

    def test_length_properties(self):
        """Test aggregate length properties."""
        # Add different amounts of tokens to each cache
        for i, cache in enumerate(self.caches):
            source_keys = mx.random.normal((1, 4, i + 2, 32))
            source_values = mx.random.normal((1, 4, i + 2, 32))
            cache.update_source(source_keys, source_values)

            target_keys = mx.random.normal((1, 4, i + 1, 32))
            target_values = mx.random.normal((1, 4, i + 1, 32))
            cache.update_target(target_keys, target_values)

        # Check maximum lengths
        self.assertEqual(self.caches.total_source_length, 5)  # max is 3+2
        self.assertEqual(self.caches.total_target_length, 4)  # max is 3+1


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for realistic streaming scenarios."""

    def test_streaming_generation_flow(self):
        """Test a complete streaming generation flow."""
        cache = StreamingCache(cache_type="kv", target_position_offset=10000)

        # Simulate streaming input processing
        for i in range(3):
            # Process input chunk
            input_keys = mx.random.normal((1, 8, 2, 64))
            input_values = mx.random.normal((1, 8, 2, 64))
            cache.update_source(input_keys, input_values)

        self.assertEqual(cache.source_length, 6)

        # Switch to generation mode
        cache.set_mode(read_mode=False)
        self.assertTrue(cache.is_merged)

        # Generate some tokens
        for i in range(4):
            gen_keys = mx.random.normal((1, 8, 1, 64))
            gen_values = mx.random.normal((1, 8, 1, 64))
            cache.update_target(gen_keys, gen_values)

        self.assertEqual(cache.target_length, 4)
        self.assertEqual(cache.offset, 10)  # 6 source + 4 target

        # Process more input (interrupt generation)
        cache.set_mode(read_mode=True)
        self.assertFalse(cache.is_merged)

        more_input_keys = mx.random.normal((1, 8, 3, 64))
        more_input_values = mx.random.normal((1, 8, 3, 64))
        cache.update_source(more_input_keys, more_input_values)

        self.assertEqual(cache.source_length, 9)

        # Resume generation with updated context
        cache.set_mode(read_mode=False)
        self.assertTrue(cache.is_merged)
        self.assertEqual(cache.offset, 13)  # 9 source + 4 target

    def test_position_offset_handling(self):
        """Test that position offsets are properly maintained."""
        position_offset = 5000
        cache = StreamingCache(cache_type="kv", target_position_offset=position_offset)

        # Add source tokens
        source_keys = mx.random.normal((1, 8, 5, 64))
        source_values = mx.random.normal((1, 8, 5, 64))
        cache.update_source(source_keys, source_values)

        # Position offset should be preserved in state
        self.assertEqual(cache.stream_state.target_position_offset, position_offset)

        # Save and restore state
        state_dict = cache.state_dict()
        new_cache = StreamingCache(cache_type="kv")
        new_cache.load_state_dict(state_dict)

        self.assertEqual(new_cache.stream_state.target_position_offset, position_offset)


if __name__ == "__main__":
    unittest.main()
