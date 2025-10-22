# Copyright © 2023-2024 Apple Inc.

import sys
from pathlib import Path
import unittest

import mlx.core as mx
import mlx.nn as nn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlx_lm.models.streaming_cache import DualStreamingCache


class TestDualStreamingCache(unittest.TestCase):
    """Test suite for DualStreamingCache"""

    def setUp(self):
        """Set up test fixtures"""
        self.cache = DualStreamingCache()
        # Standard dimensions for testing
        # (batch_size, num_heads, seq_len, head_dim)
        self.batch_size = 1
        self.num_heads = 4
        self.head_dim = 64

    def _create_kv(self, seq_len):
        """Helper to create random key/value pairs"""
        shape = (self.batch_size, self.num_heads, seq_len, self.head_dim)
        keys = mx.random.normal(shape)
        values = mx.random.normal(shape)
        return keys, values

    def test_initialization(self):
        """Test cache initializes correctly"""
        self.assertEqual(self.cache.source_offset, 0)
        self.assertEqual(self.cache.target_offset, 0)
        self.assertEqual(self.cache.offset, 0)
        self.assertFalse(self.cache._is_merged)

    def test_source_cache_update(self):
        """Test updating source cache"""
        keys, values = self._create_kv(10)

        source_k, source_v = self.cache.update_source(keys, values)

        self.assertEqual(source_k.shape, (self.batch_size, self.num_heads, 10, self.head_dim))
        self.assertEqual(source_v.shape, (self.batch_size, self.num_heads, 10, self.head_dim))
        self.assertEqual(self.cache.source_offset, 10)
        self.assertEqual(self.cache.target_offset, 0)

    def test_target_cache_update(self):
        """Test updating target cache"""
        keys, values = self._create_kv(5)

        target_k, target_v = self.cache.update_target(keys, values)

        self.assertEqual(target_k.shape, (self.batch_size, self.num_heads, 5, self.head_dim))
        self.assertEqual(target_v.shape, (self.batch_size, self.num_heads, 5, self.head_dim))
        self.assertEqual(self.cache.source_offset, 0)
        self.assertEqual(self.cache.target_offset, 5)

    def test_incremental_source_update(self):
        """Test incrementally adding to source cache"""
        # Add first chunk
        keys1, values1 = self._create_kv(10)
        self.cache.update_source(keys1, values1)
        self.assertEqual(self.cache.source_offset, 10)

        # Add second chunk
        keys2, values2 = self._create_kv(5)
        source_k, source_v = self.cache.update_source(keys2, values2)

        self.assertEqual(self.cache.source_offset, 15)
        self.assertEqual(source_k.shape[-2], 15)  # Should return all 15 tokens

    def test_incremental_target_update(self):
        """Test incrementally adding to target cache"""
        # Add first token
        keys1, values1 = self._create_kv(1)
        self.cache.update_target(keys1, values1)
        self.assertEqual(self.cache.target_offset, 1)

        # Add second token
        keys2, values2 = self._create_kv(1)
        target_k, target_v = self.cache.update_target(keys2, values2)

        self.assertEqual(self.cache.target_offset, 2)
        self.assertEqual(target_k.shape[-2], 2)

    def test_merge_empty_target(self):
        """Test merging when target cache is empty"""
        keys, values = self._create_kv(10)
        self.cache.update_source(keys, values)

        self.cache.merge_source_target()

        self.assertTrue(self.cache._is_merged)
        merged_k, merged_v = self.cache.get_merged()
        self.assertEqual(merged_k.shape[-2], 10)  # Only source tokens

    def test_merge_with_target(self):
        """Test merging source and target caches"""
        # Add source tokens
        source_k, source_v = self._create_kv(10)
        self.cache.update_source(source_k, source_v)

        # Add target tokens
        target_k, target_v = self._create_kv(3)
        self.cache.update_target(target_k, target_v)

        # Merge
        self.cache.merge_source_target()

        self.assertTrue(self.cache._is_merged)
        merged_k, merged_v = self.cache.get_merged()

        # Should have 10 source + 3 target = 13 total
        self.assertEqual(merged_k.shape[-2], 13)
        self.assertEqual(merged_v.shape[-2], 13)

    def test_merge_idempotent(self):
        """Test that merging twice doesn't break anything"""
        source_k, source_v = self._create_kv(10)
        self.cache.update_source(source_k, source_v)

        self.cache.merge_source_target()
        self.cache.merge_source_target()  # Should be no-op

        merged_k, merged_v = self.cache.get_merged()
        self.assertEqual(merged_k.shape[-2], 10)

    def test_separate_after_merge(self):
        """Test separating merged cache"""
        # Setup: source + target + merge
        source_k, source_v = self._create_kv(10)
        self.cache.update_source(source_k, source_v)

        target_k, target_v = self._create_kv(3)
        self.cache.update_target(target_k, target_v)

        self.cache.merge_source_target()
        self.assertTrue(self.cache._is_merged)

        # Separate
        self.cache.separate_source_target()

        self.assertFalse(self.cache._is_merged)
        self.assertEqual(self.cache.source_offset, 10)
        self.assertEqual(self.cache.target_offset, 3)

    def test_separate_idempotent(self):
        """Test that separating twice doesn't break anything"""
        source_k, source_v = self._create_kv(10)
        self.cache.update_source(source_k, source_v)

        self.cache.merge_source_target()
        self.cache.separate_source_target()
        self.cache.separate_source_target()  # Should be no-op

        self.assertFalse(self.cache._is_merged)

    def test_cannot_update_while_merged(self):
        """Test that updating while merged raises error"""
        source_k, source_v = self._create_kv(10)
        self.cache.update_source(source_k, source_v)
        self.cache.merge_source_target()

        # Try to update source while merged
        with self.assertRaises(RuntimeError):
            new_k, new_v = self._create_kv(5)
            self.cache.update_source(new_k, new_v)

        # Try to update target while merged
        with self.assertRaises(RuntimeError):
            new_k, new_v = self._create_kv(1)
            self.cache.update_target(new_k, new_v)

    def test_get_merged_without_merging(self):
        """Test that getting merged state without merging raises error"""
        source_k, source_v = self._create_kv(10)
        self.cache.update_source(source_k, source_v)

        with self.assertRaises(RuntimeError):
            self.cache.get_merged()

    def test_get_source(self):
        """Test getting source keys/values"""
        keys, values = self._create_kv(10)
        self.cache.update_source(keys, values)

        source_k, source_v = self.cache.get_source()

        self.assertEqual(source_k.shape[-2], 10)
        self.assertEqual(source_v.shape[-2], 10)

    def test_get_target(self):
        """Test getting target keys/values"""
        keys, values = self._create_kv(5)
        self.cache.update_target(keys, values)

        target_k, target_v = self.cache.get_target()

        self.assertEqual(target_k.shape[-2], 5)
        self.assertEqual(target_v.shape[-2], 5)

    def test_get_empty_caches(self):
        """Test getting from empty caches"""
        source_k, source_v = self.cache.get_source()
        self.assertIsNone(source_k)
        self.assertIsNone(source_v)

        target_k, target_v = self.cache.get_target()
        self.assertIsNone(target_k)
        self.assertIsNone(target_v)

    def test_reset_target(self):
        """Test resetting target cache only"""
        source_k, source_v = self._create_kv(10)
        self.cache.update_source(source_k, source_v)

        target_k, target_v = self._create_kv(5)
        self.cache.update_target(target_k, target_v)

        self.cache.reset_target()

        # Source should be intact
        self.assertEqual(self.cache.source_offset, 10)
        # Target should be reset
        self.assertEqual(self.cache.target_offset, 0)

    def test_reset(self):
        """Test resetting both caches"""
        source_k, source_v = self._create_kv(10)
        self.cache.update_source(source_k, source_v)

        target_k, target_v = self._create_kv(5)
        self.cache.update_target(target_k, target_v)

        self.cache.reset()

        self.assertEqual(self.cache.source_offset, 0)
        self.assertEqual(self.cache.target_offset, 0)
        self.assertFalse(self.cache._is_merged)

    def test_total_offset(self):
        """Test that total offset is sum of source and target"""
        source_k, source_v = self._create_kv(10)
        self.cache.update_source(source_k, source_v)

        target_k, target_v = self._create_kv(5)
        self.cache.update_target(target_k, target_v)

        self.assertEqual(self.cache.offset, 15)

    def test_streaming_workflow(self):
        """Test a complete streaming workflow"""
        # Step 1: Read source chunk 1
        source_k1, source_v1 = self._create_kv(5)
        self.cache.update_source(source_k1, source_v1)

        # Step 2: Merge and generate first target token
        self.cache.merge_source_target()
        self.cache.separate_source_target()

        target_k1, target_v1 = self._create_kv(1)
        self.cache.update_target(target_k1, target_v1)

        # Step 3: Read source chunk 2
        source_k2, source_v2 = self._create_kv(5)
        self.cache.update_source(source_k2, source_v2)

        # Step 4: Merge and generate second target token
        self.cache.merge_source_target()
        self.cache.separate_source_target()

        target_k2, target_v2 = self._create_kv(1)
        self.cache.update_target(target_k2, target_v2)

        # Verify final state
        self.assertEqual(self.cache.source_offset, 10)  # 5 + 5
        self.assertEqual(self.cache.target_offset, 2)    # 1 + 1
        self.assertFalse(self.cache._is_merged)

    def test_state_serialization(self):
        """Test state getter/setter for serialization"""
        source_k, source_v = self._create_kv(10)
        self.cache.update_source(source_k, source_v)

        target_k, target_v = self._create_kv(5)
        self.cache.update_target(target_k, target_v)

        # Get state
        state = self.cache.state

        # Create new cache and restore state
        new_cache = DualStreamingCache()
        new_cache.state = state

        # Verify
        self.assertEqual(new_cache.source_offset, 10)
        self.assertEqual(new_cache.target_offset, 5)

    def test_repr(self):
        """Test string representation"""
        source_k, source_v = self._create_kv(10)
        self.cache.update_source(source_k, source_v)

        target_k, target_v = self._create_kv(5)
        self.cache.update_target(target_k, target_v)

        repr_str = repr(self.cache)

        self.assertIn("source_offset=10", repr_str)
        self.assertIn("target_offset=5", repr_str)
        self.assertIn("is_merged=False", repr_str)


class TestDualStreamingCacheEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""

    def setUp(self):
        self.cache = DualStreamingCache()
        self.batch_size = 1
        self.num_heads = 4
        self.head_dim = 64

    def _create_kv(self, seq_len):
        shape = (self.batch_size, self.num_heads, seq_len, self.head_dim)
        return mx.random.normal(shape), mx.random.normal(shape)

    def test_large_batch_size(self):
        """Test with larger batch size"""
        batch_size = 8
        shape = (batch_size, self.num_heads, 10, self.head_dim)
        keys = mx.random.normal(shape)
        values = mx.random.normal(shape)

        self.cache.update_source(keys, values)

        source_k, source_v = self.cache.get_source()
        self.assertEqual(source_k.shape[0], batch_size)

    def test_many_incremental_updates(self):
        """Test many small incremental updates"""
        # Add 100 tokens one at a time
        for _ in range(100):
            keys, values = self._create_kv(1)
            self.cache.update_target(keys, values)

        self.assertEqual(self.cache.target_offset, 100)

    def test_alternating_merge_separate(self):
        """Test alternating between merge and separate"""
        source_k, source_v = self._create_kv(10)
        self.cache.update_source(source_k, source_v)

        for i in range(5):
            target_k, target_v = self._create_kv(1)
            self.cache.update_target(target_k, target_v)

            self.cache.merge_source_target()
            merged_k, merged_v = self.cache.get_merged()
            expected_len = 10 + i + 1
            self.assertEqual(merged_k.shape[-2], expected_len)

            self.cache.separate_source_target()

    def test_reset_while_merged(self):
        """Test that reset works even when merged"""
        source_k, source_v = self._create_kv(10)
        self.cache.update_source(source_k, source_v)
        self.cache.merge_source_target()

        self.cache.reset()

        self.assertEqual(self.cache.offset, 0)
        self.assertFalse(self.cache._is_merged)


if __name__ == "__main__":
    unittest.main()
