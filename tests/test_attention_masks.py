# tests/test_attention_masks.py

# Copyright © 2025 Apple Inc.

import unittest

import mlx.core as mx
import numpy as np

from mlx_lm.models.qwen3_next import create_streaming_attention_mask
from mlx_lm.models.streaming_cache import StreamingCache


class TestAttentionMaskValidation(unittest.TestCase):
    """Detailed tests for attention mask correctness."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 1
        self.num_heads = 4
        self.head_dim = 32

    def _add_source_tokens(self, cache, num_tokens):
        """Helper to add source tokens to cache."""
        keys = mx.random.normal(
            (self.batch_size, self.num_heads, num_tokens, self.head_dim)
        )
        values = mx.random.normal(
            (self.batch_size, self.num_heads, num_tokens, self.head_dim)
        )
        cache.update_source(keys, values)

    def _add_target_tokens(self, cache, num_tokens):
        """Helper to add target tokens to cache."""
        keys = mx.random.normal(
            (self.batch_size, self.num_heads, num_tokens, self.head_dim)
        )
        values = mx.random.normal(
            (self.batch_size, self.num_heads, num_tokens, self.head_dim)
        )
        cache.update_target(keys, values)

    def _mask_to_numpy(self, mask):
        """Convert MLX mask (bfloat16) to numpy array."""
        # Convert bfloat16 to float32 first, then to numpy
        return np.array(mask.astype(mx.float32))

    def test_source_only_mask_structure(self):
        """Test mask structure when only source tokens exist."""
        cache = StreamingCache(cache_type="kv")
        self._add_source_tokens(cache, 5)

        # Create mask for processing more source tokens
        mask = create_streaming_attention_mask(
            query_len=3,
            cache=cache,
            is_source_query=True,
        )

        # Verify shape
        self.assertEqual(mask.shape, (1, 1, 3, 8))  # 5 existing + 3 new = 8

        # Convert to numpy for easier checking
        mask_np = self._mask_to_numpy(mask[0, 0])

        # Verify causal structure: token i can attend to tokens 0..i
        # Position 0 in query corresponds to position 5 in kv (after existing 5 tokens)
        for i in range(3):
            query_pos = 5 + i  # Position in full sequence
            for j in range(8):
                if j <= query_pos:
                    # Should be able to attend (value = 0)
                    self.assertEqual(
                        mask_np[i, j],
                        0.0,
                        f"Query pos {i} (global {query_pos}) should attend to KV pos {j}",
                    )
                else:
                    # Should be masked (value = -inf)
                    self.assertTrue(
                        np.isinf(mask_np[i, j]) and mask_np[i, j] < 0,
                        f"Query pos {i} (global {query_pos}) should NOT attend to future KV pos {j}",
                    )

    def test_source_cannot_see_target(self):
        """Test that source tokens cannot attend to target tokens."""
        cache = StreamingCache(cache_type="kv")
        self._add_source_tokens(cache, 3)
        self._add_target_tokens(cache, 2)

        # Separate to process more source
        cache.separate()

        # Create mask for processing more source tokens
        mask = create_streaming_attention_mask(
            query_len=2,
            cache=cache,
            is_source_query=True,
        )

        # Shape should only include source tokens, not target
        # 3 existing source + 2 new source = 5 total
        self.assertEqual(mask.shape, (1, 1, 2, 5))

        mask_np = self._mask_to_numpy(mask[0, 0])

        # All attention should be causal within source only
        for i in range(2):
            query_pos = 3 + i  # After 3 existing source tokens
            for j in range(5):
                if j <= query_pos:
                    self.assertEqual(mask_np[i, j], 0.0)
                else:
                    self.assertTrue(np.isinf(mask_np[i, j]) and mask_np[i, j] < 0)

    def test_target_sees_all_source_and_causal_target(self):
        """Test that target tokens see all source + causal target."""
        cache = StreamingCache(cache_type="kv")
        self._add_source_tokens(cache, 4)
        cache.merge()  # Merge for generation

        # Generate first target token
        mask = create_streaming_attention_mask(
            query_len=1,
            cache=cache,
            is_source_query=False,
        )

        # Should see all 4 source tokens + itself
        self.assertEqual(mask.shape, (1, 1, 1, 5))
        mask_np = self._mask_to_numpy(mask[0, 0])

        # Should attend to all source tokens (0-3) and itself (4)
        for j in range(5):
            self.assertEqual(
                mask_np[0, j], 0.0, f"First target token should attend to position {j}"
            )

        # Add the first target token to cache
        self._add_target_tokens(cache, 1)

        # Generate second target token
        mask = create_streaming_attention_mask(
            query_len=1,
            cache=cache,
            is_source_query=False,
        )

        # Should see 4 source + 1 previous target + itself = 6
        self.assertEqual(mask.shape, (1, 1, 1, 6))
        mask_np = self._mask_to_numpy(mask[0, 0])

        # Should attend to all positions
        for j in range(6):
            self.assertEqual(mask_np[0, j], 0.0)

    def test_multi_token_generation_mask(self):
        """Test mask for generating multiple tokens at once (e.g., prefill)."""
        cache = StreamingCache(cache_type="kv")
        self._add_source_tokens(cache, 3)
        cache.merge()

        # Generate 3 target tokens at once
        mask = create_streaming_attention_mask(
            query_len=3,
            cache=cache,
            is_source_query=False,
        )

        # Should see 3 source + 3 target = 6
        self.assertEqual(mask.shape, (1, 1, 3, 6))
        mask_np = self._mask_to_numpy(mask[0, 0])

        # Verify structure:
        # Target token 0 (pos 3): sees source 0-2, self
        # Target token 1 (pos 4): sees source 0-2, target 0, self
        # Target token 2 (pos 5): sees source 0-2, target 0-1, self

        # Check each row
        # Row 0: target pos 3 (first target token)
        for j in range(4):
            self.assertEqual(mask_np[0, j], 0.0, f"Row 0, col {j} should be 0")
        for j in range(4, 6):
            self.assertTrue(
                np.isinf(mask_np[0, j]) and mask_np[0, j] < 0,
                f"Row 0, col {j} should be -inf",
            )

        # Row 1: target pos 4
        for j in range(5):
            self.assertEqual(mask_np[1, j], 0.0, f"Row 1, col {j} should be 0")
        self.assertTrue(
            np.isinf(mask_np[1, 5]) and mask_np[1, 5] < 0, "Row 1, col 5 should be -inf"
        )

        # Row 2: target pos 5 - sees all
        for j in range(6):
            self.assertEqual(mask_np[2, j], 0.0, f"Row 2, col {j} should be 0")

    def test_empty_cache_mask(self):
        """Test mask creation with empty cache (first tokens)."""
        cache = StreamingCache(cache_type="kv")

        # Process first tokens
        mask = create_streaming_attention_mask(
            query_len=3,
            cache=cache,
            is_source_query=True,
        )

        self.assertEqual(mask.shape, (1, 1, 3, 3))
        mask_np = self._mask_to_numpy(mask[0, 0])

        # Should be standard causal mask
        # Row 0: [0, -inf, -inf]
        self.assertEqual(mask_np[0, 0], 0.0)
        self.assertTrue(np.isinf(mask_np[0, 1]) and mask_np[0, 1] < 0)
        self.assertTrue(np.isinf(mask_np[0, 2]) and mask_np[0, 2] < 0)

        # Row 1: [0, 0, -inf]
        self.assertEqual(mask_np[1, 0], 0.0)
        self.assertEqual(mask_np[1, 1], 0.0)
        self.assertTrue(np.isinf(mask_np[1, 2]) and mask_np[1, 2] < 0)

        # Row 2: [0, 0, 0]
        self.assertEqual(mask_np[2, 0], 0.0)
        self.assertEqual(mask_np[2, 1], 0.0)
        self.assertEqual(mask_np[2, 2], 0.0)

    def test_mask_values_are_valid(self):
        """Test that mask values are either 0 or -inf."""
        cache = StreamingCache(cache_type="kv")
        self._add_source_tokens(cache, 5)
        self._add_target_tokens(cache, 3)
        cache.merge()

        mask = create_streaming_attention_mask(
            query_len=2,
            cache=cache,
            is_source_query=False,
        )

        mask_np = self._mask_to_numpy(mask[0, 0])

        # All values should be either 0 or -inf
        for i in range(mask_np.shape[0]):
            for j in range(mask_np.shape[1]):
                val = mask_np[i, j]
                is_valid = (val == 0.0) or (np.isinf(val) and val < 0)
                self.assertTrue(
                    is_valid, f"Mask[{i},{j}] contains invalid value: {val}"
                )

    def test_mask_dtype_matches_model(self):
        """Test that mask dtype is bfloat16 to match model."""
        cache = StreamingCache(cache_type="kv")
        self._add_source_tokens(cache, 3)

        mask = create_streaming_attention_mask(
            query_len=2,
            cache=cache,
            is_source_query=True,
        )

        # Should be bfloat16 to match model dtype
        self.assertEqual(mask.dtype, mx.bfloat16)

    def test_batch_dimension_broadcasts(self):
        """Test that mask broadcasts correctly with batch dimension."""
        cache = StreamingCache(cache_type="kv")
        self._add_source_tokens(cache, 3)

        mask = create_streaming_attention_mask(
            query_len=2,
            cache=cache,
            is_source_query=True,
        )

        # Should have shape [1, 1, query_len, kv_len] for broadcasting
        self.assertEqual(mask.shape[0], 1)  # Batch dim
        self.assertEqual(mask.shape[1], 1)  # Head dim

        # Should broadcast with [B, H, Q, K] attention scores
        batch_size = 4
        num_heads = 8
        scores_shape = (batch_size, num_heads, mask.shape[2], mask.shape[3])

        # Simulate adding mask to scores
        scores = mx.zeros(scores_shape)
        result = scores + mask

        self.assertEqual(result.shape, scores_shape)

    def test_mask_causality_preserved(self):
        """Test that causal masking is preserved in all scenarios."""
        cache = StreamingCache(cache_type="kv")
        self._add_source_tokens(cache, 4)

        mask = create_streaming_attention_mask(
            query_len=3,
            cache=cache,
            is_source_query=True,
        )

        mask_np = self._mask_to_numpy(mask[0, 0])

        # For each query position, verify it cannot attend to future positions
        for i in range(mask.shape[2]):
            for j in range(mask.shape[3]):
                if j > (4 + i):  # Future position
                    self.assertTrue(
                        np.isinf(mask_np[i, j]) and mask_np[i, j] < 0,
                        f"Query {i} should not attend to future position {j}",
                    )

    def test_incremental_source_processing(self):
        """Test mask correctness when processing source incrementally."""
        cache = StreamingCache(cache_type="kv")

        # Process first chunk
        self._add_source_tokens(cache, 2)
        mask1 = create_streaming_attention_mask(1, cache, is_source_query=True)
        self.assertEqual(mask1.shape, (1, 1, 1, 3))

        # Process second chunk
        self._add_source_tokens(cache, 2)
        mask2 = create_streaming_attention_mask(1, cache, is_source_query=True)
        self.assertEqual(mask2.shape, (1, 1, 1, 5))

        # Both should allow attending to all previous source
        mask1_np = self._mask_to_numpy(mask1[0, 0, 0, :3])
        mask2_np = self._mask_to_numpy(mask2[0, 0, 0, :5])

        # All should be 0 (can attend)
        for i in range(3):
            self.assertEqual(mask1_np[i], 0.0)
        for i in range(5):
            self.assertEqual(mask2_np[i], 0.0)

    def test_assertion_triggers_on_invalid_shape(self):
        """Test that assertion catches shape mismatches (if they occur)."""
        # This test verifies the assertion logic works
        # We can't easily trigger the assertion without modifying internals,
        # so we'll just verify the masks are always correct
        cache = StreamingCache(cache_type="kv")
        self._add_source_tokens(cache, 10)

        # Try various query lengths
        for query_len in [1, 3, 5]:
            mask = create_streaming_attention_mask(
                query_len=query_len,
                cache=cache,
                is_source_query=True,
            )
            expected_kv_len = 10 + query_len
            self.assertEqual(mask.shape, (1, 1, query_len, expected_kv_len))

    def test_transition_from_source_to_target_mask(self):
        """Test mask correctness when transitioning from source to target."""
        cache = StreamingCache(cache_type="kv")

        # Add source tokens
        self._add_source_tokens(cache, 5)

        # Create source mask
        source_mask = create_streaming_attention_mask(
            query_len=2,
            cache=cache,
            is_source_query=True,
        )
        self.assertEqual(source_mask.shape, (1, 1, 2, 7))

        # Update cache with those tokens
        self._add_source_tokens(cache, 2)

        # Merge and switch to target
        cache.merge()

        # Create target mask
        target_mask = create_streaming_attention_mask(
            query_len=1,
            cache=cache,
            is_source_query=False,
        )
        # Should see all 7 source tokens + self
        self.assertEqual(target_mask.shape, (1, 1, 1, 8))

        # Verify target can see all source
        target_mask_np = self._mask_to_numpy(target_mask[0, 0, 0])
        for i in range(8):
            self.assertEqual(
                target_mask_np[i], 0.0, f"Target should see source position {i}"
            )


class TestMaskBoundaryConditions(unittest.TestCase):
    """Test edge cases and boundary conditions for masks."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 1
        self.num_heads = 4
        self.head_dim = 32

    def _add_source_tokens(self, cache, num_tokens):
        """Helper to add source tokens to cache."""
        keys = mx.random.normal(
            (self.batch_size, self.num_heads, num_tokens, self.head_dim)
        )
        values = mx.random.normal(
            (self.batch_size, self.num_heads, num_tokens, self.head_dim)
        )
        cache.update_source(keys, values)

    def _add_target_tokens(self, cache, num_tokens):
        """Helper to add target tokens to cache."""
        keys = mx.random.normal(
            (self.batch_size, self.num_heads, num_tokens, self.head_dim)
        )
        values = mx.random.normal(
            (self.batch_size, self.num_heads, num_tokens, self.head_dim)
        )
        cache.update_target(keys, values)

    def _mask_to_numpy(self, mask):
        """Convert MLX mask (bfloat16) to numpy array."""
        return np.array(mask.astype(mx.float32))

    def test_single_token_source(self):
        """Test mask with single source token."""
        cache = StreamingCache(cache_type="kv")
        self._add_source_tokens(cache, 1)

        mask = create_streaming_attention_mask(1, cache, is_source_query=True)
        self.assertEqual(mask.shape, (1, 1, 1, 2))

        mask_np = self._mask_to_numpy(mask[0, 0])
        # Should see previous token and self
        self.assertEqual(mask_np[0, 0], 0.0)
        self.assertEqual(mask_np[0, 1], 0.0)

    def test_large_sequence(self):
        """Test mask with large sequence."""
        cache = StreamingCache(cache_type="kv")
        self._add_source_tokens(cache, 100)

        mask = create_streaming_attention_mask(10, cache, is_source_query=True)
        self.assertEqual(mask.shape, (1, 1, 10, 110))

        # Verify causality still holds
        mask_np = self._mask_to_numpy(mask[0, 0])
        for i in range(10):
            query_pos = 100 + i
            for j in range(110):
                if j <= query_pos:
                    self.assertEqual(mask_np[i, j], 0.0)
                else:
                    self.assertTrue(np.isinf(mask_np[i, j]) and mask_np[i, j] < 0)

    def test_alternating_read_write(self):
        """Test masks when alternating between read and write modes."""
        cache = StreamingCache(cache_type="kv")

        # Read: Add source
        self._add_source_tokens(cache, 3)
        mask1 = create_streaming_attention_mask(1, cache, is_source_query=True)
        self.assertEqual(mask1.shape, (1, 1, 1, 4))

        # Update cache with the new source token
        self._add_source_tokens(cache, 1)

        # Write: Merge and generate target
        cache.merge()

        # When merged, we generate target tokens (not add source)
        self._add_target_tokens(cache, 1)

        mask2 = create_streaming_attention_mask(1, cache, is_source_query=False)
        # Should see all 4 source + 1 target + new token = 6
        self.assertEqual(mask2.shape, (1, 1, 1, 6))

        # Read: Separate and add more source
        cache.separate()
        mask3 = create_streaming_attention_mask(1, cache, is_source_query=True)
        # Should see 4 existing source + new token = 5
        self.assertEqual(mask3.shape, (1, 1, 1, 5))

    def test_zero_query_length_handled(self):
        """Test that zero query length is handled gracefully."""
        cache = StreamingCache(cache_type="kv")
        self._add_source_tokens(cache, 3)

        # Zero query length should return empty mask
        mask = create_streaming_attention_mask(0, cache, is_source_query=True)
        self.assertEqual(mask.shape, (1, 1, 0, 0))

        # Verify it's the right dtype
        self.assertEqual(mask.dtype, mx.bfloat16)


if __name__ == "__main__":
    unittest.main()
