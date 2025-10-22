# Copyright © 2023-2024 Apple Inc.

import sys
from pathlib import Path
import unittest

import mlx.core as mx
import mlx.nn as nn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlx_lm.models.qwen2_streaming import (
    Model,
    ModelArgs,
    StreamingAttention,
    StreamingTransformerBlock,
    Qwen2ModelStreaming,
)
from mlx_lm.models.streaming_cache import DualStreamingCache
from mlx_lm.models.cache import KVCache


class TestStreamingAttention(unittest.TestCase):
    """Test suite for StreamingAttention module"""

    def setUp(self):
        """Set up test fixtures"""
        self.args = ModelArgs(
            model_type="qwen2",
            hidden_size=256,
            num_hidden_layers=2,
            intermediate_size=512,
            num_attention_heads=4,
            num_key_value_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=1000,
            max_position_embeddings=2048,
            rope_theta=10000.0,
        )
        self.attn = StreamingAttention(self.args)
        self.batch_size = 1
        self.seq_len = 10

    def test_initialization(self):
        """Test attention module initializes correctly"""
        self.assertEqual(self.attn.n_heads, 4)
        self.assertEqual(self.attn.n_kv_heads, 4)
        self.assertIsNotNone(self.attn.rope)

    def test_forward_no_cache(self):
        """Test forward pass without cache"""
        x = mx.random.normal((self.batch_size, self.seq_len, self.args.hidden_size))
        output = self.attn(x)

        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.args.hidden_size))

    def test_forward_with_standard_cache(self):
        """Test forward pass with standard KVCache"""
        cache = KVCache()
        x = mx.random.normal((self.batch_size, self.seq_len, self.args.hidden_size))

        output = self.attn(x, cache=cache)

        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.args.hidden_size))
        self.assertEqual(cache.offset, self.seq_len)

    def test_forward_with_dual_cache_reading(self):
        """Test forward pass with DualStreamingCache in reading mode"""
        cache = DualStreamingCache()
        x = mx.random.normal((self.batch_size, self.seq_len, self.args.hidden_size))

        output = self.attn(x, cache=cache, is_reading=True)

        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.args.hidden_size))
        self.assertEqual(cache.source_offset, self.seq_len)
        self.assertEqual(cache.target_offset, 0)

    def test_forward_with_dual_cache_writing(self):
        """Test forward pass with DualStreamingCache in writing mode"""
        cache = DualStreamingCache()

        # First add some source tokens
        x_source = mx.random.normal((self.batch_size, 10, self.args.hidden_size))
        self.attn(x_source, cache=cache, is_reading=True)

        # Now add target tokens
        x_target = mx.random.normal((self.batch_size, 1, self.args.hidden_size))
        output = self.attn(x_target, cache=cache, is_reading=False)

        self.assertEqual(output.shape, (self.batch_size, 1, self.args.hidden_size))
        self.assertEqual(cache.source_offset, 10)
        self.assertEqual(cache.target_offset, 1)

    def test_position_ids_override(self):
        """Test that position_ids parameter is used when provided"""
        cache = DualStreamingCache()
        x = mx.random.normal((self.batch_size, 5, self.args.hidden_size))

        # Custom position IDs starting from 100
        position_ids = mx.array([[100, 101, 102, 103, 104]])

        output = self.attn(x, cache=cache, position_ids=position_ids, is_reading=True)

        self.assertEqual(output.shape, (self.batch_size, 5, self.args.hidden_size))
        # Cache should still update normally
        self.assertEqual(cache.source_offset, 5)

    def test_incremental_reading(self):
        """Test incremental source token processing"""
        cache = DualStreamingCache()

        # Add first chunk
        x1 = mx.random.normal((self.batch_size, 5, self.args.hidden_size))
        self.attn(x1, cache=cache, is_reading=True)
        self.assertEqual(cache.source_offset, 5)

        # Add second chunk
        x2 = mx.random.normal((self.batch_size, 5, self.args.hidden_size))
        self.attn(x2, cache=cache, is_reading=True)
        self.assertEqual(cache.source_offset, 10)

    def test_incremental_writing(self):
        """Test incremental target token generation"""
        cache = DualStreamingCache()

        # Add source
        x_source = mx.random.normal((self.batch_size, 10, self.args.hidden_size))
        self.attn(x_source, cache=cache, is_reading=True)

        # Add target tokens one by one
        for i in range(5):
            x_target = mx.random.normal((self.batch_size, 1, self.args.hidden_size))
            self.attn(x_target, cache=cache, is_reading=False)
            self.assertEqual(cache.target_offset, i + 1)


class TestStreamingTransformerBlock(unittest.TestCase):
    """Test suite for StreamingTransformerBlock"""

    def setUp(self):
        self.args = ModelArgs(
            model_type="qwen2",
            hidden_size=256,
            num_hidden_layers=2,
            intermediate_size=512,
            num_attention_heads=4,
            num_key_value_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=1000,
        )
        self.block = StreamingTransformerBlock(self.args)
        self.batch_size = 1

    def test_forward_no_cache(self):
        """Test forward pass without cache"""
        x = mx.random.normal((self.batch_size, 10, self.args.hidden_size))
        output = self.block(x)

        self.assertEqual(output.shape, x.shape)

    def test_forward_with_dual_cache(self):
        """Test forward pass with DualStreamingCache"""
        cache = DualStreamingCache()
        x = mx.random.normal((self.batch_size, 10, self.args.hidden_size))

        output = self.block(x, cache=cache, is_reading=True)

        self.assertEqual(output.shape, x.shape)
        self.assertEqual(cache.source_offset, 10)


class TestQwen2ModelStreaming(unittest.TestCase):
    """Test suite for Qwen2ModelStreaming"""

    def setUp(self):
        self.args = ModelArgs(
            model_type="qwen2",
            hidden_size=256,
            num_hidden_layers=2,
            intermediate_size=512,
            num_attention_heads=4,
            num_key_value_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=1000,
        )
        self.model = Qwen2ModelStreaming(self.args)
        self.batch_size = 1

    def test_initialization(self):
        """Test model initializes correctly"""
        self.assertEqual(len(self.model.layers), 2)
        self.assertEqual(self.model.vocab_size, 1000)

    def test_forward_no_cache(self):
        """Test forward pass without cache"""
        inputs = mx.random.randint(0, 1000, (self.batch_size, 10))
        output = self.model(inputs)

        self.assertEqual(output.shape, (self.batch_size, 10, self.args.hidden_size))

    def test_forward_with_dual_caches(self):
        """Test forward pass with list of DualStreamingCaches"""
        caches = [DualStreamingCache() for _ in range(self.args.num_hidden_layers)]
        inputs = mx.random.randint(0, 1000, (self.batch_size, 10))

        output = self.model(inputs, cache=caches, is_reading=True)

        self.assertEqual(output.shape, (self.batch_size, 10, self.args.hidden_size))
        # All caches should be updated
        for cache in caches:
            self.assertEqual(cache.source_offset, 10)
            self.assertEqual(cache.target_offset, 0)

    def test_streaming_workflow(self):
        """Test complete streaming read-write workflow"""
        caches = [DualStreamingCache() for _ in range(self.args.num_hidden_layers)]

        # Phase 1: Read source chunk 1
        source_chunk1 = mx.random.randint(0, 1000, (self.batch_size, 5))
        self.model(source_chunk1, cache=caches, is_reading=True)

        # Verify source cache updated
        for cache in caches:
            self.assertEqual(cache.source_offset, 5)
            self.assertEqual(cache.target_offset, 0)

        # Phase 2: Read source chunk 2
        source_chunk2 = mx.random.randint(0, 1000, (self.batch_size, 5))
        self.model(source_chunk2, cache=caches, is_reading=True)

        # Verify source cache updated
        for cache in caches:
            self.assertEqual(cache.source_offset, 10)
            self.assertEqual(cache.target_offset, 0)

        # Phase 3: Generate target tokens
        for i in range(3):
            target_token = mx.random.randint(0, 1000, (self.batch_size, 1))
            self.model(target_token, cache=caches, is_reading=False)

            # Verify target cache updated
            for cache in caches:
                self.assertEqual(cache.source_offset, 10)
                self.assertEqual(cache.target_offset, i + 1)


class TestModel(unittest.TestCase):
    """Test suite for top-level Model class"""

    def setUp(self):
        self.args = ModelArgs(
            model_type="qwen2",
            hidden_size=256,
            num_hidden_layers=2,
            intermediate_size=512,
            num_attention_heads=4,
            num_key_value_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=1000,
        )
        self.model = Model(self.args)
        self.batch_size = 1

    def test_initialization(self):
        """Test model initializes correctly"""
        self.assertEqual(self.model.model_type, "qwen2")
        self.assertIsNotNone(self.model.model)

    def test_forward_returns_logits(self):
        """Test forward pass returns correct logits shape"""
        inputs = mx.random.randint(0, 1000, (self.batch_size, 10))
        logits = self.model(inputs)

        self.assertEqual(logits.shape, (self.batch_size, 10, self.args.vocab_size))

    def test_make_cache(self):
        """Test make_cache creates DualStreamingCache instances"""
        caches = self.model.make_cache()

        self.assertEqual(len(caches), self.args.num_hidden_layers)
        for cache in caches:
            self.assertIsInstance(cache, DualStreamingCache)

    def test_streaming_generation(self):
        """Test streaming generation with dual caches"""
        caches = self.model.make_cache()

        # Read phase
        source = mx.random.randint(0, 1000, (self.batch_size, 10))
        logits_source = self.model(source, cache=caches, is_reading=True)

        self.assertEqual(logits_source.shape, (self.batch_size, 10, self.args.vocab_size))

        # Write phase
        target = mx.random.randint(0, 1000, (self.batch_size, 1))
        logits_target = self.model(target, cache=caches, is_reading=False)

        self.assertEqual(logits_target.shape, (self.batch_size, 1, self.args.vocab_size))

        # Verify caches
        for cache in caches:
            self.assertEqual(cache.source_offset, 10)
            self.assertEqual(cache.target_offset, 1)

    def test_position_ids_propagation(self):
        """Test that position_ids are properly propagated through the model"""
        caches = self.model.make_cache()
        inputs = mx.random.randint(0, 1000, (self.batch_size, 5))

        # Custom position IDs
        position_ids = mx.array([[0, 1, 2, 3, 4]])

        logits = self.model(inputs, cache=caches, position_ids=position_ids, is_reading=True)

        self.assertEqual(logits.shape, (self.batch_size, 5, self.args.vocab_size))

    def test_separate_position_spaces(self):
        """Test that source and target use separate position ID spaces"""
        caches = self.model.make_cache()

        # Source with position IDs [0, 1, 2, ..., 9]
        source = mx.random.randint(0, 1000, (self.batch_size, 10))
        source_pos = mx.arange(10).reshape(1, -1)
        self.model(source, cache=caches, position_ids=source_pos, is_reading=True)

        # Target with position IDs [0, 1, 2] (separate space)
        for i in range(3):
            target = mx.random.randint(0, 1000, (self.batch_size, 1))
            target_pos = mx.array([[i]])
            self.model(target, cache=caches, position_ids=target_pos, is_reading=False)

        # Verify final cache state
        for cache in caches:
            self.assertEqual(cache.source_offset, 10)
            self.assertEqual(cache.target_offset, 3)


class TestIntegrationWithCache(unittest.TestCase):
    """Integration tests with cache operations"""

    def setUp(self):
        self.args = ModelArgs(
            model_type="qwen2",
            hidden_size=256,
            num_hidden_layers=2,
            intermediate_size=512,
            num_attention_heads=4,
            num_key_value_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=1000,
        )
        self.model = Model(self.args)

    def test_cache_merge_for_attention(self):
        """Test that caches can be merged for attention computation"""
        caches = self.model.make_cache()

        # Add source
        source = mx.random.randint(0, 1000, (1, 10))
        self.model(source, cache=caches, is_reading=True)

        # Add target
        target = mx.random.randint(0, 1000, (1, 3))
        self.model(target, cache=caches, is_reading=False)

        # Merge caches
        for cache in caches:
            cache.merge_source_target()
            merged_k, merged_v = cache.get_merged()
            # Should have 10 source + 3 target = 13 total
            self.assertEqual(merged_k.shape[-2], 13)

    def test_cache_reset_target(self):
        """Test resetting target cache while keeping source"""
        caches = self.model.make_cache()

        # Add source and target
        source = mx.random.randint(0, 1000, (1, 10))
        self.model(source, cache=caches, is_reading=True)

        target = mx.random.randint(0, 1000, (1, 5))
        self.model(target, cache=caches, is_reading=False)

        # Reset target caches
        for cache in caches:
            cache.reset_target()
            self.assertEqual(cache.source_offset, 10)
            self.assertEqual(cache.target_offset, 0)

    def test_multiple_generations_same_source(self):
        """Test generating multiple target sequences from same source"""
        caches = self.model.make_cache()

        # Add source once
        source = mx.random.randint(0, 1000, (1, 10))
        self.model(source, cache=caches, is_reading=True)

        # Generate first target sequence
        for _ in range(3):
            target = mx.random.randint(0, 1000, (1, 1))
            self.model(target, cache=caches, is_reading=False)

        # Verify first sequence
        for cache in caches:
            self.assertEqual(cache.target_offset, 3)

        # Reset for second sequence
        for cache in caches:
            cache.reset_target()

        # Generate second target sequence
        for _ in range(5):
            target = mx.random.randint(0, 1000, (1, 1))
            self.model(target, cache=caches, is_reading=False)

        # Verify second sequence
        for cache in caches:
            self.assertEqual(cache.source_offset, 10)  # Source unchanged
            self.assertEqual(cache.target_offset, 5)   # New target sequence


if __name__ == "__main__":
    unittest.main()
