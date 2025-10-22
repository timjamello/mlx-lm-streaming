# Copyright © 2025 Apple Inc.

import unittest
from typing import Generator, List
from unittest.mock import MagicMock, Mock, patch

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.streaming_cache import StreamingCache, StreamingCacheList
from mlx_lm.streaming_generate import (
    StreamingState,
    streaming_generate,
    streaming_generate_step,
)


class MockModel(nn.Module):
    """Mock model for testing streaming generation."""

    def __init__(self, vocab_size: int = 100):
        super().__init__()
        self.vocab_size = vocab_size
        self.call_count = 0
        self.last_cache_state = None

    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        """Simulate model forward pass."""
        self.call_count += 1

        batch_size, seq_len = inputs.shape

        # Update cache if provided (simulate attention layers)
        if cache is not None and isinstance(cache, StreamingCacheList):
            # Track cache state
            self.last_cache_state = {
                "merged": cache.is_merged,
                "read_mode": cache[0].stream_state.read_mode if cache else None,
                "source_len": cache.total_source_length,
                "target_len": cache.total_target_length,
            }

            # Simulate updating each cache layer with mock keys/values
            for layer_cache in cache:
                # Create mock keys and values (num_heads=4, head_dim=32)
                num_heads = 4
                head_dim = 32
                mock_keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
                mock_values = mx.random.normal(
                    (batch_size, num_heads, seq_len, head_dim)
                )

                # Update the cache (this will route to update_source or update_target)
                layer_cache.update_and_fetch(mock_keys, mock_values)

        # Return mock logits
        logits = mx.random.uniform(
            low=-10, high=10, shape=(batch_size, seq_len, self.vocab_size)
        )
        return logits

    def make_cache(self, streaming: bool = False) -> StreamingCacheList:
        """Create mock streaming cache."""
        if streaming:
            # Create 4 layers of streaming cache (mix of linear and attention)
            caches = [
                StreamingCache(cache_type="kv", position_offset=10000) for _ in range(4)
            ]
            return StreamingCacheList(caches)
        return None


class TestStreamingGenerate(unittest.TestCase):
    """Test streaming generation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = MockModel(vocab_size=100)
        self.tokenizer = Mock()
        self.tokenizer.encode = lambda x: list(range(len(x.split())))
        self.tokenizer.decode = lambda tokens: f"[{','.join(map(str, tokens))}]"

    def test_basic_streaming(self):
        """Test basic streaming without input stream."""
        prompt = [1, 2, 3]
        cache = self.model.make_cache(streaming=True)

        tokens_generated = []
        modes_seen = []

        for token, is_input, mode in streaming_generate_step(
            self.model,
            mx.array(prompt),
            cache,
            wait_k=1,
            max_tokens=5,
            temperature=0.0,
        ):
            if not is_input:
                tokens_generated.append(token)
            modes_seen.append(mode)

        # Should have processed input then generated output
        self.assertIn("read", modes_seen)
        self.assertIn("write", modes_seen)
        self.assertEqual(len(tokens_generated), 5)

        # Cache should have both source and target tokens
        self.assertGreater(cache.total_source_length, 0)
        self.assertGreater(cache.total_target_length, 0)

    def test_streaming_with_input_chunks(self):
        """Test streaming with dynamic input arrival."""

        def input_generator() -> Generator[List[int], None, None]:
            """Simulate input arriving in chunks."""
            chunks = [
                [4, 5],  # Chunk 1
                [6, 7, 8],  # Chunk 2
                [9],  # Chunk 3
            ]
            for chunk in chunks:
                yield chunk

        prompt = [1, 2, 3]
        cache = self.model.make_cache(streaming=True)

        source_tokens = []
        target_tokens = []

        for token, is_input, mode in streaming_generate_step(
            self.model,
            mx.array(prompt),
            cache,
            wait_k=2,  # Wait for 2 chunks before generating
            max_tokens=3,
            input_stream=input_generator(),
        ):
            if is_input:
                source_tokens.append(token)
            else:
                target_tokens.append(token)

        # Should have processed initial prompt + first chunk before generation
        self.assertEqual(source_tokens[:3], prompt)  # Initial prompt
        self.assertIn(4, source_tokens)  # First chunk
        self.assertIn(5, source_tokens)

        # Should have generated tokens
        self.assertEqual(len(target_tokens), 3)

    def test_wait_k_policy(self):
        """Test that generation waits for k chunks."""

        chunks_yielded = []

        def tracked_input_generator():
            chunks = [[4, 5], [6, 7], [8, 9]]
            for i, chunk in enumerate(chunks):
                chunks_yielded.append(i)
                yield chunk

        prompt = [1, 2, 3]
        cache = self.model.make_cache(streaming=True)

        first_generation_after_chunks = None
        chunks_before_first_gen = 0

        for token, is_input, mode in streaming_generate_step(
            self.model,
            mx.array(prompt),
            cache,
            wait_k=3,  # Wait for 3 chunks total (including prompt)
            max_tokens=2,
            input_stream=tracked_input_generator(),
        ):
            if not is_input and first_generation_after_chunks is None:
                first_generation_after_chunks = len(chunks_yielded)
                chunks_before_first_gen = len(chunks_yielded)

        # Should have waited for 2 additional chunks (3 total with prompt)
        # Plus one more chunk is pulled at the start of the generation loop
        self.assertEqual(chunks_before_first_gen, 3)

    def test_cache_mode_switching(self):
        """Test that cache properly switches between read and write modes."""
        prompt = [1, 2, 3]
        cache = self.model.make_cache(streaming=True)

        def single_chunk_generator():
            yield [4, 5, 6]

        mode_sequence = []

        for token, is_input, mode in streaming_generate_step(
            self.model,
            mx.array(prompt),
            cache,
            wait_k=1,
            max_tokens=2,
            input_stream=single_chunk_generator(),
        ):
            mode_sequence.append((mode, cache.is_merged))

        # Check mode transitions
        read_modes = [(m, merged) for m, merged in mode_sequence if m == "read"]
        write_modes = [(m, merged) for m, merged in mode_sequence if m == "write"]

        # During read, cache should not be merged
        for mode, merged in read_modes:
            self.assertFalse(merged, "Cache should not be merged during read")

        # During write, cache should be merged
        for mode, merged in write_modes:
            self.assertTrue(merged, "Cache should be merged during write")

    def test_position_offsets(self):
        """Test that position offsets are maintained correctly."""
        prompt = [1, 2, 3]
        cache = self.model.make_cache(streaming=True)

        # Track source and target lengths
        source_len_history = []
        target_len_history = []

        for token, is_input, mode in streaming_generate_step(
            self.model,
            mx.array(prompt),
            cache,
            wait_k=1,
            max_tokens=3,
        ):
            source_len_history.append(cache.total_source_length)
            target_len_history.append(cache.total_target_length)

        # Source length should be 3 after processing the prompt chunk
        # (tokens are processed in chunks, not one-by-one)
        self.assertEqual(source_len_history[0], 3)  # After prompt chunk processed
        self.assertEqual(source_len_history[2], 3)  # Still 3 after yielding all tokens

        # Target length should increase during generation
        first_gen_idx = next(i for i, tl in enumerate(target_len_history) if tl > 0)
        self.assertGreater(first_gen_idx, 2)  # Generation starts after prompt
        self.assertEqual(target_len_history[-1], 3)  # Generated 3 tokens

    def test_empty_input_stream(self):
        """Test handling of empty input stream."""
        prompt = [1, 2, 3]
        cache = self.model.make_cache(streaming=True)

        def empty_generator():
            return
            yield  # Never yields anything

        tokens = []
        for token, is_input, mode in streaming_generate_step(
            self.model,
            mx.array(prompt),
            cache,
            wait_k=5,  # Wait for 5 chunks but won't get them
            max_tokens=2,
            input_stream=empty_generator(),
        ):
            tokens.append((token, is_input))

        # Should still process prompt and generate
        input_tokens = [t for t, is_inp in tokens if is_inp]
        output_tokens = [t for t, is_inp in tokens if not is_inp]

        self.assertEqual(len(input_tokens), 3)  # Just the prompt
        self.assertEqual(len(output_tokens), 2)  # Still generates

    def test_high_level_interface(self):
        """Test the high-level streaming_generate function."""

        def input_chunks():
            yield [10, 11]
            yield [12, 13]

        source, target = streaming_generate(
            self.model,
            self.tokenizer,
            "hello world",
            input_stream=input_chunks(),
            wait_k=2,
            max_tokens=5,
            verbose=False,
        )

        # Should have processed prompt + input chunks
        self.assertGreater(len(source), 2)  # At least the prompt
        self.assertEqual(len(target), 5)  # Generated 5 tokens

    def test_interruption_simulation(self):
        """Test that streaming can handle interruption-like scenarios."""
        prompt = [1, 2, 3]
        cache = self.model.make_cache(streaming=True)

        def interrupting_input():
            # Start with normal input
            yield [4, 5]
            # Simulate interruption with new input
            yield [99, 99, 99]  # "Actually, wait..."

        tokens_before_interrupt = []
        tokens_after_interrupt = []
        seen_interrupt = False

        for token, is_input, mode in streaming_generate_step(
            self.model,
            mx.array(prompt),
            cache,
            wait_k=1,
            max_tokens=10,
            input_stream=interrupting_input(),
        ):
            if is_input and token == 99:
                seen_interrupt = True

            if not seen_interrupt:
                tokens_before_interrupt.append((token, is_input))
            else:
                tokens_after_interrupt.append((token, is_input))

        # Should have processed tokens before and after interruption
        self.assertGreater(len(tokens_before_interrupt), 0)
        self.assertGreater(len(tokens_after_interrupt), 0)
        self.assertTrue(seen_interrupt)

        # Cache should maintain consistency
        self.assertGreater(cache.total_source_length, 0)
        self.assertGreater(cache.total_target_length, 0)


if __name__ == "__main__":
    unittest.main()
