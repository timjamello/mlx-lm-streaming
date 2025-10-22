# Copyright © 2025 Apple Inc.

"""
Streaming generation for models that can process input while generating output.
Based on the StreamingLLM paper's approach to simultaneous input/output processing.
"""

from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .models.streaming_cache import StreamingCacheList
from .sample_utils import apply_top_p


@dataclass
class StreamingState:
    """Track the state of streaming generation."""

    source_tokens: List[int]
    target_tokens: List[int]
    source_chunks: List[List[int]]  # Chunks of input as they arrive
    current_chunk_idx: int
    wait_k: int
    max_tokens: int
    temperature: float
    top_p: float


def streaming_generate_step(
    model: nn.Module,
    prompt: mx.array,
    cache: StreamingCacheList,
    *,
    wait_k: int = 5,
    max_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    input_stream: Optional[Generator[List[int], None, None]] = None,
    eos_token_ids: Optional[List[int]] = None,
) -> Generator[Tuple[int, bool, str], None, None]:
    """
    Generate tokens while processing streaming input.

    This implements the core StreamingLLM algorithm:
    1. Process initial prompt
    2. Wait for k input chunks before starting generation
    3. Alternate between processing new input and generating output
    4. Handle dynamic input arrival during generation

    Args:
        model: The model with streaming support
        prompt: Initial prompt tokens
        cache: StreamingCacheList for managing dual caches
        wait_k: Number of input chunks to wait before generation
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        input_stream: Generator yielding new input token chunks
        eos_token_ids: List of EOS token IDs to stop generation

    Yields:
        Tuple of (token, is_input, mode) where:
        - token: The token ID
        - is_input: True if processing input, False if generating
        - mode: "read" or "write" to indicate current mode
    """

    # Initialize state
    state = StreamingState(
        source_tokens=[],
        target_tokens=[],
        source_chunks=[],
        current_chunk_idx=0,
        wait_k=wait_k,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    # Process initial prompt as first chunk
    initial_chunk = prompt.tolist() if isinstance(prompt, mx.array) else prompt
    state.source_chunks.append(initial_chunk)

    # Helper function to process input chunk
    def process_input_chunk(chunk: List[int]) -> None:
        """Process a chunk of input tokens."""
        if not chunk:
            return

        # Set cache to read mode
        cache.set_mode(read_mode=True)

        # Process the chunk
        chunk_array = mx.array(chunk)[None, :]  # Add batch dimension
        logits = model(chunk_array, cache=cache)
        mx.eval(logits)  # Force evaluation

        state.source_tokens.extend(chunk)

        # Yield tokens to indicate processing
        for token in chunk:
            yield token, True, "read"

    # Helper function to generate one token
    def generate_token() -> Optional[int]:
        """Generate a single output token."""
        # Set cache to write mode (automatically merges)
        cache.set_mode(read_mode=False)

        # Use last generated token or a start token
        if state.target_tokens:
            input_token = mx.array([[state.target_tokens[-1]]])
        else:
            # First generation - use last source token
            input_token = mx.array([[state.source_tokens[-1]]])

        # Forward pass
        logits = model(input_token, cache=cache)
        logits = logits[:, -1, :]  # Get last token logits

        # Sample
        if temperature > 0:
            # Apply temperature and top-p filtering
            logits = logits / temperature
            logits = apply_top_p(logits, top_p)
            # Sample from filtered distribution
            token = mx.random.categorical(logits).item()
        else:
            token = mx.argmax(logits, axis=-1).item()

        state.target_tokens.append(token)
        return token

    # Process initial prompt
    yield from process_input_chunk(initial_chunk)

    # Main streaming loop
    generated_count = 0
    input_chunks_processed = 1  # Initial prompt counts as first chunk

    # If we have an input stream, collect chunks until wait_k
    if input_stream is not None:
        none_count = 0
        max_none_retries = 100  # Prevent infinite loop on None chunks

        try:
            # Collect more input chunks until we hit wait_k
            while input_chunks_processed < wait_k:
                new_chunk = next(input_stream)

                # Handle None (no input available yet)
                if new_chunk is None:
                    none_count += 1
                    if none_count >= max_none_retries:
                        print(
                            f"[WARNING] Received {none_count} consecutive None chunks. "
                            f"Starting generation with {input_chunks_processed} chunks "
                            f"(expected {wait_k})."
                        )
                        break
                    continue

                # Process actual chunk
                if new_chunk:
                    state.source_chunks.append(new_chunk)
                    yield from process_input_chunk(new_chunk)
                    input_chunks_processed += 1
                    none_count = 0  # Reset on successful chunk

        except StopIteration:
            # Input stream ended early
            if input_chunks_processed < wait_k:
                print(
                    f"[WARNING] Input stream ended after {input_chunks_processed} chunks, "
                    f"expected {wait_k}. Starting generation early."
                )

    # Now start generation, interleaving with any remaining input
    eos_encountered = False

    while input_stream is not None or (
        generated_count < max_tokens and not eos_encountered
    ):
        # Check if there's new input to process
        has_new_input = False
        if input_stream is not None:
            try:
                new_chunk = next(input_stream)
                # Handle None (no input available) vs empty list vs actual input
                if new_chunk is not None and new_chunk:
                    state.source_chunks.append(new_chunk)
                    yield from process_input_chunk(new_chunk)
                    input_chunks_processed += 1
                    has_new_input = True
                    # Reset EOS when new input arrives - allows multiple generation turns
                    eos_encountered = False
            except StopIteration:
                input_stream = None  # No more input

        # Generate if we didn't just process new input, haven't hit token limit, and haven't hit EOS
        if not has_new_input and generated_count < max_tokens and not eos_encountered:
            # Generate a token
            token = generate_token()
            if token is None:
                break

            yield token, False, "write"
            generated_count += 1

            # Check for EOS token
            if eos_token_ids and token in eos_token_ids:
                eos_encountered = True

    # Final cache cleanup
    cache.separate_all()


def streaming_generate(
    model: nn.Module,
    tokenizer,
    prompt: Union[str, List[int]],
    input_stream: Optional[Generator[List[int], None, None]] = None,
    wait_k: int = 5,
    max_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    verbose: bool = False,
) -> Tuple[List[int], List[int]]:
    """
    High-level streaming generation interface.

    Args:
        model: Model with streaming support
        tokenizer: Tokenizer for encoding/decoding
        prompt: Initial prompt (string or token list)
        input_stream: Generator yielding new input chunks
        wait_k: Wait-k policy parameter
        max_tokens: Maximum generation length
        temperature: Sampling temperature
        top_p: Top-p sampling
        verbose: Print tokens as they're processed

    Returns:
        Tuple of (source_tokens, target_tokens)
    """
    # Encode prompt if needed
    if isinstance(prompt, str):
        prompt_tokens = tokenizer.encode(prompt)
    else:
        prompt_tokens = prompt

    # Create streaming cache
    cache = model.make_cache(streaming=True)

    # Collect results
    source_tokens = []
    target_tokens = []

    # Run streaming generation
    for token, is_input, _mode in streaming_generate_step(
        model,
        mx.array(prompt_tokens),
        cache,
        wait_k=wait_k,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        input_stream=input_stream,
    ):
        if is_input:
            source_tokens.append(token)
            if verbose:
                print(f"[INPUT] {tokenizer.decode([token])}", end="", flush=True)
        else:
            target_tokens.append(token)
            if verbose:
                print(f"[GEN] {tokenizer.decode([token])}", end="", flush=True)

    if verbose:
        print()  # New line at end

    return source_tokens, target_tokens
