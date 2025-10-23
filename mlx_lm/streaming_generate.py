# Copyright © 2023-2024 Apple Inc.
# Streaming generation for StreamingLLM

from typing import Dict, Generator, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .models.streaming_cache import DualStreamingCache
from .sample_utils import make_sampler
from .streaming_stopping_criteria import (
    EOSStoppingCriteria,
    MaxLengthStoppingCriteria,
    WordBoundaryStoppingCriteria,
)
from .streaming_utils import StreamingState, calculate_wait_words


def stream_generate_streaming(
    model,
    tokenizer,
    source_token_ids: mx.array,
    source_seg_len: List[int],
    wait_k: int = 3,
    max_new_words: Optional[int] = None,
    max_tokens: int = 1024,
    assistant_start_tokens: Optional[mx.array] = None,
    pe_cache_length: int = 0,
    split_mode: str = "word",
    end_token: str = "<|im_end|>",
    temp: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
    **kwargs,
) -> Generator[Dict, None, None]:
    """
    Generate tokens using streaming policy (wait-k).

    This implements the core streaming generation algorithm from StreamingLLM,
    alternating between reading source tokens and writing target tokens based
    on the wait-k policy.

    Reference: StreamingLLM's _sample_streaming in generation/generate.py:947-1206

    Args:
        model: The streaming model (with DualStreamingCache support)
        tokenizer: Tokenizer instance
        source_token_ids: Source token IDs (1, seq_len)
        source_seg_len: List of token lengths for each source segment/word
        wait_k: Wait-k parameter (wait for k source words before generating)
        max_new_words: Maximum number of target words to generate
        max_tokens: Maximum total tokens to generate
        assistant_start_tokens: Token IDs for assistant message start
        pe_cache_length: Starting position ID for target tokens (default 0)
        split_mode: Segmentation mode ('word' or 'sentence')
        end_token: End instruction token
        temp: Sampling temperature
        top_p: Top-p sampling parameter
        repetition_penalty: Repetition penalty factor
        repetition_context_size: Context size for repetition penalty

    Yields:
        Dictionary with:
        - text: Generated text for current word
        - token_ids: Generated token IDs
        - is_final: Whether this is the final word
        - source_words_read: Number of source words read so far
        - target_words_generated: Number of target words generated
        - mode: Current mode ('read' or 'write')
        - word_complete: True when a word has been completed (write mode), False otherwise

    Algorithm:
        1. Initialize dual caches and state
        2. While not finished:
           IF reading mode:
             - Read next source word chunk
             - Update source cache
             - Switch to writing if wait-k satisfied
           ELSE (writing mode):
             - Generate tokens until word boundary
             - Update target cache
             - Switch to reading if more source available

    Example:
        >>> # Prepare input
        >>> prepared = prepare_streaming_input(
        ...     "Hello world, how are you?",
        ...     tokenizer,
        ...     wait_k=2
        ... )
        >>> # Generate
        >>> for chunk in stream_generate_streaming(
        ...     model,
        ...     tokenizer,
        ...     prepared["source_token_ids"],
        ...     prepared["source_seg_len"],
        ...     wait_k=2
        ... ):
        ...     print(chunk["text"], end="", flush=True)
    """
    # Create sampler
    sampler = make_sampler(
        temp=temp,
        top_p=top_p,
        min_p=kwargs.get("min_p", 0.0),
        top_k=kwargs.get("top_k", 0),
    )

    # Initialize dual caches for all layers
    if hasattr(model, "make_cache"):
        caches = model.make_cache()  # List of DualStreamingCache
    else:
        num_layers = len(model.layers)
        caches = [DualStreamingCache() for _ in range(num_layers)]

    # Initialize streaming state
    state = StreamingState(
        source_seg_len=source_seg_len,
        wait_k=wait_k,
        max_target_words=max_new_words,
        pe_cache_length=pe_cache_length,
    )

    # Initialize stopping criteria
    word_boundary_criteria = WordBoundaryStoppingCriteria(
        tokenizer=tokenizer, max_tokens_per_word=50, end_tokens=[end_token]
    )

    max_length_criteria = MaxLengthStoppingCriteria(max_length=max_tokens)

    eos_criteria = EOSStoppingCriteria(eos_token_id=tokenizer.eos_token_id)

    # Track generated tokens
    all_target_tokens = []
    current_word_tokens = []
    total_tokens_generated = 0

    # Get assistant start tokens (if provided)
    if assistant_start_tokens is not None:
        next_input = assistant_start_tokens
    else:
        # Use a simple start token
        # Some tokenizers don't have a BOS token, fall back to EOS or first generated token
        if hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None:
            next_input = mx.array([[tokenizer.bos_token_id]])
        elif hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            # Use EOS as start token (common pattern for some tokenizers)
            next_input = mx.array([[tokenizer.eos_token_id]])
        else:
            # Fallback: use token ID 0
            next_input = mx.array([[0]])

    # Source position tracking
    source_pos_offset = 0

    # === MAIN STREAMING LOOP ===
    while not state.finished:

        # === READING PHASE ===
        if state.is_reading and state.should_read_next_source():
            # Calculate how many source tokens to read
            tokens_to_read = state.get_source_tokens_to_read()

            # Skip if no tokens to read (e.g., empty segment)
            if tokens_to_read == 0:
                # Mark as read and continue
                state.mark_source_read()
                # Check if we can start writing
                if state.check_wait_k_policy():
                    state.switch_to_writing()
                    current_word_tokens = []
                continue

            # Get source chunk
            source_chunk = source_token_ids[
                :, source_pos_offset : source_pos_offset + tokens_to_read
            ]

            # Safety check: ensure chunk is not empty
            if source_chunk.shape[1] == 0:
                # This shouldn't happen, but handle it gracefully
                print(
                    f"Warning: Empty source_chunk despite tokens_to_read={tokens_to_read}"
                )
                print(f"  source_token_ids.shape: {source_token_ids.shape}")
                print(f"  source_pos_offset: {source_pos_offset}")
                print(f"  Skipping this segment...")
                state.mark_source_read()
                if state.check_wait_k_policy():
                    state.switch_to_writing()
                    current_word_tokens = []
                continue

            # Create position IDs for this chunk
            position_ids = mx.arange(
                source_pos_offset, source_pos_offset + tokens_to_read
            ).reshape(1, -1)

            # Forward pass in reading mode
            _ = model(
                source_chunk, cache=caches, position_ids=position_ids, is_reading=True
            )

            # Update state
            source_pos_offset += tokens_to_read
            state.mark_source_read()

            # Yield read progress
            yield {
                "text": "",
                "token_ids": [],
                "is_final": False,
                "source_words_read": state.source_words_read,
                "target_words_generated": state.target_words_generated,
                "mode": "read",
                "word_complete": False,
            }

            # Check if we can start writing
            if state.check_wait_k_policy():
                state.switch_to_writing()
                # Reset for next word
                current_word_tokens = []

        # === WRITING PHASE ===
        elif not state.is_reading and state.should_write_next_target():
            print(f"\n[DEBUG] Entering WRITING phase - source_words_read={state.source_words_read}, target_words_generated={state.target_words_generated}")
            token_count = 0
            word_finished = False

            # Generate tokens until word boundary
            while not word_finished:
                # Create position IDs for target
                target_pos = pe_cache_length + len(current_word_tokens)
                position_ids = mx.array([[target_pos]])

                # Forward pass in writing mode
                logits = model(
                    next_input,
                    cache=caches,
                    position_ids=position_ids,
                    is_reading=False,
                )

                # Sample next token
                logits = logits[:, -1, :]

                # Apply repetition penalty if specified
                if repetition_penalty and repetition_penalty != 1.0:
                    if len(all_target_tokens) > 0:
                        context = all_target_tokens[-repetition_context_size:]
                        logits = apply_repetition_penalty(
                            logits, context, repetition_penalty
                        )

                next_token = sampler(logits)
                token_count += 1
                total_tokens_generated += 1

                # DEBUG: Show what's being generated
                token_id = int(next_token[0])
                token_text = tokenizer.decode([token_id])
                print(f"[DEBUG WRITE] Token #{token_count}: id={token_id}, text='{token_text}', target_pos={target_pos}, is_reading=False, current_word_len={len(current_word_tokens)}")

                # Add to current word
                current_word_tokens.append(token_id)

                # Check stopping criteria
                current_word_array = mx.array(current_word_tokens)

                # 1. Check word boundary
                should_stop, remove_last = word_boundary_criteria(
                    current_word_array, token_count
                )

                # 2. Check EOS
                if eos_criteria(int(next_token[0])):
                    should_stop = True

                # 3. Check max length
                if max_length_criteria(total_tokens_generated):
                    should_stop = True
                    state.finished = True

                if should_stop:
                    word_finished = True

                    # Handle token removal (for space detection)
                    if remove_last and len(current_word_tokens) > 1:
                        removed_token = current_word_tokens.pop()
                        # Need to pop from cache too
                        for cache in caches:
                            cache.target_cache.offset -= 1

                    # Add to all tokens
                    all_target_tokens.extend(current_word_tokens)

                    # Decode word
                    word_text = tokenizer.decode(current_word_tokens)

                    # Mark word as generated
                    state.mark_target_written()

                    # Yield generated word
                    yield {
                        "text": word_text,
                        "token_ids": current_word_tokens.copy(),
                        "is_final": state.finished
                        or not state.should_write_next_target(),
                        "source_words_read": state.source_words_read,
                        "target_words_generated": state.target_words_generated,
                        "mode": "write",
                        "word_complete": True,
                    }

                    # Check if we should switch back to reading
                    if not state.finished and state.should_read_next_source():
                        state.switch_to_reading()
                    elif not state.should_read_next_source():
                        # No more source to read, stay in writing mode
                        current_word_tokens = []
                    else:
                        # Finished
                        state.finished = True

                else:
                    # Update next_input for next iteration
                    next_input = next_token.reshape(1, 1)

        else:
            # No more reading or writing to do
            state.finished = True


def apply_repetition_penalty(
    logits: mx.array, context_tokens: List[int], penalty: float
) -> mx.array:
    """
    Apply repetition penalty to logits.

    Args:
        logits: Logits array (batch, vocab_size)
        context_tokens: List of recent token IDs
        penalty: Penalty factor (> 1 reduces repetition)

    Returns:
        Modified logits
    """
    if len(context_tokens) == 0 or penalty == 1.0:
        return logits

    # Apply penalty to tokens in context
    for token_id in set(context_tokens):
        if logits[0, token_id] < 0:
            logits[0, token_id] *= penalty
        else:
            logits[0, token_id] /= penalty

    return logits


def generate_streaming(
    model,
    tokenizer,
    prompt: str,
    wait_k: int = 3,
    max_new_words: Optional[int] = None,
    max_tokens: int = 1024,
    system_prompt: str = "",
    split_mode: str = "word",
    verbose: bool = False,
    **kwargs,
) -> str:
    """
    Convenient wrapper for streaming generation.

    Args:
        model: The streaming model
        tokenizer: Tokenizer instance
        prompt: Input prompt text
        wait_k: Wait-k parameter
        max_new_words: Maximum words to generate
        max_tokens: Maximum tokens to generate
        system_prompt: System prompt
        split_mode: How to split text ('word' or 'sentence')
        verbose: Whether to print progress
        **kwargs: Additional generation parameters

    Returns:
        Generated text as a single string

    Example:
        >>> from mlx_lm.models.qwen2_streaming import Model, ModelArgs
        >>> model = Model(args)
        >>> text = generate_streaming(
        ...     model,
        ...     tokenizer,
        ...     "Translate to French: Hello world",
        ...     wait_k=3
        ... )
    """
    from .streaming_data_utils import prepare_streaming_input

    # Prepare input
    prepared = prepare_streaming_input(
        source_text=prompt,
        tokenizer=tokenizer,
        wait_k=wait_k,
        system_prompt=system_prompt,
        split_mode=split_mode,
        add_space=kwargs.get("add_space", False),
        pe_cache_length=kwargs.get("pe_cache_length", 0),
    )

    if verbose:
        print("=" * 50)
        print(f"Streaming generation with wait-k={wait_k}")
        print("=" * 50)

    # Generate
    generated_text = ""
    for chunk in stream_generate_streaming(
        model=model,
        tokenizer=tokenizer,
        source_token_ids=prepared["source_token_ids"],
        source_seg_len=prepared["source_seg_len"],
        wait_k=wait_k,
        max_new_words=max_new_words,
        max_tokens=max_tokens,
        assistant_start_tokens=prepared["assistant_start_tokens"],
        split_mode=split_mode,
        end_token=prepared["metadata"]["end_token"],
        **kwargs,
    ):
        if chunk["mode"] == "write" and chunk["text"]:
            generated_text += chunk["text"]
            if verbose:
                print(chunk["text"], end="", flush=True)

    if verbose:
        print()
        print("=" * 50)
        print(f"Generated {len(generated_text.split())} words")
        print("=" * 50)

    return generated_text
