# Copyright © 2023-2024 Apple Inc.
# Streaming utilities for StreamingLLM implementation

from typing import List, Optional, Tuple
import mlx.core as mx


def calculate_wait_words(
    source_seg_len: List[int],
    target_word_idx: int,
    wait_k: int
) -> int:
    """
    Calculate how many source words to wait for before generating target word.

    This implements the wait-k policy from the StreamingLLM paper.

    Args:
        source_seg_len: List of token lengths for each source word/segment
        target_word_idx: Index of the target word being generated (0-indexed)
        wait_k: Wait-k parameter (number of source words to wait)

    Returns:
        Number of source words to wait for (index in source_seg_len)

    Example:
        >>> source_seg_len = [5, 3, 4, 2, 6]  # 5 source words
        >>> wait_k = 2
        >>> # For target word 0: wait for words 0,1 (min(2+0, 5-1) = 2)
        >>> calculate_wait_words(source_seg_len, 0, wait_k)
        2
        >>> # For target word 1: wait for words 0,1,2 (min(2+1, 5-1) = 3)
        >>> calculate_wait_words(source_seg_len, 1, wait_k)
        3
        >>> # For target word 3: wait for all (min(2+3, 5-1) = 4)
        >>> calculate_wait_words(source_seg_len, 3, wait_k)
        4
    """
    source_word_total = len(source_seg_len)
    # Wait for wait_k + target_word_idx source words, but not more than available
    # -1 because we typically don't count the instruction as a "word"
    source_words_waited = min(wait_k + target_word_idx, source_word_total - 1)
    return source_words_waited


def generate_wait_words_list(
    source_seg_len: List[int],
    target_seg_len: List[int],
    wait_k: int
) -> List[int]:
    """
    Generate wait words list for all target words.

    Args:
        source_seg_len: List of token lengths for each source word
        target_seg_len: List of token lengths for each target word
        wait_k: Wait-k parameter

    Returns:
        List of indices indicating how many source words to wait for each target word

    Example:
        >>> source_seg_len = [10, 5, 3, 4, 2]  # 5 source words
        >>> target_seg_len = [4, 3, 5]  # 3 target words
        >>> wait_k = 2
        >>> generate_wait_words_list(source_seg_len, target_seg_len, wait_k)
        [2, 3, 4]  # Wait for 2, 3, 4 source words respectively
    """
    wait_tokens_list = []
    source_word_total = len(source_seg_len)

    for tgt_idx in range(len(target_seg_len)):
        source_words_waited = min(wait_k + tgt_idx, source_word_total)
        wait_tokens_list.append(source_words_waited)

    return wait_tokens_list


def create_streaming_attention_mask(
    total_len: int,
    source_seg_len: List[int],
    target_seg_len: List[int],
    wait_tokens_list: List[int],
    dtype: mx.Dtype = mx.float32
) -> mx.array:
    """
    Create custom attention mask for streaming generation.

    This mask ensures that:
    1. Target tokens can only attend to source tokens that have been "waited for"
    2. Target tokens attend to all previous target tokens (causal)
    3. Source tokens attend to all previous source tokens (causal)

    Args:
        total_len: Total sequence length (padded)
        source_seg_len: List of token lengths for each source word
        target_seg_len: List of token lengths for each target word
        wait_tokens_list: List of source word indices waited for each target word
        dtype: Data type for the mask

    Returns:
        Attention mask of shape (total_len, total_len)

    The mask uses -inf for positions that cannot be attended to.

    Reference: Based on StreamingLLM's generate_attention_mask in
               models/Qwen2_5/qwen_streaming.py:711-752
    """
    # Start with upper triangular (causal) mask
    # 1 in upper triangle (including diagonal), 0 below
    causal_mask = mx.tri(total_len, total_len, k=0, dtype=dtype)

    # Convert to attention mask format (0 = can attend, -inf = cannot attend)
    # Invert: upper triangle should be -inf
    inf_value = -3e38
    attn_mask = (1 - causal_mask) * inf_value

    # Calculate positions
    source_token_len = sum(source_seg_len)
    target_token_len = sum(target_seg_len)
    actual_total_len = source_token_len + target_token_len

    # Mask out padding positions (if any)
    if actual_total_len < total_len:
        attn_mask = mx.concatenate([
            attn_mask[:actual_total_len, :],
            mx.full((total_len - actual_total_len, total_len), inf_value, dtype=dtype)
        ], axis=0)

    # Apply streaming constraints for target tokens
    streaming_start = source_token_len

    for index, num_tokens in enumerate(target_seg_len):
        # How many source words can this target word attend to?
        wait_words = wait_tokens_list[index]

        # Calculate how many source tokens that corresponds to
        wait_tokens = sum(source_seg_len[:wait_words])

        # Mask out future source tokens (tokens beyond wait_tokens)
        # Target word cannot attend to source tokens it hasn't "waited" for
        if wait_tokens < source_token_len:
            attn_mask[
                streaming_start:streaming_start + num_tokens,
                wait_tokens:source_token_len
            ] = inf_value

        streaming_start += num_tokens

    # Shift source attention mask in streaming part up by one
    # This aligns the attention for the target tokens properly
    if target_token_len > 1:
        attn_mask[
            source_token_len:source_token_len + target_token_len - 1,
            :source_token_len
        ] = attn_mask[
            source_token_len + 1:source_token_len + target_token_len,
            :source_token_len
        ]

    return attn_mask


def segment_by_words(
    text: str,
    tokenizer,
    include_spaces: bool = True
) -> Tuple[List[str], List[int]]:
    """
    Segment text into words and calculate token lengths for each word.

    Args:
        text: Input text to segment
        tokenizer: Tokenizer to use
        include_spaces: Whether to include leading spaces in token count

    Returns:
        Tuple of (word_list, token_lengths)

    Example:
        >>> text = "Hello world, how are you?"
        >>> words, lengths = segment_by_words(text, tokenizer)
        >>> # words: ["Hello", "world", ",", "how", "are", "you", "?"]
        >>> # lengths: [2, 3, 1, 2, 2, 2, 1]  # example token counts
    """
    # Split by whitespace
    words = text.split()

    token_lengths = []
    for i, word in enumerate(words):
        # Add leading space for words after the first (if include_spaces)
        if i > 0 and include_spaces:
            word_with_space = " " + word
        else:
            word_with_space = word

        # Tokenize and get length
        tokens = tokenizer.encode(word_with_space, add_special_tokens=False)
        token_lengths.append(len(tokens))

    return words, token_lengths


def calculate_position_ids(
    source_seg_len: List[int],
    target_seg_len: List[int],
    current_source_words: int,
    current_target_words: int,
    pe_cache_length: int = 0
) -> Tuple[mx.array, mx.array]:
    """
    Calculate position IDs for source and target tokens in streaming mode.

    Args:
        source_seg_len: List of token lengths for each source word
        target_seg_len: List of token lengths for each target word
        current_source_words: Number of source words processed so far
        current_target_words: Number of target words generated so far
        pe_cache_length: Starting position ID for target tokens (default 0)

    Returns:
        Tuple of (source_position_ids, target_position_ids)

    Example:
        >>> source_seg_len = [10, 5, 3]  # 3 source words
        >>> target_seg_len = [4, 3]  # 2 target words
        >>> # After reading 2 source words and generating 1 target word:
        >>> src_pos, tgt_pos = calculate_position_ids(
        ...     source_seg_len, target_seg_len,
        ...     current_source_words=2,
        ...     current_target_words=1,
        ...     pe_cache_length=0
        ... )
        >>> # src_pos: [0,1,2,...,14]  # 15 total tokens (10+5)
        >>> # tgt_pos: [0,1,2,3]  # 4 tokens for first target word
    """
    # Source position IDs: sequential from 0
    source_token_len = sum(source_seg_len[:current_source_words])
    source_position_ids = mx.arange(source_token_len)

    # Target position IDs: sequential from pe_cache_length
    target_token_len = sum(target_seg_len[:current_target_words])
    target_position_ids = mx.arange(
        pe_cache_length,
        pe_cache_length + target_token_len
    )

    return source_position_ids, target_position_ids


class StreamingState:
    """
    Tracks the state during streaming generation.

    This class maintains all the bookkeeping needed for streaming generation:
    - Which source words have been read
    - Which target words have been generated
    - Current mode (reading or writing)
    - Wait-k policy tracking
    """

    def __init__(
        self,
        source_seg_len: List[int],
        wait_k: int,
        max_target_words: Optional[int] = None,
        pe_cache_length: int = 0
    ):
        """
        Initialize streaming state.

        Args:
            source_seg_len: List of token lengths for each source word
            wait_k: Wait-k parameter
            max_target_words: Maximum number of target words to generate
            pe_cache_length: Starting position for target position IDs
        """
        self.source_seg_len = source_seg_len
        self.wait_k = wait_k
        self.max_target_words = max_target_words
        self.pe_cache_length = pe_cache_length

        # State tracking
        self.source_words_read = 0  # How many source words processed
        self.target_words_generated = 0  # How many target words generated
        self.is_reading = True  # Start in reading mode
        self.finished = False

        # Lagging tracking (for metrics)
        self.wait_lagging = []  # Track lagging for each target word

    def should_read_next_source(self) -> bool:
        """Check if we should read the next source word."""
        return (
            self.source_words_read < len(self.source_seg_len) and
            not self.finished
        )

    def should_write_next_target(self) -> bool:
        """Check if we should write the next target word."""
        if self.max_target_words is not None:
            return self.target_words_generated < self.max_target_words
        return True  # No limit

    def get_source_tokens_to_read(self) -> int:
        """Get number of tokens in next source word to read."""
        if self.source_words_read < len(self.source_seg_len):
            return self.source_seg_len[self.source_words_read]
        return 0

    def mark_source_read(self):
        """Mark current source word as read."""
        self.source_words_read += 1

    def mark_target_written(self):
        """Mark current target word as written."""
        self.target_words_generated += 1
        # Record lagging (how many source words we've waited for)
        self.wait_lagging.append(self.source_words_read)

    def switch_to_writing(self):
        """Switch from reading to writing mode."""
        self.is_reading = False

    def switch_to_reading(self):
        """Switch from writing to reading mode."""
        self.is_reading = True

    def check_wait_k_policy(self) -> bool:
        """
        Check if wait-k policy is satisfied for next target word.

        Returns:
            True if we have waited for enough source words
        """
        required_source_words = min(
            self.wait_k + self.target_words_generated,
            len(self.source_seg_len)
        )
        return self.source_words_read >= required_source_words

    def __repr__(self):
        return (
            f"StreamingState("
            f"source={self.source_words_read}/{len(self.source_seg_len)}, "
            f"target={self.target_words_generated}, "
            f"mode={'READ' if self.is_reading else 'WRITE'}, "
            f"finished={self.finished})"
        )
