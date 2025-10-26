import time
from typing import List, Optional, Tuple

import mlx.core as mx


def calculate_wait_words(
    source_seg_len: List[int], target_word_idx: int, wait_k: int
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
    source_words_waited = min(wait_k + target_word_idx, source_word_total - 1)
    return source_words_waited


def generate_wait_words_list(
    source_seg_len: List[int], target_seg_len: List[int], wait_k: int
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
        >>> source_seg_len = [10, 5, 3, 4, 2]
        >>> target_seg_len = [4, 3, 5]
        >>> wait_k = 2
        >>> generate_wait_words_list(source_seg_len, target_seg_len, wait_k)
        [2, 3, 4]
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
    dtype: mx.Dtype = mx.float32,
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
    causal_mask = mx.tri(total_len, total_len, k=0, dtype=dtype)

    inf_value = -3e38
    attn_mask = (1 - causal_mask) * inf_value

    source_token_len = sum(source_seg_len)
    target_token_len = sum(target_seg_len)
    actual_total_len = source_token_len + target_token_len

    if actual_total_len < total_len:
        attn_mask = mx.concatenate(
            [
                attn_mask[:actual_total_len, :],
                mx.full(
                    (total_len - actual_total_len, total_len), inf_value, dtype=dtype
                ),
            ],
            axis=0,
        )

    streaming_start = source_token_len

    for index, num_tokens in enumerate(target_seg_len):
        wait_words = wait_tokens_list[index]

        wait_tokens = sum(source_seg_len[:wait_words])

        if wait_tokens < source_token_len:
            attn_mask[
                streaming_start : streaming_start + num_tokens,
                wait_tokens:source_token_len,
            ] = inf_value

        streaming_start += num_tokens

    if target_token_len > 1:
        attn_mask[
            source_token_len : source_token_len + target_token_len - 1,
            :source_token_len,
        ] = attn_mask[
            source_token_len + 1 : source_token_len + target_token_len,
            :source_token_len,
        ]

    return attn_mask


def segment_by_words(
    text: str, tokenizer, include_spaces: bool = True
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
        >>>
        >>>
    """
    words = text.split()

    token_lengths = []
    for i, word in enumerate(words):
        if i > 0 and include_spaces:
            word_with_space = " " + word
        else:
            word_with_space = word

        tokens = tokenizer.encode(word_with_space, add_special_tokens=False)
        token_lengths.append(len(tokens))

    return words, token_lengths


def calculate_position_ids(
    source_seg_len: List[int],
    target_seg_len: List[int],
    current_source_words: int,
    current_target_words: int,
    pe_cache_length: int = 0,
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
        >>> source_seg_len = [10, 5, 3]
        >>> target_seg_len = [4, 3]
        >>>
        >>> src_pos, tgt_pos = calculate_position_ids(
        ...     source_seg_len, target_seg_len,
        ...     current_source_words=2,
        ...     current_target_words=1,
        ...     pe_cache_length=0
        ... )
        >>>
        >>>
    """
    source_token_len = sum(source_seg_len[:current_source_words])
    source_position_ids = mx.arange(source_token_len)

    target_token_len = sum(target_seg_len[:current_target_words])
    target_position_ids = mx.arange(pe_cache_length, pe_cache_length + target_token_len)

    return source_position_ids, target_position_ids


class StreamingState:
    """
    Tracks the state during streaming generation.

    This class maintains all the bookkeeping needed for streaming generation:
    - Which source words have been read
    - Which target words have been generated
    - Current mode (reading or writing)
    - Wait-k policy tracking
    - Source token arrival timestamps for attention reweighting
    """

    def __init__(
        self,
        wait_k: int,
        source_seg_len: Optional[List[int]] = None,
        max_target_words: Optional[int] = None,
        pe_cache_length: int = 0,
    ):
        """
        Initialize streaming state.

        Args:
            wait_k: Wait-k parameter
            source_seg_len: List of token lengths for each source word (optional for queue-based streaming)
            max_target_words: Maximum number of target words to generate
            pe_cache_length: Starting position for target position IDs
        """
        self.source_seg_len = source_seg_len if source_seg_len is not None else []
        self.wait_k = wait_k
        self.max_target_words = max_target_words
        self.pe_cache_length = pe_cache_length

        self.source_words_read = 0
        self.target_words_generated = 0
        self.is_reading = True
        self.finished = False
        self.source_stream_ended = False

        self.wait_lagging = []

        # Track arrival time for each source token (for attention reweighting)
        self.source_token_timestamps = []  # List of timestamps, one per source token

    def add_source_segment(self, token_length: int):
        """Add a new source segment as it's read from the queue."""
        self.source_seg_len.append(token_length)
        # Track timestamp for each token in this segment
        current_time = time.time()
        for _ in range(token_length):
            self.source_token_timestamps.append(current_time)

    def mark_source_stream_ended(self):
        """Mark that the source stream has ended (no more input coming)."""
        self.source_stream_ended = True

    def should_read_next_source(self) -> bool:
        """Check if we should read the next source word."""
        return not self.source_stream_ended and not self.finished

    def should_write_next_target(self) -> bool:
        """Check if we should write the next target word."""
        if self.max_target_words is not None:
            return self.target_words_generated < self.max_target_words
        return True

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
        required_source_words = self.wait_k + self.target_words_generated

        # If stream has ended, cap at what we have
        if self.source_stream_ended and len(self.source_seg_len) > 0:
            required_source_words = min(
                required_source_words, len(self.source_seg_len) - 1
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


def compute_recency_bias(
    source_timestamps: List[float],
    source_length: int,
    target_length: int,
    current_time: float,
    recency_window: float = 1.0,
    source_boost: float = 1000.0,
    target_dampening: float = 100000000.0,
) -> Optional[mx.array]:
    """
    Compute attention bias to boost recent source tokens and dampen target momentum.

    Args:
        source_timestamps: List of timestamps for each source token (may be shorter than source_length)
        source_length: Actual number of source tokens in the cache
        target_length: Number of target tokens generated so far
        current_time: Current timestamp
        recency_window: Time window (seconds) for considering tokens as "recent"
        source_boost: Attention boost for recent source tokens
        target_dampening: Attention dampening for target tokens (to reduce momentum)

    Returns:
        Bias matrix of shape (1, source_length + target_length) or None if no bias needed
    """
    if source_length == 0 or target_length == 0:
        return None

    total_length = source_length + target_length

    # Create bias matrix: (1, total_length)
    # We only need to bias the last target token (the one being generated)
    bias = mx.zeros((1, total_length))

    # Compute recency for each source token that we have a timestamp for
    # Note: source_timestamps may be shorter than source_length due to template tokens
    num_timestamped = min(len(source_timestamps), source_length)
    timestamp_offset = source_length - num_timestamped  # Where our timestamps start

    for i in range(num_timestamped):
        timestamp = source_timestamps[i]
        age = current_time - timestamp
        if age < recency_window:
            # Recent tokens get a boost
            # Use exponential decay: newer = stronger boost
            recency_factor = mx.exp(-age / recency_window)
            bias[0, timestamp_offset + i] = source_boost * recency_factor

    # Dampen target tokens to reduce momentum
    if target_length > 0:
        bias[0, source_length:] = -target_dampening

    return bias
