# Copyright © 2023-2024 Apple Inc.
# Stopping criteria for streaming generation

from typing import Optional, List, Tuple
import mlx.core as mx


class WordBoundaryStoppingCriteria:
    """
    Stopping criteria that detects word boundaries and punctuation marks.

    This is used to determine when to switch from writing (generating target tokens)
    back to reading (processing source tokens) in the streaming generation loop.

    Based on StreamingLLM's StopTokenCriteria from:
    generation/Stopping_criteria.py:9-33

    The criteria looks for:
    1. Word boundaries (spaces)
    2. Punctuation marks (. , : ; ? !)
    3. End instruction tokens
    4. Maximum token limit
    """

    def __init__(
        self,
        tokenizer,
        max_tokens_per_word: int = 50,
        end_tokens: Optional[List[str]] = None,
        terminating_punctuation: Optional[List[str]] = None
    ):
        """
        Initialize stopping criteria.

        Args:
            tokenizer: Tokenizer for decoding tokens
            max_tokens_per_word: Maximum tokens to generate for one word
            end_tokens: List of special end tokens (e.g., ['<|end|>', '<|im_end|>'])
            terminating_punctuation: List of punctuation that ends a word
        """
        self.tokenizer = tokenizer
        self.max_tokens_per_word = max_tokens_per_word

        self.end_tokens = end_tokens or []
        self.terminating_punctuation = terminating_punctuation or [
            ".", ",", ":", ";", "?", "!"
        ]

    def __call__(
        self,
        generated_ids: mx.array,
        token_count: int
    ) -> Tuple[bool, bool]:
        """
        Check if we should stop generating and switch back to reading.

        Args:
            generated_ids: Array of generated token IDs for current word (1D)
            token_count: Number of tokens generated for current word

        Returns:
            Tuple of (should_stop, remove_last_token)
            - should_stop: True if we should stop generating this word
            - remove_last_token: True if the last token should be removed
              (happens when we detect a space, indicating the token belongs
               to the next word)

        Logic:
        1. If we find a space after the first token, stop (it's a word boundary)
        2. If we find punctuation, stop
        3. If we find an end token, stop
        4. If we've hit max tokens, stop
        5. If stopped due to space, remove the last token (it starts next word)
        """
        should_stop = False
        remove_last_token = False

        # Need at least one token
        if token_count == 0:
            return should_stop, remove_last_token

        # Decode tokens to text
        # Handle both 1D and 2D arrays
        if len(generated_ids.shape) == 2:
            token_ids = generated_ids[0]
        else:
            token_ids = generated_ids

        # Convert MLX array to Python list for tokenizer
        token_list = token_ids.tolist()

        # Decode all generated tokens
        full_text = self.tokenizer.decode(token_list)

        # Decode just the last token
        if len(token_list) > 0:
            last_token_text = self.tokenizer.decode([token_list[-1]])
        else:
            last_token_text = ""

        # Check stopping conditions

        # 1. Check for word boundary (space) - but skip first character
        #    to avoid stopping on initial space
        if len(full_text) > 1 and ' ' in full_text[1:]:
            should_stop = True
            # If we found a space, the last token likely starts the next word
            if len(token_list) >= 2:
                remove_last_token = True

        # 2. Check for terminating punctuation
        if last_token_text.strip() in self.terminating_punctuation:
            should_stop = True

        # 3. Check for end instruction tokens
        for end_token in self.end_tokens:
            if end_token in last_token_text:
                should_stop = True

        # 4. Check max tokens limit
        if token_count >= self.max_tokens_per_word:
            should_stop = True

        return should_stop, remove_last_token


class MaxLengthStoppingCriteria:
    """
    Simple stopping criteria based on total generated length.
    """

    def __init__(self, max_length: int):
        """
        Initialize max length criteria.

        Args:
            max_length: Maximum total tokens to generate
        """
        self.max_length = max_length

    def __call__(self, total_generated: int) -> bool:
        """
        Check if we've hit the maximum length.

        Args:
            total_generated: Total number of tokens generated so far

        Returns:
            True if we should stop generation
        """
        return total_generated >= self.max_length


class EOSStoppingCriteria:
    """
    Stopping criteria based on EOS token.
    """

    def __init__(self, eos_token_id: int):
        """
        Initialize EOS criteria.

        Args:
            eos_token_id: The EOS token ID
        """
        self.eos_token_id = eos_token_id

    def __call__(self, last_token_id: int) -> bool:
        """
        Check if the last token is EOS.

        Args:
            last_token_id: The last generated token ID

        Returns:
            True if we should stop generation
        """
        return last_token_id == self.eos_token_id


class CombinedStoppingCriteria:
    """
    Combines multiple stopping criteria with OR logic.
    """

    def __init__(self, criteria: List):
        """
        Initialize combined criteria.

        Args:
            criteria: List of stopping criteria objects
        """
        self.criteria = criteria

    def add_criteria(self, criterion):
        """Add a new stopping criterion."""
        self.criteria.append(criterion)

    def __call__(self, **kwargs) -> bool:
        """
        Check all criteria - stop if ANY return True.

        Args:
            **kwargs: Arguments passed to each criterion

        Returns:
            True if any criterion says to stop
        """
        for criterion in self.criteria:
            try:
                if criterion(**kwargs):
                    return True
            except TypeError:
                # Criterion doesn't accept these kwargs, skip it
                continue
        return False
