# Copyright © 2023-2024 Apple Inc.
# Data utilities for streaming generation

from typing import Dict, List, Optional, Tuple

import mlx.core as mx


class StreamingDataPreparator:
    """
    Prepares data for streaming generation.

    This class handles:
    1. Chat template construction
    2. Word/token segmentation
    3. Position ID calculation
    4. Metadata extraction for streaming

    Based on StreamingLLM's StreamingDataCollator from:
    dataloader_hf.py

    Simplified for inference/generation use cases.
    """

    def __init__(
        self,
        tokenizer,
        system_prompt: str = "",
        user_template: str = "<|im_start|>user\n{content}<|im_end|>\n",
        assistant_template: str = "<|im_start|>assistant\n",
        end_token: str = "<|im_end|>\n",
        split_mode: str = "word",
        add_space: bool = False,
    ):
        """
        Initialize data preparator.

        Args:
            tokenizer: Tokenizer instance
            system_prompt: System prompt text
            user_template: Template for user messages (with {content} placeholder)
            assistant_template: Template for assistant messages (start)
            end_token: End token for messages
            split_mode: How to split text ('word' or 'sentence')
            add_space: Whether to add leading spaces to words (for LLaMA-style tokenizers)
        """
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.assistant_template = assistant_template
        self.end_token = end_token
        self.split_mode = split_mode
        self.add_space = add_space

        # Precompute template token lengths
        self.assistant_tokens = self._tokenize(assistant_template)
        self.end_tokens = self._tokenize(end_token)

        # Calculate instruction lengths
        if system_prompt:
            system_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            self.system_token_len = len(self._tokenize(system_text))
        else:
            self.system_token_len = 0

        self.user_template_len = len(
            self._tokenize(user_template.replace("{content}", ""))
        )

    def _tokenize(self, text: str) -> List[int]:
        """Tokenize text without adding special tokens."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def prepare_source_text(
        self, source_text: str, include_system: bool = True
    ) -> Tuple[str, List[int], List[int]]:
        """
        Prepare source text for streaming generation.

        Args:
            source_text: Source text to process
            include_system: Whether to include system prompt

        Returns:
            Tuple of (formatted_text, token_ids, segment_lengths)
            - formatted_text: Formatted with chat template
            - token_ids: Tokenized IDs
            - segment_lengths: Token length of each word/segment
        """
        # Format with chat template
        if include_system and self.system_prompt:
            system_text = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
        else:
            system_text = ""

        user_message = self.user_template.format(content=source_text)
        full_text = system_text + user_message

        # Segment source text into words
        source_segments, segment_lengths = self._segment_text(source_text)

        # Tokenize full text
        token_ids = self._tokenize(full_text)

        # Calculate segment lengths including template overhead
        full_segment_lengths = self._calculate_segment_lengths_with_template(
            source_segments, include_system
        )

        return full_text, token_ids, full_segment_lengths

    def _segment_text(self, text: str) -> Tuple[List[str], List[int]]:
        """
        Segment text into words/sentences and calculate token lengths.

        Args:
            text: Text to segment

        Returns:
            Tuple of (segments, token_lengths)
        """
        if self.split_mode == "word":
            # Split by whitespace
            segments = text.split()
        elif self.split_mode == "sentence":
            # Simple sentence splitting (can be enhanced)
            import re

            segments = re.split(r"([.?!])\s+", text)
            # Re-attach punctuation
            segments = [
                segments[i] + segments[i + 1] if i + 1 < len(segments) else segments[i]
                for i in range(0, len(segments), 2)
            ]
        else:
            raise ValueError(f"Unknown split_mode: {self.split_mode}")

        # Calculate token length for each segment
        token_lengths = []
        for i, segment in enumerate(segments):
            # Add leading space for words after the first (if needed)
            if i > 0 and self.add_space and self.split_mode == "word":
                segment_with_space = " " + segment
            else:
                segment_with_space = segment

            tokens = self._tokenize(segment_with_space)
            token_lengths.append(len(tokens))

        return segments, token_lengths

    def _calculate_segment_lengths_with_template(
        self, source_segments: List[str], include_system: bool
    ) -> List[int]:
        """
        Calculate segment lengths by tokenizing progressively and finding boundaries.

        This matches StreamingLLM's approach where we tokenize cumulative text
        to find exact token boundaries.
        """
        segment_lengths = []

        # Build full text components
        if include_system and self.system_prompt:
            system_text = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
        else:
            system_text = ""

        user_prefix = self.user_template.split("{content}")[0]  # "<|im_start|>user\n"
        user_suffix = self.user_template.split("{content}")[1]  # "<|im_end|>\n"

        # Track cumulative tokenization
        cumulative_text = system_text + user_prefix
        prev_token_count = len(self._tokenize(cumulative_text))

        # First segment is the template overhead
        segment_lengths.append(prev_token_count)

        # Tokenize progressively to find exact boundaries
        for i, segment in enumerate(source_segments):
            # Build text with proper spacing
            if i > 0 and self.split_mode == "word":
                cumulative_text += " " + segment
            else:
                cumulative_text += segment

            # Tokenize and find difference
            current_token_count = len(self._tokenize(cumulative_text))
            segment_token_len = current_token_count - prev_token_count

            segment_lengths.append(segment_token_len)
            prev_token_count = current_token_count

        # Add end token
        cumulative_text += user_suffix
        final_token_count = len(self._tokenize(cumulative_text))
        end_token_len = final_token_count - prev_token_count
        segment_lengths.append(end_token_len)

        return segment_lengths

    def get_assistant_start_tokens(self) -> mx.array:
        """
        Get token IDs for assistant message start.

        Returns:
            MLX array of assistant start token IDs
        """
        return mx.array(self.assistant_tokens)

    def prepare_metadata(
        self, source_seg_len: List[int], wait_k: int, pe_cache_length: int = 0
    ) -> Dict:
        """
        Prepare metadata needed for streaming generation.

        Args:
            source_seg_len: List of token lengths for source segments
            wait_k: Wait-k parameter
            pe_cache_length: Starting position ID for target tokens

        Returns:
            Dictionary with metadata:
            - source_seg_len: Source segment lengths
            - wait_k: Wait-k parameter
            - pe_cache_length: Position encoding offset
            - split_mode: Segmentation mode
            - assistant_tokens: Assistant start tokens
            - end_token: End token
        """
        return {
            "source_seg_len": source_seg_len,
            "wait_k": wait_k,
            "pe_cache_length": pe_cache_length,
            "split_mode": self.split_mode,
            "assistant_tokens": self.assistant_tokens,
            "end_token": self.end_token,
        }


def prepare_streaming_input(
    source_text: str,
    tokenizer,
    wait_k: int = 3,
    system_prompt: str = "",
    split_mode: str = "word",
    add_space: bool = False,
    pe_cache_length: int = 0,
) -> Dict:
    """
    Convenient function to prepare all streaming inputs.

    Args:
        source_text: Source text to process
        tokenizer: Tokenizer instance
        wait_k: Wait-k parameter
        system_prompt: System prompt
        split_mode: How to split text ('word' or 'sentence')
        add_space: Whether to add leading spaces
        pe_cache_length: Position encoding offset for target

    Returns:
        Dictionary with all prepared data:
        - source_token_ids: mx.array of source token IDs
        - source_seg_len: List of segment lengths
        - assistant_start_tokens: mx.array of assistant start tokens
        - metadata: Additional metadata dict
    """
    preparator = StreamingDataPreparator(
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        split_mode=split_mode,
        add_space=add_space,
    )

    # Prepare source
    formatted_text, token_ids, seg_lengths = preparator.prepare_source_text(
        source_text, include_system=bool(system_prompt)
    )

    # Get assistant tokens
    assistant_tokens = preparator.get_assistant_start_tokens()

    # Prepare metadata
    metadata = preparator.prepare_metadata(
        source_seg_len=seg_lengths, wait_k=wait_k, pe_cache_length=pe_cache_length
    )

    return {
        "source_token_ids": mx.array(token_ids).reshape(1, -1),  # Add batch dimension
        "source_seg_len": seg_lengths,
        "assistant_start_tokens": assistant_tokens.reshape(1, -1),
        "metadata": metadata,
        "formatted_source_text": formatted_text,
    }
