# Copyright © 2023-2024 Apple Inc.

import sys
from pathlib import Path
import unittest
from unittest.mock import Mock

import mlx.core as mx

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlx_lm.streaming_utils import (
    calculate_wait_words,
    generate_wait_words_list,
    create_streaming_attention_mask,
    StreamingState
)
from mlx_lm.streaming_stopping_criteria import (
    WordBoundaryStoppingCriteria,
    MaxLengthStoppingCriteria,
    EOSStoppingCriteria
)


class TestWaitKPolicy(unittest.TestCase):
    """Test wait-k policy calculations"""

    def test_calculate_wait_words_basic(self):
        """Test basic wait-k calculation"""
        source_seg_len = [10, 5, 3, 4, 2]  # 5 source words

        # First target word with wait_k=2
        result = calculate_wait_words(source_seg_len, target_word_idx=0, wait_k=2)
        self.assertEqual(result, 2)  # min(2+0, 5-1) = 2

        # Second target word
        result = calculate_wait_words(source_seg_len, target_word_idx=1, wait_k=2)
        self.assertEqual(result, 3)  # min(2+1, 5-1) = 3

        # Last target word
        result = calculate_wait_words(source_seg_len, target_word_idx=3, wait_k=2)
        self.assertEqual(result, 4)  # min(2+3, 5-1) = 4

    def test_wait_k_saturates(self):
        """Test that wait-k saturates at max source words"""
        source_seg_len = [10, 5, 3]  # 3 source words

        # Large target index should saturate
        result = calculate_wait_words(source_seg_len, target_word_idx=10, wait_k=2)
        self.assertEqual(result, 2)  # min(2+10, 3-1) = 2

    def test_generate_wait_words_list(self):
        """Test generating wait words for all target words"""
        source_seg_len = [10, 5, 3, 4]
        target_seg_len = [4, 3, 5]  # 3 target words

        result = generate_wait_words_list(source_seg_len, target_seg_len, wait_k=2)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], 2)  # First target: wait for 2 source words
        self.assertEqual(result[1], 3)  # Second target: wait for 3
        self.assertEqual(result[2], 4)  # Third target: wait for 4


class TestStreamingAttentionMask(unittest.TestCase):
    """Test streaming attention mask generation"""

    def test_mask_basic_structure(self):
        """Test that mask has correct shape and basic causal structure"""
        total_len = 20
        source_seg_len = [2, 3, 2]  # 7 source tokens
        target_seg_len = [2, 3]  # 5 target tokens
        wait_tokens_list = [1, 2]  # Wait lists

        mask = create_streaming_attention_mask(
            total_len, source_seg_len, target_seg_len, wait_tokens_list
        )

        self.assertEqual(mask.shape, (total_len, total_len))

    def test_mask_prevents_future_attention(self):
        """Test that mask prevents attending to future tokens"""
        total_len = 15
        source_seg_len = [3, 3, 3]  # 9 source tokens
        target_seg_len = [2, 2]  # 4 target tokens
        wait_tokens_list = [1, 2]

        mask = create_streaming_attention_mask(
            total_len, source_seg_len, target_seg_len, wait_tokens_list
        )

        # Upper triangle should be masked (-inf)
        inf_value = -3e38
        # Check a few positions in upper triangle
        self.assertLess(float(mask[0, 5]), inf_value / 2)  # Future position
        self.assertLess(float(mask[1, 10]), inf_value / 2)  # Future position


class TestStreamingState(unittest.TestCase):
    """Test StreamingState tracking"""

    def setUp(self):
        self.source_seg_len = [10, 5, 3, 4, 2]
        self.wait_k = 2
        self.state = StreamingState(
            source_seg_len=self.source_seg_len,
            wait_k=self.wait_k,
            max_target_words=5
        )

    def test_initialization(self):
        """Test state initializes correctly"""
        self.assertEqual(self.state.source_words_read, 0)
        self.assertEqual(self.state.target_words_generated, 0)
        self.assertTrue(self.state.is_reading)
        self.assertFalse(self.state.finished)

    def test_should_read_next_source(self):
        """Test should_read logic"""
        self.assertTrue(self.state.should_read_next_source())

        # After reading all source
        self.state.source_words_read = len(self.source_seg_len)
        self.assertFalse(self.state.should_read_next_source())

    def test_should_write_next_target(self):
        """Test should_write logic"""
        self.assertTrue(self.state.should_write_next_target())

        # After reaching max
        self.state.target_words_generated = 5
        self.assertFalse(self.state.should_write_next_target())

    def test_get_source_tokens_to_read(self):
        """Test getting tokens for next source word"""
        tokens = self.state.get_source_tokens_to_read()
        self.assertEqual(tokens, 10)  # First word

        self.state.mark_source_read()
        tokens = self.state.get_source_tokens_to_read()
        self.assertEqual(tokens, 5)  # Second word

    def test_mode_switching(self):
        """Test switching between read and write modes"""
        self.assertTrue(self.state.is_reading)

        self.state.switch_to_writing()
        self.assertFalse(self.state.is_reading)

        self.state.switch_to_reading()
        self.assertTrue(self.state.is_reading)

    def test_check_wait_k_policy(self):
        """Test wait-k policy checking"""
        # Initially, need to wait for wait_k words
        self.state.source_words_read = 1
        self.assertFalse(self.state.check_wait_k_policy())

        self.state.source_words_read = 2
        self.assertTrue(self.state.check_wait_k_policy())

    def test_lagging_tracking(self):
        """Test that lagging is tracked correctly"""
        self.state.source_words_read = 2
        self.state.mark_target_written()

        self.assertEqual(len(self.state.wait_lagging), 1)
        self.assertEqual(self.state.wait_lagging[0], 2)


class TestStoppingCriteria(unittest.TestCase):
    """Test stopping criteria"""

    def setUp(self):
        # Create a mock tokenizer
        self.tokenizer = Mock()

    def test_word_boundary_on_space(self):
        """Test stopping on word boundary (space)"""
        # Mock tokenizer to return text with space in middle (indicating word boundary)
        # First call (full_text): returns "hello world" (has space after first char)
        # Second call (last_token): returns "world"
        call_count = [0]
        def decode_mock(ids):
            call_count[0] += 1
            if call_count[0] == 1:  # First call for full text
                return "hello world"  # Has space in position > 1
            else:  # Second call for last token
                return "world"

        self.tokenizer.decode = Mock(side_effect=decode_mock)

        criteria = WordBoundaryStoppingCriteria(
            tokenizer=self.tokenizer,
            max_tokens_per_word=50
        )

        # Generate tokens that decode to text with space
        tokens = mx.array([1, 2, 3])
        should_stop, remove_last = criteria(tokens, token_count=3)

        self.assertTrue(should_stop)
        self.assertTrue(remove_last)  # Space detected, remove last token

    def test_stopping_on_punctuation(self):
        """Test stopping on punctuation"""
        self.tokenizer.decode = Mock(side_effect=lambda ids: "." if len(ids) == 1 else "hello.")

        criteria = WordBoundaryStoppingCriteria(
            tokenizer=self.tokenizer,
            max_tokens_per_word=50
        )

        tokens = mx.array([1])
        should_stop, remove_last = criteria(tokens, token_count=1)

        self.assertTrue(should_stop)
        self.assertFalse(remove_last)  # Punctuation, don't remove

    def test_stopping_on_max_tokens(self):
        """Test stopping when max tokens reached"""
        self.tokenizer.decode = Mock(return_value="word")

        criteria = WordBoundaryStoppingCriteria(
            tokenizer=self.tokenizer,
            max_tokens_per_word=3
        )

        tokens = mx.array([1, 2, 3])
        should_stop, remove_last = criteria(tokens, token_count=3)

        self.assertTrue(should_stop)

    def test_max_length_criteria(self):
        """Test max length stopping criteria"""
        criteria = MaxLengthStoppingCriteria(max_length=100)

        self.assertFalse(criteria(50))
        self.assertFalse(criteria(99))
        self.assertTrue(criteria(100))
        self.assertTrue(criteria(101))

    def test_eos_criteria(self):
        """Test EOS stopping criteria"""
        eos_token_id = 2
        criteria = EOSStoppingCriteria(eos_token_id=eos_token_id)

        self.assertFalse(criteria(1))
        self.assertTrue(criteria(2))
        self.assertFalse(criteria(3))


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for streaming scenarios"""

    def test_simple_streaming_workflow(self):
        """Test a simple streaming workflow"""
        # Setup
        source_seg_len = [5, 3, 4]  # 3 source words
        wait_k = 2
        state = StreamingState(source_seg_len, wait_k, max_target_words=2)

        # === PHASE 1: Read first source word ===
        self.assertTrue(state.should_read_next_source())
        tokens = state.get_source_tokens_to_read()
        self.assertEqual(tokens, 5)
        state.mark_source_read()

        # Can't write yet (need wait_k=2 words)
        self.assertFalse(state.check_wait_k_policy())

        # === PHASE 2: Read second source word ===
        tokens = state.get_source_tokens_to_read()
        self.assertEqual(tokens, 3)
        state.mark_source_read()

        # Now can write
        self.assertTrue(state.check_wait_k_policy())
        state.switch_to_writing()

        # === PHASE 3: Write first target word ===
        state.mark_target_written()
        self.assertEqual(state.target_words_generated, 1)

        # === PHASE 4: Can read third source ===
        state.switch_to_reading()
        self.assertTrue(state.should_read_next_source())
        state.mark_source_read()

        # === PHASE 5: Write second target word ===
        state.switch_to_writing()
        state.mark_target_written()

        # Reached max target words
        self.assertFalse(state.should_write_next_target())

    def test_wait_k_policy_enforcement(self):
        """Test that wait-k policy is enforced correctly"""
        source_seg_len = [5, 3, 4, 2, 6]  # 5 source words
        wait_k = 3
        state = StreamingState(source_seg_len, wait_k)

        # For first target word, need wait_k words
        for i in range(wait_k):
            self.assertFalse(state.check_wait_k_policy())
            state.mark_source_read()

        self.assertTrue(state.check_wait_k_policy())

        # Generate first target
        state.mark_target_written()

        # For second target word, need wait_k + 1 words
        self.assertFalse(state.check_wait_k_policy())
        state.mark_source_read()  # Read 4th word
        self.assertTrue(state.check_wait_k_policy())


if __name__ == "__main__":
    unittest.main()
