"""
Tests for Phase 4: CLI Integration & Examples

This test module covers:
1. stream_generate_streaming_llm API wrapper
2. CLI argument parsing for streaming flags
3. Public API exports
4. Integration tests
"""

import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

import mlx.core as mx
import mlx.nn as nn


class TestStreamGenerateStreamingLLMWrapper(unittest.TestCase):
    """Test the stream_generate_streaming_llm wrapper function."""

    def setUp(self):
        """Set up test fixtures."""
        from mlx_lm.generate import stream_generate_streaming_llm
        from mlx_lm.tokenizer_utils import TokenizerWrapper

        self.stream_generate_streaming_llm = stream_generate_streaming_llm

        # Create mock model
        self.mock_model = MagicMock(spec=nn.Module)

        # Create mock tokenizer
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        self.mock_tokenizer._tokenizer = self.mock_tokenizer

    @patch("mlx_lm.streaming_generate.stream_generate_streaming")
    @patch("mlx_lm.streaming_data_utils.StreamingDataPreparator")
    def test_wrapper_with_string_prompt(
        self, mock_preparator_class, mock_stream_generate
    ):
        """Test wrapper with string prompt."""
        # Setup mocks
        mock_preparator = MagicMock()
        mock_preparator.prepare_source_text.return_value = (
            "formatted",
            [1, 2, 3],
            [1, 1, 1],
        )
        mock_preparator_class.return_value = mock_preparator

        # Mock streaming output
        mock_stream_generate.return_value = [
            {
                "text": "hello",
                "token": 10,
                "word_complete": True,
                "source_words_read": 1,
                "target_words_generated": 1,
            }
        ]

        # Call wrapper
        prompt = "Test prompt"
        responses = list(
            self.stream_generate_streaming_llm(
                model=self.mock_model,
                tokenizer=self.mock_tokenizer,
                prompt=prompt,
                wait_k=3,
            )
        )

        # Verify preparator was called
        mock_preparator.prepare_source_text.assert_called_once_with(prompt)

        # Verify streaming function was called
        mock_stream_generate.assert_called_once()
        call_kwargs = mock_stream_generate.call_args[1]
        self.assertEqual(call_kwargs["wait_k"], 3)
        self.assertEqual(call_kwargs["source_token_ids"], [1, 2, 3])
        self.assertEqual(call_kwargs["source_seg_len"], [1, 1, 1])

        # Verify responses (includes final response with finish_reason)
        self.assertEqual(len(responses), 2)  # 1 word + 1 final
        self.assertTrue(hasattr(responses[0], "word_complete"))
        self.assertEqual(responses[0].word_complete, True)
        # Final response has finish_reason
        self.assertEqual(responses[1].finish_reason, "stop")

    @patch("mlx_lm.streaming_generate.stream_generate_streaming")
    @patch("mlx_lm.streaming_data_utils.StreamingDataPreparator")
    def test_wrapper_with_token_list_prompt(
        self, mock_preparator_class, mock_stream_generate
    ):
        """Test wrapper with list of token IDs."""
        # Setup mocks
        mock_preparator = MagicMock()
        mock_preparator_class.return_value = mock_preparator

        mock_stream_generate.return_value = [
            {
                "text": "word",
                "token": 5,
                "word_complete": True,
                "source_words_read": 1,
                "target_words_generated": 1,
            }
        ]

        # Call with token list
        prompt = [1, 2, 3, 4, 5]
        responses = list(
            self.stream_generate_streaming_llm(
                model=self.mock_model,
                tokenizer=self.mock_tokenizer,
                prompt=prompt,
                wait_k=2,
            )
        )

        # Verify preparator was NOT called for source text
        mock_preparator.prepare_source_text.assert_not_called()

        # Verify streaming function received token IDs
        call_kwargs = mock_stream_generate.call_args[1]
        self.assertEqual(call_kwargs["source_token_ids"], prompt)
        # Each token treated as a word
        self.assertEqual(call_kwargs["source_seg_len"], [1, 1, 1, 1, 1])

    @patch("mlx_lm.streaming_generate.stream_generate_streaming")
    @patch("mlx_lm.streaming_data_utils.StreamingDataPreparator")
    def test_wrapper_with_mx_array_prompt(
        self, mock_preparator_class, mock_stream_generate
    ):
        """Test wrapper with mx.array prompt."""
        mock_preparator = MagicMock()
        mock_preparator_class.return_value = mock_preparator

        mock_stream_generate.return_value = []

        # Call with mx.array
        prompt = mx.array([10, 20, 30])
        list(
            self.stream_generate_streaming_llm(
                model=self.mock_model,
                tokenizer=self.mock_tokenizer,
                prompt=prompt,
            )
        )

        # Verify conversion to list
        call_kwargs = mock_stream_generate.call_args[1]
        self.assertEqual(call_kwargs["source_token_ids"], [10, 20, 30])

    @patch("mlx_lm.streaming_generate.stream_generate_streaming")
    @patch("mlx_lm.streaming_data_utils.StreamingDataPreparator")
    def test_wrapper_parameters_passed_through(
        self, mock_preparator_class, mock_stream_generate
    ):
        """Test that all parameters are passed to underlying function."""
        mock_preparator = MagicMock()
        mock_preparator.prepare_source_text.return_value = ("", [1], [1])
        mock_preparator_class.return_value = mock_preparator

        mock_stream_generate.return_value = []

        # Call with all parameters
        list(
            self.stream_generate_streaming_llm(
                model=self.mock_model,
                tokenizer=self.mock_tokenizer,
                prompt="test",
                wait_k=5,
                max_new_words=100,
                max_tokens_per_word=30,
                temp=0.8,
                top_p=0.95,
                min_p=0.05,
                repetition_penalty=1.2,
                repetition_context_size=40,
            )
        )

        call_kwargs = mock_stream_generate.call_args[1]
        self.assertEqual(call_kwargs["wait_k"], 5)
        self.assertEqual(call_kwargs["max_new_words"], 100)
        self.assertEqual(call_kwargs["max_tokens_per_word"], 30)
        self.assertEqual(call_kwargs["temp"], 0.8)
        self.assertEqual(call_kwargs["top_p"], 0.95)
        self.assertEqual(call_kwargs["repetition_penalty"], 1.2)
        self.assertEqual(call_kwargs["repetition_context_size"], 40)


class TestCLIArgumentParsing(unittest.TestCase):
    """Test CLI argument parsing for streaming flags."""

    def test_streaming_flags_added_to_parser(self):
        """Test that streaming flags are added to argument parser."""
        from mlx_lm.generate import setup_arg_parser

        parser = setup_arg_parser()

        # Parse with streaming flags
        args = parser.parse_args(
            [
                "--streaming",
                "--wait-k",
                "5",
                "--max-new-words",
                "100",
                "--max-tokens-per-word",
                "30",
            ]
        )

        self.assertTrue(args.streaming)
        self.assertEqual(args.wait_k, 5)
        self.assertEqual(args.max_new_words, 100)
        self.assertEqual(args.max_tokens_per_word, 30)

    def test_streaming_default_values(self):
        """Test default values for streaming flags."""
        from mlx_lm.generate import setup_arg_parser

        parser = setup_arg_parser()
        args = parser.parse_args([])

        self.assertFalse(args.streaming)
        self.assertEqual(args.wait_k, 3)
        self.assertIsNone(args.max_new_words)
        self.assertEqual(args.max_tokens_per_word, 50)

    def test_streaming_flag_boolean(self):
        """Test --streaming flag is boolean."""
        from mlx_lm.generate import setup_arg_parser

        parser = setup_arg_parser()

        # Without flag
        args = parser.parse_args([])
        self.assertFalse(args.streaming)

        # With flag
        args = parser.parse_args(["--streaming"])
        self.assertTrue(args.streaming)


class TestPublicAPIExports(unittest.TestCase):
    """Test that streaming functions are exported in public API."""

    def test_stream_generate_streaming_llm_exported(self):
        """Test stream_generate_streaming_llm is exported from mlx_lm."""
        import mlx_lm

        self.assertTrue(hasattr(mlx_lm, "stream_generate_streaming_llm"))
        self.assertTrue(callable(mlx_lm.stream_generate_streaming_llm))

    def test_existing_exports_still_available(self):
        """Test that existing exports are not broken."""
        import mlx_lm

        # Check existing functions
        self.assertTrue(hasattr(mlx_lm, "generate"))
        self.assertTrue(hasattr(mlx_lm, "stream_generate"))
        self.assertTrue(hasattr(mlx_lm, "batch_generate"))
        self.assertTrue(hasattr(mlx_lm, "load"))
        self.assertTrue(hasattr(mlx_lm, "convert"))


class TestIntegration(unittest.TestCase):
    """Integration tests for Phase 4."""

    @patch("mlx_lm.streaming_generate.stream_generate_streaming")
    @patch("mlx_lm.streaming_data_utils.StreamingDataPreparator")
    @patch("mlx_lm.tokenizer_utils.TokenizerWrapper")
    def test_end_to_end_string_to_response(
        self, mock_tokenizer_wrapper, mock_preparator_class, mock_stream_generate
    ):
        """Test end-to-end flow from string prompt to GenerationResponse."""
        from mlx_lm.generate import stream_generate_streaming_llm

        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer_wrapper.return_value = mock_tokenizer

        mock_preparator = MagicMock()
        mock_preparator.prepare_source_text.return_value = (
            "formatted_text",
            [1, 2, 3, 4, 5],
            [2, 3],  # 2 words
        )
        mock_preparator_class.return_value = mock_preparator

        # Simulate streaming output
        mock_stream_generate.return_value = [
            {
                "text": "Bonjour",
                "token": 100,
                "word_complete": True,
                "source_words_read": 1,
                "target_words_generated": 1,
            },
            {
                "text": ",",
                "token": 101,
                "word_complete": True,
                "source_words_read": 2,
                "target_words_generated": 2,
            },
        ]

        # Create mock model
        mock_model = MagicMock()

        # Generate
        responses = list(
            stream_generate_streaming_llm(
                model=mock_model,
                tokenizer=mock_tokenizer,
                prompt="Translate to French: Hello",
                wait_k=3,
                max_new_words=10,
            )
        )

        # Verify responses (2 words + 1 final = 3 total)
        self.assertEqual(len(responses), 3)

        # First response
        self.assertEqual(responses[0].text, "Bonjour")
        self.assertEqual(responses[0].token, 100)
        self.assertTrue(responses[0].word_complete)
        self.assertEqual(responses[0].source_words_read, 1)
        self.assertEqual(responses[0].target_words_generated, 1)

        # Second response
        self.assertEqual(responses[1].text, ",")
        self.assertEqual(responses[1].token, 101)
        self.assertTrue(responses[1].word_complete)
        self.assertEqual(responses[1].source_words_read, 2)
        self.assertEqual(responses[1].target_words_generated, 2)

        # Third response (final with finish_reason)
        self.assertEqual(responses[2].finish_reason, "stop")

    def test_generation_response_attributes(self):
        """Test that GenerationResponse has correct attributes."""
        from mlx_lm.generate import GenerationResponse

        # Create a response
        response = GenerationResponse(
            text="hello",
            token=10,
            logprobs=mx.array([0.0]),
            from_draft=False,
            prompt_tokens=5,
            prompt_tps=100.0,
            generation_tokens=1,
            generation_tps=50.0,
            peak_memory=1.5,
            finish_reason=None,
        )

        # Test standard attributes
        self.assertEqual(response.text, "hello")
        self.assertEqual(response.token, 10)
        self.assertEqual(response.prompt_tokens, 5)
        self.assertEqual(response.generation_tokens, 1)

        # Test that we can add streaming attributes
        response.word_complete = True
        response.source_words_read = 3
        response.target_words_generated = 1

        self.assertTrue(response.word_complete)
        self.assertEqual(response.source_words_read, 3)
        self.assertEqual(response.target_words_generated, 1)


class TestExampleScripts(unittest.TestCase):
    """Smoke tests for example scripts."""

    def test_streaming_basic_imports(self):
        """Test that streaming_basic.py can be imported."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "streaming_basic",
            "mlx_lm/examples/streaming_basic.py",
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            # Just check it can be loaded
            self.assertIsNotNone(module)

    def test_streaming_cli_demo_imports(self):
        """Test that streaming_cli_demo.py can be imported."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "streaming_cli_demo",
            "mlx_lm/examples/streaming_cli_demo.py",
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            self.assertIsNotNone(module)

    def test_streaming_realtime_imports(self):
        """Test that streaming_realtime.py can be imported."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "streaming_realtime",
            "mlx_lm/examples/streaming_realtime.py",
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            self.assertIsNotNone(module)

    def test_streaming_visualization_imports(self):
        """Test that streaming_visualization.py can be imported."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "streaming_visualization",
            "mlx_lm/examples/streaming_visualization.py",
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            self.assertIsNotNone(module)


if __name__ == "__main__":
    unittest.main()
