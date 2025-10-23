#!/usr/bin/env python3
"""
StreamingLLM Visualization Example

This example demonstrates the streaming input/output visualization pattern
from StreamingLLM's evaluate/streaming_eval.py, showing both source and
target text on separate lines as streaming progresses.
"""

import sys

from mlx_lm import load_streaming, stream_generate_streaming_llm


def streaming_visualization_demo(model, tokenizer, source_text, wait_k=3):
    """
    Visualize streaming generation showing both input and output.

    Matches the pattern from StreamingLLM's evaluate/streaming_eval.py:
    - Print streaming-input on one line
    - Print streaming-output on another line
    - Update as generation progresses

    Args:
        model: The language model
        tokenizer: The tokenizer
        source_text: Source text to process
        wait_k: Wait-k policy parameter
    """
    print("=" * 80)
    print(f"STREAMING VISUALIZATION (wait-k={wait_k})")
    print("=" * 80)

    # Print the streaming input
    print("streaming-input:")
    print(source_text)
    print()

    # Print streaming output header
    print("streaming-output:")

    # Track number of words for final stats
    word_count = 0

    # Stream generation
    for response in stream_generate_streaming_llm(
        model=model,
        tokenizer=tokenizer,
        prompt=source_text,
        wait_k=wait_k,
        max_new_words=100,
        temp=0.7,  # Greedy for consistency
    ):
        # Only print when a complete word is generated
        if hasattr(response, "word_complete") and response.word_complete:
            # Print just the new word, not the entire accumulated text
            print(response.text, end=" ", flush=True)
            word_count += 1

    # Final newline after generation completes
    print()
    print()
    print("=" * 80)

    if hasattr(response, "generation_tps"):
        print(f"Generation speed: {response.generation_tps:.2f} tokens/sec")
        print(f"Words generated: {word_count}")
        print(f"Peak memory: {response.peak_memory:.2f} GB")


def streaming_with_progress_demo(model, tokenizer, source_text, wait_k=3):
    """
    Enhanced visualization showing streaming progress with word counts.

    Shows:
    - streaming-input with word count
    - streaming-output with live updates
    - Progress indicators
    """
    print("=" * 80)
    print(f"STREAMING WITH PROGRESS (wait-k={wait_k})")
    print("=" * 80)

    # Count source words
    source_words = source_text.split()
    source_word_count = len(source_words)

    print(f"streaming-input: ({source_word_count} words)")
    print(source_text)
    print()

    print("streaming-output:")

    word_count = 0
    max_source_read = 0

    for response in stream_generate_streaming_llm(
        model=model,
        tokenizer=tokenizer,
        prompt=source_text,
        wait_k=wait_k,
        max_new_words=100,
        temp=0.0,
    ):
        if hasattr(response, "word_complete") and response.word_complete:
            word_count += 1
            max_source_read = max(max_source_read, response.source_words_read)

            # Print just the new word
            print(response.text, end=" ", flush=True)

    print()
    print()
    print("=" * 80)


def side_by_side_visualization(model, tokenizer, source_text, wait_k=3):
    """
    Side-by-side visualization of source and target with word alignment.

    Shows which source words have been read as each target word is generated.
    """
    print("=" * 80)
    print(f"SIDE-BY-SIDE VISUALIZATION (wait-k={wait_k})")
    print("=" * 80)

    source_words = source_text.split()

    print("Source words:")
    for i, word in enumerate(source_words, 1):
        print(f"  {i:2d}. {word}")
    print()

    print("Streaming generation:")
    print(f"{'Target Word':<20} {'Source Read':<15} {'Lag':<10}")
    print("-" * 50)

    output_text = ""

    for response in stream_generate_streaming_llm(
        model=model,
        tokenizer=tokenizer,
        prompt=source_text,
        wait_k=wait_k,
        max_new_words=100,
        temp=0.0,
    ):
        if hasattr(response, "word_complete") and response.word_complete:
            word = response.text
            source_read = response.source_words_read
            target_gen = response.target_words_generated
            lag = source_read - target_gen

            output_text += word + " "
            print(f"{word:<20} {source_read:>2d}/{len(source_words):<10} lag={lag}")

    print()
    print("Final output:")
    print(output_text.strip())
    print("=" * 80)


def batch_comparison(model, tokenizer, examples, wait_k=3):
    """
    Compare multiple examples showing input/output pairs.

    Similar to StreamingLLM's evaluation format.
    """
    print("=" * 80)
    print(f"BATCH STREAMING EVALUATION (wait-k={wait_k})")
    print("=" * 80)
    print()

    for i, source_text in enumerate(examples, 1):
        print(f"Example {i}/{len(examples)}")
        print("-" * 80)

        # Print streaming input
        print("streaming-input:")
        print(source_text)
        print()

        # Print streaming output
        print("streaming-output:")

        for response in stream_generate_streaming_llm(
            model=model,
            tokenizer=tokenizer,
            prompt=source_text,
            wait_k=wait_k,
            max_new_words=50,
            temp=0.0,
        ):
            if hasattr(response, "word_complete") and response.word_complete:
                print(response.text, end=" ", flush=True)

        print()
        print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="StreamingLLM Visualization Examples")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model path",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["simple", "progress", "side-by-side", "batch", "all"],
        help="Visualization mode",
    )
    parser.add_argument(
        "--wait-k",
        type=int,
        default=3,
        help="Wait-k value",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Translate to French: The quick brown fox jumps over the lazy dog.",
        help="Input prompt",
    )

    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer = load_streaming(args.model)
    print("Model loaded successfully!\n")

    if args.mode == "simple" or args.mode == "all":
        streaming_visualization_demo(model, tokenizer, args.prompt, wait_k=args.wait_k)
        print()

    if args.mode == "progress" or args.mode == "all":
        streaming_with_progress_demo(model, tokenizer, args.prompt, wait_k=args.wait_k)
        print()

    if args.mode == "side-by-side" or args.mode == "all":
        side_by_side_visualization(model, tokenizer, args.prompt, wait_k=args.wait_k)
        print()

    if args.mode == "batch" or args.mode == "all":
        examples = [
            "Translate to Spanish: Good morning.",
            "Translate to German: How are you?",
            "Translate to French: Thank you very much.",
        ]
        batch_comparison(model, tokenizer, examples, wait_k=args.wait_k)


if __name__ == "__main__":
    main()
