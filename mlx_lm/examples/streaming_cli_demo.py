#!/usr/bin/env python3
"""
CLI-based StreamingLLM Demo

This example shows how to use streaming generation from the command line
with different wait-k values and demonstrates the effect of the policy.
"""

import argparse
from mlx_lm import load, stream_generate_streaming_llm


def demo_wait_k_effect(model, tokenizer, prompt, wait_k_values):
    """Demonstrate how different wait-k values affect streaming."""
    print("\n" + "=" * 80)
    print("COMPARING DIFFERENT WAIT-K VALUES")
    print("=" * 80)

    for wait_k in wait_k_values:
        print(f"\n{'─' * 80}")
        print(f"wait-k = {wait_k}")
        print(f"{'─' * 80}")

        words = []
        source_progress = []

        for response in stream_generate_streaming_llm(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            wait_k=wait_k,
            max_new_words=15,
            temp=0.0,  # Greedy for consistency
        ):
            if hasattr(response, "word_complete") and response.word_complete:
                words.append(response.text)
                source_progress.append(response.source_words_read)

        # Show the streaming pattern
        print(f"Generated: {' '.join(words)}")
        print(f"Source words read when each target word was generated:")
        for i, (word, src_read) in enumerate(zip(words, source_progress), 1):
            print(f"  Word {i:2d} ('{word:12s}'): {src_read} source words")


def main():
    parser = argparse.ArgumentParser(description="StreamingLLM CLI Demo")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model path or HuggingFace model ID",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Translate to Spanish: The quick brown fox jumps over the lazy dog.",
        help="Input prompt",
    )
    parser.add_argument(
        "--wait-k",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="Wait-k values to compare (default: 1 3 5)",
    )

    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

    print(f"\nPrompt: {args.prompt}")

    demo_wait_k_effect(model, tokenizer, args.prompt, args.wait_k)


if __name__ == "__main__":
    main()
