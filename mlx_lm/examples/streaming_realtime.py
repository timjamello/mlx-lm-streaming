#!/usr/bin/env python3
"""
Real-time StreamingLLM Example

This example demonstrates real-time streaming generation that simulates
simultaneous translation or live transcription scenarios using the queue-based API.
"""

import sys
import time
from queue import Queue
from threading import Thread

from mlx_lm import load, stream_generate


def feed_text_realtime(queue: Queue, source_text: str, delay: float = 0.5):
    """
    Feed text to queue word-by-word with delay to simulate real-time input.

    Args:
        queue: Queue to feed text into
        source_text: Source text to process
        delay: Delay between words (simulates real-time STT)
    """
    words = source_text.split()
    for word in words:
        time.sleep(delay)
        queue.put(word + " ")
    queue.put(None)


def realtime_streaming_demo(model, tokenizer, source_text, wait_k=3, delay=0.5):
    """
    Simulate real-time streaming where output is displayed as soon as
    words are generated.

    Args:
        model: The language model
        tokenizer: The tokenizer
        source_text: Source text to process
        wait_k: Wait-k policy parameter
        delay: Delay between word arrivals (seconds) to simulate real-time
    """
    print("\n" + "=" * 80)
    print("REAL-TIME STREAMING SIMULATION")
    print("=" * 80)
    print(f"Source: {source_text}")
    print(f"Wait-k: {wait_k}")
    print(f"\nOutput (streaming):")
    print("-" * 80)

    # Create queue and start feeding thread
    source_queue = Queue()
    feed_thread = Thread(
        target=feed_text_realtime, args=(source_queue, source_text, delay)
    )
    feed_thread.start()

    start_time = time.time()
    word_count = 0

    for response in stream_generate(
        model=model,
        tokenizer=tokenizer,
        source_queue=source_queue,
        wait_k=wait_k,
        max_new_words=50,
        system_prompt="Translate the following English text to French",
        temp=0.7,
    ):
        if hasattr(response, "word_complete") and response.word_complete:
            word_count += 1

            # Print the word in real-time
            print(f"{response.text}", end=" ", flush=True)

    elapsed = time.time() - start_time

    print("\n" + "-" * 80)
    print(f"\nGenerated {word_count} words in {elapsed:.2f} seconds")
    print(f"Average: {word_count / elapsed:.2f} words/sec")

    if hasattr(response, "generation_tps"):
        print(f"Token generation speed: {response.generation_tps:.2f} tokens/sec")
        print(f"Peak memory: {response.peak_memory:.2f} GB")

    feed_thread.join()


def interactive_mode(model, tokenizer):
    """
    Interactive mode where user can enter prompts and see streaming results.
    """
    print("\n" + "=" * 80)
    print("INTERACTIVE STREAMING MODE")
    print("=" * 80)
    print("Enter source text to see streaming generation in action.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            prompt = input("Source text> ").strip()

            if prompt.lower() in ["quit", "exit", "q"]:
                print("Exiting...")
                break

            if not prompt:
                continue

            # Ask for wait-k value
            wait_k_input = input("Wait-k (default=3)> ").strip()
            wait_k = int(wait_k_input) if wait_k_input else 3

            print(f"\nStreaming output (wait-k={wait_k}):")
            print("-" * 80)

            # Create queue and feed text
            source_queue = Queue()

            def feed_text():
                words = prompt.split()
                for word in words:
                    source_queue.put(word + " ")
                source_queue.put(None)

            feed_thread = Thread(target=feed_text)
            feed_thread.start()

            for response in stream_generate(
                model=model,
                tokenizer=tokenizer,
                source_queue=source_queue,
                wait_k=wait_k,
                max_new_words=30,
                system_prompt="Translate to French",
                temp=0.7,
            ):
                if hasattr(response, "word_complete") and response.word_complete:
                    print(f"{response.text}", end=" ", flush=True)

            feed_thread.join()
            print("\n" + "-" * 80 + "\n")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Exiting...")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Real-time StreamingLLM Demo")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model path",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--wait-k",
        type=int,
        default=3,
        help="Wait-k value (default: 3)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Delay between words in seconds (default: 0.3)",
    )

    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)
    print("Model loaded successfully!\n")

    if args.interactive:
        interactive_mode(model, tokenizer)
    else:
        # Demo with predefined examples
        examples = [
            "Translate to French: The weather is beautiful today.",
            "Translate to German: I love machine learning and artificial intelligence.",
            "Summarize this: Artificial intelligence has made tremendous progress in recent years.",
        ]

        for i, example in enumerate(examples, 1):
            print(f"\n\nExample {i}/{len(examples)}")
            realtime_streaming_demo(
                model, tokenizer, example, wait_k=args.wait_k, delay=args.delay
            )
            time.sleep(1)  # Pause between examples


if __name__ == "__main__":
    main()
