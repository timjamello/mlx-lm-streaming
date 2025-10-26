#!/usr/bin/env python3
"""
Basic StreamingLLM Example

This example demonstrates basic streaming generation with the wait-k policy
using the queue-based API.
"""

from queue import Queue
from threading import Thread

from mlx_lm import load, stream_generate


def feed_source_text(queue: Queue, text: str):
    """
    Feed source text to the queue.

    For simplicity, we feed the entire text at once. In a real scenario,
    this would be coming from an STT pipeline word-by-word.
    """
    # Split text into words and feed them
    words = text.split()
    for word in words:
        queue.put(word + " ")

    # Signal end of stream
    queue.put(None)


def main():
    # Load a Qwen2.5 model (or any compatible streaming model)
    print("Loading model...")
    model_path = "Qwen/Qwen2.5-0.5B-Instruct"
    model, tokenizer = load(model_path)

    # Example source text
    source_text = "Hello, how are you doing today?"

    print("\n" + "=" * 60)
    print("STREAMING GENERATION (wait-k=3)")
    print("=" * 60)
    print(f"Source: {source_text}\n")

    # Create queue and feed text in background
    source_queue = Queue()
    feed_thread = Thread(target=feed_source_text, args=(source_queue, source_text))
    feed_thread.start()

    # Generate with streaming
    words = []
    for response in stream_generate(
        model=model,
        tokenizer=tokenizer,
        source_queue=source_queue,
        wait_k=3,  # Wait for 3 source words before generating
        max_new_words=20,  # Generate up to 20 words
        system_prompt="Translate the following English text to French",
        temp=0.7,  # Sampling temperature
    ):
        # Only print when a complete word is generated
        if hasattr(response, "word_complete") and response.word_complete:
            words.append(response.text)
            print(f"Word {len(words)}: '{response.text}'")
            print(f"  → Source words read: {response.source_words_read}")
            print(f"  → Target words generated: {response.target_words_generated}")
            print()

    print("=" * 60)
    print("FINAL OUTPUT:")
    print(" ".join(words))
    print("=" * 60)

    if hasattr(response, "generation_tps"):
        print(f"\nGeneration speed: {response.generation_tps:.2f} tokens/sec")
        print(f"Peak memory: {response.peak_memory:.2f} GB")

    # Wait for feed thread
    feed_thread.join()


if __name__ == "__main__":
    main()
