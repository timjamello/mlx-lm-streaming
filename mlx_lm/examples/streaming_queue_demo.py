#!/usr/bin/env python3
"""
Queue-Based Streaming Demo

This example demonstrates the queue-based streaming API where source text
is fed incrementally from a queue, simulating real-time scenarios like
live STT (speech-to-text) to translation.
"""

import os
import sys
import time
from queue import Queue
from threading import Thread

from mlx_lm import load, stream_generate


def basic_queue_demo():
    """Basic demonstration of queue-based streaming with live visualization."""
    print("=" * 80)
    print("QUEUE-BASED STREAMING DEMO")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    model_path = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    model, tokenizer = load(model_path)
    print("Model loaded!")

    # Create source queue
    source_queue = Queue()

    # Example source text chunks (simulating STT output)
    source_texts = [
        "Hello, ",
        "how are ",
        "you doing ",
        "today? ",
        "I hope ",
        "you are ",
        "well. ",
        "It's been ",
        "a while since ",
        "we last spoke, ",
        "and I wanted ",
        "to check in ",
        "and see how ",
        "things are going ",
        "for you. ",
        "If you have ",
        "a few minutes, ",
        "I'd love to ",
        "catch up and ",
        "hear about ",
        "any recent ",
        "updates or ",
        "changes in your ",
        "life. ",
        "Take care ",
        "and stay safe.",
    ]

    # Shared list to track what's been fed (for display)
    fed_texts = []

    def stt_feed_with_tracking():
        for text in source_texts:
            time.sleep(0.2)
            source_queue.put(text)
            fed_texts.append(text)
        source_queue.put(None)

    # Start STT simulator in background thread
    stt_thread = Thread(target=stt_feed_with_tracking)
    stt_thread.start()

    # Clear screen and set up display
    if os.name == "nt":  # Windows
        os.system("cls")
    else:  # Unix/Linux/Mac
        sys.stdout.write("\033[2J\033[H")  # Clear screen and move to top

    # Print header
    print("=" * 80)
    print("QUEUE-BASED STREAMING DEMO (wait-k=7)")
    print("=" * 80)
    print()

    # Save cursor position
    sys.stdout.write("\033[s")

    # Reserve space for the display
    print("input-stream (incoming):")
    print()  # Space for input
    print()  # Separator
    print("translation-output:")
    print()  # Space for output
    print()  # Extra space

    # Move back to saved position
    sys.stdout.write("\033[u")

    # Track displayed content
    current_output_text = ""
    last_fed_count = 0

    # Stream generation
    for response in stream_generate(
        model=model,
        tokenizer=tokenizer,
        source_queue=source_queue,
        wait_k=7,
        system_prompt="Translate the following English text to French. Do nothing else. Do not add commentary.",
        temp=0.0,  # Greedy decoding
    ):
        needs_redraw = False

        # Display what's actually been fed to the queue
        if len(fed_texts) > last_fed_count:
            last_fed_count = len(fed_texts)
            needs_redraw = True

        # Track output as it's generated
        if hasattr(response, "word_complete") and response.word_complete:
            current_output_text += response.text
            needs_redraw = True

        if needs_redraw:
            # Restore cursor and redraw display
            sys.stdout.write("\033[u")
            sys.stdout.write("\033[J")

            # Build input text from what's been fed
            current_input_text = "".join(fed_texts)

            print("input-stream (incoming):")
            print(current_input_text if current_input_text else "")
            print()
            print("translation-output:")
            print(current_output_text if current_output_text else "")

            sys.stdout.flush()

    # Final summary
    print()
    print()
    print("=" * 80)
    print("✓ Translation complete!")
    print("=" * 80)

    # Wait for STT thread to finish
    stt_thread.join()


def main():
    try:
        basic_queue_demo()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
