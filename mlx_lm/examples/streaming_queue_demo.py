#!/usr/bin/env python3
"""
Queue-Based Streaming Demo - Real-Time AI Assistant

This example demonstrates the queue-based streaming API where the AI assistant
listens to a continuous stream of speech-to-text input and responds in real-time,
adapting to sudden changes in conversation direction.
"""

import os
import sys
import time
from queue import Queue
from threading import Thread

from mlx_lm import load, stream_generate


def basic_queue_demo():
    """Real-time AI assistant demo with live visualization."""
    print("=" * 80)
    print("REAL-TIME AI ASSISTANT DEMO")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    model_path = "mlx-community/Qwen3-VL-32B-Instruct-8bit"
    model, tokenizer = load(model_path)
    print("Model loaded!")

    # Create source queue
    source_queue = Queue()

    # Realistic STT stream: proper utterances with variable delays
    # Format: (text, delay_before_next_in_seconds)
    # Longer delays allow assistant to actually start responding before interruptions
    source_utterances = [
        ("Hey assistant,", 0.6),
        ("I wanted to tell you about", 0.7),
        ("this amazing thing that happened to me yesterday.", 1.2),
        ("I was walking through the park", 0.9),
        (
            "and I saw this really beautiful bird.",
            5.0,
        ),  # Long pause - let assistant start responding
        ("It had these bright blue feathers and...", 4.0),  # Thinking pause
        ("wait, actually,", 0.2),  # Quick interruption mid-thought
        ("nevermind that.", 0.5),  # Fast follow-up
        ("Can you help me figure out", 0.8),
        (
            "what to make for dinner tonight?",
            5.5,
        ),  # Question pause - let assistant respond
        ("I have chicken,", 0.9),
        ("some vegetables,", 0.6),
        ("and rice in the fridge.", 1.2),
        ("I'm thinking maybe a stir fry or...", 5.0),  # Thinking/trailing off
        ("hmm,", 0.6),
        ("actually on second thought,", 0.3),  # Quick topic change
        (
            "I just remembered I need to finish my work presentation first.",
            5.5,
        ),  # New topic - let assistant respond
        ("Do you have any tips", 0.9),
        ("for making a presentation more engaging?", 4.5),  # Question pause
        ("I have to present to about twenty people tomorrow morning", 1.2),
        ("and I'm pretty nervous about it.", 4.0),  # Emotional pause
        ("Should I use a lot of slides", 0.9),
        ("or keep it minimal?", 5.0),  # Question pause
        ("Oh wait,", 0.2),  # Quick interruption
        ("speaking of tomorrow,", 0.7),
        ("what's the weather supposed to be like?", 4.0),  # Another question
        ("I need to know if I should bring an umbrella or not.", 2.5),
        ("Though I guess I could just check my phone for that.", 1.5),
        ("Anyway, back to the presentation,", 0.8),
        ("I really want to make a good impression", 1.0),
        ("because this could lead to a promotion.", 2.0),
        ("What do you think?", 0.0),
    ]

    # Shared list to track what chunks have actually been READ by generator
    chunks_read = []

    # Create a wrapper queue that tracks what's been consumed
    from queue import Queue as OrigQueue

    class TrackingQueue(OrigQueue):
        def get(self, *args, **kwargs):
            item = super().get(*args, **kwargs)
            if item is not None:
                chunks_read.append(item)
            return item

    source_queue = TrackingQueue()

    def stt_feed():
        for text, delay in source_utterances:
            source_queue.put(text + " ")
            time.sleep(delay)
        source_queue.put(None)

    # Start STT simulator in background thread
    stt_thread = Thread(target=stt_feed)
    stt_thread.start()

    # Clear screen and set up display
    if os.name == "nt":  # Windows
        os.system("cls")
    else:  # Unix/Linux/Mac
        sys.stdout.write("\033[2J\033[H")  # Clear screen and move to top

    # Print header
    print("=" * 80)
    print("REAL-TIME AI ASSISTANT (wait-k=5)")
    print("=" * 80)
    print()

    # Save cursor position
    sys.stdout.write("\033[s")

    # Reserve space for the display
    print("user-speech (live stt):")
    print()  # Space for input
    print()  # Separator
    print("assistant-response (streaming):")
    print()  # Space for output
    print()  # Extra space

    # Move back to saved position
    sys.stdout.write("\033[u")

    # Track displayed content
    current_output_text = ""
    last_chunk_count = 0

    # Stream generation with real-time assistant prompt
    for response in stream_generate(
        model=model,
        tokenizer=tokenizer,
        source_queue=source_queue,
        wait_k=5,
        system_prompt=(
            """You are a real-time AI assistant listening to live speech-to-text input.
The user's speech is arriving incrementally as they speak, and you must
respond naturally and adaptively in real-time. The user may change topics
suddenly, interrupt themselves, or shift direction mid-sentence - this is
normal in live conversation. Be helpful, concise, and responsive to whatever
the user is currently saying. Adapt quickly to topic changes and maintain a
natural, conversational tone. Do not narrate, IMMEDIATELY RESPOND TO THE LATEST THING THE USER SAYS.
Interrupt yourself mid-sentence if you must. When interrupted, DO NOT NARRATE THE INTERRUPTION. Do not say "you said nevermind that", simply pivot.
Do not add commentary. Do not add notes.

If the user has moved on, YOU MUST MOVE ON TOO. Stop yourself mid-sentence often if the user has changed the subject.

DO NOT ramble on when the user has already moved on. You do not need to write complete sentences.

EXAMPLE: "You have ten events on your calendar - to fix the battery, you'll need to shake the device."

EXAMPLE: "The flight leaves in 30 minutes — to reset the router, press and hold the power button for ten seconds."

EXAMPLE: "My dentist appointment is at three — to save the document, click File > Save As."

EXAMPLE: "We need to finish the budget before Friday — to pair your headphones, open Bluetooth settings and select the device."

EXAMPLE: "He's waiting outside with the keys — to clear the cache, go to Settings > Storage > Clear Cache."

EXAMPLE: "Dinner's burning on the stove — to export the file as PDF, choose Export > PDF."

EXAMPLE: "I have to sign the contract now — to update the app, visit the App Store and tap Update."

EXAMPLE: "The baby is crying in the next room — to change your password, go to Account > Security > Change Password."

EXAMPLE: "They're calling from the school office — to reconnect to Wi‑Fi, choose the network and enter the password."

EXAMPLE: "We need to evacuate the building — to enable two‑factor authentication, open Security settings and follow the prompts."

EXAMPLE: "My car won't start in the parking lot — to back up your photos, enable cloud sync in Photos settings."
"""
        ),
        temp=0.7,  # Greedy decoding
        top_p=0.8,
    ):
        needs_redraw = False

        # Check if new chunks have been READ from queue
        if len(chunks_read) > last_chunk_count:
            last_chunk_count = len(chunks_read)
            needs_redraw = True

        # Track output as it's generated
        if hasattr(response, "word_complete") and response.word_complete:
            current_output_text += response.text + ""
            needs_redraw = True

        if needs_redraw:
            # Restore cursor and redraw display
            sys.stdout.write("\033[u")
            sys.stdout.write("\033[J")

            # Display ONLY chunks that have been ACTUALLY READ from queue
            current_input_text = "".join(chunks_read)

            print("user-speech (live stt):")
            print(current_input_text if current_input_text else "")
            print()
            print("assistant-response (streaming):")
            print(current_output_text if current_output_text else "")

            sys.stdout.flush()

    # Final summary
    print()
    print()
    print("=" * 80)
    print("✓ Conversation complete!")
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
