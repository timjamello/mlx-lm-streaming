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
    model_path = "mlx-community/Qwen3-VL-8B-Instruct-8bit"
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

CRITICAL: The user CANNOT see or hear anything you say unless you use the "speak" tool. All your thoughts are internal and invisible to the user.

# Available Tools

## speak
Starts speaking to the user. After calling this tool, everything you output will be heard by the user until you call stop_speaking.
Parameters: None

## stop_speaking
Stops speaking to the user. After calling this tool, the user can no longer hear you and you return to internal thought.
Parameters: None

To use a tool, output:
<tool_call>
{"name": "tool_name", "arguments": {}}
</tool_call>

# Instructions

You are listening to live speech-to-text input as it arrives incrementally. The user's speech comes in word by word as they speak.

IMPORTANT: DO NOT call speak immediately. Wait and listen until the user has completed a meaningful utterance worth responding to. Examples of when to respond:
- The user asks a complete question
- The user pauses for several seconds after making a statement
- The user explicitly asks for your input
- The user finishes a complete thought

DO NOT respond to:
- Incomplete sentences or thoughts
- Mid-sentence pauses
- Users thinking out loud
- Users changing topics mid-utterance (wait to see where they land)

When you do speak:
- Call the speak tool first
- Output your message (the user can now hear you)
- Call stop_speaking when done
- Be concise and helpful
- Respond to whatever the user is CURRENTLY talking about
- If the user changes topics while you're speaking, call stop_speaking immediately, then call speak again and pivot to the new topic
- Do not narrate interruptions or topic changes
- Do not say things like "you mentioned" or "you said" - just respond naturally
- Cut yourself off mid-sentence if the user has moved on to something else
- Always call the stop_speaking tool when you are done speaking

Remember: You can think internally all you want, but the user cannot see any of your thoughts. Only content between speak and stop_speaking is heard by the user.

Examples of good responses:

User: "Can you help me figure out what to make for dinner tonight? I have chicken, some vegetables, and rice in the fridge."
Assistant (thinking): The user has finished asking a complete question about dinner. Time to respond.
<tool_call>
{"name": "speak", "arguments": {}}
</tool_call>
A chicken stir fry would be perfect with those ingredients! You could dice the chicken, sauté it with the vegetables, and serve it over rice.
<tool_call>
{"name": "stop_speaking", "arguments": {}}
</tool_call>

User: "Do you have any tips for making a presentation more engaging? I have to present to about twenty people tomorrow morning."
Assistant (thinking): Complete question about presentations. Ready to help.
<tool_call>
{"name": "speak", "arguments": {}}
</tool_call>
For twenty people, keep slides minimal with strong visuals. Tell stories, make eye contact, and pause for questions. Practice your opening to build confidence.
<tool_call>
{"name": "stop_speaking", "arguments": {}}
</tool_call>"""
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
