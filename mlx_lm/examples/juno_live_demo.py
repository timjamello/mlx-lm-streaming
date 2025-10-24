#!/usr/bin/env python3
"""
Juno Live Demo - Full STT → Streaming LLM → TTS Pipeline

This demonstrates Juno as a conversational assistant that:
1. Continuously listens via STT (Speech-to-Text)
2. Processes conversation stream in real-time
3. Decides when to speak using tool calls
4. Outputs speech via TTS (Text-to-Speech)

Architecture:
- 3 separate processes (STT, LLM, TTS) to avoid MLX conflicts
- IPC via multiprocessing.Queue for communication
- Each process has exclusive MLX access
"""

import asyncio
import multiprocessing as mp
import time
import sys
import logging
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(processName)s] %(levelname)s: %(message)s'
)


# ==================== STT Process ====================

def stt_process(output_queue: mp.Queue, stop_event: mp.Event):
    """
    STT Process: Continuously listens and transcribes speech.
    Sends transcribed text to LLM process via queue.
    """
    import sounddevice as sd
    import numpy as np
    from collections import deque

    logger = logging.getLogger("STT")
    logger.info("STT Process starting...")

    # Simplified STT for demo - just simulates receiving speech
    # In real implementation, this would use Parakeet MLX

    # Audio parameters
    sample_rate = 16000
    chunk_duration = 1.0  # 1 second chunks
    chunk_size = int(sample_rate * chunk_duration)

    # Simulated conversation (in reality, this would be from microphone + Parakeet)
    simulated_input = [
        "[Timothy]: Hey Juno, can you hear me?",
        "[Timothy]: I wanted to ask you about something.",
        "[Timothy]: Can you tell me about the weather today?",
        "[Timothy]: Actually, never mind. Let's talk about cooking instead.",
        "[Timothy]: What's your favorite recipe for pasta carbonara?",
        "[Timothy]: Thanks! That's helpful."
    ]

    try:
        for i, text in enumerate(simulated_input):
            if stop_event.is_set():
                break

            # Simulate real-time speech arrival (1 second between segments)
            time.sleep(2.0)

            logger.info(f"Transcribed: {text}")

            # Send to LLM process
            output_queue.put({
                'type': 'transcription',
                'text': text,
                'speaker': 'Timothy',
                'timestamp': time.time()
            })

        # Signal end of stream
        output_queue.put({'type': 'end'})
        logger.info("STT Process completed")

    except KeyboardInterrupt:
        logger.info("STT Process interrupted")
    except Exception as e:
        logger.error(f"STT Process error: {e}")


# ==================== TTS Process ====================

def tts_process(input_queue: mp.Queue, stop_event: mp.Event):
    """
    TTS Process: Receives text chunks and speaks them.
    Uses Kokoro MLX for text-to-speech.
    """
    logger = logging.getLogger("TTS")
    logger.info("TTS Process starting...")

    # Simplified TTS for demo - just prints to console
    # In real implementation, this would use Kokoro MLX + sounddevice

    try:
        while not stop_event.is_set():
            try:
                # Non-blocking get with timeout
                message = input_queue.get(timeout=0.1)

                if message['type'] == 'speak':
                    text = message['text']
                    logger.info(f"🔊 SPEAKING: {text}")

                    # Simulate speech duration (in real version, this would be actual playback)
                    time.sleep(len(text) * 0.05)  # ~50ms per character

                elif message['type'] == 'stop':
                    logger.info("🔇 Stopped speaking")

                elif message['type'] == 'end':
                    logger.info("TTS Process shutting down")
                    break

            except mp.queues.Empty:
                continue

    except KeyboardInterrupt:
        logger.info("TTS Process interrupted")
    except Exception as e:
        logger.error(f"TTS Process error: {e}")


# ==================== LLM Process ====================

def llm_process(
    stt_queue: mp.Queue,
    tts_queue: mp.Queue,
    stop_event: mp.Event,
    model_path: str,
    wait_k: int
):
    """
    LLM Process: Runs Juno's streaming model.
    - Receives transcriptions from STT
    - Processes them with streaming generation
    - Detects tool calls and sends audio to TTS
    """
    logger = logging.getLogger("LLM")
    logger.info("LLM Process starting...")

    try:
        from mlx_lm import load_streaming
        from mlx_lm.streaming_data_utils import prepare_streaming_input
        from mlx_lm.streaming_generate import stream_generate_streaming
        import mlx.core as mx

        # Load model
        logger.info(f"Loading model: {model_path}")
        model, tokenizer = load_streaming(model_path)
        logger.info("Model loaded successfully")

        # System prompt
        JUNO_SYSTEM_PROMPT = """You are Juno, an AI assistant observing a conversation.

## Critical Rules:
1. You are LISTENING to a conversation between speakers (e.g., [Timothy], [Alice])
2. You are ONLY "Juno" - you are NOT the other speakers
3. DO NOT respond unless someone addresses you directly
4. When addressed, use tools to control your speech

## Tools Available:
<tools>
speak: Begin speaking (only when addressed!)
stop_speaking: Stop speaking
</tools>

To call: <tool_call>{"name": "function_name"}</tool_call>

## Response Protocol:
1. When addressed, call: <tool_call>{"name": "speak", "arguments": {}}</tool_call>
2. Generate your response (be concise!)
3. IMMEDIATELY call: <tool_call>{"name": "stop_speaking", "arguments": {}}</tool_call>

Remember: You are an OBSERVER until addressed."""

        # Accumulate conversation stream
        conversation_buffer = []
        is_speaking = False
        current_speech_buffer = []

        while not stop_event.is_set():
            try:
                # Check for new transcription
                message = stt_queue.get(timeout=0.1)

                if message['type'] == 'end':
                    logger.info("Received end signal from STT")
                    break

                if message['type'] == 'transcription':
                    text = message['text']
                    conversation_buffer.append(text)

                    logger.info(f"📝 Added to stream: {text}")

                    # Build full conversation context
                    full_conversation = " ".join(conversation_buffer)

                    # Prepare streaming input
                    prepared = prepare_streaming_input(
                        source_text=full_conversation,
                        tokenizer=tokenizer,
                        wait_k=wait_k,
                        system_prompt=JUNO_SYSTEM_PROMPT,
                        split_mode="word",
                    )

                    # Run streaming generation
                    logger.info("🤖 Processing with streaming LLM...")

                    for chunk in stream_generate_streaming(
                        model=model,
                        tokenizer=tokenizer,
                        source_token_ids=prepared["source_token_ids"],
                        source_seg_len=prepared["source_seg_len"],
                        wait_k=wait_k,
                        max_new_words=100,
                        assistant_start_tokens=prepared["assistant_start_tokens"],
                        pe_cache_length=0,
                        split_mode="word",
                        end_token="<|im_end|>",
                        temp=0.7,
                        top_p=0.9,
                    ):
                        # Check for tool calls in generated text
                        if chunk.get("mode") == "write" and chunk.get("text"):
                            text = chunk["text"]

                            # Detect tool calls
                            if "<tool_call>" in text and '"name": "speak"' in text:
                                if not is_speaking:
                                    logger.info("🎤 Juno starting to speak")
                                    is_speaking = True
                                    current_speech_buffer = []

                            elif "<tool_call>" in text and '"name": "stop_speaking"' in text:
                                if is_speaking:
                                    logger.info("🛑 Juno stopping")

                                    # Send accumulated speech to TTS
                                    speech_text = " ".join(current_speech_buffer)
                                    if speech_text.strip():
                                        tts_queue.put({
                                            'type': 'speak',
                                            'text': speech_text
                                        })

                                    tts_queue.put({'type': 'stop'})
                                    is_speaking = False
                                    current_speech_buffer = []

                            else:
                                # Regular text - accumulate if speaking
                                if is_speaking and text.strip():
                                    # Filter out tool call JSON
                                    if not text.strip().startswith("{") and not text.strip().endswith("}"):
                                        current_speech_buffer.append(text)
                                        logger.info(f"💬 Juno: {text}")

                    # Break after processing this segment (in real version, this would be continuous)
                    # For demo, we process one at a time

            except mp.queues.Empty:
                continue

        # Cleanup
        tts_queue.put({'type': 'end'})
        logger.info("LLM Process completed")

    except KeyboardInterrupt:
        logger.info("LLM Process interrupted")
    except Exception as e:
        logger.error(f"LLM Process error: {e}", exc_info=True)


# ==================== Main Coordinator ====================

def main():
    """
    Main coordinator that launches all processes and manages communication.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Juno Live Demo - STT → LLM → TTS")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen3-VL-32B-Instruct-4bit",
        help="Model path"
    )
    parser.add_argument(
        "--wait-k",
        type=int,
        default=7,
        help="Wait-k value for streaming"
    )

    args = parser.parse_args()

    logger = logging.getLogger("MAIN")
    logger.info("=" * 80)
    logger.info("Juno Live Demo - Full Pipeline")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Wait-k: {args.wait_k}")
    logger.info("=" * 80)

    # Create IPC queues
    stt_to_llm_queue = mp.Queue()
    llm_to_tts_queue = mp.Queue()
    stop_event = mp.Event()

    # Create processes
    processes = []

    try:
        # Start STT process
        stt_proc = mp.Process(
            target=stt_process,
            args=(stt_to_llm_queue, stop_event),
            name="STT"
        )
        stt_proc.start()
        processes.append(stt_proc)
        logger.info("✓ STT Process started")

        # Start TTS process
        tts_proc = mp.Process(
            target=tts_process,
            args=(llm_to_tts_queue, stop_event),
            name="TTS"
        )
        tts_proc.start()
        processes.append(tts_proc)
        logger.info("✓ TTS Process started")

        # Start LLM process (last, as it takes time to load)
        llm_proc = mp.Process(
            target=llm_process,
            args=(stt_to_llm_queue, llm_to_tts_queue, stop_event, args.model, args.wait_k),
            name="LLM"
        )
        llm_proc.start()
        processes.append(llm_proc)
        logger.info("✓ LLM Process started")

        logger.info("\n🎤 Demo running... Press Ctrl+C to stop\n")

        # Wait for processes to complete
        for proc in processes:
            proc.join()

        logger.info("\n✓ All processes completed")

    except KeyboardInterrupt:
        logger.info("\n⚠ Interrupted by user")
        stop_event.set()

        # Give processes time to clean up
        time.sleep(1)

        # Terminate if still running
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)

        logger.info("✓ Cleanup complete")

    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        stop_event.set()

        for proc in processes:
            if proc.is_alive():
                proc.terminate()


if __name__ == "__main__":
    # Required for multiprocessing on macOS
    mp.set_start_method('spawn', force=True)
    main()
