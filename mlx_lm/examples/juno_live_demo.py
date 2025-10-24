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
    STT Process: Continuously listens to microphone and transcribes speech.
    Uses Parakeet MLX for real-time transcription.
    """
    import sounddevice as sd
    import numpy as np
    from collections import deque
    import asyncio
    import tempfile
    import soundfile as sf
    from pathlib import Path

    # Import Parakeet STT
    from parakeet_mlx import from_pretrained

    logger = logging.getLogger("STT")
    logger.info("STT Process starting...")

    # Audio parameters
    sample_rate = 16000
    chunk_duration = 3.0  # 3 second chunks for better transcription
    chunk_size = int(sample_rate * chunk_duration)

    try:
        # Load Parakeet model (MLX - exclusive to this process!)
        logger.info("Loading Parakeet STT model...")
        asr_model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
        logger.info("Parakeet model loaded successfully")

        logger.info("\n🎤 Listening to microphone... Speak now!\n")

        # Continuous capture loop
        while not stop_event.is_set():
            # Capture audio from microphone
            logger.debug(f"Recording {chunk_duration}s chunk...")
            audio_chunk = sd.rec(
                chunk_size,
                samplerate=sample_rate,
                channels=1,
                dtype='int16',
                blocking=True
            )

            if stop_event.is_set():
                break

            # Check if there's significant audio (simple energy check)
            audio_float = audio_chunk.astype(np.float32) / 32768.0
            energy = np.sqrt(np.mean(audio_float ** 2))

            # Only transcribe if there's some energy (not silence)
            if energy < 0.01:
                logger.debug("Silence detected, skipping transcription")
                continue

            logger.info("Audio detected, transcribing...")

            # Save to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                audio_float_squeezed = audio_float.squeeze()
                sf.write(tmp_file.name, audio_float_squeezed, sample_rate)
                tmp_path = tmp_file.name

            try:
                # Transcribe with Parakeet
                result = asr_model.transcribe(tmp_path)

                # Extract text
                if hasattr(result, 'text'):
                    text = result.text.strip()
                else:
                    text = str(result).strip()

                # Only send if there's actual text
                if text and len(text) > 0:
                    logger.info(f"✅ Transcribed: {text}")

                    # Send to LLM process
                    output_queue.put({
                        'type': 'transcription',
                        'text': f"[Timothy]: {text}",
                        'speaker': 'Timothy',
                        'timestamp': time.time()
                    })
                else:
                    logger.debug("Empty transcription, skipping")

            finally:
                # Clean up temp file
                Path(tmp_path).unlink(missing_ok=True)

    except KeyboardInterrupt:
        logger.info("STT Process interrupted")
    except Exception as e:
        logger.error(f"STT Process error: {e}", exc_info=True)
    finally:
        # Signal end of stream
        output_queue.put({'type': 'end'})
        logger.info("STT Process completed")


# ==================== TTS Process ====================

def tts_process(input_queue: mp.Queue, stop_event: mp.Event):
    """
    TTS Process: Receives text chunks and speaks them.
    Uses Kokoro MLX for text-to-speech with real audio playback.
    """
    import sounddevice as sd
    import numpy as np
    from mlx_audio.tts.models.kokoro import KokoroPipeline
    from mlx_audio.tts.utils import load_model

    logger = logging.getLogger("TTS")
    logger.info("TTS Process starting...")

    try:
        # Load Kokoro TTS model (MLX - exclusive to this process!)
        logger.info("Loading Kokoro TTS model...")
        model_id = "prince-canuma/Kokoro-82M"
        lang_code = "a"  # American English
        voice = "af_heart"

        model = load_model(model_id)
        pipeline = KokoroPipeline(
            lang_code=lang_code,
            model=model,
            repo_id=model_id
        )
        logger.info("Kokoro TTS model loaded successfully")

        sample_rate = 24000  # Kokoro outputs at 24kHz

        while not stop_event.is_set():
            try:
                # Non-blocking get with timeout
                message = input_queue.get(timeout=0.1)

                if message['type'] == 'speak':
                    text = message['text']
                    logger.info(f"🔊 SPEAKING: {text}")

                    # Generate audio using Kokoro MLX
                    generator = pipeline(text, voice=voice, speed=1.32)

                    # Collect all audio chunks
                    audio_chunks = []
                    for i, (gs, ps, audio_chunk) in enumerate(generator):
                        # MLX returns audio in shape (1, samples), extract the first channel
                        if audio_chunk.ndim > 1:
                            audio_chunk = audio_chunk[0]
                        audio_chunks.append(audio_chunk)

                    # Concatenate and play
                    if audio_chunks:
                        full_audio = np.concatenate(audio_chunks)

                        # Play audio and wait for completion
                        sd.play(full_audio, sample_rate)
                        sd.wait()

                        logger.info("✅ Finished speaking")

                elif message['type'] == 'stop':
                    logger.info("🔇 Stopped speaking")
                    # Stop any ongoing playback
                    sd.stop()

                elif message['type'] == 'end':
                    logger.info("TTS Process shutting down")
                    break

            except mp.queues.Empty:
                continue

    except KeyboardInterrupt:
        logger.info("TTS Process interrupted")
    except Exception as e:
        logger.error(f"TTS Process error: {e}", exc_info=True)


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
        from mlx_lm import load
        from mlx_lm.streaming_data_utils import prepare_streaming_input
        from mlx_lm.streaming_generate import stream_generate_streaming
        import mlx.core as mx

        # Load model
        logger.info(f"Loading model: {model_path}")
        model, tokenizer = load(model_path)
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
    logger.info("🎙️  JUNO LIVE DEMO - Full Pipeline with Real Audio")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Wait-k: {args.wait_k}")
    logger.info("")
    logger.info("This demo will:")
    logger.info("  1. Listen to your microphone continuously (STT)")
    logger.info("  2. Process speech with streaming LLM")
    logger.info("  3. Speak responses through your speakers (TTS)")
    logger.info("")
    logger.info("💡 TIP: Say 'Hey Juno' or 'Juno' to address her!")
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
