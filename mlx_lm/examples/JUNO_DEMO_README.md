# Juno Live Demo - Full Pipeline

This demo showcases Juno as a real-time conversational assistant with:
- **Continuous STT** (Speech-to-Text)
- **Streaming LLM** processing
- **Tool-controlled TTS** (Text-to-Speech)

## Architecture

The demo uses **3 separate processes** to avoid MLX thread-safety issues:

```
┌─────────────────┐
│  STT Process    │  ← Continuously listens to microphone
│  (Parakeet MLX) │     Transcribes speech
└────────┬────────┘     Sends text to LLM
         │
         │ Queue (IPC)
         ▼
┌─────────────────┐
│  LLM Process    │  ← Receives transcription stream
│  (Qwen Stream)  │     Processes with streaming generation
└────────┬────────┘     Detects tool calls (speak/stop_speaking)
         │              Sends speech to TTS
         │ Queue (IPC)
         ▼
┌─────────────────┐
│  TTS Process    │  ← Receives text chunks
│  (Kokoro MLX)   │     Speaks them aloud
└─────────────────┘
```

**Key Design Decision**: Each process has **exclusive MLX access** - no thread locks needed!

## Current Status

### ✅ Implemented (Demo Version)

- [x] Multi-process architecture with IPC queues
- [x] STT process (simulated with pre-recorded conversation)
- [x] LLM process with streaming generation
- [x] TTS process (simulated with console output)
- [x] Tool call detection (`<speak>` and `<stop_speaking>`)
- [x] Basic conversation flow

### 🚧 TODO (Production Version)

To make this production-ready, integrate your existing implementations:

#### STT Process (`stt_engine.py`)

Replace `stt_process()` function with:

```python
def stt_process(output_queue: mp.Queue, stop_event: mp.Event):
    """Real STT using Parakeet MLX"""
    from core.stt_engine import STTEngine
    import sounddevice as sd
    import numpy as np

    # Initialize STT engine (MLX exclusive to this process)
    stt = STTEngine(
        model_name="mlx-community/parakeet-tdt-0.6b-v3",
        enable_diarization=False,  # Skip for speed
        enable_voice_profiles=True
    )

    # Audio capture parameters
    sample_rate = 16000
    chunk_duration = 2.0  # 2-second chunks

    # Capture loop
    while not stop_event.is_set():
        # Capture audio from microphone
        audio_chunk = sd.rec(
            int(chunk_duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='int16'
        )
        sd.wait()

        # Convert to bytes
        audio_bytes = audio_chunk.tobytes()

        # Transcribe
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            stt.transcribe(audio_bytes, sample_rate=sample_rate)
        )
        loop.close()

        if result:
            output_queue.put({
                'type': 'transcription',
                'text': f"[Timothy]: {result}",
                'speaker': 'Timothy',
                'timestamp': time.time()
            })
```

#### TTS Process (`tts_engine.py`)

Replace `tts_process()` function with:

```python
def tts_process(input_queue: mp.Queue, stop_event: mp.Event):
    """Real TTS using Kokoro MLX"""
    from core.tts_engine import TTSEngine

    # Initialize TTS engine (MLX exclusive to this process)
    tts = TTSEngine()

    while not stop_event.is_set():
        message = input_queue.get(timeout=0.1)

        if message['type'] == 'speak':
            text = message['text']

            # Generate and play speech (synchronous in this process)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(tts.speak(text, play_audio=True))
            loop.close()

        elif message['type'] == 'end':
            break
```

#### Metal Lock Removal

Since each process has exclusive MLX access, you can **remove the `metal_lock`** from:
- `STTEngine._process_audio_sync()`
- `TTSEngine._generate_audio()`

The process boundary provides natural isolation!

## Usage

### Demo Version (Current)

```bash
python mlx_lm/examples/juno_live_demo.py --model mlx-community/Qwen3-VL-32B-Instruct-4bit --wait-k 7
```

### Production Version (After Integration)

1. Copy your `stt_engine.py` and `tts_engine.py` to the project
2. Update the imports in `juno_live_demo.py`
3. Replace `stt_process()` and `tts_process()` with real implementations
4. Run:

```bash
python mlx_lm/examples/juno_live_demo.py \
    --model mlx-community/Qwen3-VL-32B-Instruct-4bit \
    --wait-k 7
```

## Example Output

```
[MAIN] ================================================================================
[MAIN] Juno Live Demo - Full Pipeline
[MAIN] ================================================================================
[MAIN] Model: mlx-community/Qwen3-VL-32B-Instruct-4bit
[MAIN] Wait-k: 7
[MAIN] ================================================================================
[MAIN] ✓ STT Process started
[MAIN] ✓ TTS Process started
[MAIN] ✓ LLM Process started

[STT] Transcribed: [Timothy]: Hey Juno, can you hear me?
[LLM] 📝 Added to stream: [Timothy]: Hey Juno, can you hear me?
[LLM] 🤖 Processing with streaming LLM...
[LLM] 🎤 Juno starting to speak
[LLM] 💬 Juno: Yes, I can hear you!
[LLM] 🛑 Juno stopping
[TTS] 🔊 SPEAKING: Yes, I can hear you!
[TTS] 🔇 Stopped speaking
```

## Key Advantages

1. **No Thread Contention**: Each MLX operation is isolated to its own process
2. **True Parallelism**: STT, LLM, and TTS can run simultaneously
3. **Fault Isolation**: If one process crashes, others can continue
4. **Simple IPC**: Queues handle all communication cleanly

## Performance Notes

- **STT Process**: Runs Parakeet MLX continuously (~0.3x RTF typical)
- **LLM Process**: Runs streaming generation (~20-40 tokens/sec typical)
- **TTS Process**: Runs Kokoro MLX as needed (~0.8x RTF typical)

Total system load: All 3 processes share GPU efficiently via Metal scheduler.

## Next Steps

1. Integrate real STT with microphone input
2. Integrate real TTS with audio playback
3. Add VAD (Voice Activity Detection) to reduce transcription latency
4. Add speaker diarization if needed
5. Tune `wait_k` parameter for optimal responsiveness
6. Add interruption handling (stop TTS when new speech detected)

## Troubleshooting

### "Model not found" error

Download the model first:
```bash
huggingface-cli download mlx-community/Qwen3-VL-32B-Instruct-4bit
```

### Processes hang on exit

This is normal with MLX - press Ctrl+C twice if needed.

### Audio device errors

Check available audio devices:
```python
import sounddevice as sd
print(sd.query_devices())
```

Set default device:
```python
sd.default.device = 0  # Or your device ID
```
