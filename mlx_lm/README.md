# MLX Streaming LLM

MLX Streaming LLM is a focused Python package for streaming text generation using large language models on Apple Silicon with MLX.

This library implements the **StreamingLLM wait-k policy**, enabling:
* Incremental input processing (word-by-word or token-by-token)
* Real-time output generation with configurable latency
* Simultaneous translation and live transcription capabilities
* Separate position encodings for source and target tokens

## Quick Start

```python
from mlx_lm import load, stream_generate

# Load a streaming model
model, tokenizer = load("Qwen/Qwen2.5-0.5B-Instruct")

# Stream generation
for response in stream_generate(
    model=model,
    tokenizer=tokenizer,
    prompt="Translate to French: Hello, how are you?",
    wait_k=3,
    max_new_words=20,
):
    if response.word_complete:
        print(response.text, end=' ', flush=True)
```

## What's Different

This is a **focused fork** of mlx-lm specifically for streaming generation:

**Removed**: Batch generation, fine-tuning, quantization, model conversion, 90+ non-streaming model architectures

**Added**: Streaming generation core, dual cache system, wait-k policy implementation, live visualization tools

## Documentation

See the main [README.md](../README.md) in the repository root for complete documentation.

## Examples

Explore `examples/` for demonstrations:
- `streaming_basic.py` - Basic streaming usage
- `streaming_cli_demo.py` - Compare wait-k values
- `streaming_visualization.py` - Visualize streams
- `streaming_realtime.py` - Real-time simulation
- `juno_live_demo.py` - Live conversation assistant

## API Reference

### Main Functions

- **`load(model_path)`** - Load a streaming-compatible model
- **`stream_generate(...)`** - Stream generation with wait-k policy

### Key Parameters

- **`wait_k`** - Number of source words to wait before generating (default: 3)
- **`max_new_words`** - Maximum words to generate
- **`system_prompt`** - Task description
- **`temp`** - Sampling temperature

See [examples/README.md](examples/README.md) for detailed parameter documentation.
