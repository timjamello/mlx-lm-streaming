# MLX Streaming LLM

**Real-time streaming text generation with MLX** - Implementation of StreamingLLM's wait-k policy for Apple Silicon.

MLX Streaming LLM is a Python package for streaming text generation using large language models on Apple Silicon with MLX. Unlike traditional batch processing, this library enables **incremental input processing** and **real-time output generation** with configurable latency control.

## Key Features

- **Streaming Input Processing**: Process source text incrementally rather than in batches
- **Wait-k Policy**: Configurable lag control between input and output (wait-k algorithm)
- **Separate Position Encodings**: Independent position IDs for source and target tokens
- **Dual Cache System**: Efficient separate KV caches for streaming generation
- **MLX Optimized**: Native support for Apple Silicon with MLX acceleration
- **Compatible Models**: Qwen2.5 family with streaming support

## What is StreamingLLM?

StreamingLLM implements the wait-k policy where the model:
1. **Reads** source text incrementally (word-by-word or token-by-token)
2. **Waits** for k source words before generating each target word
3. **Generates** output with controllable latency

This enables applications like:
- **Simultaneous translation** - Translate speech in real-time
- **Live transcription** - Process and respond to streaming text
- **Real-time summarization** - Generate summaries as content arrives

## Installation

```bash
# Install from source
git clone https://github.com/yourusername/mlx-streaming-llm.git
cd mlx-streaming-llm
pip install -e .
```

**Requirements**:
- Python 3.8+
- Apple Silicon Mac (M1/M2/M3/M4)
- MLX >= 0.29.2

## Quick Start

```python
from mlx_lm import load, stream_generate

# Load a streaming-compatible model
model, tokenizer = load("Qwen/Qwen2.5-0.5B-Instruct")

# Stream generation with wait-k=3
for response in stream_generate(
    model=model,
    tokenizer=tokenizer,
    prompt="Translate to French: Hello, how are you doing today?",
    wait_k=3,  # Wait for 3 source words before generating
    max_new_words=20,
    temp=0.7,
):
    if response.word_complete:
        print(response.text, end=' ', flush=True)
```

## Python API

### Basic Streaming

```python
from mlx_lm import load, stream_generate

model, tokenizer = load("Qwen/Qwen2.5-0.5B-Instruct")

# Generate with streaming
for response in stream_generate(
    model=model,
    tokenizer=tokenizer,
    prompt="Your source text here",
    wait_k=3,
    max_new_words=50,
    system_prompt="Translate to French",  # Optional task description
    temp=0.0,  # Greedy decoding
):
    if response.word_complete:
        # Access streaming metadata
        print(f"Word: {response.text}")
        print(f"  Source words read: {response.source_words_read}")
        print(f"  Target words: {response.target_words_generated}")
        print(f"  Speed: {response.generation_tps:.1f} tok/s")
```

### Response Object

The `GenerationResponse` object includes:
- `text` - Generated text segment
- `token` - Token ID
- `word_complete` - True when word boundary reached
- `source_words_read` - Number of source words processed
- `target_words_generated` - Number of target words generated
- `generation_tps` - Generation speed (tokens/sec)
- `peak_memory` - Peak memory usage (GB)

### Key Parameters

#### `wait_k` (int, default=3)
Number of source words to wait for before generating each target word.
- **Lower** (1-2): Faster response, less context
- **Higher** (5-7): More context, better quality
- **Typical**: 3-5 for balanced performance

#### `max_new_words` (int, optional)
Maximum number of words to generate. If None, generates until source exhausted.

#### `max_tokens_per_word` (int, default=50)
Safety limit for tokens per word. Prevents runaway generation.

#### `system_prompt` (str)
Task description kept separate from source text (not segmented).

#### Temperature & Sampling
- `temp` (float, default=0.0) - Sampling temperature
- `top_p` (float, default=1.0) - Nucleus sampling
- `repetition_penalty` (float, optional) - Penalty for repetition

## Examples

See the `mlx_lm/examples/` directory for detailed examples:

- **`streaming_basic.py`** - Introduction to streaming generation
- **`streaming_cli_demo.py`** - Compare different wait-k values
- **`streaming_visualization.py`** - Visualize input/output streams
- **`streaming_realtime.py`** - Real-time streaming simulation
- **`juno_live_demo.py`** - Multi-process live conversation assistant

Run any example:
```bash
python mlx_lm/examples/streaming_basic.py
```

## Supported Models

Currently supports Qwen2.5 family models with streaming variants:
- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen2.5-1.5B-Instruct`
- `Qwen/Qwen2.5-3B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`

And other Qwen3 variants. Check `mlx_lm/models/` for supported architectures.

## How It Works

Traditional LLM generation processes the entire input before generating output:

```
Input:  [The][quick][brown][fox] → Process → Output: [Le][rapide][renard][brun]
```

StreamingLLM processes incrementally with wait-k policy:

```
Input:  [The][quick][brown]... → Generate: [Le]
Input:  [The][quick][brown][fox]... → Generate: [Le][rapide]
...
```

This is achieved through:
1. **Separate Position IDs** - Source: 0,1,2,3... Target: 0,1,2,3...
2. **Dual KV Caches** - Independent source and target caches
3. **Wait-k Policy** - Controlled lag between input and output

## Technical Details

### Streaming Generation Algorithm

The core algorithm follows StreamingLLM's approach:

```python
while not finished:
    if reading_mode:
        # Read next source word chunk
        process_source_tokens(...)
        update_source_cache(...)

        # Check if can start writing
        if source_words_read >= wait_k + target_words:
            switch_to_writing()

    else:  # writing_mode
        # Generate tokens until word boundary
        token = sample_next_token(...)
        update_target_cache(...)

        if is_word_boundary(token):
            yield word
            switch_to_reading()
```

### Cache Management

The `DualStreamingCache` maintains:
- **Source cache**: KV states for input tokens
- **Target cache**: KV states for generated tokens
- **Merged cache**: Combined for attention computation

Caches are merged before each forward pass and separated after.

## Performance

Typical performance on Apple Silicon (M1 Max):
- **Speed**: 20-40 tokens/sec (model dependent)
- **Memory**: ~2-8 GB depending on model size
- **Latency**: Controlled by wait-k (typical: 200-500ms per word)

## Differences from mlx-lm

This library is a **focused fork** of mlx-lm specifically for streaming generation:

**Removed**:
- Batch generation
- Fine-tuning (LoRA, full model)
- Quantization (AWQ, GPTQ)
- Model conversion utilities
- Server/API functionality
- Non-streaming models (90+ architectures)

**Added**:
- Streaming generation core (`streaming_generate.py`)
- Dual cache system (`streaming_cache.py`)
- Wait-k policy implementation
- Streaming data utilities
- Live visualization tools

## References

- **StreamingLLM Paper**: [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)
- **Wait-k Policy**: [Simultaneous Translation with Flexible Policy via Restricted Imitation Learning](https://aclanthology.org/P19-1289/)
- **MLX**: [Apple's MLX Framework](https://github.com/ml-explore/mlx)
- **Original StreamingLLM**: [MIT Implementation](https://github.com/mit-han-lab/streaming-llm)

## Contributing

Contributions are welcome! This project focuses specifically on streaming generation capabilities. Please see `CONTRIBUTING.md` for guidelines.

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- Based on [mlx-lm](https://github.com/ml-explore/mlx-lm) by Apple's MLX team
- Implements techniques from [StreamingLLM](https://arxiv.org/abs/2309.17453) by MIT HAN Lab
- Built with [MLX](https://github.com/ml-explore/mlx) framework
