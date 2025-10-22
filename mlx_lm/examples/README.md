# StreamingLLM Examples

This directory contains example scripts demonstrating how to use StreamingLLM functionality in mlx-lm.

## Overview

StreamingLLM enables streaming text generation with a wait-k policy, where the model processes source text incrementally and generates output with a configurable lag. This is useful for:

- **Simultaneous translation**: Translate speech or text in real-time with minimal lag
- **Live transcription**: Generate summaries or responses as text is being input
- **Real-time text processing**: Process and respond to streaming text inputs

## Examples

### 1. Basic Streaming (`streaming_basic.py`)

**Purpose**: Introduction to streaming generation with the wait-k policy.

**Usage**:
```bash
python mlx_lm/examples/streaming_basic.py
```

**What it demonstrates**:
- Loading a streaming-compatible model
- Basic streaming generation with wait-k=3
- Accessing streaming metadata (source words read, target words generated)
- Word-level output streaming

**Output**:
```
Word 1: 'Bonjour'
  → Source words read: 3
  → Target words generated: 1

Word 2: ','
  → Source words read: 4
  → Target words generated: 2
...
```

### 2. CLI Demo (`streaming_cli_demo.py`)

**Purpose**: Compare different wait-k values and understand their impact.

**Usage**:
```bash
# Use default wait-k values (1, 3, 5)
python mlx_lm/examples/streaming_cli_demo.py

# Custom wait-k values
python mlx_lm/examples/streaming_cli_demo.py --wait-k 1 2 3 4 5

# Custom prompt
python mlx_lm/examples/streaming_cli_demo.py \
    --prompt "Translate to Spanish: Hello world" \
    --wait-k 2 4 6
```

**What it demonstrates**:
- Effect of different wait-k values on streaming behavior
- Tracking source words read for each generated target word
- Comparing streaming patterns side-by-side

**Key insight**: Higher wait-k values mean the model waits for more source context before generating each word, potentially improving quality but increasing latency.

### 3. Real-time Streaming (`streaming_realtime.py`)

**Purpose**: Simulate real-time streaming scenarios.

**Usage**:
```bash
# Demo mode with predefined examples
python mlx_lm/examples/streaming_realtime.py

# Interactive mode
python mlx_lm/examples/streaming_realtime.py --interactive

# Custom settings
python mlx_lm/examples/streaming_realtime.py \
    --wait-k 5 \
    --delay 0.5
```

**What it demonstrates**:
- Real-time word-by-word output
- Interactive streaming mode
- Performance metrics (words/sec, tokens/sec)
- Simulated processing delays

**Interactive mode**:
```
Prompt> Translate to French: Good morning
Wait-k (default=3)> 3

Streaming output (wait-k=3):
--------------------------------------------------------------------------------
Bonjour matin ...
--------------------------------------------------------------------------------
```

## Using Streaming from CLI

You can also use streaming mode directly from the mlx-lm CLI:

```bash
# Basic streaming
mlx_lm generate \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --prompt "Translate to French: Hello world" \
    --streaming \
    --wait-k 3

# With custom parameters
mlx_lm generate \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --prompt "Summarize this text: ..." \
    --streaming \
    --wait-k 5 \
    --max-new-words 50 \
    --max-tokens-per-word 30 \
    --temp 0.7
```

## Using Streaming in Your Code

### Basic API Usage

```python
from mlx_lm import load, stream_generate_streaming_llm

# Load model
model, tokenizer = load("Qwen/Qwen2.5-0.5B-Instruct")

# Stream generation
for response in stream_generate_streaming_llm(
    model=model,
    tokenizer=tokenizer,
    prompt="Translate to French: Hello, how are you?",
    wait_k=3,
    max_new_words=20,
    temp=0.7,
):
    if response.word_complete:
        print(response.text, end=' ', flush=True)
```

### Accessing Metadata

```python
for response in stream_generate_streaming_llm(...):
    if response.word_complete:
        print(f"Word: {response.text}")
        print(f"  Source words read: {response.source_words_read}")
        print(f"  Target words generated: {response.target_words_generated}")
        print(f"  Generation speed: {response.generation_tps:.2f} tok/s")
```

### Low-level API

For more control, use the lower-level streaming generation API:

```python
from mlx_lm.streaming_generate import stream_generate_streaming
from mlx_lm.streaming_data_utils import StreamingDataPreparator

# Prepare data
preparator = StreamingDataPreparator(tokenizer)
formatted_text, token_ids, seg_lens = preparator.prepare_source_text(prompt)

# Stream with low-level control
for output in stream_generate_streaming(
    model=model,
    tokenizer=tokenizer,
    source_token_ids=token_ids,
    source_seg_len=seg_lens,
    wait_k=3,
    max_new_words=50,
):
    if output['word_complete']:
        print(output['text'], output['token'])
```

## Parameters

### `wait_k` (int, default=3)
Number of source words to wait for before generating each target word.
- **Lower values** (1-2): Faster response, less context
- **Higher values** (5-7): More context, potentially better quality
- **Typical range**: 3-5 for balanced performance

### `max_new_words` (int, default=None)
Maximum number of words to generate. If None, generates until source is exhausted.

### `max_tokens_per_word` (int, default=50)
Safety limit for tokens per word. Prevents runaway generation.

### `temp` (float, default=0.0)
Sampling temperature:
- **0.0**: Greedy decoding (deterministic)
- **0.7-0.9**: Balanced sampling
- **1.0+**: More random/creative

## Requirements

- mlx-lm with StreamingLLM support
- A streaming-compatible model (e.g., Qwen2.5)
- Python 3.8+

## Troubleshooting

### Model doesn't support streaming
**Error**: `AttributeError: 'Model' object has no attribute 'make_cache'`

**Solution**: Use a streaming-compatible model architecture (currently Qwen2.5).

### Slow generation
**Issue**: Streaming is slower than expected

**Solutions**:
- Reduce `wait_k` for faster response
- Reduce `max_tokens_per_word`
- Use a smaller model
- Ensure MLX is using GPU acceleration (Apple Silicon)

### Empty output
**Issue**: No text is generated

**Solutions**:
- Check that `max_new_words` is not too small
- Verify the prompt is properly formatted
- Try with `temp > 0.0` for non-greedy sampling

## Additional Resources

- [StreamingLLM Paper](https://arxiv.org/abs/2309.17453)
- [mlx-lm Documentation](https://github.com/ml-explore/mlx-lm)
- [Wait-k Policy Explanation](https://aclanthology.org/P19-1289/)

## Contributing

Have an interesting use case or example? Contributions are welcome!
