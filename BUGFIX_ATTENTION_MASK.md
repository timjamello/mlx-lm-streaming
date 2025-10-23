# Critical Fix: Attention Mask During Generation (WRITE Mode)

## Problem Summary

The model was hallucinating and repeating translations because it **couldn't see the source tokens during generation**. It kept re-translating "The quick brown fox..." because a causal attention mask was blocking access to the source context.

## Observed Behavior

```
Step   Target Word          Src Read   Tgt Gen    Lag
1      Le                   7          1          6
2       renard              8          2          6
...
10      pares.              16         10         6
11      \nNote:             17         11         6     ← Starts hallucinating
12      There               18         12         6
13      seems               19         13         6
...
31      \nLa                37         31         6     ← Restarts translation
32      rapide              38         32         6
33      renard              39         33         6
40      \nCorrection:       46         40         6     ← Yet another restart
41      \nLe                47         41         6
```

The model would:
1. Correctly translate the first sentence
2. Start meta-commenting in English
3. Restart the translation
4. Repeat this pattern endlessly

## Root Cause

**File**: `mlx_lm/models/qwen2_streaming.py:242` (BEFORE FIX)

```python
mask = create_attention_mask(h, cache[0])  # ← ALWAYS creates causal mask

for layer, c in zip(self.layers, cache):
    h = layer(h, mask, c, position_ids, is_reading)
```

During WRITE mode (generation), the cache structure is:
```
merged_cache = [source_tokens_0...86, target_tokens_0...N]
                 positions 0-86       positions 87+
```

With a **causal mask**, when generating target token at position 90:
- Can attend to positions 0-90 (causal)
- BUT the causal mask is applied to the MERGED cache
- This creates a standard upper-triangular mask that treats everything as a sequence

The problem: The model needs to attend to **ALL source tokens** (positions 0-86) regardless of the current generation position, but a standard causal mask prevents this for tokens at later positions.

## Original StreamingLLM Implementation

**File**: `/tmp/StreamingLLM/models/Qwen2_5/qwen_streaming.py:870-879`

```python
elif generate_mode == 'streaming':
    _wait = None
    assert ReadAction is not None
    if ReadAction:  # Reading source
        history_source_length = source_key_values.get_seq_length()
        current_length = history_source_length + inputs_embeds.shape[1]
        cache_position = torch.arange(history_source_length, current_length, dtype=torch.long, device=cache_position.device)
        causal_mask = self._update_causal_mask(attention_mask[:,:current_length], inputs_embeds, cache_position, past_key_values, output_attentions)
    else:  # Writing target
        causal_mask = None  # ← NO MASK during generation!
```

Then at **line 428-430**:
```python
if attention_mask is not None:
    causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = attn_weights + causal_mask
# If attention_mask is None, skip mask application = full attention
```

## The Fix

**File**: `mlx_lm/models/qwen2_streaming.py:242-248` (AFTER FIX)

```python
# Only use causal mask during READ mode
# During WRITE mode, mask = None to allow full attention to source tokens
# Matches StreamingLLM's qwen_streaming.py:879 where causal_mask = None
if is_reading:
    mask = create_attention_mask(h, cache[0])
else:
    mask = None  # No mask during generation = attend to all tokens

for layer, c in zip(self.layers, cache):
    h = layer(h, mask, c, position_ids, is_reading)
```

## Why This Works

### READ Mode (is_reading=True)
- Processing source tokens sequentially
- Need causal mask so token N can't see token N+1
- Standard causal attention is correct

### WRITE Mode (is_reading=False)
- Generating target tokens
- Cache contains: [all_source_tokens, previous_target_tokens]
- **Need to see ALL source tokens** (no restrictions)
- **Need causal attention over previous target tokens only**
- Setting `mask = None` in MLX means:
  - In `scaled_dot_product_attention()`, no mask is applied
  - Model can attend to all positions in the cache
  - KV cache structure naturally provides causal behavior for target tokens (because we only have K/V for previously generated tokens)

The merged cache naturally provides the right behavior:
- Source cache: Contains all source tokens
- Target cache: Contains only previously generated tokens (causal by construction)
- No mask needed because the cache structure itself enforces causality

## Expected Behavior After Fix

With `mask = None` during generation:
- Model can see **all source tokens** (positions 0-86)
- Model can see **all previously generated target tokens** (causal by cache construction)
- Should produce coherent, continuous translation
- No hallucination or meta-commentary
- No restarts or repetition

## Testing

Run the trace visualization:

```bash
python mlx_lm/examples/streaming_visualization.py --mode trace --wait-k 7
```

Expected output:
- Coherent French translation from start to finish
- No English meta-commentary
- No repeated translations
- Smooth progression through all 86 source words

## Technical Details

### Why mask=None Works in MLX

In MLX's `scaled_dot_product_attention()`:
```python
if mask is not None:
    scores = scores + mask  # Apply mask (adds -inf to blocked positions)
weights = mx.softmax(scores, axis=-1)
```

When `mask=None`, the attention scores are unmasked, allowing full attention across the entire cache.

The causality for target tokens is maintained because:
1. Target cache only contains previously generated tokens
2. Current token only sees keys/values that have been added to the cache
3. Future target tokens don't exist in the cache yet

This is different from training where you have the full target sequence upfront and need an explicit causal mask.

## Changes Made

- `mlx_lm/models/qwen2_streaming.py`:
  - Line 242-248: Conditional mask creation based on `is_reading` flag
  - Added detailed comments explaining the behavior
  - Matches original StreamingLLM implementation (line 879)

## References

- Original StreamingLLM: `/tmp/StreamingLLM/models/Qwen2_5/qwen_streaming.py:879`
- Attention layer: `/tmp/StreamingLLM/models/Qwen2_5/qwen_streaming.py:428-430`
