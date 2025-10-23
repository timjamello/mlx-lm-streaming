# Fix: Segment Count Bug (Reading Beyond Source Words)

## Problem

The streaming generation was reading **88 segments** when there were only **86 actual source words**:

```
Step   Target Word          Src Read   Tgt Gen    Lag
...
80      machines            86         80         6
81      à                   87         81         6
82      écrire              88         82         6
```

This caused the model to:
1. Read segments beyond the actual content
2. Process the end token `<|im_end|>` as if it were a source word
3. Get confused and hallucinate/generate meta-commentary

## Root Cause

The segment structure includes:
- **Segment 0**: Template overhead (system prompt + user prefix)
- **Segments 1-86**: The 86 actual source words
- **Segment 87**: End token `<|im_end|>`

Total: **88 segments**, but only **86 are actual words**

The code was iterating through ALL 88 segments, treating the end token as a "source word" to be read.

## Original StreamingLLM Implementation

From `generation/generate.py:1171`:

```python
if source_words >= len(source_seg_len)-1:  # all source tokens have been loaded, '-1' to remove instruct
    ReadAction = False
```

They cap at **`len(source_seg_len) - 1`** to exclude the end token from being read as a separate segment.

## The Fix

**File**: `mlx_lm/streaming_utils.py`

### 1. Cap source reading at len-1

```python
def should_read_next_source(self) -> bool:
    """Check if we should read the next source word."""
    # Cap at len-1 to exclude the end token (last segment)
    # Matches StreamingLLM's generation/generate.py:1171
    return (
        self.source_words_read < len(self.source_seg_len) - 1 and  # ← CHANGED
        not self.finished
    )
```

### 2. Cap wait-k policy at len-1

```python
def check_wait_k_policy(self) -> bool:
    """
    Check if wait-k policy is satisfied for next target word.
    """
    # Cap at len-1 to exclude the end token from wait-k calculation
    # Matches StreamingLLM's generation/generate.py:1171
    required_source_words = min(
        self.wait_k + self.target_words_generated,
        len(self.source_seg_len) - 1  # ← CHANGED
    )
    return self.source_words_read >= required_source_words
```

## Expected Behavior After Fix

For 86-word input:
- **Segment 0**: Instruction (always read first)
- **Segments 1-86**: The 86 words (read progressively)
- **Segment 87**: End token (NOT read as a separate segment)

With `wait_k=7`:
- Read segments 0-7 (instruction + 7 words)
- Generate target word 1
- Read segment 8
- Generate target word 2
- ...
- Read segment 86
- Generate remaining words
- **Stop at 86**, never reading segment 87

Final `Src Read` should cap at **86**, not go to 87 or 88.

## Changes Made

- `mlx_lm/streaming_utils.py`:
  - `StreamingState.should_read_next_source()`: Added `- 1` to cap
  - `StreamingState.check_wait_k_policy()`: Added `- 1` to cap

Both changes match the original StreamingLLM implementation.

## Testing

Run the trace visualization to verify:

```bash
python mlx_lm/examples/streaming_visualization.py --mode trace --wait-k 7
```

Expected output:
- `Src Read` should cap at 86 (not 87 or 88)
- No hallucination/meta-commentary
- Coherent French translation throughout
