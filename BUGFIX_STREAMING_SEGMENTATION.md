# Critical Bug Fix: Streaming Prompt Segmentation

## Problem Summary

The streaming LLM was producing garbled, nonsensical output because the **instruction/task description was being segmented as source words** instead of being kept as a system prompt.

## Example of Garbled Output

```
Input: "The quick brown fox jumps over the lazy dog..."

Output: "2.  "I  am  a  professional  and  I  brown  fox  jumps  over  the  lazy  dog..."
```

The model was trying to "translate" words like "Translate", "the", "following", "English", "paragraph" instead of understanding them as instructions.

## Root Cause

### Before Fix

The visualization script was passing the complete instruction + source text as the prompt:

```python
# streaming_visualization.py (WRONG)
prompt=f"Translate the following English paragraph to French: {source_text}"
```

The `stream_generate_streaming_llm` function would then segment this **entire string** word-by-word:
- Word 0: "Translate"
- Word 1: "the"
- Word 2: "following"
- Word 3: "English"
- ...
- Word 7: "The"
- Word 8: "quick"
- Word 9: "brown"
- ...

With `wait_k=3`, the model would:
1. Read "Translate the following" (words 0-2)
2. Try to generate output (???)
3. Read "English paragraph to" (words 3-5)
4. Generate more confusion

### Comparison with Original StreamingLLM

The original StreamingLLM implementation **separates** instruction from source:

```json
{
    "Instruct": "<|im_start|>system\nTranslate the following English paragraph to French<|im_end|>\n",
    "user_Instruct": "<|im_start|>user\n",
    ...
}
```

**Correct structure:**
```
<|im_start|>system
Translate the following English paragraph to French<|im_end|>
<|im_start|>user
The quick brown fox jumps over...<|im_end|>     ← Only this gets segmented!
<|im_start|>assistant
[French translation]
```

## The Fix (Option 2)

### 1. Added `system_prompt` Parameter

**File:** `mlx_lm/generate.py`

```python
def stream_generate_streaming_llm(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: Union[str, mx.array, List[int]],
    wait_k: int = 3,
    max_new_words: Optional[int] = None,
    max_tokens_per_word: int = 50,
    system_prompt: str = "Translate the following English paragraph to French",  # ← NEW
    temp: float = 0.0,
    ...
):
```

### 2. Passed to StreamingDataPreparator

```python
preparator = StreamingDataPreparator(tokenizer, system_prompt=system_prompt)
```

### 3. Updated Visualization Script

**File:** `mlx_lm/examples/streaming_visualization.py`

```python
# Before (WRONG):
prompt=f"Translate the following English paragraph to French: {source_text}",

# After (CORRECT):
prompt=source_text,  # Just the source text, no instruction
system_prompt="Translate the following English paragraph to French",
```

## Changes Made

### Modified Files

1. **mlx_lm/generate.py**
   - Added `system_prompt` parameter to `stream_generate_streaming_llm()`
   - Updated docstring with clear explanation
   - Updated example code
   - Passed `system_prompt` to `StreamingDataPreparator`

2. **mlx_lm/examples/streaming_visualization.py**
   - Updated all 4 visualization functions:
     - `live_streaming_visualization()`
     - `simple_streaming_visualization()`
     - `side_by_side_visualization()`
     - `word_by_word_trace()`
   - Separated source text from instruction
   - Added explicit `system_prompt` parameter

## Expected Behavior After Fix

Now the model should:
1. Understand the task from the system prompt (not segmented)
2. Process only the actual source text word-by-word
3. Generate coherent French translation with proper wait-k behavior

### Correct Flow with wait_k=3

- **Read:** "The quick brown" (words 0-2)
- **Generate:** "Le rapide brun"
- **Read:** "fox jumps over" (words 3-5)
- **Generate:** "renard saute sur"
- **Read:** "the lazy dog" (words 6-8)
- **Generate:** "le chien paresseux"
- ...

## Testing

Run the visualization to verify the fix:

```bash
python mlx_lm/examples/streaming_visualization.py --mode live --wait-k 3
```

Expected output should now be coherent French translation, not garbled text.

## Key Takeaway

**The instruction/task description must be kept separate from the source text in streaming generation.**

Only the actual content to be processed should be segmented word-by-word. The instruction that describes what to do with that content goes in the system prompt.
