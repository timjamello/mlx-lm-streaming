#!/usr/bin/env python3
"""
Basic StreamingLLM Example

This example demonstrates basic streaming generation with the wait-k policy.
The model processes source text incrementally and generates output with a
configurable lag.
"""

from mlx_lm import load, stream_generate_streaming_llm


def main():
    # Load a Qwen2.5 model (or any compatible streaming model)
    print("Loading model...")
    model_path = "Qwen/Qwen2.5-0.5B-Instruct"
    model, tokenizer = load(model_path)

    # Example prompt
    prompt = "Translate the following to French: Hello, how are you doing today?"

    print("\n" + "=" * 60)
    print("STREAMING GENERATION (wait-k=3)")
    print("=" * 60)

    # Generate with streaming
    words = []
    for response in stream_generate_streaming_llm(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        wait_k=3,  # Wait for 3 source words before generating
        max_new_words=20,  # Generate up to 20 words
        temp=0.7,  # Sampling temperature
    ):
        # Only print when a complete word is generated
        if hasattr(response, "word_complete") and response.word_complete:
            words.append(response.text)
            print(f"Word {len(words)}: '{response.text}'")
            print(f"  → Source words read: {response.source_words_read}")
            print(f"  → Target words generated: {response.target_words_generated}")
            print()

    print("=" * 60)
    print("FINAL OUTPUT:")
    print(" ".join(words))
    print("=" * 60)

    if hasattr(response, "generation_tps"):
        print(f"\nGeneration speed: {response.generation_tps:.2f} tokens/sec")
        print(f"Peak memory: {response.peak_memory:.2f} GB")


if __name__ == "__main__":
    main()
