"""
Test script for Qwen3Next streaming implementation.

This script tests the hybrid Mamba+Attention streaming architecture.
"""

import sys
from mlx_lm import load_streaming, stream_generate_streaming_llm


def test_qwen3next_streaming(model_path="Qwen/Qwen3-0.5B-Chat"):
    """
    Test Qwen3Next streaming with a simple translation task.

    Args:
        model_path: HuggingFace model path for Qwen3Next model
    """
    print("=" * 80)
    print("QWEN3NEXT STREAMING TEST")
    print("=" * 80)
    print(f"\nLoading model: {model_path}")
    print("This model uses hybrid architecture: Mamba + Attention layers")
    print()

    try:
        model, tokenizer = load_streaming(model_path)
        print("✓ Model loaded successfully!")
        print()

        # Print model info
        num_layers = len(model.layers)
        mamba_layers = sum(1 for l in model.layers if l.is_linear)
        attn_layers = num_layers - mamba_layers
        print(f"Model architecture:")
        print(f"  Total layers: {num_layers}")
        print(f"  Mamba layers: {mamba_layers}")
        print(f"  Attention layers: {attn_layers}")
        print()

        # Test cache creation
        caches = model.make_cache()
        print(f"Cache types:")
        for i, cache in enumerate(caches[:8]):  # Show first 8
            cache_type = type(cache).__name__
            print(f"  Layer {i}: {cache_type}")
        if len(caches) > 8:
            print(f"  ... ({len(caches)} total layers)")
        print()

        # Test translation
        source_text = "The quick brown fox jumps over the lazy dog."
        system_prompt = "Translate the following English text to French"
        wait_k = 3

        print("=" * 80)
        print("STREAMING TRANSLATION TEST")
        print("=" * 80)
        print(f"Source: {source_text}")
        print(f"System: {system_prompt}")
        print(f"wait-k: {wait_k}")
        print()
        print("Generated output:")
        print("-" * 80)

        word_count = 0
        for response in stream_generate_streaming_llm(
            model,
            tokenizer,
            source_text,
            wait_k=wait_k,
            system_prompt=system_prompt,
            temp=0.0,  # Greedy for reproducibility
        ):
            if response.word_complete:
                word_count += 1
                print(f"{response.text} ", end="", flush=True)

                # Print progress every 5 words
                if word_count % 5 == 0:
                    print(f"\n[{response.source_words_read} src words read, {response.target_words_generated} tgt words generated]", flush=True)

        print()
        print("-" * 80)
        print(f"\n✓ Test completed successfully!")
        print(f"  Total target words generated: {word_count}")
        print()

    except Exception as e:
        print(f"\n✗ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_cache_types():
    """
    Test that caches are correctly mixed (Mamba + Dual).
    """
    print("=" * 80)
    print("CACHE TYPE TEST")
    print("=" * 80)

    try:
        from mlx_lm.models.qwen3_next_streaming import Model, ModelArgs
        from mlx_lm.models.streaming_cache import DualStreamingCache
        from mlx_lm.models.cache import MambaCache

        # Create a simple model config
        args = ModelArgs(
            model_type="qwen3_next",
            hidden_size=512,
            num_hidden_layers=8,
            intermediate_size=1024,
            num_attention_heads=8,
            linear_num_value_heads=4,
            linear_num_key_heads=4,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            linear_conv_kernel_dim=4,
            num_experts=0,
            num_experts_per_tok=0,
            decoder_sparse_step=1,
            shared_expert_intermediate_size=512,
            mlp_only_layers=[],
            moe_intermediate_size=512,
            rms_norm_eps=1e-6,
            vocab_size=1000,
            num_key_value_heads=4,
            rope_theta=10000.0,
            partial_rotary_factor=0.5,
            max_position_embeddings=2048,
            head_dim=64,
            full_attention_interval=4,  # Layers 0,1,2 Mamba, 3 Attention, repeat
        )

        model = Model(args)
        caches = model.make_cache()

        print(f"Total layers: {len(caches)}")
        print()

        # Check pattern: Mamba, Mamba, Mamba, Attention, repeat
        expected_pattern = []
        for i in range(len(caches)):
            is_linear = (i + 1) % 4 != 0
            expected_type = MambaCache if is_linear else DualStreamingCache
            expected_pattern.append(expected_type.__name__)

            actual_type = type(caches[i])
            match = "✓" if actual_type == expected_type else "✗"
            print(f"  Layer {i}: {actual_type.__name__:20} {match}")

        print()
        print("Expected pattern (full_attention_interval=4):")
        print("  Layers 0,1,2: Mamba")
        print("  Layer 3: Attention")
        print("  Layers 4,5,6: Mamba")
        print("  Layer 7: Attention")
        print()

        # Verify all correct
        all_correct = all(
            type(caches[i]).__name__ == expected_pattern[i]
            for i in range(len(caches))
        )

        if all_correct:
            print("✓ All cache types match expected pattern!")
        else:
            print("✗ Cache types don't match expected pattern")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test Qwen3Next streaming")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.5B-Chat",
        help="Model path (default: Qwen/Qwen3-0.5B-Chat)",
    )
    parser.add_argument(
        "--cache-test-only",
        action="store_true",
        help="Only run cache type test (no model loading)",
    )

    args = parser.parse_args()

    if args.cache_test_only:
        success = test_cache_types()
    else:
        # Run cache test first
        print("\nRunning cache type verification...")
        print()
        cache_success = test_cache_types()
        print()

        # Then full model test
        print("\nRunning full model test...")
        print()
        model_success = test_qwen3next_streaming(args.model)

        success = cache_success and model_success

    print()
    print("=" * 80)
    if success:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 80)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
