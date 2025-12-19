#!/usr/bin/env python3
"""Test OpenRouter integration with ShinkaEvolve."""

from shinka.llm.query import query

def test_openrouter():
    """Test a simple query to OpenRouter using the free Llama model."""
    print("Testing OpenRouter integration...")
    print("-" * 50)

    model_name = "meta-llama/llama-3.3-70b-instruct:free"
    msg = "What is the meaning of life? Please answer in one sentence."
    system_msg = "You are a helpful assistant."

    print(f"Model: {model_name}")
    print(f"User message: {msg}")
    print("-" * 50)

    try:
        result = query(
            model_name=model_name,
            msg=msg,
            system_msg=system_msg,
            msg_history=[],
            temperature=0.7,
            max_tokens=100,
        )

        print(f"\nResponse: {result.content}")
        print(f"\nTokens used:")
        print(f"  Input: {result.input_tokens}")
        print(f"  Output: {result.output_tokens}")
        print(f"  Total: {result.input_tokens + result.output_tokens}")
        print(f"\nCost: ${result.cost:.6f}")
        print("\n✓ OpenRouter integration test passed!")

    except Exception as e:
        print(f"\n✗ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    test_openrouter()
