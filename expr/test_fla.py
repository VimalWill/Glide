from fla.ops.gla import fused_chunk_gla
from fla.ops.gla import fused_recurrent_gla

import torch

def test_fla_ops():
    """Test FLA GLA operations with both chunk and recurrent modes."""
    # Setup test parameters
    batch_size = 2
    num_heads = 4
    seq_len = 16
    head_dim = 64

    # Create input tensors (b, h, n, d) format
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
    g = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)

    # Apply normalizations (as done in the model)
    q = torch.softmax(q, dim=-1)
    k = torch.softmax(k, dim=-1)
    g = torch.nn.functional.logsigmoid(g) / 16.0

    # Ensure contiguous
    q, k, v, g = (x.contiguous() for x in (q, k, v, g))

    scale = 1.0

    # Test fused_chunk_gla
    print("Testing fused_chunk_gla...")
    o_chunk, state_chunk = fused_chunk_gla(
        q, k, v, g,
        scale=scale,
        initial_state=None,
        output_final_state=True
    )

    assert o_chunk.shape == (batch_size, num_heads, seq_len, head_dim), \
        f"Expected shape {(batch_size, num_heads, seq_len, head_dim)}, got {o_chunk.shape}"
    assert state_chunk is not None, "Recurrent state should not be None"
    print(f"  ✓ Output shape: {o_chunk.shape}")
    print(f"  ✓ State shape: {state_chunk.shape}")

    # Test fused_recurrent_gla (single step)
    print("\nTesting fused_recurrent_gla...")
    q_single = q[:, :, :1, :]  # Single timestep
    k_single = k[:, :, :1, :]
    v_single = v[:, :, :1, :]
    g_single = g[:, :, :1, :]

    o_recurrent, state_recurrent = fused_recurrent_gla(
        q_single, k_single, v_single, g_single,
        scale=scale,
        initial_state=None,
        output_final_state=True
    )

    assert o_recurrent.shape == (batch_size, num_heads, 1, head_dim), \
        f"Expected shape {(batch_size, num_heads, 1, head_dim)}, got {o_recurrent.shape}"
    assert state_recurrent is not None, "Recurrent state should not be None"
    print(f"  ✓ Output shape: {o_recurrent.shape}")
    print(f"  ✓ State shape: {state_recurrent.shape}")

    # Test recurrent mode with state passing
    print("\nTesting recurrent with state passing...")
    state = None
    outputs = []
    for t in range(seq_len):
        q_t = q[:, :, t:t+1, :]
        k_t = k[:, :, t:t+1, :]
        v_t = v[:, :, t:t+1, :]
        g_t = g[:, :, t:t+1, :]

        o_t, state = fused_recurrent_gla(
            q_t, k_t, v_t, g_t,
            scale=scale,
            initial_state=state,
            output_final_state=True
        )
        outputs.append(o_t)

    o_recurrent_full = torch.cat(outputs, dim=2)
    print(f"  ✓ Recurrent full sequence shape: {o_recurrent_full.shape}")

    # Compare chunk vs recurrent (they should be close but may differ slightly)
    diff = torch.abs(o_chunk - o_recurrent_full).mean()
    print(f"\nMean absolute difference (chunk vs recurrent): {diff.item():.6f}")

    print("\n✅ All tests passed!")

    return o_chunk, o_recurrent_full


if __name__ == "__main__":
    test_fla_ops()