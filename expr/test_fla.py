import torch
from fla.ops.gla.naive import naive_recurrent_gla

def main():
    # Test naive_recurrent_gla with sample inputs (CPU-compatible)
    batch_size = 2
    seq_len = 128
    hidden_dim = 64
    num_heads = 4

    # Create random input tensors
    q = torch.randn(batch_size, num_heads, seq_len, hidden_dim)
    k = torch.randn(batch_size, num_heads, seq_len, hidden_dim)
    v = torch.randn(batch_size, num_heads, seq_len, hidden_dim)
    gk = torch.randn(batch_size, num_heads, seq_len, hidden_dim)  # gating for keys

    print(f"Testing naive_recurrent_gla (CPU-compatible implementation)")
    print(f"\nInput shapes:")
    print(f"  q: {q.shape}")
    print(f"  k: {k.shape}")
    print(f"  v: {v.shape}")
    print(f"  gk: {gk.shape}")

    # Run naive_recurrent_gla
    output, final_state = naive_recurrent_gla(q, k, v, gk, output_final_state=True)

    print(f"\nOutput shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output device: {output.device}")
    if final_state is not None:
        print(f"Final state shape: {final_state.shape}")
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()
