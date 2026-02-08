import torch
from lolcats.models import LolcatsConfig, LolcatsModelForCausalLM

def test_lolcats():
    config = LolcatsConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        # linear attention settings
        feature_map="elu",
        attn_mode="fused_chunk",
        expand_k=1,
        expand_v=1,
    )

    model = LolcatsModelForCausalLM(config)
    model.eval()

    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)

    print(f"Logits shape: {outputs.logits.shape}")
    print(f"Past KV type: {type(outputs.past_key_values)}")

    assert outputs.logits.shape == (batch_size, seq_len, config.vocab_size), \
        f"Expected logits shape {(batch_size, seq_len, config.vocab_size)}, got {outputs.logits.shape}"

    # Test with output_attentions to exercise hybrid attention path
    with torch.no_grad():
        outputs_attn = model(input_ids=input_ids, output_attentions=True, use_cache=True)

    print(f"Attentions returned: {len(outputs_attn.attentions)} layers")
    print(f"Logits shape (attn mode): {outputs_attn.logits.shape}")

    # Test autoregressive decoding (single token with past KV)
    past_kv = outputs.past_key_values
    next_token = torch.randint(0, config.vocab_size, (batch_size, 1))

    with torch.no_grad():
        decode_out = model(input_ids=next_token, past_key_values=past_kv, use_cache=True)

    print(f"Decode logits shape: {decode_out.logits.shape}")
    assert decode_out.logits.shape == (batch_size, 1, config.vocab_size)

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_lolcats()
