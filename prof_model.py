from time import time
import json
import torch
from transformers import AutoTokenizer

import glide_exp.llama.glide_llama_modelling  # triggers AutoModel registration
from glide_exp.llama.glide_llama_modelling import GlideForCausalLM


def decode_latency(model, tokenizer, prompt: str, n_tokens: int):
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    per_token_ms = []

    with torch.no_grad():
        # prefill full prompt in one shot
        out = model(input_ids, use_cache=True)
        past = out.past_key_values
        next_token = out.logits[:, -1:, :].argmax(dim=-1)

        # decode one token at a time
        for i in range(n_tokens):
            torch.cuda.synchronize()
            t0 = time()
            out = model(next_token, past_key_values=past, use_cache=True)
            torch.cuda.synchronize()
            ms = round((time() - t0) * 1e3, 3)
            per_token_ms.append(ms)
            print(f"  token {i+1}/{n_tokens}: {ms} ms", flush=True)
            past = out.past_key_values
            next_token = out.logits[:, -1:, :].argmax(dim=-1)

    return per_token_ms


def main():
    tokenizer_path = "meta-llama/Meta-Llama-3-8B"

    print("Initializing model with random weights ...")
    from glide_exp.llama.glide_llama_modelling.glide_config import GlideConfig
    config = GlideConfig(
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=14336,
        vocab_size=128256,
        max_position_embeddings=131072,
        rope_theta=500000.0,
    )
    model = GlideForCausalLM(config).to(dtype=torch.bfloat16, device="cuda")
    model.eval()

    # override window size on all attention layers
    window_size = 512
    for layer in model.model.layers:
        layer.self_attn.window_size = window_size
    print(f"Window size set to {window_size} on all layers.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_tokens = 500

    # warmup with a long prompt to compile flex_attention at the target seq length
    warmup_prompt = "The quick brown fox jumps over the lazy dog. " * 10
    print("Warming up ...")
    with torch.no_grad():
        ids = tokenizer(warmup_prompt, return_tensors="pt").input_ids.to("cuda")
        out = model(ids, use_cache=True)
        past = out.past_key_values
        tok = out.logits[:, -1:, :].argmax(dim=-1)
        for _ in range(10):
            out = model(tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            tok = out.logits[:, -1:, :].argmax(dim=-1)
    torch.cuda.synchronize()
    print("Warmup done.")

    prompts = {
        "prompt_20k": "The quick brown fox jumps over the lazy dog. " * 2200,
    }

    results = {}
    for name, prompt in prompts.items():
        per_token_ms = decode_latency(model, tokenizer, prompt, n_tokens)
        avg = round(sum(per_token_ms) / len(per_token_ms), 3)
        print(f"{name}: avg={avg} ms/tok  min={min(per_token_ms)}  max={max(per_token_ms)}")
        results[name] = per_token_ms

    with open("prof_model_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to prof_model_results.json")


if __name__ == "__main__":
    main()
