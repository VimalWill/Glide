from time import time
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import glide_exp.llama.glide_llama_modelling  # triggers AutoModel registration
from glide_exp.llama.glide_llama_modelling import GlideForCausalLM


def set_profiling(model, enabled: bool):
    for layer in model.model.layers:
        layer.self_attn.profiling = enabled


def collect_attn_times(model):
    li = sum(l.self_attn.time_li_attn for l in model.model.layers)
    sa = sum(l.self_attn.time_sa_attn for l in model.model.layers)
    return li * 1e3, sa * 1e3   # seconds -> ms


def decode_latency(model, tokenizer, prompt: str, n_tokens: int):
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    per_token_ms = []
    per_token_li_ms = []
    per_token_sa_ms = []

    with torch.no_grad():
        # prefill full prompt in one shot
        out = model(input_ids, use_cache=True)
        past = out.past_key_values
        next_token = out.logits[:, -1:, :].argmax(dim=-1)

        set_profiling(model, True)
        # decode one token at a time
        for _ in range(n_tokens):
            torch.cuda.synchronize()
            t0 = time()
            out = model(next_token, past_key_values=past, use_cache=True)
            torch.cuda.synchronize()
            per_token_ms.append(round((time() - t0) * 1e3, 3))
            li_ms, sa_ms = collect_attn_times(model)
            per_token_li_ms.append(round(li_ms, 3))
            per_token_sa_ms.append(round(sa_ms, 3))
            past = out.past_key_values
            next_token = out.logits[:, -1:, :].argmax(dim=-1)
        set_profiling(model, False)

    return per_token_ms, per_token_li_ms, per_token_sa_ms


def main():
    glide_path = "/u/vwilliam/Glide/checkpoints/liger/best/"
    tokenizer_path = "meta-llama/Meta-Llama-3-8B"

    print(f"Loading model from {glide_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        glide_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    # override window size on all attention layers
    window_size = 20000
    for layer in model.model.layers:
        layer.self_attn.window_size = window_size
    print(f"Window size set to {window_size} on all layers.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_tokens = 500

    # warmup with a long prompt to compile flex_attention at the target seq length
    warmup_prompt = "The quick brown fox jumps over the lazy dog. " * 2200
    print("Warming up (long prefill + 50 decode steps) ...")
    with torch.no_grad():
        ids = tokenizer(warmup_prompt, return_tensors="pt").input_ids.to("cuda")
        out = model(ids, use_cache=True)
        past = out.past_key_values
        tok = out.logits[:, -1:, :].argmax(dim=-1)
        for _ in range(50):
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
        per_token_ms, per_token_li_ms, per_token_sa_ms = decode_latency(model, tokenizer, prompt, n_tokens)
        avg = round(sum(per_token_ms) / len(per_token_ms), 3)
        avg_li = round(sum(per_token_li_ms) / len(per_token_li_ms), 3)
        avg_sa = round(sum(per_token_sa_ms) / len(per_token_sa_ms), 3)
        print(f"{name}: total={avg} ms/tok  GLA={avg_li}  SWA={avg_sa}")
        results[name] = {
            "total_ms":   per_token_ms,
            "gla_ms":     per_token_li_ms,
            "swa_ms":     per_token_sa_ms,
        }

    with open("prof_model_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to prof_model_results.json")


if __name__ == "__main__":
    main()
