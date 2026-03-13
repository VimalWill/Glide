from time import time
import json
import torch
from transformers import AutoTokenizer

import glide_exp.llama.glide_llama_modelling  # triggers AutoModel registration
from glide_exp.llama.glide_llama_modelling import GlideForCausalLM


def estimate_e2e_latency(model: GlideForCausalLM, tokenizer: AutoTokenizer, prompt: str, n_tokens: int):
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    # per_token_ms: one entry per token (prefill tokens + decode tokens)
    per_token_ms = []
    past = None

    with torch.no_grad():
        # prefill — one token at a time to observe per-token latency ramp
        for i in range(prompt_len):
            tok = input_ids[:, i:i+1]
            torch.cuda.synchronize()
            t0 = time()
            out = model(tok, past_key_values=past, use_cache=True)
            torch.cuda.synchronize()
            per_token_ms.append(round((time() - t0) * 1e3, 3))
            past = out.past_key_values

        next_token = out.logits[:, -1:, :].argmax(dim=-1)

        # decode — one token at a time
        for _ in range(n_tokens):
            torch.cuda.synchronize()
            t0 = time()
            out = model(next_token, past_key_values=past, use_cache=True)
            torch.cuda.synchronize()
            per_token_ms.append(round((time() - t0) * 1e3, 3))
            past = out.past_key_values
            next_token = out.logits[:, -1:, :].argmax(dim=-1)

    prefill_ms = sum(per_token_ms[:prompt_len])
    decode_ms  = sum(per_token_ms[prompt_len:])
    return {
        "prompt_tokens":       prompt_len,
        "generated_tokens":    n_tokens,
        "per_token_ms":        per_token_ms,          # all prompt_len + n_tokens values
        "prefill_total_ms":    round(prefill_ms, 3),
        "decode_avg_ms":       round(decode_ms / n_tokens, 3),
        "total_ms":            round(prefill_ms + decode_ms, 3),
        "throughput_tok_s":    round(n_tokens / (decode_ms / 1e3), 2),
    }


def parse_prof_averages(prof):
    """Extract CUDA time (ms) for each record_function label."""
    out = {}
    events = prof.key_averages()
    # detect correct attribute name
    sample = events[0] if events else None
    cuda_attr = None
    for attr in ("self_cuda_time_total", "cuda_time_total", "device_time_total", "self_cpu_time_total"):
        if sample is not None and hasattr(sample, attr):
            cuda_attr = attr
            break
    print(f"[profiler] using attribute: {cuda_attr}")
    for evt in events:
        if evt.key.startswith("#") or not evt.key[0].isalpha():
            continue
        out[evt.key] = round(getattr(evt, cuda_attr, 0) / 1e3, 3)  # us -> ms
    return out


def main():
    glide_path = "/u/vwilliam/Glide/checkpoints/liger/best/"
    tokenizer_path = "meta-llama/Meta-Llama-3-8B"

    print(f"Loading model from {glide_path} ...")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        glide_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = [
        "The quick brown fox jumps over the lazy dog. " * 10,  # ~100 tokens
        "The quick brown fox jumps over the lazy dog. " * 50,  # ~500 tokens
    ]
    n_tokens = 500
    warmup_prompt = "Hello world. " * 10

    # warmup (profiler not active)
    print("Warming up ...")
    with torch.no_grad():
        ids = tokenizer(warmup_prompt, return_tensors="pt").input_ids.to("cuda")
        model(ids, use_cache=True)

    results = []
    for prompt in prompts:
        row = estimate_e2e_latency(model, tokenizer, prompt, n_tokens)
        print(json.dumps({k: v for k, v in row.items() if k != "per_token_ms"}, indent=2))
        results.append(row)

    with open("prof_model_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} entries to prof_model_results.json")


if __name__ == "__main__":
    main()