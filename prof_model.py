from time import time
import json
import torch
from torch.profiler import profile, ProfilerActivity
from transformers import AutoTokenizer

import glide_exp.llama.glide_llama_modelling  # triggers AutoModel registration
from glide_exp.llama.glide_llama_modelling import GlideForCausalLM


def estimate_e2e_latency(model: GlideForCausalLM, tokenizer: AutoTokenizer, prompt: str, n_tokens: int):
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        # prefill
        torch.cuda.synchronize()
        t_prefill_start = time()
        out = model(input_ids, use_cache=True)
        torch.cuda.synchronize()
        t_prefill_end = time()

        past = out.past_key_values
        next_token = out.logits[:, -1:, :].argmax(dim=-1)

        # decode n_tokens one at a time
        torch.cuda.synchronize()
        t_decode_start = time()
        for _ in range(n_tokens):
            out = model(next_token, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_token = out.logits[:, -1:, :].argmax(dim=-1)
        torch.cuda.synchronize()
        t_decode_end = time()

    prefill_ms = (t_prefill_end - t_prefill_start) * 1e3
    decode_ms  = (t_decode_end  - t_decode_start)  * 1e3
    total_ms   = prefill_ms + decode_ms

    return {
        "prompt_tokens":       prompt_len,
        "generated_tokens":    n_tokens,
        "prefill_ms":          round(prefill_ms, 3),
        "decode_total_ms":     round(decode_ms, 3),
        "decode_per_token_ms": round(decode_ms / n_tokens, 3),
        "total_ms":            round(total_ms, 3),
        "throughput_tok_s":    round(n_tokens / (decode_ms / 1e3), 2),
    }


def parse_prof_averages(prof):
    """Extract cuda_time_total (ms) for each record_function label."""
    out = {}
    for evt in prof.key_averages():
        if evt.key.startswith("#") or not evt.key[0].isalpha():
            continue
        out[evt.key] = round(evt.self_cuda_time_total / 1e3, 3)  # us -> ms
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
        "The quick brown fox jumps over the lazy dog. " * 20,   # ~200 tokens
        "The quick brown fox jumps over the lazy dog. " * 100,  # ~1000 tokens
    ]
    n_tokens = 50
    warmup_prompt = "Hello world. " * 10

    # warmup (profiler not active)
    print("Warming up ...")
    with torch.no_grad():
        ids = tokenizer(warmup_prompt, return_tensors="pt").input_ids.to("cuda")
        model(ids, use_cache=True)

    results = []
    for prompt in prompts:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            row = estimate_e2e_latency(model, tokenizer, prompt, n_tokens)
        row["prof_breakdown_ms"] = parse_prof_averages(prof)
        print(json.dumps(row, indent=2))
        results.append(row)

    with open("prof_model_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} entries to prof_model_results.json")


if __name__ == "__main__":
    main()