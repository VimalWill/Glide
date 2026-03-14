import csv
import os
import time
import torch
from transformers import AutoTokenizer
from glide_exp.llama.glide_llama_modelling import GlideForCausalLM

DEFAULT_PROMPT = (
    "The history of artificial intelligence dates back to antiquity, "
    "with myths, stories, and rumors of artificial beings endowed with "
    "intelligence or consciousness by master craftsmen. The seeds of modern AI "
    "were planted by classical philosophers who attempted to describe the process "
    "of human thinking as the mechanical manipulation of symbols."
)


def set_window_size(model: GlideForCausalLM, window_size: int) -> None:
    """Override window_size on every attention layer at runtime."""
    for layer in model.model.layers:
        layer.self_attn.window_size = window_size


def estimate_decode_stage_per_token_latency(
    model: GlideForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str = DEFAULT_PROMPT,
    n_tokens: int = 500,
    warmup: int = 5,
    window_size: int | None = None,
) -> list[float]:
    """
    Returns per-token latency (ms) for each of the `n_tokens` decode steps.

    Procedure:
      1. Optionally override window_size on all attention layers.
      2. Tokenize `prompt` and run a prefill pass to build the KV cache.
      3. Decode one token at a time, measuring each step individually.
      4. Return list of latencies (warmup steps discarded).
    """
    model.eval()
    device = next(model.parameters()).device
    use_cuda = device.type == "cuda"

    if window_size is not None:
        set_window_size(model, window_size)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        # --- Prefill ---
        outputs = model(input_ids=input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1:, :].argmax(dim=-1)

        # --- Decode loop ---
        latencies = []

        for step in range(warmup + n_tokens):
            if use_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start_event.record()
            else:
                t0 = time.perf_counter()

            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )

            if use_cuda:
                end_event.record()
                torch.cuda.synchronize()
                elapsed_ms = start_event.elapsed_time(end_event)
            else:
                elapsed_ms = (time.perf_counter() - t0) * 1e3

            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1:, :].argmax(dim=-1)

            if step >= warmup:
                latencies.append(elapsed_ms)
                print(f"\r  token {len(latencies)}/{n_tokens}  {elapsed_ms:.1f} ms", end="", flush=True)

        print()

    return latencies


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--window-size", type=int, default=None,
                        help="Override attention window size (e.g. 20000)")
    parser.add_argument("--n-tokens", type=int, default=500)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--out", type=str, default="latency_results.csv")
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = GlideForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16)
    model = model.cuda().eval()

    ws = args.window_size or "model_default"
    write_header = not os.path.exists(args.out)

    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "token_idx", "latency_ms"])
        if write_header:
            writer.writeheader()

        for r in range(args.rounds):
            print(f"\n--- Round {r+1}/{args.rounds} | window={ws} ---")
            latencies = estimate_decode_stage_per_token_latency(
                model, tokenizer,
                prompt=args.prompt,
                n_tokens=args.n_tokens,
                window_size=args.window_size,
            )
            mean_ms = sum(latencies) / len(latencies)
            print(f"Mean: {mean_ms:.2f} ms/tok  ({1000/mean_ms:.1f} tok/s)")
            for i, lat in enumerate(latencies):
                writer.writerow({"round": f"round {r+1}", "token_idx": i, "latency_ms": round(lat, 4)})

    print(f"\nSaved to {args.out}")
