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
) -> float:
    """
    Estimates per-token latency (ms) for the decode (autoregressive) stage.

    Procedure:
      1. Optionally override window_size on all attention layers.
      2. Tokenize `prompt` and run a prefill pass to build the KV cache.
      3. Decode one token at a time for `n_tokens` steps, measuring wall-clock
         time using CUDA events (GPU) or time.perf_counter (CPU).
      4. Return mean latency in milliseconds, excluding `warmup` steps.

    Args:
        model:       GlideForCausalLM instance (on device, eval mode).
        tokenizer:   Matching tokenizer for the model.
        prompt:      Prefill text (tokenized internally).
        n_tokens:    Number of decode steps to measure.
        warmup:      Initial decode steps to discard (Triton autotuner, etc.).
        window_size: If set, overrides window_size on all attention layers.

    Returns:
        Mean per-token decode latency in milliseconds.
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

    return sum(latencies) / len(latencies)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--window-size", type=int, default=None,
                        help="Override attention window size (e.g. 20000)")
    parser.add_argument("--n-tokens", type=int, default=200)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = GlideForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16)
    model = model.cuda().eval()

    latency = estimate_decode_stage_per_token_latency(
        model, tokenizer,
        prompt=args.prompt,
        n_tokens=args.n_tokens,
        window_size=args.window_size,
    )
    ws = args.window_size or "model default"
    print(f"Window size : {ws}")
    print(f"Decode latency: {latency:.2f} ms/tok  ({1000/latency:.1f} tok/s)")

    csv_path = "latency_results.csv"
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint", "window_size", "n_tokens", "latency_ms", "throughput_tok_s"])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "checkpoint": args.checkpoint,
            "window_size": ws,
            "n_tokens": args.n_tokens,
            "latency_ms": round(latency, 4),
            "throughput_tok_s": round(1000 / latency, 2),
        })
    print(f"Saved to {csv_path}")
