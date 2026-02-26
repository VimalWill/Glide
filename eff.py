import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from glide_exp.llama.glide_llama_modelling import GlideForCausalLM
import os
from tqdm import tqdm

N_RUNS = 5

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda:0'


model_configs = [
    {"name": "Glide-Llama-8b", "path": "/u/vwilliam/Glide/checkpoints/liger/best/", "cls": GlideForCausalLM},
]


decode_lengths = [1024, 2048, 4096, 8192, 16384, 32768]
prompt_text = "LLaMa is a powerful model for long context understanding."

all_results = []

for config in model_configs:
    print(f"\n===== Testing {config['name']} =====")

    model = config["cls"].from_pretrained(config["path"], torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(config["path"])

    results = []

    # warmup
    input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        model.generate(input_ids, max_new_tokens=10, do_sample=False)
    torch.cuda.synchronize()

    for gen_len in tqdm(decode_lengths, desc=f"Benchmarking {config['name']}"):
        print(f"Generating {gen_len} tokens...")

        input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


        run_latencies = []
        for _ in range(N_RUNS):
            with torch.no_grad():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

                model.generate(
                    input_ids=input_ids,
                    max_new_tokens=gen_len,
                    do_sample=False,
                    use_cache=True
                )

                end_event.record()
                torch.cuda.synchronize()
                run_latencies.append(start_event.elapsed_time(end_event) / 1000.0)

        latency_mean = np.mean(run_latencies)
        latency_std  = np.std(run_latencies)
        max_mem_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
        tps_mean = gen_len / latency_mean

        print(f"{config['name']} | Tokens: {gen_len}, Time: {latency_mean:.2f}±{latency_std:.3f}s, TPS: {tps_mean:.1f}, Mem: {max_mem_gb:.2f}GB")
        results.append((config["name"], gen_len, latency_mean, latency_std, tps_mean, max_mem_gb))

    all_results.extend(results)


df = pd.DataFrame(all_results, columns=["Model", "Decoding Length", "Latency (s)", "Latency Std", "Tokens/sec", "Memory (GB)"])
print("\n=== Benchmark Results ===")
print(df.to_string(index=False))
df.to_csv("eff_results.csv", index=False)
print("\nSaved to eff_results.csv")