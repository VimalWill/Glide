import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import time
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from Liger.liger.models.liger_gla.modeling_liger_gla import LigerGLAForCausalLM
from Liger.liger.models.liger_gsa import LigerGSAForCausalLM
import os
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda:0'


model_configs = [
    {"name": "Liger-GLA-8B", "path": "linear-moe-hub/Liger-GLA-8B", "cls": LigerGLAForCausalLM},
    {"name": "Liger-GSA-8B", "path": "linear-moe-hub/Liger-GSA-8B", "cls": LigerGSAForCausalLM},
    # {"name": "Falcon-Mamba-7B", "path": "xx/falcon-mamba-7b", "cls": AutoModelForCausalLM},
    # {"name": "LLaMA-3-8B", "path": "xx/Meta-Llama-3-8B", "cls": AutoModelForCausalLM}
]


decode_lengths = [1024, 2048, 4096, 8192, 16384, 32768]
prompt_text = "LLaMa is a powerful model for long context understanding."

all_results = []

for config in model_configs:
    print(f"\n===== Testing {config['name']} =====")

    model = config["cls"].from_pretrained(config["path"], torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(config["path"])

    results = []

    for gen_len in tqdm(decode_lengths, desc=f"Benchmarking {config['name']}"):
        print(f"Generating {gen_len} tokens...")

        input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)


        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


        with torch.no_grad():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=gen_len,
                do_sample=False,
                use_cache=True
            )

            end_event.record()
            torch.cuda.synchronize()
            latency = start_event.elapsed_time(end_event) / 1000.0  

        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / 1024 / 1024 / 1024

        print(f"{config['name']} | Tokens: {gen_len}, Time: {latency:.2f}s, Mem: {max_mem_gb:.2f}GB")
        results.append((config["name"], gen_len, latency, max_mem_gb))

    all_results.extend(results)


df = pd.DataFrame(all_results, columns=["Model", "Decoding Length", "Latency Time (s)", "Memory Usage (GB)"])
print("\n=== Benchmark Results ===")
print(df.to_string(index=False))