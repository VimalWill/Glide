import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from glide_exp.llama.glide_llama_modelling import GlideForCausalLM, GlideConfig
import os
from tqdm import tqdm
from typing import Dict, Optional

liger_configuration: Optional[Dict] = {
            0:  {"window_size": 64},
            1:  {"window_size": 64},
            2:  {"window_size": 64},
            3:  {"window_size": 64},
            4:  {"window_size": 64},
            5:  {"window_size": 64},
            6:  {"window_size": 64},
            7:  {"window_size": 64},
            8:  {"window_size": 64},
            9:  {"window_size": 64},
            10: {"window_size": 64},
            11: {"window_size": 64},
            12: {"window_size": 64},
            13: {"window_size": 64},
            14: {"window_size": 64},
            15: {"window_size": 64},
            16: {"window_size": 64},
            17: {"window_size": 64},
            18: {"window_size": 64},
            19: {"window_size": 64},
            20: {"window_size": 64},
            21: {"window_size": 64},
            22: {"window_size": 64},
            23: {"window_size": 64},
            24: {"window_size": 64},
            25: {"window_size": 64},
            26: {"window_size": 64},
            27: {"window_size": 64},
            28: {"window_size": 64},
            29: {"window_size": 64},
            30: {"window_size": 64},
            31: {"window_size": 64},
        },

Glide_w_4_configuration: Optional[Dict] = {
            0:  {"window_size": 48},
            1:  {"window_size": 48},
            2:  {"window_size": 48},
            3:  {"window_size": 48},
            4:  {"window_size": 48},
            5:  {"window_size": 48},
            6:  {"window_size": 48},
            7:  {"window_size": 48},
            8:  {"window_size": 48},
            9:  {"window_size": 48},
            10: {"window_size": 48},
            11: {"window_size": 48},
            12: {"window_size": 48},
            13: {"window_size": 48},
            14: {"window_size": 48},
            15: {"window_size": 48},
            16: {"window_size": 48},
            17: {"window_size": 48},
            18: {"window_size": 48},
            19: {"window_size": 48},
            20: {"window_size": 48},
            21: {"window_size": 48},
            22: {"window_size": 48},
            23: {"window_size": 48},
            24: {"window_size": 48},
            25: {"window_size": 48},
            26: {"window_size": 48},
            27: {"window_size": 48},
            28: {"window_size": 48},
            29: {"window_size": 48},
            30: {"window_size": 48},
            31: {"window_size": 48},
        },

Glide_w_2_configuration: Optional[Dict] = {
            0:  {"window_size": 32},
            1:  {"window_size": 32},
            2:  {"window_size": 32},
            3:  {"window_size": 32},
            4:  {"window_size": 32},
            5:  {"window_size": 32},
            6:  {"window_size": 32},
            7:  {"window_size": 32},
            8:  {"window_size": 32},
            9:  {"window_size": 32},
            10: {"window_size": 32},
            11: {"window_size": 32},
            12: {"window_size": 32},
            13: {"window_size": 32},
            14: {"window_size": 32},
            15: {"window_size": 32},
            16: {"window_size": 32},
            17: {"window_size": 32},
            18: {"window_size": 32},
            19: {"window_size": 32},
            20: {"window_size": 32},
            21: {"window_size": 32},
            22: {"window_size": 32},
            23: {"window_size": 32},
            24: {"window_size": 32},
            25: {"window_size": 32},
            26: {"window_size": 32},
            27: {"window_size": 32},
            28: {"window_size": 32},
            29: {"window_size": 32},
            30: {"window_size": 32},
            31: {"window_size": 32},
        },

Glide_3w_4_configuration: Optional[Dict] = {
            0:  {"window_size": 16},
            1:  {"window_size": 16},
            2:  {"window_size": 16},
            3:  {"window_size": 16},
            4:  {"window_size": 16},
            5:  {"window_size": 16},
            6:  {"window_size": 16},
            7:  {"window_size": 16},
            8:  {"window_size": 16},
            9:  {"window_size": 16},
            10: {"window_size": 16},
            11: {"window_size": 16},
            12: {"window_size": 16},
            13: {"window_size": 16},
            14: {"window_size": 16},
            15: {"window_size": 16},
            16: {"window_size": 16},
            17: {"window_size": 16},
            18: {"window_size": 16},
            19: {"window_size": 16},
            20: {"window_size": 16},
            21: {"window_size": 16},
            22: {"window_size": 16},
            23: {"window_size": 16},
            24: {"window_size": 16},
            25: {"window_size": 16},
            26: {"window_size": 16},
            27: {"window_size": 16},
            28: {"window_size": 16},
            29: {"window_size": 16},
            30: {"window_size": 16},
            31: {"window_size": 16},
        },

N_RUNS = 5
EXPIR_ID = "window_0"
file_name = f"eff_results_{EXPIR_ID}.csv"
print(f"Running efficiency benchmark, results will be saved to {file_name}...")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda:0'

# Llama-3-8B architecture constants for memory estimation
_N_LAYERS   = 32
_N_KV_HEADS = 8    # GQA: 8 KV heads
_HEAD_DIM   = 128
_BYTES      = 2    # bfloat16

def estimate_softmax_attn_memory_bytes(layer_config, seq_len):
    """
    KV cache memory (bytes, bfloat16, K+V) for softmax attention during decoding.
      - Full attention  : linear  → n_layers * n_kv_heads * seq_len * head_dim * 2 (K+V)
      - Window attention: constant → n_layers * n_kv_heads * window_size * head_dim * 2 (K+V)
    layer_config=None signals full (standard) softmax attention.
    """
    if layer_config is None:
        return _N_LAYERS * _N_KV_HEADS * seq_len * _HEAD_DIM * _BYTES * 2
    total = 0
    for cfg in layer_config.values():
        window = cfg["window_size"]
        total += _N_KV_HEADS * window * _HEAD_DIM * _BYTES * 2
    return total


model_configs = [
    {"name": "liger-Llama-8b",     "path": "/u/vwilliam/Glide/checkpoints/liger/best/", "cls": GlideForCausalLM,      "layer_config": liger_configuration},
    {"name": "Glide-Llama-8b-w48", "path": "/u/vwilliam/Glide/checkpoints/liger/best/", "cls": GlideForCausalLM,      "layer_config": Glide_w_4_configuration},
    {"name": "Glide-Llama-8b-w32", "path": "/u/vwilliam/Glide/checkpoints/liger/best/", "cls": GlideForCausalLM,      "layer_config": Glide_w_2_configuration},
    {"name": "Glide-Llama-8b-w16", "path": "/u/vwilliam/Glide/checkpoints/liger/best/", "cls": GlideForCausalLM,      "layer_config": Glide_3w_4_configuration},
    {"name": "Llama-3-8b",         "path": "meta-llama/Meta-Llama-3-8B",                "cls": AutoModelForCausalLM,  "layer_config": None},
]


decode_lengths = [1024, 2048, 4096, 8192, 16384, 32768]
prompt_text = "LLaMa is a powerful model for long context understanding."

all_results = []

for config in model_configs:
    print(f"\n===== Testing {config['name']} =====")
    model = config["cls"].from_pretrained(config["path"], torch_dtype=torch.bfloat16).to(device).eval()
    if config["layer_config"] is not None:
        for layer_idx, layer_cfg in config["layer_config"].items():
            attn = model.model.layers[layer_idx].self_attn
            if hasattr(attn, "window_size"):
                attn.window_size = layer_cfg["window_size"]
    tokenizer = AutoTokenizer.from_pretrained(config["path"])

    results = []

    # warmup
    input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
    prompt_len = input_ids.shape[1]
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
        softmax_attn_bytes = estimate_softmax_attn_memory_bytes(config["layer_config"], prompt_len + gen_len)

        print(f"{config['name']} | Tokens: {gen_len}, Time: {latency_mean:.2f}±{latency_std:.3f}s, TPS: {tps_mean:.1f}, Mem: {max_mem_gb:.2f}GB, SoftmaxAttnMem: {softmax_attn_bytes/1024**3:.4f}GB")
        results.append((config["name"], gen_len, latency_mean, latency_std, tps_mean, max_mem_gb, softmax_attn_bytes))

    all_results.extend(results)


df = pd.DataFrame(all_results, columns=["Model", "Decoding Length", "Latency (s)", "Latency Std", "Tokens/sec", "Memory (GB)", "Softmax Attn Memory (bytes)"])
print("\n=== Benchmark Results ===")
print(df.to_string(index=False))
df.to_csv(file_name, index=False)
print(f"\nSaved to {file_name}")