import argparse
import torch
from glide_exp.llama.glide_llama_modelling import GlideForCausalLM, GlideConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

# Stub missing transformers symbols before lm_eval imports them
import transformers as _transformers
if not hasattr(_transformers, "Qwen2AudioForConditionalGeneration"):
    _transformers.Qwen2AudioForConditionalGeneration = type(
        "Qwen2AudioForConditionalGeneration", (), {}
    )

import lm_eval
from lm_eval.models.huggingface import HFLM
import json
import os

EXPR_ID = "31_w/4"
RESULTS_FILE = "experiments.json"

TASKS = [
    "wikitext",
    "piqa",
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "winogrande",
]
TASKS_5SHOT = ["mmlu"]

metric_map = {
    "piqa":          ("acc,none",),
    "arc_easy":      ("acc,none",),
    "arc_challenge": ("acc_norm,none",),
    "hellaswag":     ("acc_norm,none",),
    "winogrande":    ("acc,none",),
    "wikitext":      ("word_perplexity,none",),
}


def _eval_model(model, tokenizer, batch_size: int = 8):
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, max_length=1024)

    with torch.no_grad():
        results = lm_eval.simple_evaluate(
            model=lm, tasks=TASKS, num_fewshot=None, log_samples=False,
        )
        results_5shot = lm_eval.simple_evaluate(
            model=lm, tasks=TASKS_5SHOT, num_fewshot=5, log_samples=False,
        )

    results["results"].update(results_5shot["results"])
    return results


def _extract_metrics(results: dict) -> dict:
    task_results = results["results"]
    out = {}
    for task, (primary,) in metric_map.items():
        if task not in task_results:
            continue
        out[task] = task_results[task].get(primary)
    mmlu_accs = [
        v.get("acc,none", v.get("acc_norm,none"))
        for k, v in task_results.items()
        if k.startswith("mmlu") and isinstance(v, dict)
    ]
    if mmlu_accs:
        out["mmlu"] = sum(mmlu_accs) / len(mmlu_accs)
    return out


def _print_results(metrics: dict):
    print("\n" + "=" * 60)
    print(f"{'Task':<20} {'Value':>8}")
    print("=" * 60)
    for task, val in metrics.items():
        print(f"{task:<20} {val:>8.4f}" if isinstance(val, float) else f"{task:<20} {str(val):>8}")
    print("=" * 60)


def _save_results(expr_id: str, metrics: dict):
    data = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            data = json.load(f)
    data[expr_id] = metrics
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {RESULTS_FILE} under key '{expr_id}'")


def _load_and_eval(name: str, path: str, model_cls, out_path: str):
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {name}  ({path})")
    print("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = model_cls.from_pretrained(path, torch_dtype=torch.bfloat16)
    model = model.cuda()
    model.eval()
    results = _eval_model(model, tokenizer)
    metrics = _extract_metrics(results)
    _print_results(metrics)
    _save_results(name, metrics)
    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--glide-only", action="store_true")
    args = parser.parse_args()

    glide_path = "/u/vwilliam/Glide/checkpoints/liger/best/"
    baseline_path = "meta-llama/Meta-Llama-3-8B"

    if not args.baseline_only:
        _load_and_eval("Glide (liger)", glide_path, GlideForCausalLM, "eval_results_glide.json")

    if not args.glide_only:
        _load_and_eval("Llama-3-8B (baseline)", baseline_path, AutoModelForCausalLM, "eval_results_baseline.json")


if __name__ == "__main__":
    main()
