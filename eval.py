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


TASKS = [
    "wikitext",
]

# Metrics reported per task (used when printing results)
NORM_TASKS = {"arc_challenge", "hellaswag"}


def _eval_model(model, tokenizer, batch_size: int = 8):
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, max_length=1024)

    with torch.no_grad():
        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=TASKS,
            num_fewshot=None,  # use task defaults (0-shot for most)
            log_samples=False,
        )
    return results


def _print_results(results: dict):
    print("\n" + "=" * 60)
    print(f"{'Task':<20} {'Metric':<20} {'Value':>8}")
    print("=" * 60)

    task_results = results["results"]

    metric_map = {
        "piqa":          ("acc,none",     "acc_norm,none"),
        "arc_easy":      ("acc,none",     "acc_norm,none"),
        "arc_challenge": ("acc_norm,none","acc_norm,none"),
        "hellaswag":     ("acc_norm,none","acc_norm,none"),
        "winogrande":    ("acc,none",     "acc,none"),
        "wikitext":      ("word_perplexity,none", "word_perplexity,none"),
    }

    for task, (primary, _) in metric_map.items():
        if task not in task_results:
            continue
        tr = task_results[task]
        val = tr.get(primary, tr.get("acc,none", "N/A"))
        label = "PPL" if task == "wikitext" else ("acc_norm" if "norm" in primary else "acc")
        if isinstance(val, float):
            print(f"{task:<20} {label:<20} {val:>8.4f}")
        else:
            print(f"{task:<20} {label:<20} {str(val):>8}")

    # MMLU: aggregate over subtasks
    mmlu_accs = [
        v.get("acc,none", v.get("acc_norm,none"))
        for k, v in task_results.items()
        if k.startswith("mmlu") and isinstance(v, dict)
    ]
    if mmlu_accs:
        avg = sum(mmlu_accs) / len(mmlu_accs)
        print(f"{'mmlu (avg)':<20} {'acc':<20} {avg:>8.4f}")

    print("=" * 60)


def _load_and_eval(name: str, path: str, model_cls, out_path: str):
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {name}  ({path})")
    print("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = model_cls.from_pretrained(path, torch_dtype=torch.bfloat16)
    model = model.cuda()
    model.eval()
    results = _eval_model(model, tokenizer)
    _print_results(results)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Full results saved to {out_path}")
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
