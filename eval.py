from glide_exp.llama.glide_llama_modelling import GlideForCausalLM, GlideConfig
from transformers import AutoTokenizer

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
    "piqa",
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "winogrande",
    "mmlu",
    "wikitext",
]

# Metrics reported per task (used when printing results)
NORM_TASKS = {"arc_challenge", "hellaswag"}


def _eval_model(model: GlideForCausalLM, tokenizer, batch_size: int = 8):
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)

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


def main():
    path = "/u/vwilliam/Glide/checkpoints/liger/best/"

    print(f"Loading model from {path} ...")
    model = GlideForCausalLM.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = model.cuda()
    model.eval()

    results = _eval_model(model, tokenizer)

    _print_results(results)

    # Also dump full JSON for archiving
    out_path = "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
