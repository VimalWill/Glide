import torch
from transformers import AutoConfig, AutoTokenizer

from glide_exp.llama import GlideForCausalLM
from glide_exp.llama import GlideConfig

import lm_eval
from lm_eval.models.huggingface import HFLM


def test_lm_eval():
    # Load LLaMA config from HuggingFace and create GlideConfig
    pretrained_path = "meta-llama/Llama-3.1-8B"
    base_config = AutoConfig.from_pretrained(pretrained_path)
    config = GlideConfig()
    config.__dict__.update(base_config.__dict__)

    print("Loading model...")
    model = GlideForCausalLM.from_pretrained(
        pretrained_path, config=config, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    model.eval()

    # Wrap in lm-eval's HFLM
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size="auto")

    # Run evaluation
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["hellaswag", "wikitext"],
        batch_size="auto",
    )

    # Print results
    for task, metrics in results["results"].items():
        print(f"\n{task}:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    test_lm_eval()
