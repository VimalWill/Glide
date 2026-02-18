import torch
from transformers import AutoTokenizer
from glide_exp.llama.glide_llama_modelling import GlideForCausalLM

PATH = "/u/vwilliam/Glide/checkpoints/liger/best/"

print("Loading model in bfloat16 (same as eval.py)...")
model = GlideForCausalLM.from_pretrained(PATH, torch_dtype=torch.bfloat16).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained(PATH)
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
print(f"dtype: {next(model.parameters()).dtype}")

# --- test 1: various seq lengths ---
for seq_len in [7, 63, 65, 128, 256, 512, 1024, 2048]:
    ids = torch.randint(0, model.config.vocab_size, (1, seq_len), device="cuda")
    with torch.no_grad():
        out = model(ids)
    print(f"seq_len={seq_len:5d}  logits={tuple(out.logits.shape)}  ok")

# --- test 2: batch > 1 (mimics lm_eval batching) ---
ids = torch.randint(0, model.config.vocab_size, (4, 128), device="cuda")
with torch.no_grad():
    out = model(ids)
print(f"batch=4, seq_len=128  logits={tuple(out.logits.shape)}  ok")

# --- test 3: with attention mask (lm_eval passes this for padded batches) ---
ids = torch.randint(0, model.config.vocab_size, (4, 128), device="cuda")
mask = torch.ones(4, 128, device="cuda")
mask[0, :10] = 0  # simulate left-padding
with torch.no_grad():
    out = model(ids, attention_mask=mask)
print(f"batch=4, seq_len=128, attention_mask=ok  logits={tuple(out.logits.shape)}  ok")

print("\nAll forward passes passed.")
