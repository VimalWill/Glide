from glide.glide_llama import GlideModelForCausalLM
from glide.glide_llama.config import GlideConfig
import torch 

def main():
    config = GlideConfig()

    with torch.device('cuda')
        model = GlideModelForCausalLM(config).to(dtype=torch.bfloat16)

    print(model)

if __name__ == "__main__":
    main()