from glide.glide_llama import GlideModelForCausalLM
from glide.glide_llama.config import GlideConfig

def main():
    config = GlideConfig()
    model = GlideModelForCausalLM(config)

    print(model)

if __name__ == "__main__":
    main()