from .glide_config import GlideMistralConfig
from .glide_mistral import GlideMistralForCausalLM

from transformers import AutoConfig, AutoModelForCausalLM
AutoConfig.register("liger_mistral_gla", GlideMistralConfig)
AutoModelForCausalLM.register(GlideMistralConfig, GlideMistralForCausalLM)
