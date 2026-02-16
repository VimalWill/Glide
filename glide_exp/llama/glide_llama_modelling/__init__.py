from .glide_config import GlideConfig
from .glide_lama import GlideForCausalLM

from transformers import AutoConfig, AutoModelForCausalLM
AutoConfig.register("glide", GlideConfig)
AutoModelForCausalLM.register(GlideConfig, GlideForCausalLM)