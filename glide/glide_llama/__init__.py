from .modelling_glide_llama import *
from .config import GlideConfig
from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("glide-lolcats", GlideConfig)
AutoModelForCausalLM.register(GlideConfig, GlideModelForCausalLM)