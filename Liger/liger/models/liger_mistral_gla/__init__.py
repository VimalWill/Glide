from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_liger_mistral_gla import LigerMistralGLAConfig
from .modeling_liger_mistral_gla import LigerMistralGLAForCausalLM, LigerMistralGLAModel

AutoConfig.register(LigerMistralGLAConfig.model_type, LigerMistralGLAConfig)
AutoModel.register(LigerMistralGLAConfig, LigerMistralGLAModel)
AutoModelForCausalLM.register(LigerMistralGLAConfig, LigerMistralGLAForCausalLM)


__all__ = ['LigerMistralGLAConfig', 'LigerMistralGLAForCausalLM', 'LigerMistralGLAModel']