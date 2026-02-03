from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_liger_gla import LigerGLAConfig
from .modeling_liger_gla import LigerGLAForCausalLM, LigerGLAModel

AutoConfig.register(LigerGLAConfig.model_type, LigerGLAConfig)
AutoModel.register(LigerGLAConfig, LigerGLAModel)
AutoModelForCausalLM.register(LigerGLAConfig, LigerGLAForCausalLM)


__all__ = ['LigerGLAConfig', 'LigerGLAForCausalLM', 'LigerGLAModel']