from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_liger_qwen3_moe_gla import LigerQwen3MoeGLAConfig
from .modeling_liger_qwen3_moe_gla import LigerQwen3MoeGLAForCausalLM, LigerQwen3MoeGLAModel

AutoConfig.register(LigerQwen3MoeGLAConfig.model_type, LigerQwen3MoeGLAConfig)
AutoModel.register(LigerQwen3MoeGLAConfig, LigerQwen3MoeGLAModel)
AutoModelForCausalLM.register(LigerQwen3MoeGLAConfig, LigerQwen3MoeGLAForCausalLM)


__all__ = ['LigerQwen3MoeGLAConfig', 'LigerQwen3MoeGLAForCausalLM', 'LigerQwen3MoeGLAModel']