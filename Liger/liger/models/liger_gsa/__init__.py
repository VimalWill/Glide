from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_liger_gsa import LigerGSAConfig
from .modeling_liger_gsa import LigerGSAForCausalLM, LigerGSAModel

AutoConfig.register(LigerGSAConfig.model_type, LigerGSAConfig)
AutoModel.register(LigerGSAConfig, LigerGSAModel)
AutoModelForCausalLM.register(LigerGSAConfig, LigerGSAForCausalLM)


__all__ = ['LigerGSAConfig', 'LigerGSAForCausalLM', 'LigerGSAModel']