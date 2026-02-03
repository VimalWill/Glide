from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_liger_hgrn2 import LigerHGRN2Config
from .modeling_liger_hgrn2 import LigerHGRN2ForCausalLM, LigerHGRN2Model

AutoConfig.register(LigerHGRN2Config.model_type, LigerHGRN2Config)
AutoModel.register(LigerHGRN2Config, LigerHGRN2Model)
AutoModelForCausalLM.register(LigerHGRN2Config, LigerHGRN2ForCausalLM)


__all__ = ['LigerHGRN2Config', 'LigerHGRN2ForCausalLM', 'LigerHGRN2Model']