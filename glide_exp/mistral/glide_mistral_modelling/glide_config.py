# -*- coding: utf-8 -*-

from typing import Dict, Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.models.mistral.configuration_mistral import MistralConfig


logger = logging.get_logger(__name__)

class GlideMistralConfig(MistralConfig, PretrainedConfig):
    model_type = "liger_mistral_gla"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=None,
        hidden_act="silu",
        max_position_embeddings=4096 * 32,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        sliding_window=4096,
        attention_dropout=0.0,
        # linear attention
        expand_k: int = 1,
        expand_v: int = 1,
        hidden_ratio: Optional[int] = 4,
        global_window_size = 64, 
        layer_delta_configuration: Optional[Dict] = {
            0:  {"window_size": 64},
            1:  {"window_size": 64},
            2:  {"window_size": 64},
            3:  {"window_size": 64},
            4:  {"window_size": 64},
            5:  {"window_size": 64},
            6:  {"window_size": 64},
            7:  {"window_size": 64},
            8:  {"window_size": 64},
            9:  {"window_size": 64},
            10: {"window_size": 64},
            11: {"window_size": 64},
            12: {"window_size": 64},
            13: {"window_size": 64},
            14: {"window_size": 64},
            15: {"window_size": 64},
            16: {"window_size": 64},
            17: {"window_size": 64},
            18: {"window_size": 64},
            19: {"window_size": 64},
            20: {"window_size": 64},
            21: {"window_size": 64},
            22: {"window_size": 64},
            23: {"window_size": 64},
            24: {"window_size": 64},
            25: {"window_size": 64},
            26: {"window_size": 64},
            27: {"window_size": 64},
            28: {"window_size": 64},
            29: {"window_size": 64},
            30: {"window_size": 64},
            31: {"window_size": 64},
        },
        **kwargs,
    ):
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.hidden_ratio = hidden_ratio

        self.global_window_size = global_window_size
        self.layer_delta_configuration = layer_delta_configuration or {}

        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            sliding_window=sliding_window,
            attention_dropout=attention_dropout,
            **kwargs,
        )
