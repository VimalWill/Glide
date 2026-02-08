import torch
from torch import nn
from torch import functional as F
from .config import GlideConfig
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    repeat_kv,
    apply_rotary_pos_emb,
    LlamaMLP,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
)

from einops import rearrange
from typing import Optional, Tuple
from fla.models.utils import Cache as FlaCache
from fla.ops.gla import chunk_gla
from fla.ops.gla import fused_recurrent_gla

class GlideHybridAttention(nn.Module):
    def __init__(
        self, 
        config: GlideConfig,
        layer_idx: Optional[int] = None,
        window_size: int = 64
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)
        self.pool_g = nn.AdaptiveAvgPool1d(output_size=self.head_dim * self.num_key_value_heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[FlaCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        last_state = None
        if past_key_value is not None and len(past_key_value) > self.layer_idx:
            last_state = past_key_value[self.layer_idx]

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        g = self.pool_g(k)

        # dealing with left-padding
        if attention_mask is not None:
            v = v.mul_(attention_mask[:, -v.shape[-2]:, None])

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_key_value_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_key_value_heads)
        g = rearrange(g, 'b n (h m) -> b h n m', h=self.num_key_value_heads)

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)
        g = repeat_kv(g, self.num_key_value_groups)

        sq, sk, sv = q, k, v

        # norm
        q = F.softmax(q, dim=-1)
        k = F.softmax(k, dim=-1)
        
        gate_logit_normalizer = 16
        g = F.logsigmoid(g) / gate_logit_normalizer # (b, h, n, m)

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        offsets = kwargs.get('offsets', None)
        scale = 1 
        q, k, v, g = (x.to(torch.float32).contiguous() for x in (q, k, v, g))

        if self.training or q.shape[-2] > 1:
            o_, recurrent_state = chunk_gla(q=q, k=k, v=v, g=g, scale=scale, initial_state=recurrent_state, output_final_state=True)
        else:
            o_, recurrent_state = fused_recurrent_gla(q, k, v, g, scale=scale, initial_state=recurrent_state, output_final_state=True)

        if past_key_value is not None:
            past_key_value.update(
                recurrent_state=recurrent_state,
                layer_idx=self.layer_idx,
                offset=q.shape[1]
            )
        
        q_len = hidden_states.size(-2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(sv, position_ids)
        else:
            cos, sin = position_embeddings
        sq, sk = apply_rotary_pos_emb(sq, sk, cos, sin)

        input_dtype = sq.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            sq = sq.to(target_dtype)
            sk = sk.to(target_dtype)
            sv = sv.to(target_dtype)

        window_size = 64
        if attention_mask is not None and 0.0 in attention_mask:
            pass
        else:
            attention_mask = None

        y = _flash_attention_forward( # Reashape to the expected shape for Flash Attention
            sq.transpose(1, 2),
            sk.transpose(1, 2),
            sv.transpose(1, 2),
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=0.0,
            sliding_window=window_size,
            use_top_left_mask=False,
            is_causal=True,
            target_dtype=torch.float32,
        ).transpose(1, 2)
        o_ = 0.5 * y + 0.5 * o_ 
        o = rearrange(o_.bfloat16(), 'b h n d -> b n (h d)')
        o = self.o_proj(o)

        return o, None, past_key_value
