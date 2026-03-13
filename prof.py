import importlib.util, os as _os
_spec = importlib.util.spec_from_file_location(
    "glide_llama_config",
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                  "glide_exp/llama/glide_llama_modelling/glide_config.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
GlideConfig = _mod.GlideConfig
import math
import warnings
import copy
from typing import List, Optional, Tuple, Union
from einops import rearrange, repeat


from time import time


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers.utils import logging


# ── Inline implementations (no flash_attn dependency) ─────────────────────────

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    B, H, L, D = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    return hidden_states[:, :, None, :, :].expand(B, H, n_rep, L, D).reshape(B, H * n_rep, L, D)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, _position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        base = getattr(config, 'rope_theta', 10000.0)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, x, position_ids):
        inv = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        pos = position_ids[:, None, :].float()
        freqs = (inv @ pos).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

# ──────────────────────────────────────────────────────────────────────────────

from fla.models.utils import Cache as FlaCache
from fla.ops.gla import chunk_gla, fused_recurrent_gla
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

logger = logging.get_logger(__name__)

_compiled_flex_attention = torch.compile(flex_attention)


def sliding_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int = 1024,
    causal: bool = True,
    scale: float = None,
    block_mask_cache: dict = None
) -> torch.Tensor:

    B, H, L, D = q.shape
    device = q.device

    if causal:
        def mask_mod(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            window_mask = q_idx - kv_idx <= window_size
            return causal_mask & window_mask
    else:
        def mask_mod(b, h, q_idx, kv_idx):
            return torch.abs(q_idx - kv_idx) <= window_size

    cache_key = (B, H, L, window_size, causal, device)
    if block_mask_cache is not None and cache_key in block_mask_cache:
        block_mask = block_mask_cache[cache_key]
    else:
        block_mask = create_block_mask(mask_mod, B, H, L, L, device=device)
        if block_mask_cache is not None:
            block_mask_cache[cache_key] = block_mask

    output = _compiled_flex_attention(q, k, v, block_mask=block_mask, scale=scale)
    return output

class LigerAttention(nn.Module):
    def __init__(
        self, 
        config: GlideConfig,
        layer_idx: Optional[int] = None,
        window_size: int = 64,
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

        self.window_size = window_size
        self.block_mask_cache = {}
        self.time_io = None
        self.time_kv_io = None
        self.time_compute_softmax_attn = None
        self.time_compute_li_attn = None
        self.time_combination = None

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

        torch.cuda.synchronize()
        t_io_start = time()
        q = self.q_proj(hidden_states)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        torch.cuda.synchronize()
        t_q_end = time()

        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        g = self.pool_g(k)

        # dealing with left-padding
        if attention_mask is not None:
            v = v.mul_(attention_mask[:, -v.shape[-2]:, None])

        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_key_value_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_key_value_heads)
        g = rearrange(g, 'b n (h m) -> b h n m', h=self.num_key_value_heads)

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)
        g = repeat_kv(g, self.num_key_value_groups)
        torch.cuda.synchronize()
        t_io_end = time()

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

        torch.cuda.synchronize()
        t_li_start = time()
        if self.training:
            o_, recurrent_state = chunk_gla(q=q, k=k, v=v, g=g, scale=scale, initial_state=recurrent_state, output_final_state=True)
        else:
            o_, recurrent_state = fused_recurrent_gla(q, k, v, g, scale=scale, initial_state=recurrent_state, output_final_state=True)
        torch.cuda.synchronize()
        t_li_end = time()

        if past_key_value is not None:
            past_key_value.update(
                recurrent_state=recurrent_state,
                layer_idx=self.layer_idx,
                offset=q.shape[1]
            )
        
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
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

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            sq = sq.to(target_dtype)
            sk = sk.to(target_dtype)
            sv = sv.to(target_dtype)

        torch.cuda.synchronize()
        t_sa_start = time()
        y = sliding_window_attention(
            sq, sk, sv,
            window_size=self.window_size,
            causal=True,
            block_mask_cache=self.block_mask_cache,
        )
        torch.cuda.synchronize()
        t_sa_end = time()

        torch.cuda.synchronize()
        t_com_start = time()
        o_ = 0.5 * y + 0.5 * o_
        torch.cuda.synchronize()
        t_com_end = time()
        o = rearrange(o_.to(self.o_proj.weight.dtype), 'b h n d -> b n (h d)')
        o = self.o_proj(o)

        # update latency breakdown
        self.time_io = t_q_end - t_io_start
        self.time_kv_io = t_io_end - t_q_end
        self.time_compute_li_attn = t_li_end - t_li_start
        self.time_compute_softmax_attn = t_sa_end - t_sa_start
        self.time_combination = t_com_end - t_com_start

        return o, None, past_key_value

def print_latency_breakdown(hidden_size=512, num_heads=8, num_key_value_heads=None, seq_len=512, batch_size=1, window_size=64, warmup=3, runs=10):
    device = 'cuda'
    if num_key_value_heads is None:
        num_key_value_heads = num_heads

    glide_config = GlideConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
    )

    attn = LigerAttention(glide_config, layer_idx=0, window_size=window_size).to(device=device, dtype=torch.bfloat16).eval()

    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    cos, sin = attn.rotary_emb(hidden_states, position_ids)
    position_embeddings = (cos, sin)

    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            attn(hidden_states, position_ids=position_ids, position_embeddings=position_embeddings)

    # measure
    times = {"kv_io": 0.0, "li_attn": 0.0, "sa_attn": 0.0, "combo": 0.0}
    with torch.no_grad():
        for _ in range(runs):
            attn(hidden_states, position_ids=position_ids, position_embeddings=position_embeddings)
            times["kv_io"]   += attn.time_kv_io
            times["li_attn"] += attn.time_compute_li_attn
            times["sa_attn"] += attn.time_compute_softmax_attn
            times["combo"]   += attn.time_combination

    for k in times:
        times[k] /= runs

    total = sum(times.values())

    flops = estimate_flops(hidden_size, num_heads, num_key_value_heads, seq_len, batch_size, window_size)
    total_flops = sum(flops.values())
    total_tflops = (total_flops / total) / 1e12 if total > 0 else 0.0

    print(f"\nLatency Breakdown  (window={window_size}, B={batch_size}, L={seq_len}, H={hidden_size}, heads={num_heads}, avg over {runs} runs)")
    print(f"  KV I/O       (k+v+g):    {times['kv_io']*1e3:7.3f} ms  ({times['kv_io']/total*100:5.1f}%)")
    print(f"  Linear Attn  (GLA):      {times['li_attn']*1e3:7.3f} ms  ({times['li_attn']/total*100:5.1f}%)")
    print(f"  SWA          (window):   {times['sa_attn']*1e3:7.3f} ms  ({times['sa_attn']/total*100:5.1f}%)")
    print(f"  Combine      (0.5+0.5):  {times['combo']*1e3:7.3f} ms  ({times['combo']/total*100:5.1f}%)")
    print(f"  {'─'*45}")
    print(f"  Total:                   {total*1e3:7.3f} ms  |  {total_flops/1e9:.2f} GFLOPs  |  {total_tflops:.2f} TFLOP/s")

    return {
        "window_size": window_size,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "num_key_value_heads": num_key_value_heads,
        "runs": runs,
        "kv_io_ms":     round(times["kv_io"]   * 1e3, 4),
        "li_attn_ms":   round(times["li_attn"] * 1e3, 4),
        "sa_attn_ms":   round(times["sa_attn"] * 1e3, 4),
        "combo_ms":     round(times["combo"]   * 1e3, 4),
        "total_ms":     round(total             * 1e3, 4),
        "total_gflops": round(total_flops       / 1e9, 4),
        "total_tflops": round(total_tflops,             4),
    }

def estimate_flops(hidden_size, num_heads, num_key_value_heads, seq_len, batch_size, window_size):
    """
    FLOPs counted as multiply-adds × 2.

    KV I/O  : k_proj + v_proj + pool_g  (3 linear layers over kv heads)
    GLA     : recurrent state update (outer product k*v^T) + gating + output query (q*S)
              per head per token ≈ 4 * head_dim^2
    SWA     : QK^T + softmax·V  (each token attends to `window_size` tokens)
              per head ≈ 4 * L * window_size * head_dim
    Combine : elementwise 0.5*y + 0.5*o_  → 2 * B * L * hidden_size
    """
    B, L, H, Hkv = batch_size, seq_len, num_heads, num_key_value_heads
    D = hidden_size // H           # head dim

    # KV I/O: k_proj + v_proj + pool_g — each is (hidden_size → Hkv*D) linear
    kv_proj_flops = 3 * 2 * B * L * hidden_size * (Hkv * D)

    # GLA: fused_recurrent with H heads, head_dim D
    # per head per token: outer product (2*D^2) + gate (D^2) + output (2*D^2) ≈ 5*D^2
    gla_flops = 5 * 2 * B * H * L * D * D   # ×2 for mul-add

    # SWA: QK^T and AV each = 2 * B * H * L * window_size * D
    swa_flops = 4 * B * H * L * window_size * D

    # Combine: two scalar multiplies + add per element
    combine_flops = 3 * B * L * hidden_size

    return {
        "kv_io":   kv_proj_flops,
        "li_attn": gla_flops,
        "sa_attn": swa_flops,
        "combo":   combine_flops,
    }





if __name__ == "__main__":
    import json, argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--flops-only", action="store_true", help="estimate FLOPs without running GPU")
    args = parser.parse_args()

    config_path = "prof_config.json"
    with open(config_path) as f:
        cfg = json.load(f)

    B   = cfg["batch_size"]
    H   = cfg["hidden_size"]
    Nh  = cfg["num_heads"]
    Nkv = cfg["num_key_value_heads"]
    L   = cfg["seq_len"]

    if args.flops_only:
        # KV I/O bytes per token: load K + V cache for the sliding window (bf16 = 2 bytes)
        # Cache stored with Nkv heads; multiply by num_layers for whole-model cost
        num_layers = cfg.get("num_layers", 1)
        D = H // Nh   # head dim
        def kv_io_bytes(window_size):
            per_layer = 2 * window_size * Nkv * D * 2   # K + V, Nkv heads, D dims, bf16
            return per_layer, per_layer * num_layers

        results = []
        print(f"\nKV I/O per token  (hidden={H}, heads={Nh}, kv_heads={Nkv}, head_dim={D}, layers={num_layers})")
        print(f"  {'window':>8}  {'per-layer B':>12}  {'per-layer MB':>13}  {'whole-model MB':>15}")
        print(f"  {'─'*57}")
        for ws_key, fracs in cfg["window_sizes"].items():
            for frac in fracs:
                kb_layer, kb_model = kv_io_bytes(frac)
                print(f"  {frac:>8}  {kb_layer:>12,}  {kb_layer/1e6:>13.3f}  {kb_model/1e6:>15.3f}")
                results.append({
                    "window_group":       int(ws_key),
                    "window_size":        frac,
                    "kv_io_bytes_layer":  kb_layer,
                    "kv_io_mb_layer":     round(kb_layer / 1e6, 4),
                    "kv_io_mb_model":     round(kb_model / 1e6, 4),
                })
        out_path = "flops_results.json"
        with open(out_path, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nSaved {len(results)} entries to {out_path}")

    else:
        results = []
        for ws_key, fracs in cfg["window_sizes"].items():
            for frac in fracs:
                row = print_latency_breakdown(
                    hidden_size=H,
                    num_heads=Nh,
                    num_key_value_heads=Nkv,
                    seq_len=L,
                    batch_size=B,
                    window_size=frac,
                    warmup=cfg["warmup"],
                    runs=cfg["runs"],
                )
                row["window_group"] = int(ws_key)
                results.append(row)

        out_path = "prof_results.json"
        with open(out_path, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nSaved {len(results)} entries to {out_path}")


