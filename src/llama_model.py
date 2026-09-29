import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
from src.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import (
    logging,
)
from src.kv_pruning_utils import init_snapkv,init_H2O
try:
    from pygini import gini
except ImportError:
    def gini(*args, **kwargs):
        raise ImportError("pygini is required only for method=\"gini\"")

import math
import os
import json


def _write_attn_trace(**payload):
    trace_path = os.environ.get("THINK_ATTN_TRACE")
    if not trace_path:
        return
    try:
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _apply_rotary_compat(rotary_emb, value_states, position_ids, kv_seq_len,
                         query_states, key_states):
    """RoPE compat shim — dispatches between LLaMA (x, position_ids) and Mistral 4.40
    (x, seq_len) rotary_emb signatures. Returns (q_rot, k_rot, cos, sin)."""
    if 'mistral' in type(rotary_emb).__module__:
        from transformers.models.mistral.modeling_mistral import (
            apply_rotary_pos_emb as _mistral_apply,
        )
        cos, sin = rotary_emb(value_states, seq_len=int(kv_seq_len))
        q_rot, k_rot = _mistral_apply(query_states, key_states, cos, sin, position_ids)
        return q_rot, k_rot, cos, sin
    cos, sin = rotary_emb(value_states, position_ids)
    q_rot, k_rot = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    return q_rot, k_rot, cos, sin


def _write_prepare_trace(**payload):
    trace_path = os.environ.get("THINK_PREP_TRACE")
    if not trace_path:
        return
    try:
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


logger = logging.get_logger(__name__)


# Prefer the fused SDPA kernels for the prefill output branch. The eager
# (Q @ K^T) path materialises a (B, H, T, T) fp16 matrix which is ~55 GB for
# Qwen2.5-7B at LongBench T=31500 and OOMs even on an 80 GB A100. SDPA with
# is_causal=True dispatches to flash / memory-efficient kernels that never
# realise the score matrix; peak falls to ~1 GB at the same shape. The
# prefill logits are discarded anyway (only attn_output propagates), so the
# accuracy-sensitive scoring path (update_thinkv → key_pruner_iterative) is
# untouched — it scores with window Q × full K inside kv_pruning_utils.
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    _SDPA_AVAILABLE = True
    _SDPA_BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
except Exception:
    _SDPA_AVAILABLE = False
    _SDPA_BACKENDS = None


def _prefill_output_sdpa(query_states, key_states, value_states):
    """Fused prefill attention output via torch SDPA (no T×T materialisation).

    Returns attn_output of shape (B, H, T, D) — same as the eager softmax
    path would produce, but without a (B, H, T, T) intermediate.

    Caller is responsible for skipping the downstream softmax/matmul block
    (attn_weights is None after this). LongBench single-batch prompts need
    only is_causal=True; explicit 4-D HF attention_mask is a causal mask
    anyway, so passing it would force SDPA to its eager fallback. If batched
    padding support is added later, fork here.
    """
    with sdpa_kernel(_SDPA_BACKENDS):
        return F.scaled_dot_product_attention(
            query_states, key_states, value_states, is_causal=True
        )


def llama_attn_forward_H2O(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """★★★ 우리 방법의 핵심 attention forward ★★★

    대분기 (q_len 기반):
      - q_len == kv_seq_len  → prefill (첫 호출, 전체 문장)
          └→ :125-142 update_thinkv 호출해서 eviction + iterative mask 계산,
              그 결과를 past_key_value.update_think()로 8h 압축 저장.
      - q_len == 1  → decode (답 한 단어씩 생성)
          └→ :148-161 새 K/V를 8h로 slice해서 cache에 append,
              :163-199 pruned + recent 두 갈래로 attention 계산.

    공통 후처리: softmax → V matmul → o_proj (:204-216).
    """
    bsz, q_len, _ = hidden_states.size()

    # [A] 처음 호출 시 이 attention 모듈에 H2OKVCluster 하나 붙임.
    #     → src/kv_pruning_utils.py:1323 init_H2O
    #     이후 self.kv_cluster.update_thinkv(...)로 우리 pruner 호출.
    init_H2O(self)

    if getattr(self.config, "pretraining_tp", 1) > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // getattr(self.config, "pretraining_tp", 1)
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // getattr(self.config, "pretraining_tp", 1), dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(getattr(self.config, "pretraining_tp", 1))]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(getattr(self.config, "pretraining_tp", 1))]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(getattr(self.config, "pretraining_tp", 1))]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    # [B] 표준 Q/K/V projection 후 shape 정리.
    #     Q: [B, 32, q_len, 128], K/V: [B, 8, q_len, 128]  ← GQA: K/V 원본은 8 head
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    query_states_pre = query_states    # pre-RoPE 백업 (legacy 모드용; 우리 메인 경로는 미사용)
    key_states_pre = key_states

    # kv_seq_len: 지금까지 본 총 token 수 (decode면 self.kv_seq_len에 누적됨)
    kv_seq_len = key_states.shape[-2]
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        if hasattr(self, "kv_seq_len"):
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    # [C] RoPE (위치 회전) + GQA broadcast.
    #     repeat_kv: 8 → 32 head (view 연산, 추가 메모리 없음).
    #     저장 시점에는 다시 8로 slice → 메모리 절약 유지. decode path의 :154에서 슬라이스.
    query_states, key_states, cos, sin = _apply_rotary_compat(
        self.rotary_emb, value_states, position_ids, kv_seq_len, query_states, key_states,
    )
    key_states = repeat_kv(key_states, self.num_key_value_groups)           # [B, 32, q_len, 128]
    value_states = repeat_kv(value_states, self.num_key_value_groups)       # [B, 32, q_len, 128]
    key_states_pre = repeat_kv(key_states_pre, self.num_key_value_groups)   # H2O-path [MAIN-METHOD-ANCHOR]

    _write_attn_trace(event='attn_forward_state', layer_idx=int(self.layer_idx) if self.layer_idx is not None else None, q_len=int(q_len), kv_seq_len=int(kv_seq_len), past_is_none=bool(past_key_value is None), method_mode=str(getattr(self.config, 'think_pruner_mode', 'think')))

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        # ====================================================================
        # [D] Prefill 분기 : q_len == kv_seq_len이면 "이번이 첫 forward, 전체 문장을 본다"
        #     → 우리 방법의 오프라인 압축 로직 실행.
        # ====================================================================
        if key_states.shape[-2] == kv_seq_len:
            _write_attn_trace(event='update_thinkv_call', layer_idx=int(self.layer_idx) if self.layer_idx is not None else None, q_len=int(q_len), kv_seq_len=int(kv_seq_len), key_len=int(key_states.shape[-2]), mode=str(getattr(self.config, 'think_pruner_mode', 'think')))
            self.kv_seq_len = kv_seq_len
            self.kv_cluster.refresh_interval = getattr(self.config, 'think_refresh_interval', 500)
            # [D-1] 압축 본체 호출 (token eviction + channel mask 계산).
            #       → src/kv_pruning_utils.py:1145 H2OKVCluster.update_thinkv
            #         pruner_mode='gqa_iter32_think' → :1281 key_pruner_iterative
            #       반환:
            #         kv_pruned : [1, 8, past=96, kept=77]  (오래된 토큰, 채널도 자른 K)
            #         kv_recent : [1, 8, window=32, 128]    (최근 창, 온전한 채널)
            #         mask      : [1, 8, 128] bool           (각 group이 남긴 채널 마스크)
            #         value_states_compress : [1, 8, 128, 128]  (V; 8 head로 slice됨)
            _pruner_mode = getattr(self.config, 'think_pruner_mode', 'think')
            _is_leankfair = _pruner_mode.endswith('_leankfair')
            if _is_leankfair:
                # Fully separated prefill path for LeanK-fair mode. update_thinkv is NOT
                # touched; sink_size comes from config (default 4, matches StreamingLLM/LeanK).
                _sink_size = int(getattr(self.config, 'sink_size', 4))
                kv_sink, kv_pruned, kv_recent, mask, value_states_compress = self.kv_cluster.update_thinkv_leankfair(
                    key_states, query_states, value_states, attention_mask, self.num_key_value_groups,
                    query_states_pre=query_states_pre, key_states_pre=key_states_pre,
                    o_proj_weight=self.o_proj.weight.detach(), k_proj_weight=self.k_proj.weight.detach(),
                    cos=cos, sin=sin, pruner_mode=_pruner_mode,
                    tau=getattr(self.config, 'think_pruner_tau', 1.0),
                    layer_idx=self.layer_idx, sink_size=_sink_size,
                )
                past_key_value.update_think_leankfair(
                    kv_sink, kv_pruned, kv_recent, mask, value_states_compress,
                    self.layer_idx, cache_kwargs,
                )
            else:
                kv_pruned, kv_recent, mask, value_states_compress = self.kv_cluster.update_thinkv(
                    key_states, query_states, value_states, attention_mask, self.num_key_value_groups,
                    query_states_pre=query_states_pre, key_states_pre=key_states_pre, o_proj_weight=self.o_proj.weight.detach(),
                    k_proj_weight=self.k_proj.weight.detach(),
                    cos=cos, sin=sin,
                    pruner_mode=_pruner_mode, tau=getattr(self.config, 'think_pruner_tau', 1.0), layer_idx=self.layer_idx
                )
                if _pruner_mode.startswith('seg_'):
                    past_key_value.update_think_segmented(kv_pruned, kv_recent, value_states_compress, self.layer_idx, cache_kwargs)
                else:
                    _has_residual = any(_pruner_mode.endswith(s) for s in ('_resavg', '_resavg_scaled', '_resconst', '_resrand', '_resabs', '_resl2'))
                    _gqa_pruned = _pruner_mode.startswith('gqa_')
                    # [D-2] 압축 결과를 cache에 저장.
                    #       → src/cache_utils.py:155 DynamicCache.update_think
                    #         key_cache_pruned / key_cache / value_cache / mask / gqa_pruned 리스트에 layer별로 저장.
                    past_key_value.update_think(kv_pruned, kv_recent, mask, value_states_compress, self.layer_idx, cache_kwargs, residual=_has_residual, gqa_pruned=_gqa_pruned)
        # ====================================================================
        # [E] Decode 분기 : q_len == 1 이면 "이미 prefill 끝났고, 지금 한 단어 생성 중"
        #     새 단어의 K/V를 cache에 append (8 head로 저장).
        # ====================================================================
        else:
            _write_attn_trace(event='cache_update_only', layer_idx=int(self.layer_idx) if self.layer_idx is not None else None, q_len=int(q_len), kv_seq_len=int(kv_seq_len), key_len=int(key_states.shape[-2]), mode=str(getattr(self.config, 'think_pruner_mode', 'think')))
            self.kv_seq_len += q_len
            # gqa_pruned 레이어라면 cache는 8h로 저장돼있음.
            # 새 K/V가 repeat_kv로 32h인데, 8h로 다시 slice해서 concat → 메모리 8h 유지.
            # attention 계산 시점에만 :180에서 32h로 view 확장.
            _layer_is_gqa = (hasattr(past_key_value, 'gqa_pruned')
                             and self.layer_idx < len(past_key_value.gqa_pruned)
                             and past_key_value.gqa_pruned[self.layer_idx])
            if _layer_is_gqa:
                # [E-1] 8h로 slice (stride로 중복 제거: 4개씩 같은 원소니까 ::4로 꺼냄)
                key_states_8h = key_states[:, ::self.num_key_value_groups, :, :].contiguous()
                value_states_8h = value_states[:, ::self.num_key_value_groups, :, :].contiguous()
                # [E-2] cache에 append.
                #       → src/cache_utils.py:119 DynamicCache.update (단순 concat)
                key_states_cached, value_states_cached = past_key_value.update(
                    key_states_8h, value_states_8h, self.layer_idx, cache_kwargs
                )
                # [E-2.5] Optional decode-time flush (Tier 1 real compression).
                # Off by default — only methods that opt in via
                # ``config.decode_flush_enabled = True`` pay this cost. When on,
                # once key_cache[L] grows past ``decode_flush_threshold`` tokens,
                # the oldest ones get channel-compressed via the prefill mask and
                # migrated into key_cache_pruned[L]. See
                # ``DynamicCache.flush_recent_if_needed`` for the invariants.
                if getattr(self.config, 'decode_flush_enabled', False):
                    _recent = int(getattr(self.config, 'decode_flush_recent_size',
                                          getattr(self.kv_cluster, 'recent_size', 32)))
                    _threshold = int(getattr(self.config, 'decode_flush_threshold', 2 * _recent))
                    flushed = past_key_value.flush_recent_if_needed(
                        self.layer_idx, recent_size=_recent, threshold=_threshold
                    )
                    if flushed:
                        # Rebind key_states_cached to the trimmed recent buffer so
                        # attn_weights2 (recent path) covers only the new recent window.
                        # key_cache_pruned absorbed the flushed tokens — attn_weights1
                        # will naturally pick them up via the pruned-past branch.
                        key_states_cached = past_key_value.key_cache[self.layer_idx]
                        value_states_cached = past_key_value.value_cache[self.layer_idx]
                # [E-3] attention 계산용으로 32h view 확장 (메모리는 여전히 8h)
                key_states = repeat_kv(key_states_cached, self.num_key_value_groups)
                value_states = repeat_kv(value_states_cached, self.num_key_value_groups)
            else:
                # 비-GQA 레이어 (legacy). 그냥 32h로 저장.
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # ========================================================================
    # [F] Decode attention 계산 (q_len == 1 일 때만).
    #     두 갈래의 logit을 concat:
    #       attn_weights1: 오래된 토큰(pruned K) ← 채널만 kept_dim짜리
    #       attn_weights2: 최근 + 방금 생성된 토큰(recent K) ← 전체 채널
    #     concat([w1, w2]) → softmax → V matmul.  prefill은 이 분기 타지 않음.
    # ========================================================================
    if query_states.shape[-2] ==1:
        _is_segmented = hasattr(past_key_value, 'segments') and self.layer_idx < len(past_key_value.segments) and past_key_value.segments[self.layer_idx] is not None and len(past_key_value.segments[self.layer_idx]) > 0
        if _is_segmented:
            # (segment 모드, 우리 메인 method는 사용 X — seg_kenergy 등 legacy)
            score_parts = []
            _seg_mode = getattr(self.config, 'decode_sqrt_mode', 'd')
            for seg_k, seg_mask in past_key_value.segments[self.layer_idx]:
                seg_mask_exp = seg_mask.unsqueeze(2)
                q_seg = query_states[seg_mask_exp].view(query_states.shape[0], query_states.shape[1], query_states.shape[2], -1)
                _seg_kept = q_seg.shape[-1]
                if _seg_mode == 'kept':
                    _seg_div = math.sqrt(_seg_kept)
                elif isinstance(_seg_mode, (int, float)):
                    _seg_div = float(_seg_mode) * math.sqrt(self.head_dim)
                else:
                    _seg_div = math.sqrt(self.head_dim)
                score_parts.append(torch.matmul(q_seg, seg_k.transpose(2, 3)) / _seg_div)
            score_parts.append(torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim))
            attn_weights = torch.cat(score_parts, dim=-1)
        else:
            # ★ 우리 메인 method가 타는 경로 ★
            # [F-1] cache에서 꺼내기.
            key_pruned = past_key_value.key_cache_pruned[self.layer_idx]   # [1, 8, past, kept]
            base_mask = past_key_value.mask[self.layer_idx]                 # [1, 8, 128]
            _is_residual = past_key_value.residual_mode[self.layer_idx] if self.layer_idx < len(past_key_value.residual_mode) else False
            _is_gqa = past_key_value.gqa_pruned[self.layer_idx] if self.layer_idx < len(past_key_value.gqa_pruned) else False
            # [F-2] GQA면 key_pruned와 mask를 32 head view로 확장.
            if _is_gqa:
                key_pruned = repeat_kv(key_pruned, self.num_key_value_groups)              # [1, 32, past, kept] (view)
                base_mask = base_mask.repeat_interleave(self.num_key_value_groups, dim=1)  # [1, 32, 128]
            mask = base_mask.unsqueeze(2)
            # [F-3] Q에서도 same mask로 kept 채널만 추출 → [1, 32, 1, kept]
            masked_query_states = query_states[mask].view(query_states.shape[0], query_states.shape[1], query_states.shape[2], -1)
            if _is_residual:
                # (residual 모드, 우리 메인 method는 사용 X)
                _pruner_mode_dec = getattr(self.config, 'think_pruner_mode', '')
                drop_mask = (~base_mask).unsqueeze(2)
                dropped_query_states = query_states[drop_mask].view(query_states.shape[0], query_states.shape[1], query_states.shape[2], -1)
                if _pruner_mode_dec.endswith('_resabs'):
                    drop_avg = dropped_query_states.abs().mean(dim=-1, keepdim=True)
                elif _pruner_mode_dec.endswith('_resl2'):
                    drop_avg = dropped_query_states.pow(2).mean(dim=-1, keepdim=True).sqrt()
                elif _pruner_mode_dec.endswith('_resavg_scaled'):
                    import math as _math
                    drop_avg = dropped_query_states.mean(dim=-1, keepdim=True) * _math.sqrt(dropped_query_states.shape[-1])
                else:
                    drop_avg = dropped_query_states.mean(dim=-1, keepdim=True)
                masked_query_states = torch.cat([masked_query_states, drop_avg], dim=-1)
            # Decode-time attention temperature. Defaults to 'd' (sqrt(head_dim))
            # because matching the training distribution empirically outperforms the
            # information-theoretically "correct" sqrt(kept_dim). See
            # docs/SQRT_TEMPERATURE_CALIBRATION.md for the causal experiments.
            # Still configurable via config.decode_sqrt_mode for future ablations:
            #   'd'    -> sqrt(head_dim)            (default, training-matched)
            #   'kept' -> sqrt(kept_dim)            (unit-variance, was B5 fix)
            #   float  -> alpha * sqrt(head_dim)    (temperature sweep)
            _mode = getattr(self.config, 'decode_sqrt_mode', 'd')
            _kept_dim = masked_query_states.shape[-1]
            if _mode == 'kept':
                _div = math.sqrt(_kept_dim)
            elif _mode == 'd':
                _div = math.sqrt(self.head_dim)
            elif isinstance(_mode, (int, float)):
                _div = float(_mode) * math.sqrt(self.head_dim)
            else:
                _div = math.sqrt(_kept_dim)
            # [F-4] 두 갈래 attention logit 계산:
            #       - attn_weights1: masked Q × pruned K (오래된 토큰, kept 채널만)
            #       - attn_weights2: Q × recent K (최근 창 + 이번에 append된 새 토큰)
            #       둘 다 학습 분포 맞추기 위해 sqrt(head_dim=128)로 나눔 (default='d').
            attn_weights1 = torch.matmul(masked_query_states, key_pruned.transpose(2, 3)) / _div
            attn_weights2 = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            # [F-4b] LeanK-fair: 3-way attention. Sink tokens stay at full d_head, so we
            # compute their logits with the ORIGINAL (unmasked) query at sqrt(head_dim).
            # The concat order [sink | middle_pruned | recent+decode] matches V's token
            # layout so softmax·V lines up without any V-side reshuffling.
            _key_sink = None
            if hasattr(past_key_value, 'key_cache_sink') and self.layer_idx < len(past_key_value.key_cache_sink):
                _key_sink = past_key_value.key_cache_sink[self.layer_idx]
            if _key_sink is not None:
                key_sink_32 = repeat_kv(_key_sink, self.num_key_value_groups)  # [1, 32, sink, D] (view)
                attn_weights_sink = torch.matmul(query_states, key_sink_32.transpose(2, 3)) / math.sqrt(self.head_dim)
            if os.environ.get('LEANKFAIR_DEBUG') and self.layer_idx == 0:
                import sys
                _sink_len = 0 if _key_sink is None else _key_sink.shape[-2]
                _has = hasattr(past_key_value, 'key_cache_sink')
                _ln = len(past_key_value.key_cache_sink) if _has else -1
                _cls = type(past_key_value).__module__ + '.' + type(past_key_value).__name__
                _is_none = True if not _has or self.layer_idx >= _ln else past_key_value.key_cache_sink[self.layer_idx] is None
                print(f"[leankfair-debug] layer={self.layer_idx} cls={_cls} has_attr={_has} sink_list_len={_ln} entry_is_none={_is_none} aw1={attn_weights1.shape[-1]} aw2={attn_weights2.shape[-1]} sink={_sink_len} vlen={value_states.shape[-2]}", file=sys.stderr, flush=True)
            # (실험용 probe 훅, 평상시 capture_attn_probe=False → 바로 pass)
            if getattr(self.config, 'capture_attn_probe', False):
                _probes = getattr(self, '_attn_probes', None)
                if _probes is None:
                    _probes = []; self._attn_probes = _probes
                _probes.append({
                    'layer': int(getattr(self, 'layer_idx', -1)),
                    'kept_dim': int(_kept_dim),
                    'div': float(_div),
                    'logits1_stats': {
                        'mean_abs': float(attn_weights1.detach().abs().mean().item()),
                        'std': float(attn_weights1.detach().std().item()),
                    },
                })
            # [F-5] 두 logit 합치기 → 이후 [G]에서 softmax × V.
            if _key_sink is not None:
                attn_weights = torch.cat([attn_weights_sink, attn_weights1, attn_weights2], dim=-1)
            else:
                attn_weights = torch.cat([attn_weights1, attn_weights2], dim=-1)
        _used_sdpa_prefill = False
    else:
        # Prefill 경로에서는 압축 전 attention 그대로 계산 (표준 attention).
        # 실제로 이 logit은 버려지고 결과(attn_output)만 다음 layer로 넘어감.
        # config.prefill_sdpa=True 일 때는 SDPA로 (B,H,T,T) 피하기. 기본 False라
        # LLaMA-3 등 기존 eager 경로는 bit-identical로 보존됨.
        _default_sdpa = getattr(self.config, "model_type", None) in ("qwen2", "qwen3")
        if _SDPA_AVAILABLE and getattr(self.config, "prefill_sdpa", _default_sdpa):
            attn_output = _prefill_output_sdpa(query_states, key_states, value_states)
            attn_weights = None
            _used_sdpa_prefill = True
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            _used_sdpa_prefill = False

    if not _used_sdpa_prefill:
        # [G] 최종 attention 계산: causal mask 더하기 → softmax(fp32) → × V → o_proj.
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : attn_weights.shape[-1]]
            attn_weights = attn_weights + causal_mask

        # softmax는 fp32로 올려서 정밀도 확보 후 다시 fp16.
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        if os.environ.get('LEANKFAIR_DEBUG') and self.layer_idx == 0:
            import sys
            print(f"[leankfair-debug PRE-MATMUL] layer={self.layer_idx} q_len={q_len} kv_seq_len={kv_seq_len} aw_shape={list(attn_weights.shape)} v_shape={list(value_states.shape)} k_shape={list(key_states.shape)}", file=sys.stderr, flush=True)
        # value_states: [1, 32, T, 128] (repeat_kv view로 32 head), attn_weights: [1, 32, 1, T]
        # → attn_output: [1, 32, 1, 128]
        attn_output = torch.matmul(attn_weights, value_states)

    # (실험용 probe, 평상시 capture_attn_probe=False → pass)
    if getattr(self.config, 'capture_attn_probe', False) and query_states.shape[-2] == 1:
        with torch.no_grad():
            _p = attn_weights.detach().float()   # [B, H, 1, T]
            _p1 = _p[..., 0, :]                   # [B, H, T]
            _eps = 1e-12
            _ent = -(_p1 * (_p1 + _eps).log()).sum(dim=-1)  # [B, H]
            _max_p = _p1.max(dim=-1).values                  # [B, H]
            _argmax = _p1.argmax(dim=-1)                     # [B, H]
            if not hasattr(self, '_attn_probes') or self._attn_probes is None:
                self._attn_probes = []
            # attach softmax stats to the last probe entry (same layer)
            if self._attn_probes:
                self._attn_probes[-1].update({
                    'entropy_mean': float(_ent.mean().item()),
                    'entropy_per_head': _ent.squeeze(0).cpu().tolist(),
                    'max_prob_mean': float(_max_p.mean().item()),
                    'argmax_per_head': _argmax.squeeze(0).cpu().tolist(),
                })

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if getattr(self.config, "pretraining_tp", 1) > 1:
        attn_output = attn_output.split(self.hidden_size // getattr(self.config, "pretraining_tp", 1), dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // getattr(self.config, "pretraining_tp", 1), dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(getattr(self.config, "pretraining_tp", 1))])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


# ============================================================================
# Qwen3 adapter — 우리 KV pruning 로직은 그대로 쓰되, 다음 차이만 흡수:
#   1) Q/K에 RMSNorm (self.q_norm, self.k_norm) — Qwen3-specific
#   2) position_embeddings가 Model.forward에서 tuple(cos, sin)로 주입됨
#      (LLaMA/Qwen2는 self.rotary_emb(value_states, position_ids)를 attn 내부에서 호출)
#   3) signature : past_key_value → past_key_values 복수로 이름 바뀜 (transformers 4.51+)
#
# 테스트 상태: 코드 작성만 완료. transformers ≥ 4.51 환경에서 DGX 서버에서
#   실제 forward 동작 검증 필요. `_patch_qwen3()` 아래 docstring 참조.
# ============================================================================
def qwen3_attn_forward_H2O(
    self,
    hidden_states,
    position_embeddings=None,              # Qwen3: tuple (cos, sin)
    attention_mask=None,
    past_key_values=None,                  # renamed (was past_key_value)
    past_key_value=None,                   # legacy alias for deprecate_kwarg
    cache_position=None,
    position_ids=None,                     # fallback path if position_embeddings not supplied
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    """Qwen3-variant of :func:`llama_attn_forward_H2O`.

    Differences from llama version:
      (A) After q_proj / k_proj, apply self.q_norm / self.k_norm (RMSNorm per head).
      (B) Use supplied position_embeddings instead of computing self.rotary_emb.
      (C) past_key_values (plural) signature.
    Otherwise identical flow — see ``llama_attn_forward_H2O`` docstring.
    """
    if past_key_values is None:
        past_key_values = past_key_value       # accept both names
    past_key_value = past_key_values           # internal name used by shared logic

    bsz, q_len, _ = hidden_states.size()

    init_H2O(self)

    # Qwen3 projections + RMSNorm.
    # self.q_norm / self.k_norm operate on the head-shape (hidden_shape includes head_dim axis).
    hidden_shape = (bsz, q_len, -1, self.head_dim)
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    query_states_pre = query_states
    key_states_pre = key_states

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if hasattr(self, "kv_seq_len") and self.kv_seq_len != 0:
            kv_seq_len += self.kv_seq_len
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    # RoPE — prefer the supplied position_embeddings; fall back to self.rotary_emb if present.
    if position_embeddings is not None:
        cos, sin = position_embeddings
    elif hasattr(self, "rotary_emb"):
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        raise RuntimeError(
            "qwen3_attn_forward_H2O: neither position_embeddings nor self.rotary_emb is available"
        )
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    key_states_pre = repeat_kv(key_states_pre, self.num_key_value_groups)

    # The rest of the logic (prefill / decode branches, attention compute, o_proj)
    # is identical to llama_attn_forward_H2O. We call a shared helper instead of
    # copy-pasting ~150 lines.
    return _post_projection_attention(
        self, hidden_states, query_states, key_states, value_states,
        query_states_pre, key_states_pre,
        cos, sin, cache_position, attention_mask,
        past_key_value, kv_seq_len, q_len, bsz,
        output_attentions,
    )


def _post_projection_attention(
    self, hidden_states, query_states, key_states, value_states,
    query_states_pre, key_states_pre,
    cos, sin, cache_position, attention_mask,
    past_key_value, kv_seq_len, q_len, bsz,
    output_attentions,
):
    """Shared body of our attention forward (called by both LLaMA and Qwen3 variants).

    Handles: prefill/decode branching, update_thinkv, decode-time pruned + recent
    attention concat, softmax, value matmul, o_proj.
    """
    # (This helper will be populated in a future refactor. For now, Qwen3 users
    # should run _patch_qwen3() which prints a TODO warning until we merge.)
    raise NotImplementedError(
        "qwen3_attn_forward_H2O uses _post_projection_attention which is not yet "
        "implemented. Planned: factor the prefill/decode branch out of "
        "llama_attn_forward_H2O so both paths share it. For now, Qwen3 support "
        "is gated — see docs/experiments/DGX_QWEN3_SETUP.md."
    )


def llama_attn_forward_SnapKV(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    init_snapkv(self)

    if getattr(self.config, "pretraining_tp", 1) > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // getattr(self.config, "pretraining_tp", 1)
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // getattr(self.config, "pretraining_tp", 1), dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(getattr(self.config, "pretraining_tp", 1))]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(getattr(self.config, "pretraining_tp", 1))]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(getattr(self.config, "pretraining_tp", 1))]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    query_states_pre = query_states
    key_states_pre = key_states

    kv_seq_len = key_states.shape[-2]
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        if hasattr(self, "kv_seq_len"): 
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    query_states, key_states, cos, sin = _apply_rotary_compat(
        self.rotary_emb, value_states, position_ids, kv_seq_len, query_states, key_states,
    )
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    key_states_pre = repeat_kv(key_states_pre, self.num_key_value_groups)

    _write_attn_trace(event='attn_forward_state', layer_idx=int(self.layer_idx) if self.layer_idx is not None else None, q_len=int(q_len), kv_seq_len=int(kv_seq_len), past_is_none=bool(past_key_value is None), method_mode=str(getattr(self.config, 'think_pruner_mode', 'think')))

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        if key_states.shape[-2] == kv_seq_len:
            _write_attn_trace(event='update_thinkv_call', layer_idx=int(self.layer_idx) if self.layer_idx is not None else None, q_len=int(q_len), kv_seq_len=int(kv_seq_len), key_len=int(key_states.shape[-2]), mode=str(getattr(self.config, 'think_pruner_mode', 'think')))
            self.kv_seq_len = kv_seq_len
            self.kv_cluster.refresh_interval = getattr(self.config, 'think_refresh_interval', 500)
            kv_pruned, kv_recent, mask, value_states_compress = self.kv_cluster.update_thinkv(
                key_states, query_states, value_states, attention_mask, self.num_key_value_groups,
                query_states_pre=query_states_pre, key_states_pre=key_states_pre, o_proj_weight=self.o_proj.weight.detach(),
                k_proj_weight=self.k_proj.weight.detach(),
                cos=cos, sin=sin,
                pruner_mode=getattr(self.config, 'think_pruner_mode', 'think'), tau=getattr(self.config, 'think_pruner_tau', 1.0), layer_idx=self.layer_idx
            )
            _pruner_mode = getattr(self.config, 'think_pruner_mode', 'think')
            if _pruner_mode.startswith('seg_'):
                past_key_value.update_think_segmented(kv_pruned, kv_recent, value_states_compress, self.layer_idx, cache_kwargs)
            else:
                _has_residual = any(_pruner_mode.endswith(s) for s in ('_resavg', '_resavg_scaled', '_resconst', '_resrand', '_resabs', '_resl2'))
                _gqa_pruned = _pruner_mode.startswith('gqa_')
                past_key_value.update_think(kv_pruned, kv_recent, mask, value_states_compress, self.layer_idx, cache_kwargs, residual=_has_residual, gqa_pruned=_gqa_pruned)
        else:
            _write_attn_trace(event='cache_update_only', layer_idx=int(self.layer_idx) if self.layer_idx is not None else None, q_len=int(q_len), kv_seq_len=int(kv_seq_len), key_len=int(key_states.shape[-2]), mode=str(getattr(self.config, 'think_pruner_mode', 'think')))
            self.kv_seq_len += q_len
            # GQA-pruned layer: cache is stored at 8 KV heads. Slice incoming K/V
            # (currently repeat_kv'd to 32h) back to 8h before concat, then re-expand
            # to 32h for downstream attention.
            _layer_is_gqa = (hasattr(past_key_value, 'gqa_pruned')
                             and self.layer_idx < len(past_key_value.gqa_pruned)
                             and past_key_value.gqa_pruned[self.layer_idx])
            if _layer_is_gqa:
                key_states_8h = key_states[:, ::self.num_key_value_groups, :, :].contiguous()
                value_states_8h = value_states[:, ::self.num_key_value_groups, :, :].contiguous()
                key_states_cached, value_states_cached = past_key_value.update(
                    key_states_8h, value_states_8h, self.layer_idx, cache_kwargs
                )
                key_states = repeat_kv(key_states_cached, self.num_key_value_groups)
                value_states = repeat_kv(value_states_cached, self.num_key_value_groups)
            else:
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    if query_states.shape[-2] ==1:
        _is_segmented = hasattr(past_key_value, 'segments') and self.layer_idx < len(past_key_value.segments) and past_key_value.segments[self.layer_idx] is not None and len(past_key_value.segments[self.layer_idx]) > 0
        if _is_segmented:
            score_parts = []
            _seg_mode = getattr(self.config, 'decode_sqrt_mode', 'd')
            for seg_k, seg_mask in past_key_value.segments[self.layer_idx]:
                seg_mask_exp = seg_mask.unsqueeze(2)
                q_seg = query_states[seg_mask_exp].view(query_states.shape[0], query_states.shape[1], query_states.shape[2], -1)
                _seg_kept = q_seg.shape[-1]
                if _seg_mode == 'kept':
                    _seg_div = math.sqrt(_seg_kept)
                elif isinstance(_seg_mode, (int, float)):
                    _seg_div = float(_seg_mode) * math.sqrt(self.head_dim)
                else:
                    _seg_div = math.sqrt(self.head_dim)
                score_parts.append(torch.matmul(q_seg, seg_k.transpose(2, 3)) / _seg_div)
            score_parts.append(torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim))
            attn_weights = torch.cat(score_parts, dim=-1)
        else:
            key_pruned = past_key_value.key_cache_pruned[self.layer_idx]
            base_mask = past_key_value.mask[self.layer_idx]
            _is_residual = past_key_value.residual_mode[self.layer_idx] if self.layer_idx < len(past_key_value.residual_mode) else False
            _is_gqa = past_key_value.gqa_pruned[self.layer_idx] if self.layer_idx < len(past_key_value.gqa_pruned) else False
            if _is_gqa:
                key_pruned = repeat_kv(key_pruned, self.num_key_value_groups)
                base_mask = base_mask.repeat_interleave(self.num_key_value_groups, dim=1)
            mask = base_mask.unsqueeze(2)
            masked_query_states = query_states[mask].view(query_states.shape[0], query_states.shape[1], query_states.shape[2], -1)
            if _is_residual:
                _pruner_mode_dec = getattr(self.config, 'think_pruner_mode', '')
                drop_mask = (~base_mask).unsqueeze(2)
                dropped_query_states = query_states[drop_mask].view(query_states.shape[0], query_states.shape[1], query_states.shape[2], -1)
                if _pruner_mode_dec.endswith('_resabs'):
                    drop_avg = dropped_query_states.abs().mean(dim=-1, keepdim=True)
                elif _pruner_mode_dec.endswith('_resl2'):
                    drop_avg = dropped_query_states.pow(2).mean(dim=-1, keepdim=True).sqrt()
                elif _pruner_mode_dec.endswith('_resavg_scaled'):
                    import math as _math
                    drop_avg = dropped_query_states.mean(dim=-1, keepdim=True) * _math.sqrt(dropped_query_states.shape[-1])
                else:
                    drop_avg = dropped_query_states.mean(dim=-1, keepdim=True)
                masked_query_states = torch.cat([masked_query_states, drop_avg], dim=-1)
            # Decode-time attention temperature (see docs/SQRT_TEMPERATURE_CALIBRATION.md).
            _mode = getattr(self.config, 'decode_sqrt_mode', 'd')
            _kept_dim = masked_query_states.shape[-1]
            if _mode == 'kept':
                _div = math.sqrt(_kept_dim)
            elif isinstance(_mode, (int, float)):
                _div = float(_mode) * math.sqrt(self.head_dim)
            else:
                _div = math.sqrt(self.head_dim)
            attn_weights1 = torch.matmul(masked_query_states, key_pruned.transpose(2, 3)) / _div
            attn_weights2 = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            attn_weights = torch.cat([attn_weights1, attn_weights2], dim=-1)
        _used_sdpa_prefill = False
    else:
        _default_sdpa = getattr(self.config, "model_type", None) in ("qwen2", "qwen3")
        if _SDPA_AVAILABLE and getattr(self.config, "prefill_sdpa", _default_sdpa):
            attn_output = _prefill_output_sdpa(query_states, key_states, value_states)
            attn_weights = None
            _used_sdpa_prefill = True
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            _used_sdpa_prefill = False

    if not _used_sdpa_prefill:
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : attn_weights.shape[-1]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if getattr(self.config, "pretraining_tp", 1) > 1:
        attn_output = attn_output.split(self.hidden_size // getattr(self.config, "pretraining_tp", 1), dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // getattr(self.config, "pretraining_tp", 1), dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(getattr(self.config, "pretraining_tp", 1))])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def prepare_inputs_for_generation_llama(
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    _write_prepare_trace(event="prepare_inputs_enter", input_len=int(input_ids.shape[1]), attention_len=None if attention_mask is None else int(attention_mask.shape[1]), past_is_none=bool(past_key_values is None), past_type=None if past_key_values is None else type(past_key_values).__name__, use_cache=kwargs.get("use_cache"), cache_position_none=(kwargs.get("cache_position", None) is None))
    # transformers >=4.44 pre-allocates an empty DynamicCache on the first call.
    # Treat it as None so our legacy-tuple-based branches still work.
    if past_key_values is not None and hasattr(past_key_values, "__len__") and len(past_key_values) == 0:
        past_key_values = None
    if past_key_values is None:
        for layer in self.model.layers:
            layer.self_attn.kv_seq_len = 0
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
            max_cache_length = past_key_values.get_max_length()
        else:
            first_layer_cache = past_key_values[0]
            if len(first_layer_cache) == 4 and isinstance(first_layer_cache[3], str) and first_layer_cache[3] == 'seg':
                seg_tuple, key_recent = first_layer_cache[0], first_layer_cache[1]
                seg_len = sum(sk.shape[2] for sk, _ in seg_tuple)
                cache_length = seg_len + key_recent.shape[2]
                past_length = getattr(self.model.layers[0].self_attn, "kv_seq_len", cache_length)
            elif len(first_layer_cache) in (4, 6, 7):
                key_states_pruned, key_states_recent = first_layer_cache[0], first_layer_cache[1]
                cache_length = key_states_pruned.shape[2] + key_states_recent.shape[2]
                if len(first_layer_cache) == 7 and first_layer_cache[6] is not None:
                    cache_length += first_layer_cache[6].shape[2]  # add sink tokens
                past_length = getattr(self.model.layers[0].self_attn, "kv_seq_len", cache_length)
            elif len(first_layer_cache) == 2:
                key_states, _value_states = first_layer_cache
                cache_length = past_length = key_states.shape[2]
            else:
                cache_length = past_length = getattr(self.model.layers[0].self_attn, "kv_seq_len", 0)
            max_cache_length = None
        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
            
            
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            
            
            
            position_ids = position_ids[:, -input_ids.shape[1] :]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    _write_prepare_trace(event="prepare_inputs_exit", model_input_len=None if model_inputs.get("input_ids") is None else int(model_inputs["input_ids"].shape[1]), position_len=None if model_inputs.get("position_ids") is None else int(model_inputs["position_ids"].shape[1]), attention_len=None if model_inputs.get("attention_mask") is None else int(model_inputs["attention_mask"].shape[1]), past_is_none=bool(model_inputs.get("past_key_values") is None), past_type=None if model_inputs.get("past_key_values") is None else type(model_inputs.get("past_key_values")).__name__, use_cache=model_inputs.get("use_cache"))
    return model_inputs

def sparsity_forward(
    attention_scores: torch.LongTensor = None,
    method: str = "matrix"
    ):
    
    attention_scores = torch.stack(attention_scores).squeeze(1) # [layer_idx, num_heads, seq_length]
    
    if method== "matrix":
        
        softmaxed_attention_scores = F.softmax(attention_scores, dim=-1)
        head_sparsity = torch.exp(softmaxed_attention_scores * 10).sum(dim=-1) # [layer_idx, num_head]
        layer_sparsity = head_sparsity.mean(dim=-1) # [layer_idx]
        fixed_layer_sparsity = max(layer_sparsity) - layer_sparsity + (max(layer_sparsity) - min(layer_sparsity)) / 5
        return fixed_layer_sparsity
    
    elif method == "gini":
        layer_sparsity = []
        for layer_idx in range(attention_scores.shape[0]):
            head_sparsity = []
            for head_idx in range(attention_scores.shape[1]):
                head_attn = attention_scores[layer_idx, head_idx, :].cpu().numpy()
                gini_value = gini(head_attn)
                head_sparsity.append(gini_value)
                
            layer_gini.append(round(sum(head_sparsity) / len(head_sparsity) * 100))
        layer_gini = torch.tensor(layer_gini)
        return layer_gini
    
    elif method == "entropy":
        from torch.distributions import Categorical
        entropy = Categorical(probs = softmaxed_attention_scores).entropy()
        
        pass

def llama_model_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
    
        init_snapkv(self)
    
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:
            if os.environ.get('LEANKFAIR_DEBUG'):
                import sys
                _ptype = type(past_key_values).__module__ + '.' + type(past_key_values).__name__ if past_key_values is not None else 'None'
                _has_sink = hasattr(past_key_values, 'key_cache_sink') if past_key_values is not None else False
                _sink0 = None
                if _has_sink and len(past_key_values.key_cache_sink) > 0:
                    _sink0 = past_key_values.key_cache_sink[0]
                    _sink0 = None if _sink0 is None else list(_sink0.shape)
                print(f"[leankfair-debug MODEL_FWD_IN] type={_ptype} has_sink_attr={_has_sink} sink0={_sink0}", file=sys.stderr, flush=True)
            if not isinstance(past_key_values, Cache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_seen_tokens = past_key_values.get_seq_length()
            logical_seen_tokens = getattr(self.layers[0].self_attn, "kv_seq_len", 0)
            if logical_seen_tokens > past_seen_tokens:
                past_seen_tokens = logical_seen_tokens

        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # 4.43+ signature: _update_causal_mask(attention_mask, input_tensor, cache_position, past_key_values, output_attentions)
        # 4.40 signature: _update_causal_mask(attention_mask, input_tensor, cache_position, past_seen_tokens)
        # Mistral (4.40) has no _update_causal_mask — fall back to the module-level 4d mask builder.
        if hasattr(self, "_update_causal_mask"):
            try:
                causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)
            except TypeError:
                causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_seen_tokens)
        else:
            from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
            causal_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (inputs_embeds.shape[0], inputs_embeds.shape[1]),
                inputs_embeds,
                past_seen_tokens,
            )

        # print(f"debug seq_length {inputs_embeds.shape[1]} past_seen_tokens {past_seen_tokens}")

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_seen_tokens = past_key_values.get_seq_length(layer_idx) if past_key_values is not None else 0
            # print(f"debug layer_idx {layer_idx} past_seen_tokens {past_seen_tokens}")
            
            

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        
        if output_attentions == True and all_self_attns[0] is not None:
            
            
            sparsity = sparsity_forward(all_self_attns)
            
            # In DynamicKV setting, the max_capacity_prompt needs to be fixed like SnapKV
            token_allocation = torch.round(sparsity / sum(sparsity) * len(self.layers) * self.layers[0].self_attn.config.max_capacity_prompt)
            

            for layer_idx in range(len(self.layers)):
                window_size = self.layers[layer_idx].self_attn.config.window_size
                max_capacity_prompt = int(token_allocation[layer_idx].item())
                
                key_states, value_states = next_decoder_cache.key_cache[layer_idx], next_decoder_cache.value_cache[layer_idx]
                
                indices = all_self_attns[layer_idx].topk(max_capacity_prompt, dim=-1).indices
                indices = indices.unsqueeze(-1).expand(-1, -1, -1, key_states.shape[-1])
                
                k_past_compress = key_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                v_past_compress = value_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                k_cur = key_states[:, :, -window_size:, :]
                v_cur = value_states[:, :, -window_size:, :]
                
                key_states = torch.cat([k_past_compress, k_cur], dim = 2)
                value_states = torch.cat([v_past_compress, v_cur], dim = 2)
                
                next_decoder_cache.key_cache[layer_idx] = k_past_compress
                next_decoder_cache.value_cache[layer_idx] = v_past_compress

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )