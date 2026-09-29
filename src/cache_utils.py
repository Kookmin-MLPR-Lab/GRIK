import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


@dataclass
class Cache:
    """
    Base, abstract class for all caches. The actual data structure is specific to each subclass.
    """

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                cache to be created.

        Return:
            A tuple containing the updated key and value states.
        """
        raise NotImplementedError("Make sure to implement `update` in a subclass.")

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        raise NotImplementedError("Make sure to implement `get_seq_length` in a subclass.")

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states, if there is any."""
        raise NotImplementedError("Make sure to implement `get_max_length` in a subclass.")

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # Cache without size limit -> all cache is usable
        # Cache with size limit -> if the length cache plus the length of the new inputs is larger the maximum cache
        #   length, we will need to evict part of the cache (and thus not all cache is usable)
        max_length = self.get_max_length()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

    @property
    def seen_tokens(self):
        logger.warning_once(
            "The `seen_tokens` attribute is deprecated and will be removed in v4.41. Use the `cache_position` "
            "model input instead."
        )
        if hasattr(self, "_seen_tokens"):
            return self._seen_tokens
        else:
            return None


class DynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self) -> None:
        self.key_cache_pruned: List[torch.Tensor] = []
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.mask = []
        self.residual_mode: List[bool] = []
        self.gqa_pruned: List[bool] = []
        self.segments: List[Optional[List]] = []  # per-layer list of (pruned_K, mask) tuples
        # LeanK-fair mode: sink tokens stored at full d_head (separate from pruned-middle).
        # key_cache_sink[L] is None for layers not using leankfair mode — keeps the default
        # 2-way (past_pruned + recent) path untouched.
        self.key_cache_sink: List[Optional[torch.Tensor]] = []

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            # Keep sink list aligned even when this layer is non-leankfair.
            while len(self.key_cache_sink) < len(self.key_cache):
                self.key_cache_sink.append(None)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update_think(
        self,
        key_states_pruned: torch.Tensor,
        key_states: torch.Tensor,
        mask,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,   # kept for API parity; unused (B14)
        residual: bool = False,
        gqa_pruned: bool = False,
    ) -> None:
        """Prefill 시 압축된 K/V를 layer_idx 위치에 저장.

        호출처: src/llama_model.py:165 [D-2] (우리 메인 path)
        입력: key_pruner_iterative가 뱉은 4-tuple
          key_states_pruned : [1, 8, past, kept=77]  (오래된 토큰 + 잘린 채널)
          key_states        : [1, 8, window=32, 128] (최근 창, 온전한 채널)
          mask              : [1, 8, 128] bool        (각 group 채널 마스크)
          value_states      : [1, 8, 128, 128]        (V, 8 head slice됨)

        저장 구조 (per-layer 리스트 6개):
          self.key_cache_pruned[L] = key_states_pruned
          self.key_cache[L]        = key_states        (=recent window; decode시 append됨)
          self.value_cache[L]      = value_states
          self.mask[L]             = mask
          self.residual_mode[L]    = residual (우리 메인 path는 False)
          self.gqa_pruned[L]       = True     (우리 메인 path는 True)

        Decode 시 이 state를 읽어가는 곳:
          - src/llama_model.py:229 key_cache_pruned, mask 로드
          - src/llama_model.py:208 gqa_pruned 플래그 체크
          - src/llama_model.py:180 repeat_kv로 8h → 32h view 확장
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if len(self.key_cache) <= layer_idx:
            self.key_cache_pruned.append(key_states_pruned)
            self.mask.append(mask)
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            self.residual_mode.append(residual)
            self.gqa_pruned.append(gqa_pruned)
            self.key_cache_sink.append(None)
        else:
            # Overwrite existing layer state rather than silently ignoring (B2).
            self.key_cache_pruned[layer_idx] = key_states_pruned
            self.mask[layer_idx] = mask
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
            self.residual_mode[layer_idx] = residual
            self.gqa_pruned[layer_idx] = gqa_pruned
            # Make sure sink list stays aligned; clear any stale entry from a previous run.
            while len(self.key_cache_sink) <= layer_idx:
                self.key_cache_sink.append(None)
            self.key_cache_sink[layer_idx] = None

    def update_think_leankfair(
        self,
        key_states_sink: torch.Tensor,
        key_states_pruned: torch.Tensor,
        key_states_recent: torch.Tensor,
        mask,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,  # unused, kept for API parity
    ) -> None:
        if os.environ.get('LEANKFAIR_DEBUG') and layer_idx == 0:
            import sys
            print(f"[leankfair-debug STORE] layer={layer_idx} sink_shape={list(key_states_sink.shape)} pruned_shape={list(key_states_pruned.shape)} recent_shape={list(key_states_recent.shape)} v_shape={list(value_states.shape)}", file=sys.stderr, flush=True)
        """Prefill storage for LeanK-fair 3-way split.

        Layout matches LeanK's token-position policy:
          [sink at full d_head] | [middle at kept-dim channels] | [recent at full d_head]
        V is stored as a single contiguous 8h tensor in that same token order, so decode-
        time attention concat([w_sink, w_middle, w_recent]) lines up with V directly.

        Shapes (GQA, 8 KV heads):
          key_states_sink   : [1, 8, sink_size, head_dim]      — full channels
          key_states_pruned : [1, 8, middle_len, kept_dim]     — channel-pruned past
          key_states_recent : [1, 8, recent_size, head_dim]    — full channels, decode appends here
          mask              : [1, 8, head_dim] bool
          value_states      : [1, 8, total_len, head_dim]      — full V over all prefill tokens
        """
        if layer_idx == 0:
            self._seen_tokens += (
                key_states_sink.shape[-2] + key_states_pruned.shape[-2] + key_states_recent.shape[-2]
            )

        if len(self.key_cache) <= layer_idx:
            self.key_cache_pruned.append(key_states_pruned)
            self.mask.append(mask)
            self.key_cache.append(key_states_recent)
            self.value_cache.append(value_states)
            self.residual_mode.append(False)
            self.gqa_pruned.append(True)
            self.key_cache_sink.append(key_states_sink)
        else:
            self.key_cache_pruned[layer_idx] = key_states_pruned
            self.mask[layer_idx] = mask
            self.key_cache[layer_idx] = key_states_recent
            self.value_cache[layer_idx] = value_states
            self.residual_mode[layer_idx] = False
            self.gqa_pruned[layer_idx] = True
            while len(self.key_cache_sink) <= layer_idx:
                self.key_cache_sink.append(None)
            self.key_cache_sink[layer_idx] = key_states_sink

    def flush_recent_if_needed(self, layer_idx: int, recent_size: int, threshold: int) -> bool:
        """Decode-time flush: migrate oldest tokens from recent buffer (key_cache[L])
        into pruned-past buffer (key_cache_pruned[L]) when recent grows too large.

        Why this exists:
          SnapKV/ThinK-family pruners only compress the KV cache during prefill. In
          decode every new K is simply concat'd onto key_cache[L] at full channel
          dim, so a long generation bloats the cache back toward Full-KV memory.
          This method closes that gap by periodically re-applying the (prefill)
          channel mask to older decode tokens and moving them into
          key_cache_pruned[L] which is already at kept-channel dim.

        Invariants preserved:
          * value_cache[L] is NOT touched — its order is
            [past_prefill_tokens..., recent_prefill_tokens..., decode_tokens...]
            and the attention decode branch concatenates scores in the same
            [past | recent+decode] order. Moving K tokens from recent→past
            doesn't require moving V tokens because their V positions already
            fall within the "past" V slice by construction.
          * No re-scoring — we reuse the prefill mask. The Q history needed to
            recompute channel importance has already been discarded.
          * No token eviction here (Tier 1). Every flushed token is kept,
            just channel-compressed. Optional H2O-style eviction at flush
            time is a Tier 1.5 follow-up.

        Args:
            layer_idx: which transformer layer
            recent_size: how many tokens to keep in the recent buffer after
                flushing. Must match the prefill recent_size.
            threshold: only flush when key_cache[L].shape[-2] > threshold.
                Typical: 2 * recent_size, so flush fires after ~recent_size
                new decode tokens have accumulated.

        Returns:
            True if a flush happened, False otherwise.
        """
        # This only makes sense for GQA-compressed layers that have a
        # prefill-computed channel mask. Bail safely otherwise.
        if layer_idx >= len(self.key_cache_pruned) or self.key_cache_pruned[layer_idx] is None:
            return False
        if layer_idx >= len(self.mask) or self.mask[layer_idx] is None:
            return False

        recent_len = self.key_cache[layer_idx].shape[-2]
        if recent_len <= threshold:
            return False

        flush_count = recent_len - recent_size
        if flush_count <= 0:
            return False

        # [1] Slice oldest `flush_count` tokens from the recent buffer.
        k_old = self.key_cache[layer_idx][:, :, :flush_count, :]  # [1, H_kv, flush_count, head_dim]

        # [2] Apply the prefill channel mask to compress them down to kept_dim.
        #     mask[layer_idx]: [1, H_kv, head_dim] bool
        mask = self.mask[layer_idx]
        # Expand mask across the token axis so the boolean index matches k_old:
        mask_exp = mask.unsqueeze(2).expand(-1, -1, flush_count, -1)
        kept_dim = int(mask[0, 0].sum().item())  # same for all heads by construction
        k_compressed = k_old[mask_exp].view(k_old.shape[0], k_old.shape[1], flush_count, kept_dim)

        # [3] Concat onto pruned-past cache.
        self.key_cache_pruned[layer_idx] = torch.cat(
            [self.key_cache_pruned[layer_idx], k_compressed], dim=-2
        )

        # [4] Trim recent buffer to the last `recent_size` tokens.
        self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, flush_count:, :].contiguous()

        # Debug hook (opt-in via env var) to verify flush is actually firing.
        if os.environ.get("KVPRUNER_FLUSH_DEBUG"):
            import sys
            print(f"[flush] layer={layer_idx} moved={flush_count} "
                  f"past_now={self.key_cache_pruned[layer_idx].shape[-2]} "
                  f"recent_now={self.key_cache[layer_idx].shape[-2]}",
                  file=sys.stderr, flush=True)

        return True

    def update_think_segmented(self, segments, key_states_recent, value_states, layer_idx, cache_kwargs=None):
        """Store segmented pruned K cache (each segment has its own mask)."""
        if layer_idx == 0:
            self._seen_tokens += key_states_recent.shape[-2]
            for seg_k, _ in segments:
                self._seen_tokens += seg_k.shape[-2]
        if len(self.key_cache) <= layer_idx:
            self.segments.append(segments)
            self.key_cache_pruned.append(None)
            self.mask.append(None)
            self.key_cache.append(key_states_recent)
            self.value_cache.append(value_states)
            self.residual_mode.append(False)
            self.gqa_pruned.append(False)
            self.key_cache_sink.append(None)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache) <= layer_idx:
            return 0
        seq_len = self.key_cache[layer_idx].shape[-2]
        if layer_idx < len(self.segments) and self.segments[layer_idx]:
            for seg_k, _ in self.segments[layer_idx]:
                seq_len += seg_k.shape[-2]
        elif layer_idx < len(self.key_cache_pruned) and self.key_cache_pruned[layer_idx] is not None:
            seq_len += self.key_cache_pruned[layer_idx].shape[-2]
        return seq_len

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Beam search 시 cache를 beam index 순으로 재배열.

        참고: 우리 LongBench 실험은 num_beams=1이라 이 경로 호출 안 됨.
              하지만 라이브러리로 배포 시 beam search 사용자 위해 B1 fix로 6개 list 모두 reorder.
              (이전 버전은 key_cache / value_cache 2개만 reorder → 다른 beam들이 같은
               pruned cache를 공유해서 잘못된 생성 결과를 내는 버그 있었음.)
        """
        def _reorder(t, idx):
            if t is None or not hasattr(t, "index_select"):
                return t
            return t.index_select(0, idx.to(t.device))

        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = _reorder(self.key_cache[layer_idx], beam_idx)
            self.value_cache[layer_idx] = _reorder(self.value_cache[layer_idx], beam_idx)
            if layer_idx < len(self.key_cache_pruned):
                self.key_cache_pruned[layer_idx] = _reorder(self.key_cache_pruned[layer_idx], beam_idx)
            if layer_idx < len(self.mask):
                self.mask[layer_idx] = _reorder(self.mask[layer_idx], beam_idx)
            if layer_idx < len(self.segments) and self.segments[layer_idx] is not None:
                self.segments[layer_idx] = [(_reorder(sk, beam_idx), _reorder(sm, beam_idx))
                                             for sk, sm in self.segments[layer_idx]]
            if layer_idx < len(self.key_cache_sink) and self.key_cache_sink[layer_idx] is not None:
                self.key_cache_sink[layer_idx] = _reorder(self.key_cache_sink[layer_idx], beam_idx)

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            if layer_idx < len(self.segments) and self.segments[layer_idx]:
                # Segmented mode: (segments_tuple, key_recent, value, 'seg')
                seg_tuple = tuple((sk, sm) for sk, sm in self.segments[layer_idx])
                legacy_cache += ((seg_tuple, self.key_cache[layer_idx], self.value_cache[layer_idx], 'seg'),)
            elif layer_idx < len(self.key_cache_pruned) and self.key_cache_pruned[layer_idx] is not None:
                _res = self.residual_mode[layer_idx] if layer_idx < len(self.residual_mode) else False
                _gqa = self.gqa_pruned[layer_idx] if layer_idx < len(self.gqa_pruned) else False
                _sink = self.key_cache_sink[layer_idx] if layer_idx < len(self.key_cache_sink) else None
                if _sink is not None:
                    # 7-tuple: leankfair layer (sink at full d_head)
                    legacy_cache += ((self.key_cache_pruned[layer_idx], self.key_cache[layer_idx], self.mask[layer_idx], self.value_cache[layer_idx], _res, _gqa, _sink),)
                else:
                    legacy_cache += ((self.key_cache_pruned[layer_idx], self.key_cache[layer_idx], self.mask[layer_idx], self.value_cache[layer_idx], _res, _gqa),)
            else:
                legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                entry = past_key_values[layer_idx]
                if len(entry) == 4 and isinstance(entry[3], str) and entry[3] == 'seg':
                    seg_tuple, key_recent, value_states = entry[0], entry[1], entry[2]
                    segments = [(sk, sm) for sk, sm in seg_tuple]
                    cache.update_think_segmented(segments, key_recent, value_states, layer_idx)
                elif len(entry) == 7:
                    # leankfair layer: extra element is key_sink (full d_head)
                    key_states_pruned, key_states, mask, value_states, _res, _gqa, key_sink = entry
                    cache.update_think_leankfair(key_sink, key_states_pruned, key_states, mask, value_states, layer_idx)
                elif len(entry) == 6:
                    key_states_pruned, key_states, mask, value_states, _res, _gqa = entry
                    cache.update_think(key_states_pruned, key_states, mask, value_states, layer_idx, residual=_res, gqa_pruned=_gqa)
                elif len(entry) == 4:
                    key_states_pruned, key_states, mask, value_states = entry
                    cache.update_think(key_states_pruned, key_states, mask, value_states, layer_idx)
                elif len(entry) == 2:
                    key_states, value_states = entry
                    cache.update(key_states, value_states, layer_idx)
                else:
                    raise ValueError(f'Unsupported legacy cache entry length: {len(entry)}')
        return cache


class SinkCache(Cache):
    """
    A cache that as described in the [Attention Sinks paper](https://arxiv.org/abs/2309.17453). It allows the model to
    generate beyond the length of its context window, without losing fluency in the conversation. As it discards past
    tokens, the model will lose the ability to generate tokens that depend on the context that was discarded.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    Parameters:
        window_length (`int`):
            The length of the context window.
        num_sink_tokens (`int`):
            The number of sink tokens. See the original paper for more information.
    """

    def __init__(self, window_length: int, num_sink_tokens: int) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.window_length = window_length
        self.num_sink_tokens = num_sink_tokens
        self.cos_sin_cache = {}
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_key_rotary_pos_emb(
        self, key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        rotated_key_states = (key_states * cos) + (self._rotate_half(key_states) * sin)
        return rotated_key_states

    def _get_rerotation_cos_sin(
        self, key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if key_states.shape[-2] not in self.cos_sin_cache:
            # Upcast to float32 temporarily for better accuracy
            cos = cos.to(torch.float32)
            sin = sin.to(torch.float32)

            # Compute the cos and sin required for back- and forward-rotating to one position earlier in the sequence
            original_cos = cos[self.num_sink_tokens + key_states.shape[-2] :]
            shifted_cos = cos[self.num_sink_tokens : -key_states.shape[-2]]
            original_sin = sin[self.num_sink_tokens + key_states.shape[-2] :]
            shifted_sin = sin[self.num_sink_tokens : -key_states.shape[-2]]
            rerotation_cos = original_cos * shifted_cos + original_sin * shifted_sin
            rerotation_sin = -original_sin * shifted_cos + original_cos * shifted_sin

            self.cos_sin_cache[key_states.shape[-2]] = (
                rerotation_cos.to(key_states.dtype).unsqueeze(0),
                rerotation_sin.to(key_states.dtype).unsqueeze(0),
            )
        return self.cos_sin_cache[key_states.shape[-2]]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # Workaround to make 'key_states.shape[-2] + past_key_value.get_seq_length(self.layer_idx)' <= window_length
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""
        return self.window_length

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The following arguments can be used in `SinkCache`: `sin`,
                `cos` and `partial_rotation_size`. These arguments are used with models using RoPE, to recompute the
                rotation as the tokens are shifted.

        Return:
            A tuple containing the updated key and value states.
        """
        # Optional kwargs for `SinkCache` -- needed on models using RoPE. `partial_rotation_size` is used on models
        # with partially rotated position embeddings, like Phi or Persimmon.
        sin = cache_kwargs.get("sin")
        cos = cache_kwargs.get("cos")
        partial_rotation_size = cache_kwargs.get("partial_rotation_size")
        using_rope = cos is not None and sin is not None

        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # [bsz, num_heads, seq_len, head_dim]
        if len(self.key_cache) <= layer_idx:
            # Empty cache
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)

        elif key_states.shape[-2] + self.get_seq_length(layer_idx) < self.window_length:
            # Growing cache
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        else:
            # Shifting cache
            keys_to_keep = self.key_cache[layer_idx][
                :, :, -self.window_length + self.num_sink_tokens + key_states.shape[-2] :
            ]

            # On RoPE models, we need to recompute the Key rotation as the tokens are shifted
            if using_rope:
                rerotation_cos, rerotation_sin = self._get_rerotation_cos_sin(
                    key_states, cos[: self.window_length], sin[: self.window_length]
                )
                if partial_rotation_size is not None:
                    keys_to_keep, keys_pass = (
                        keys_to_keep[..., :partial_rotation_size],
                        keys_to_keep[..., partial_rotation_size:],
                    )
                keys_to_keep = self._apply_key_rotary_pos_emb(keys_to_keep, rerotation_cos, rerotation_sin)
                if partial_rotation_size is not None:
                    keys_to_keep = torch.cat((keys_to_keep, keys_pass), dim=-1)

            # Concatenate sink tokens, shifted & rotated tokens (if needed), and new tokens
            sink_keys = self.key_cache[layer_idx][:, :, : self.num_sink_tokens]
            self.key_cache[layer_idx] = torch.cat([sink_keys, keys_to_keep, key_states], dim=-2)

            sink_values = self.value_cache[layer_idx][:, :, : self.num_sink_tokens]
            values_to_keep = self.value_cache[layer_idx][
                :, :, -self.window_length + self.num_sink_tokens + value_states.shape[-2] :
            ]
            self.value_cache[layer_idx] = torch.cat([sink_values, values_to_keep, value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))


class StaticCache(Cache):
    """
    Static Cache class to be used with `torch.compile(model)`.

    Parameters:
        config (`PretrainedConfig):
            The configuration file defining the `max_position_embeddings`, `hidden_size` and `num_attention_heads`
            required to initialize the static cache.
        max_batch_size (`int`):
            The maximum batch size with which the model will be used.
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`torch.device`):
            The device on which the cache should be initialized. Should be the same as the layer.
        dtype (*optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
    """

    def __init__(self, config: PretrainedConfig, max_batch_size: int, max_cache_len: int, device, dtype=None) -> None:
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_cache_len = config.max_position_embeddings if max_cache_len is None else max_cache_len
        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        self.head_dim = (
            config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        )

        self.dtype = dtype if dtype is not None else torch.float32
        self.num_key_value_heads = (
            config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        )

        cache_shape = (max_batch_size, self.num_key_value_heads, self.max_cache_len, self.head_dim)
        self.key_cache: torch.Tensor = torch.zeros(cache_shape, dtype=self.dtype, device=device)
        self.value_cache: torch.Tensor = torch.zeros(cache_shape, dtype=self.dtype, device=device)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for. Kept for backward compatibility
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The `StaticCache` just needs the `q_len`
                to know how much of the cache it should overwrite.

        Return:
            A tuple containing the updated key and value states.
        """
        new_cache_positions = cache_kwargs.get("cache_position")
        k_out = self.key_cache
        v_out = self.value_cache

        k_out[:, :, new_cache_positions] = key_states
        v_out[:, :, new_cache_positions] = value_states

        return k_out, v_out

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states that were seen by the model. `layer_idx` kept for BC"""
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        # TODO: This is error prone, a filled cache may be `0.0`. Let's use a stateless integer instead, after
        # https://github.com/pytorch/pytorch/issues/120248 is fixed
        return (self.key_cache[0, 0].any(dim=-1)).sum()

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return self.max_cache_len

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        device = self.key_cache.device
        self.key_cache = self.key_cache.index_select(0, beam_idx.to(device))
        device = self.value_cache.device
        self.value_cache = self.value_cache.index_select(0, beam_idx.to(device))

    def to_legacy_cache(self):
        """Dummy function for BC. We have to keep it because otherwise the call in the forward of models will break it"""
        return None