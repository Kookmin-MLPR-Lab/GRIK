
import os
import json
import re
import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math
import hashlib

# perform qk calculation and get indices
# this version will not update in inference mode

def _write_pruning_monitor(event, **payload):
    monitor_path = os.environ.get("THINK_PRUNING_MONITOR")
    if not monitor_path:
        return
    try:
        record = {"event": event, **payload}
        with open(monitor_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass



def _mask_hash(mask: torch.Tensor) -> str:
    try:
        arr = mask.detach().to(device="cpu", dtype=torch.bool).numpy()
        return hashlib.md5(arr.tobytes()).hexdigest()
    except Exception:
        return "unavailable"


def _write_pruning_trace(event, **payload):
    trace_path = os.environ.get("THINK_PRUNING_TRACE")
    if not trace_path:
        return
    try:
        record = {"event": event, **payload}
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _tensor_to_list(tensor, digits=8):
    try:
        arr = tensor.detach().to(device="cpu", dtype=torch.float32)
        if digits is not None:
            scale = 10 ** digits
            arr = torch.round(arr * scale) / scale
        return arr.tolist()
    except Exception:
        return None


def _indices_per_head(mask: torch.Tensor):
    try:
        return [torch.nonzero(mask[0, h], as_tuple=False).squeeze(-1).detach().cpu().tolist() for h in range(mask.shape[1])]
    except Exception:
        return None


def _project_head_influence(head_out, o_proj_weight, num_heads, head_dim):
    """각 query head의 'output 기여도' 측정. 논문 §3.3의 φ_h.

    계산:  φ_h = mean_over_window( || W_O^(h) · a_h ||₂ )
    - a_h: head h의 attention 출력 (softmax(QK/√d) · V 결과)
    - W_O^(h): o_proj 가중치의 h번째 column slice → a_h가 residual stream에 더해지는 크기
    - head_out[:, qh] @ w_slice.T 하면 [B, window, D_model] → norm → window별 기여 크기,
      mean으로 window 축 축약 → head별 단일 scalar φ_h.

    호출처:
      - :738 key_pruner_iterative 루프 내부 (매 step마다 현재 mask로 φ 재계산)
      - :582 key_pruner_gqa_aware 내부 (one-shot 모드)
      - :917 update_thinkv 내부 eviction branch (token eviction용 φ)
    """
    proj = []
    for qh in range(num_heads):
        start = qh * head_dim
        end = (qh + 1) * head_dim
        w_slice = o_proj_weight[:, start:end]                 # [D_model, head_dim]
        contrib = head_out[:, qh] @ w_slice.transpose(0, 1)   # [B, window, D_model]
        proj.append(contrib.norm(dim=-1).mean(dim=-1))        # [B] per head
    return torch.stack(proj, dim=1)                           # [B, num_heads]


def _select_pair_keep_mask(score_pairs, keep_pairs):
    half = score_pairs.shape[-1]
    keep_pairs = max(1, min(int(keep_pairs), half))
    top_idx = torch.topk(score_pairs, keep_pairs, dim=-1, largest=True).indices
    pair_keep = torch.zeros_like(score_pairs, dtype=torch.bool)
    pair_keep.scatter_(-1, top_idx, True)
    return torch.cat([pair_keep, pair_keep], dim=-1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _prepare_cos_sin_for_x(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    if cos.shape[-1] != x.shape[-1] and cos.ndim >= 2 and cos.shape[-2] == x.shape[-1]:
        cos = cos.transpose(-1, -2)
        sin = sin.transpose(-1, -2)
    if cos.ndim == x.ndim - 1:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    elif cos.ndim == x.ndim - 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    return cos.to(device=x.device, dtype=x.dtype), sin.to(device=x.device, dtype=x.dtype)


def _inverse_rotary_from_cos_sin(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos, sin = _prepare_cos_sin_for_x(x, cos, sin)
    return x * cos - _rotate_half(x) * sin


def key_pruner_query_driven(kv_states, q_states, recent_size=128, ratio=0.3, layer_idx=None):
    _, _, seqlen, head_dim = kv_states.shape
    k = int(head_dim * ratio)
    queries_norm = torch.pow(q_states[..., -32:, :], 2).mean(dim=2)
    keys_norm = torch.pow(kv_states, 2).mean(dim=2)
    score = queries_norm * keys_norm
    _, indices = torch.topk(score, k, dim=-1, largest=False)
    prune_idx = indices.sort().values
    mask = torch.zeros(score.shape, dtype=torch.bool).to(kv_states.device)
    mask = mask.scatter_(-1, prune_idx, 1)
    mask_k = mask.unsqueeze(2).expand(-1, -1, seqlen - recent_size, -1)
    kv_pruned = kv_states[:, :, :seqlen - recent_size, :][~mask_k].reshape(1, -1, seqlen - recent_size, head_dim - k)
    keep_mask = ~mask
    _write_pruning_monitor(
        'think_prune',
        mode='think',
        ratio=float(ratio),
        head_dim=int(head_dim),
        keep_dim=int(keep_mask[0, 0].sum().item()),
        drop_dim=int(head_dim - keep_mask[0, 0].sum().item()),
        keep_dims_per_head=[int(keep_mask[0, h].sum().item()) for h in range(keep_mask.shape[1])],
        mask_hash=_mask_hash(keep_mask),
        recent_size=int(recent_size),
        seqlen=int(seqlen),
        pruned_shape=list(kv_pruned.shape),
    )
    _write_pruning_trace(
        'think_trace',
        layer_idx=None if layer_idx is None else int(layer_idx),
        ratio=float(ratio),
        head_dim=int(head_dim),
        seqlen=int(seqlen),
        recent_size=int(recent_size),
        queries_norm=_tensor_to_list(queries_norm.squeeze(0)),
        keys_norm=_tensor_to_list(keys_norm.squeeze(0)),
        score=_tensor_to_list(score.squeeze(0)),
        pruned_indices=_tensor_to_list(prune_idx.squeeze(0), digits=None),
        kept_indices=_indices_per_head(keep_mask),
        keep_dims_per_head=[int(keep_mask[0, h].sum().item()) for h in range(keep_mask.shape[1])],
        mask_hash=_mask_hash(keep_mask),
    )
    return kv_pruned, kv_states[:, :, seqlen - recent_size:, :], keep_mask

def key_pruner_query_driven_gqa(kv_states, q_states, num_key_value_groups, recent_size=128, ratio=0.3, mode='think_gqa_mean', layer_idx=None):
    _, num_heads, seqlen, head_dim = kv_states.shape
    k = int(head_dim * ratio)
    num_kv_heads = num_heads // num_key_value_groups
    queries_norm = torch.pow(q_states[..., -32:, :], 2).mean(dim=2).squeeze(0)
    keys_norm = torch.pow(kv_states, 2).mean(dim=2).squeeze(0)
    per_head_score = queries_norm * keys_norm
    keep_mask = torch.zeros((1, num_heads, head_dim), dtype=torch.bool, device=kv_states.device)
    group_debug = []
    for kv_h in range(num_kv_heads):
        q_start = kv_h * num_key_value_groups
        q_end = (kv_h + 1) * num_key_value_groups
        group_score = per_head_score[q_start:q_end]
        group_queries = queries_norm[q_start:q_end]
        if mode == 'think_gqa_mean':
            chosen = group_score.mean(dim=0)
            rep_q = None
        elif mode == 'think_gqa_max':
            chosen = group_score.max(dim=0).values
            rep_q = None
        elif mode == 'think_gqa_argmax':
            rep_local = int(group_queries.sum(dim=-1).argmax().item())
            rep_q = q_start + rep_local
            chosen = group_score[rep_local]
        else:
            raise ValueError(f'Unknown GQA Think mode: {mode}')
        _, idx = torch.topk(chosen, k, largest=False)
        dim_drop = idx.sort().values
        dim_keep = torch.ones(head_dim, dtype=torch.bool, device=kv_states.device)
        dim_keep[dim_drop] = False
        keep_mask[:, q_start:q_end, :] = dim_keep.unsqueeze(0).unsqueeze(0).expand(1, q_end - q_start, head_dim)
        group_debug.append({
            'kv_head': int(kv_h),
            'mode': mode,
            'q_head_range': [int(q_start), int(q_end - 1)],
            'queries_norm': _tensor_to_list(group_queries),
            'group_score': _tensor_to_list(group_score),
            'aggregated_score': _tensor_to_list(chosen),
            'selected_dims': torch.nonzero(dim_keep, as_tuple=False).squeeze(-1).detach().cpu().tolist(),
            'representative_q_head': None if rep_q is None else int(rep_q),
        })
    past_len = seqlen - recent_size
    past_mask = keep_mask.unsqueeze(2).expand(-1, -1, past_len, -1)
    keep_dim = int(keep_mask[0, 0].sum().item())
    kv_kept = kv_states[:, :, :past_len, :][past_mask].reshape(1, num_heads, past_len, keep_dim)
    kv_pruned = kv_kept
    kv_recent = kv_states[:, :, past_len:, :]
    _write_pruning_monitor(
        'think_gqa_prune',
        mode=str(mode),
        ratio=float(ratio),
        head_dim=int(head_dim),
        num_heads=int(num_heads),
        num_kv_heads=int(num_kv_heads),
        keep_dim=int(keep_dim),
        drop_dim=int(head_dim - keep_dim),
        keep_dims_per_head=[int(keep_mask[0, h].sum().item()) for h in range(keep_mask.shape[1])],
        mask_hash=_mask_hash(keep_mask),
        recent_size=int(recent_size),
        seqlen=int(seqlen),
        pruned_shape=list(kv_pruned.shape),
    )
    _write_pruning_trace(
        'think_gqa_trace',
        layer_idx=None if layer_idx is None else int(layer_idx),
        mode=str(mode),
        ratio=float(ratio),
        head_dim=int(head_dim),
        seqlen=int(seqlen),
        recent_size=int(recent_size),
        queries_norm=_tensor_to_list(queries_norm),
        keys_norm=_tensor_to_list(keys_norm),
        per_head_score=_tensor_to_list(per_head_score),
        keep_dims_per_head=[int(keep_mask[0, h].sum().item()) for h in range(keep_mask.shape[1])],
        kept_indices=_indices_per_head(keep_mask),
        mask_hash=_mask_hash(keep_mask),
        groups=group_debug,
    )
    return kv_pruned, kv_recent, keep_mask


def key_pruner_query_driven_ours(kv_states, q_states, q_states_pre, k_states_pre, value_states, o_proj_weight, k_proj_weight, num_key_value_groups, recent_size=128, ratio=0.3, mode='argmax_projout', tau=1.0, layer_idx=None, cos=None, sin=None):
    bsz, num_heads, seqlen, head_dim = kv_states.shape
    assert bsz == 1, 'official LongBench runner uses batch_size=1'
    half = head_dim // 2
    num_kv_heads = num_heads // num_key_value_groups
    drop_pairs = max(1, min(int(round(half * ratio)), half - 1))
    keep_pairs = half - drop_pairs

    q_post = q_states[..., -32:, :]
    q_pre = q_states_pre[..., -32:, :]
    k_pre = k_states_pre
    val = value_states

    q_pre_norm = q_pre.pow(2).mean(dim=2).squeeze(0)
    k_pre_energy = k_pre.pow(2).mean(dim=2).squeeze(0)
    k_pre_var = k_pre.var(dim=2, unbiased=False).squeeze(0)
    q_post_norm = q_post.pow(2).mean(dim=2).squeeze(0)
    k_post_energy = kv_states.pow(2).mean(dim=2).squeeze(0)
    k_post_var = kv_states.var(dim=2, unbiased=False).squeeze(0)
    derope_dim = q_pre_norm * k_pre_energy
    derope_pair = derope_dim[:, :half] + derope_dim[:, half:]

    invrope_dim = None
    invrope_pair = None
    need_invrope = mode in ['invrope_norm_projout_scalar']
    if need_invrope:
        if cos is None or sin is None:
            raise ValueError('invrope mode requires cos/sin tensors')
        q_inv = _inverse_rotary_from_cos_sin(q_states, cos, sin)[..., -32:, :]
        k_inv = _inverse_rotary_from_cos_sin(kv_states, cos, sin)
        q_inv_norm = q_inv.pow(2).mean(dim=2).squeeze(0)
        k_inv_norm = k_inv.pow(2).mean(dim=2).squeeze(0)
        invrope_dim = q_inv_norm * k_inv_norm
        invrope_pair = invrope_dim[:, :half] + invrope_dim[:, half:]

    postrope_dim = q_post_norm * k_post_energy
    postrope_pair = postrope_dim[:, :half] + postrope_dim[:, half:]
    prerope_kenergy_dim = k_pre_energy
    prerope_kenergy_pair = prerope_kenergy_dim[:, :half] + prerope_kenergy_dim[:, half:]
    prerope_kvariance_dim = k_pre_var
    prerope_kvariance_pair = prerope_kvariance_dim[:, :half] + prerope_kvariance_dim[:, half:]
    k_proj_row_norm = k_proj_weight.detach().float().view(num_kv_heads, head_dim, -1).norm(dim=-1)
    prerope_kwanda_abs_dim = k_proj_row_norm.repeat_interleave(num_key_value_groups, dim=0) * k_pre.abs().mean(dim=2).squeeze(0)
    prerope_kwanda_abs_pair = prerope_kwanda_abs_dim[:, :half] + prerope_kwanda_abs_dim[:, half:]
    postrope_kwanda_abs_dim = k_proj_row_norm.repeat_interleave(num_key_value_groups, dim=0) * kv_states.abs().mean(dim=2).squeeze(0)
    postrope_kwanda_abs_pair = postrope_kwanda_abs_dim[:, :half] + postrope_kwanda_abs_dim[:, half:]
    postrope_kenergy_dim = k_post_energy
    postrope_kenergy_pair = postrope_kenergy_dim[:, :half] + postrope_kenergy_dim[:, half:]
    postrope_kvariance_dim = k_post_var
    postrope_kvariance_pair = postrope_kvariance_dim[:, :half] + postrope_kvariance_dim[:, half:]
    # Q²×K_var: Q importance weighted K discriminability
    postrope_qkvar_dim = q_post_norm * k_post_var
    postrope_qkvar_pair = postrope_qkvar_dim[:, :half] + postrope_qkvar_dim[:, half:]
    # √(K²×K_var): geometric mean of energy and variance
    postrope_kgeomean_dim = (k_post_energy * k_post_var).sqrt()
    postrope_kgeomean_pair = postrope_kgeomean_dim[:, :half] + postrope_kgeomean_dim[:, half:]

    attn_logits = torch.matmul(q_post, kv_states.transpose(2, 3)) / math.sqrt(head_dim)
    attn_probs = torch.softmax(attn_logits, dim=-1)
    head_out = torch.matmul(attn_probs, val)
    attn_influence = head_out.norm(dim=-1).mean(dim=-1).squeeze(0)
    proj_influence = _project_head_influence(head_out, o_proj_weight, num_heads, head_dim).squeeze(0)

    keep_mask = torch.zeros((1, num_heads, head_dim), dtype=torch.bool, device=kv_states.device)
    group_debug = []
    for kv_h in range(num_kv_heads):
        q_start = kv_h * num_key_value_groups
        q_end = (kv_h + 1) * num_key_value_groups
        if mode in ['postrope_norm_projout_scalar']:
            group_scores = postrope_pair[q_start:q_end]
            group_scores_dim = postrope_dim[q_start:q_end]
            coord_source = 'postrope_qk'
        elif mode in ['prerope_kenergy_norm_projout_scalar']:
            group_scores = prerope_kenergy_pair[q_start:q_end]
            group_scores_dim = prerope_kenergy_dim[q_start:q_end]
            coord_source = 'prerope_kenergy'
        elif mode in ['prerope_kvariance_norm_projout_scalar']:
            group_scores = prerope_kvariance_pair[q_start:q_end]
            group_scores_dim = prerope_kvariance_dim[q_start:q_end]
            coord_source = 'prerope_kvariance'
        elif mode in ['prerope_kwanda_abs_scalar']:
            group_scores = prerope_kwanda_abs_pair[q_start:q_end]
            group_scores_dim = prerope_kwanda_abs_dim[q_start:q_end]
            coord_source = 'prerope_kwanda_abs'
        elif mode in ['postrope_kwanda_abs_norm_projout_scalar']:
            group_scores = postrope_kwanda_abs_pair[q_start:q_end]
            group_scores_dim = postrope_kwanda_abs_dim[q_start:q_end]
            coord_source = 'postrope_kwanda_abs'
        elif mode in ['postrope_kenergy_norm_projout_scalar', 'postrope_kenergy_norm_projout_scalar_resavg', 'postrope_kenergy_norm_projout_scalar_resavg_scaled', 'postrope_kenergy_norm_projout_scalar_resconst', 'postrope_kenergy_norm_projout_scalar_resrand', 'postrope_kenergy_norm_projout_scalar_resabs', 'postrope_kenergy_norm_projout_scalar_resl2']:
            group_scores = postrope_kenergy_pair[q_start:q_end]
            group_scores_dim = postrope_kenergy_dim[q_start:q_end]
            coord_source = 'postrope_kenergy'
        elif mode in ['postrope_kvariance_norm_projout_scalar']:
            group_scores = postrope_kvariance_pair[q_start:q_end]
            group_scores_dim = postrope_kvariance_dim[q_start:q_end]
            coord_source = 'postrope_kvariance'
        elif mode in ['postrope_qkvar_norm_projout_scalar', 'postrope_kgeomean_norm_projout_scalar']:
            group_scores = postrope_qkvar_pair[q_start:q_end]
            group_scores_dim = postrope_qkvar_dim[q_start:q_end]
            coord_source = 'postrope_qkvar'
        elif mode in ['postrope_kgeomean_norm_projout_scalar']:
            group_scores = postrope_kgeomean_pair[q_start:q_end]
            group_scores_dim = postrope_kgeomean_dim[q_start:q_end]
            coord_source = 'postrope_kgeomean'
        elif mode in ['invrope_norm_projout_scalar']:
            if invrope_pair is None or invrope_dim is None:
                raise ValueError('invrope mode requires cos/sin for explicit inverse rotation')
            group_scores = invrope_pair[q_start:q_end]
            group_scores_dim = invrope_dim[q_start:q_end]
            coord_source = 'invrope'
        else:
            group_scores = derope_pair[q_start:q_end]
            group_scores_dim = derope_dim[q_start:q_end]
            coord_source = 'prerope'
        group_info = {
            'kv_head': int(kv_h),
            'q_head_range': [int(q_start), int(q_end - 1)],
            'group_scores': _tensor_to_list(group_scores),
            'coord_source': coord_source,
            'attn_influence': _tensor_to_list(attn_influence[q_start:q_end]),
            'proj_influence': _tensor_to_list(proj_influence[q_start:q_end]),
        }
        if mode == 'argmax_attnout':
            chosen_local = int(attn_influence[q_start:q_end].argmax().item())
            chosen = group_scores[chosen_local]
            group_info['chosen_q_head'] = int(q_start + chosen_local)
        elif mode == 'argmax_projout':
            chosen_local = int(proj_influence[q_start:q_end].argmax().item())
            chosen = group_scores[chosen_local]
            group_info['chosen_q_head'] = int(q_start + chosen_local)
        elif mode == 'argmax_projout_scalar':
            chosen_local = int(proj_influence[q_start:q_end].argmax().item())
            chosen_q = int(q_start + chosen_local)
            chosen_dim = derope_dim[chosen_q]
            keep_dim = max(1, min(head_dim - int(round(head_dim * ratio)), head_dim - 1))
            dim_keep = torch.zeros(head_dim, dtype=torch.bool, device=kv_states.device)
            top_idx = torch.topk(chosen_dim, keep_dim, largest=True).indices
            dim_keep[top_idx] = True
            keep_mask[:, q_start:q_end, :] = dim_keep.unsqueeze(0).unsqueeze(0).expand(1, q_end - q_start, head_dim)
            group_info['chosen_q_head'] = chosen_q
            group_info['selected_dims'] = top_idx.detach().cpu().tolist()
            group_info['aggregated_dim_score'] = _tensor_to_list(chosen_dim)
            group_debug.append(group_info)
            continue
        elif mode == 'soft_attnout':
            weights = torch.softmax(attn_influence[q_start:q_end] / tau, dim=0)
            chosen = (weights[:, None] * group_scores).sum(dim=0)
            group_info['weights'] = _tensor_to_list(weights)
            group_info['chosen_q_head'] = None
        elif mode == 'soft_projout':
            weights = torch.softmax(proj_influence[q_start:q_end] / tau, dim=0)
            chosen = (weights[:, None] * group_scores).sum(dim=0)
            group_info['weights'] = _tensor_to_list(weights)
            group_info['chosen_q_head'] = None
        elif mode == 'norm_projout':
            raw = proj_influence[q_start:q_end].clamp_min(0)
            denom = raw.sum().clamp_min(1e-12)
            weights = raw / denom
            chosen = (weights[:, None] * group_scores).sum(dim=0)
            group_info['weights'] = _tensor_to_list(weights)
            group_info['chosen_q_head'] = None
        elif mode in ['norm_projout_scalar', 'postrope_norm_projout_scalar', 'invrope_norm_projout_scalar', 'prerope_kenergy_norm_projout_scalar', 'prerope_kvariance_norm_projout_scalar', 'prerope_kwanda_abs_scalar', 'postrope_kwanda_abs_norm_projout_scalar', 'postrope_kenergy_norm_projout_scalar', 'postrope_kenergy_norm_projout_scalar_resavg', 'postrope_kenergy_norm_projout_scalar_resavg_scaled', 'postrope_kenergy_norm_projout_scalar_resconst', 'postrope_kenergy_norm_projout_scalar_resrand', 'postrope_kenergy_norm_projout_scalar_resabs', 'postrope_kenergy_norm_projout_scalar_resl2', 'postrope_kvariance_norm_projout_scalar', 'postrope_qkvar_norm_projout_scalar', 'postrope_kgeomean_norm_projout_scalar']:
            raw = proj_influence[q_start:q_end].clamp_min(0)
            denom = raw.sum().clamp_min(1e-12)
            weights = raw / denom
            chosen_dim = (weights[:, None] * group_scores_dim).sum(dim=0)
            keep_dim = max(1, min(head_dim - int(round(head_dim * ratio)), head_dim - 1))
            dim_keep = torch.zeros(head_dim, dtype=torch.bool, device=kv_states.device)
            top_idx = torch.topk(chosen_dim, keep_dim, largest=True).indices
            dim_keep[top_idx] = True
            keep_mask[:, q_start:q_end, :] = dim_keep.unsqueeze(0).unsqueeze(0).expand(1, q_end - q_start, head_dim)
            group_info['weights'] = _tensor_to_list(weights)
            group_info['chosen_q_head'] = None
            group_info['selected_dims'] = top_idx.detach().cpu().tolist()
            group_info['aggregated_dim_score'] = _tensor_to_list(chosen_dim)
            group_debug.append(group_info)
            continue
        elif mode == 'top2_norm_projout':
            raw = proj_influence[q_start:q_end].clamp_min(0)
            topk = min(2, raw.numel())
            top_idx = torch.topk(raw, topk, dim=0).indices
            masked = torch.zeros_like(raw)
            masked[top_idx] = raw[top_idx]
            denom = masked.sum().clamp_min(1e-12)
            weights = masked / denom
            chosen = (weights[:, None] * group_scores).sum(dim=0)
            group_info['weights'] = _tensor_to_list(weights)
            group_info['chosen_q_head'] = None
            group_info['topk_q_heads'] = [int(q_start + i) for i in top_idx.detach().cpu().tolist()]
        elif mode == 'top3_norm_projout':
            raw = proj_influence[q_start:q_end].clamp_min(0)
            topk = min(3, raw.numel())
            top_idx = torch.topk(raw, topk, dim=0).indices
            masked = torch.zeros_like(raw)
            masked[top_idx] = raw[top_idx]
            denom = masked.sum().clamp_min(1e-12)
            weights = masked / denom
            chosen = (weights[:, None] * group_scores).sum(dim=0)
            group_info['weights'] = _tensor_to_list(weights)
            group_info['chosen_q_head'] = None
            group_info['topk_q_heads'] = [int(q_start + i) for i in top_idx.detach().cpu().tolist()]
        elif mode == 'l2_norm_projout':
            raw = proj_influence[q_start:q_end].clamp_min(0).pow(2)
            denom = raw.sum().clamp_min(1e-12)
            weights = raw / denom
            chosen = (weights[:, None] * group_scores).sum(dim=0)
            group_info['weights'] = _tensor_to_list(weights)
            group_info['chosen_q_head'] = None
        else:
            raise ValueError(f'Unknown OURS mode: {mode}')
        group_keep = _select_pair_keep_mask(chosen.unsqueeze(0), keep_pairs).squeeze(0)
        keep_mask[:, q_start:q_end, :] = group_keep.unsqueeze(0).expand(1, q_end - q_start, head_dim)
        group_info['selected_pairs'] = torch.nonzero(group_keep[:half], as_tuple=False).squeeze(-1).detach().cpu().tolist()
        group_info['selected_pair_scores'] = _tensor_to_list(chosen[group_keep[:half]])
        group_info['aggregated_pair_score'] = _tensor_to_list(chosen)
        group_debug.append(group_info)

    past_len = seqlen - recent_size
    past_mask = keep_mask.unsqueeze(2).expand(-1, -1, past_len, -1)
    keep_dim = int(keep_mask[0, 0].sum().item())
    kv_kept = kv_states[:, :, :past_len, :][past_mask].reshape(1, num_heads, past_len, keep_dim)
    _res_suffix = None
    for _s in ['_resavg_scaled', '_resavg', '_resconst', '_resrand', '_resabs', '_resl2']:
        if mode.endswith(_s):
            _res_suffix = _s
            break
    if _res_suffix is not None:
        drop_mask = (~keep_mask).unsqueeze(2).expand(-1, -1, past_len, -1)
        drop_dim = int((~keep_mask)[0, 0].sum().item())
        kv_dropped = kv_states[:, :, :past_len, :][drop_mask].reshape(1, num_heads, past_len, drop_dim)
        if _res_suffix == '_resconst':
            kv_drop_avg = kv_dropped.mean(dim=-1, keepdim=True).mean(dim=2, keepdim=True).expand(-1, -1, past_len, -1)
        elif _res_suffix == '_resrand':
            real_avg = kv_dropped.mean(dim=-1, keepdim=True)
            kv_drop_avg = torch.randn_like(real_avg) * real_avg.std()
        elif _res_suffix == '_resavg_scaled':
            kv_drop_avg = kv_dropped.mean(dim=-1, keepdim=True) * math.sqrt(drop_dim)
        elif _res_suffix == '_resabs':
            # mean of absolute values — doesn't cancel to zero
            kv_drop_avg = kv_dropped.abs().mean(dim=-1, keepdim=True)
        elif _res_suffix == '_resl2':
            # L2 norm / sqrt(d) — energy-preserving, doesn't cancel
            kv_drop_avg = kv_dropped.pow(2).mean(dim=-1, keepdim=True).sqrt()
        else:  # _resavg
            kv_drop_avg = kv_dropped.mean(dim=-1, keepdim=True)
        kv_pruned = torch.cat([kv_kept, kv_drop_avg], dim=-1)
    else:
        kv_pruned = kv_kept
    kv_recent = kv_states[:, :, past_len:, :]
    _write_pruning_monitor(
        'ours_prune',
        mode=str(mode),
        tau=float(tau),
        ratio=float(ratio),
        head_dim=int(head_dim),
        keep_dim=int(keep_dim),
        drop_dim=int(head_dim - keep_dim),
        keep_pairs=int(keep_pairs),
        drop_pairs=int(drop_pairs),
        num_heads=int(num_heads),
        num_kv_heads=int(num_kv_heads),
        keep_dims_per_head=[int(keep_mask[0, h].sum().item()) for h in range(keep_mask.shape[1])],
        mask_hash=_mask_hash(keep_mask),
        recent_size=int(recent_size),
        seqlen=int(seqlen),
        pruned_shape=list(kv_pruned.shape),
        recent_shape=list(kv_recent.shape),
        has_residual=(_res_suffix is not None),
        residual_type=_res_suffix if _res_suffix else 'none',
        residual_scale=float(math.sqrt(head_dim - keep_dim)) if _res_suffix == '_resavg_scaled' else 1.0,
        attn_influence_max=float(attn_influence.max().item()),
        proj_influence_max=float(proj_influence.max().item()),
    )
    _write_pruning_trace(
        'ours_trace',
        layer_idx=None if layer_idx is None else int(layer_idx),
        mode=str(mode),
        tau=float(tau),
        ratio=float(ratio),
        head_dim=int(head_dim),
        seqlen=int(seqlen),
        recent_size=int(recent_size),
        q_pre_norm=_tensor_to_list(q_pre_norm),
        q_post_norm=_tensor_to_list(q_post_norm),
        derope_dim=_tensor_to_list(derope_dim),
        derope_pair=_tensor_to_list(derope_pair),
        postrope_dim=_tensor_to_list(postrope_dim),
        postrope_pair=_tensor_to_list(postrope_pair),
        k_pre_energy=_tensor_to_list(k_pre_energy),
        k_pre_var=_tensor_to_list(k_pre_var),
        k_post_energy=_tensor_to_list(k_post_energy),
        k_post_var=_tensor_to_list(k_post_var),
        prerope_kenergy_dim=_tensor_to_list(prerope_kenergy_dim),
        prerope_kvariance_dim=_tensor_to_list(prerope_kvariance_dim),
        prerope_kwanda_abs_dim=_tensor_to_list(prerope_kwanda_abs_dim),
        postrope_kwanda_abs_dim=_tensor_to_list(postrope_kwanda_abs_dim),
        postrope_kenergy_dim=_tensor_to_list(postrope_kenergy_dim),
        postrope_kvariance_dim=_tensor_to_list(postrope_kvariance_dim),
        invrope_dim=_tensor_to_list(invrope_dim) if invrope_dim is not None else None,
        invrope_pair=_tensor_to_list(invrope_pair) if invrope_pair is not None else None,
        attn_influence=_tensor_to_list(attn_influence),
        proj_influence=_tensor_to_list(proj_influence),
        keep_dims_per_head=[int(keep_mask[0, h].sum().item()) for h in range(keep_mask.shape[1])],
        kept_indices=_indices_per_head(keep_mask),
        keep_pairs=int(keep_pairs),
        drop_pairs=int(drop_pairs),
        mask_hash=_mask_hash(keep_mask),
        groups=group_debug,
    )
    return kv_pruned, kv_recent, keep_mask

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def key_pruner_gqa_aware(kv_states, q_states, value_states, o_proj_weight, num_key_value_groups, recent_size=128, ratio=0.3, mode='gqa_kenergy_projout_resavg', layer_idx=None, score_window=None):
    """True GQA-aware pruner: operates on 8 KV heads, stores 8 heads.
    Supports modes:
      gqa_kenergy_projout_resavg       — K² score (legacy)
      gqa_kenergy_projout_resavg_scaled
      gqa_qkvar_projout                — Q²×√(K²×K_var) + O_proj weighted (new)

    score_window (B9): number of recent Q tokens used for influence / Q² estimation.
    Defaults to ``recent_size`` so it tracks the H2O window rather than a magic 32.
    """
    bsz, num_heads, seqlen, head_dim = kv_states.shape
    assert bsz == 1
    num_kv_heads = num_heads // num_key_value_groups

    # Clamp recent_size so seqlen < recent_size doesn't produce a negative past_len
    # (the prior code crashed with "numel overflow" on very short prompts).
    recent_size = min(int(recent_size), int(seqlen))

    # Extract unique 8 KV heads (undo repeat_kv)
    kv_8 = kv_states[:, ::num_key_value_groups, :, :]  # [1, 8, T, head_dim]

    # K characteristics per KV head (post-RoPE, 8 heads)
    k_energy = kv_8.pow(2).mean(dim=2).squeeze(0)       # [8, head_dim]
    k_var = kv_8.var(dim=2, unbiased=False).squeeze(0)   # [8, head_dim]
    k_geomean = (k_energy * k_var).sqrt()                 # [8, head_dim]

    # B9: window size for Q observation. Fall back to the H2O window / recent_size.
    if score_window is None:
        score_window = int(recent_size)
    score_window = max(1, min(score_window, q_states.shape[-2]))

    # Q head importance via proj_influence (all 32 Q heads)
    q_post = q_states[..., -score_window:, :]
    attn_logits = torch.matmul(q_post, kv_states.transpose(2, 3)) / math.sqrt(head_dim)
    attn_probs = torch.softmax(attn_logits, dim=-1)
    head_out = torch.matmul(attn_probs, value_states)
    proj_influence = _project_head_influence(head_out, o_proj_weight, num_heads, head_dim).squeeze(0)

    # Q² per head (32 heads, each different)
    q_post_norm = q_post.pow(2).mean(dim=2).squeeze(0)  # [32, head_dim]

    # (qsnr 모드) Q 시간 분산 → Q²/(Var+ε) 가 score에 쓰임. 기본 경로는 touch 하지 않음.
    use_qsnr = 'qsnr' in mode
    if use_qsnr:
        _qvar = q_post.var(dim=2, unbiased=False).squeeze(0)       # [32, D]
        _qsnr_eps = float(os.environ.get('QSNR_EPSILON', '1.0'))
        q_effective = q_post_norm / (_qvar + _qsnr_eps)             # [32, D]
    else:
        q_effective = None

    # Build mask per KV head [1, num_kv_heads, head_dim]
    keep_dim = max(1, min(head_dim - int(round(head_dim * ratio)), head_dim - 1))
    drop_dim = head_dim - keep_dim
    keep_mask = torch.zeros((1, num_kv_heads, head_dim), dtype=torch.bool, device=kv_states.device)

    use_new = mode.startswith('gqa_qkvar')
    use_think = mode.startswith('gqa_think_projout')
    use_k2only = 'k2only' in mode

    group_debug = []
    for kv_h in range(num_kv_heads):
        q_start = kv_h * num_key_value_groups
        q_end = (kv_h + 1) * num_key_value_groups

        # O_proj weights for this group
        raw = proj_influence[q_start:q_end].clamp_min(0)
        denom = raw.sum().clamp_min(1e-12)
        weights = raw / denom

        if use_k2only:
            # K² only — drops Q² window-noise factor; O_proj cancels out since K² is group-shared
            chosen_dim = k_energy[kv_h]
        elif use_qsnr:
            # Q²/(Var+ε) × K² — SNR-normalized Q replaces Q². Channels where Q fluctuates
            # a lot over the window get down-weighted; stable directions keep their weight.
            qk_score = q_effective[q_start:q_end] * k_energy[kv_h].unsqueeze(0)
            chosen_dim = (weights[:, None] * qk_score).sum(dim=0)
        elif use_new:
            # Q² × K_var, per Q head → O_proj weighted sum
            k_score = k_var[kv_h]
            score_per_q = q_post_norm[q_start:q_end] * k_score.unsqueeze(0)
            chosen_dim = (weights[:, None] * score_per_q).sum(dim=0)
        elif use_think:
            # Q² × K², per Q head → O_proj weighted sum (ThinK score + our aggregation)
            qk_score = q_post_norm[q_start:q_end] * k_energy[kv_h].unsqueeze(0)
            chosen_dim = (weights[:, None] * qk_score).sum(dim=0)
        else:
            # Legacy: Q² × K² with O_proj (same as use_think)
            qk_score = q_post_norm[q_start:q_end] * k_energy[kv_h].unsqueeze(0)
            chosen_dim = (weights[:, None] * qk_score).sum(dim=0)

        top_idx = torch.topk(chosen_dim, keep_dim, largest=True).indices
        dim_keep = torch.zeros(head_dim, dtype=torch.bool, device=kv_states.device)
        dim_keep[top_idx] = True
        keep_mask[:, kv_h, :] = dim_keep
        group_debug.append({
            'kv_head': int(kv_h),
            'q_head_range': [int(q_start), int(q_end - 1)],
            'weights': _tensor_to_list(weights),
            'selected_dims': top_idx.detach().cpu().tolist(),
        })

    # Prune on 8 KV heads
    past_len = seqlen - recent_size
    past_mask = keep_mask.unsqueeze(2).expand(-1, -1, past_len, -1)
    kv_pruned = kv_8[:, :, :past_len, :][past_mask].reshape(1, num_kv_heads, past_len, keep_dim)

    # GQA: store recent window at 8 KV heads (not 32) to preserve memory savings
    kv_recent = kv_8[:, :, past_len:, :]  # [1, 8, recent, head_dim]

    _write_pruning_monitor(
        'gqa_aware_prune',
        mode=str(mode),
        ratio=float(ratio),
        head_dim=int(head_dim),
        keep_dim=int(keep_dim),
        drop_dim=int(drop_dim),
        num_heads=int(num_heads),
        num_kv_heads=int(num_kv_heads),
        mask_hash=_mask_hash(keep_mask),
        recent_size=int(recent_size),
        seqlen=int(seqlen),
        pruned_shape=list(kv_pruned.shape),
        gqa_pruned=True,
    )
    _write_pruning_trace(
        'gqa_aware_trace',
        layer_idx=None if layer_idx is None else int(layer_idx),
        mode=str(mode),
        ratio=float(ratio),
        head_dim=int(head_dim),
        seqlen=int(seqlen),
        recent_size=int(recent_size),
        k_energy=_tensor_to_list(k_energy),
        proj_influence=_tensor_to_list(proj_influence),
        kept_indices=_indices_per_head(keep_mask),
        mask_hash=_mask_hash(keep_mask),
        groups=group_debug,
    )
    return kv_pruned, kv_recent, keep_mask


def key_pruner_iterative(kv_states, q_states, value_states, o_proj_weight, num_key_value_groups, recent_size=128, ratio=0.4, num_steps=None, mode='gqa_iter_think', layer_idx=None, score_window=None, k_proj_weight=None, rope_theta=500000.0):
    """★★ 우리 방법의 핵심 — 32단계 점진 채널 마스크 찾기 ★★

    입력:
      kv_states    : [1, 32, T, 128]  (repeat_kv된 K, GQA 이후 32 head view)
      q_states     : [1, 32, T, 128]  (Q, 32 head)
      value_states : [1, 32, T, 128]  (V, repeat_kv된 것)
      o_proj_weight: [D_model, D_model]  (output projection 가중치)
      num_steps    : 32 (iter32 모드) 혹은 4 (iter4)
      ratio        : 버릴 채널 비율 (0.4이면 최종 51 dim drop, 77 keep)
      mode         : 'gqa_iter32_think' 등

    출력:
      kv_pruned  : [1, 8, T-recent, kept=77]  (오래된 토큰 + 잘린 채널)
      kv_recent  : [1, 8, recent, 128]        (최근 window, 전체 채널)
      keep_mask  : [1, 8, 128] bool           (group별 남긴 채널)

    호출처:
      - src/kv_pruning_utils.py:1160 (update_thinkv 짧은 경로)
      - src/kv_pruning_utils.py:1282 (update_thinkv 긴 경로, 우리 메인 path)

    알고리즘 개요:
      1. Q²(각 head) + K²(각 group) 사전 계산. (step마다 재계산 X — noise 최소화)
      2. 마스크 M를 전부 True로 초기화.
      3. 32 step 반복:
         a) 현재 M으로 pruned attention 계산 (Q*K^T · mask / sqrt(kept_dim))
         b) V 곱해서 attention output 얻음 → head별 O_proj 영향력 φ_h 계산
         c) group 내 φ 정규화해서 α_h 얻음 (중요한 head 가중치)
         d) 각 group마다 채널 점수: s[d] = Σ_h α_h · Q²[h,d] · K²[group,d]
         e) step_keep 개의 상위 채널 선택 → 새 M[group] 만듦
         (step_keep은 step_ratio = ratio · step/N에 따라 점점 감소)
      4. 최종 M으로 kv_8 자르기 → kv_pruned, kv_recent 반환.
    """
    # Public "grik" name maps to the canonical paper configuration
    # (32 iterations + Wanda K-norm prior + dynamic Q^2*K^2*alpha). The
    # body below uses substring matching for backward compatibility, so
    # we expand "grik" to a tag carrying the relevant feature tokens.
    if mode == 'grik':
        mode = 'grik_iter32_think_wanda'
    if num_steps is None:
        # Parse explicit N from `gqa_iter{N}_*` (covers iter2/8/16/32/64 sweep).
        # Fall back: wanda_only → 1 step (mask is purely static), else 4.
        _m_iter = re.search(r'iter(\d+)_', mode)
        if _m_iter is not None:
            num_steps = int(_m_iter.group(1))
        elif 'wanda_only' in mode:
            num_steps = 1
        else:
            num_steps = 4
    # (B9) score_window는 Q stats에 쓸 최근 token 수. 기본=recent_size(=32).
    bsz, num_heads, seqlen, head_dim = kv_states.shape
    assert bsz == 1
    num_kv_heads = num_heads // num_key_value_groups

    # seqlen < recent_size인 짧은 prompt에서 past_len 음수 되는 crash 방지.
    recent_size = min(int(recent_size), int(seqlen))

    # [1-a] GQA 해제: 32 head → unique 8 head로. (matmul은 계속 32 head view 써야 함)
    kv_8 = kv_states[:, ::num_key_value_groups, :, :]

    # [1-b] K² = E_t[K²] per KV head. 8개 group 각각의 채널별 에너지. step 루프 밖 (constant).
    k_energy = kv_8.pow(2).mean(dim=2).squeeze(0)  # [8, D]

    # [1-c] Q² = E_t[Q²] per query head. 최근 score_window 토큰 기준. step 루프 밖 (constant).
    if score_window is None:
        score_window = int(recent_size)
    score_window = max(num_steps, min(score_window, q_states.shape[-2]))
    q_post = q_states[..., -score_window:, :]
    q_post_norm = q_post.pow(2).mean(dim=2).squeeze(0)  # [32, D]

    # (qsnr 모드용) Q의 시간 분산. Q² / (Var + ε) = 1 + SNR² 형태로 channel-별 SNR 가중.
    # mode에 'qsnr' 포함 시에만 사용; 일반 모드는 건드리지 않음.
    _use_qsnr = ('qsnr' in mode)
    if _use_qsnr:
        q_var = q_post.var(dim=2, unbiased=False).squeeze(0)      # [32, D]
        _qsnr_eps = float(os.environ.get('QSNR_EPSILON', '1.0'))
        # Q²_effective[h,d] = Q²[h,d] / (Var_t(Q[h,:,d]) + ε)
        # ε=1.0 기본. ε→0: pure SNR (분산 작은 채널 boost 극대화, 불안정),
        #            ε=1: 안정, 기존 Q²와 blend된 효과,  ε→∞: Q² normalize 효과 사라짐.
        q_effective = q_post_norm / (q_var + _qsnr_eps)            # [32, D]
    else:
        q_effective = None

    # ────────────────────────────────────────────────────────────────────────
    # Extra impact-factor precomputation (isolated: each branch multiplies the
    # existing Σ_h α_h · Q²[h,d] · K²[g,d] formula OR swaps K² for a variant;
    # α_h/O_proj aggregation is preserved in every case).
    # ────────────────────────────────────────────────────────────────────────

    # [RoPE] channel-frequency weight. LLaMA RoPE rotates pair (2i, 2i+1) at
    # angle position × inv_freq[i] with inv_freq[i] = θ^(-2i/D). Low i =
    # high inv_freq = fast rotation = local/position info. High i = low
    # inv_freq = slow rotation = long-range semantic info. We up-weight
    # slow-rotation channels linearly in pair-index so the scoring favours
    # preserving semantic channels during pruning.
    _use_rope = ('rope' in mode)
    if _use_rope:
        half = head_dim // 2
        pair_idx = torch.arange(head_dim, device=kv_states.device) // 2  # [D]
        # Linear weight in [0.5, 1.5] — slow-rotation (high i) gets up-weight.
        rope_weight = 0.5 + (pair_idx.float() + 1.0) / float(half)  # [D]
    else:
        rope_weight = None

    # [Wanda] static weight × dynamic activation, à la Wanda pruning (2023).
    # For each KV head g and channel d, ||W_k[g, d, :]||₂ is the row norm of
    # the K projection contributing to that channel. Multiplying the existing
    # Q² × K² by this norm gates channels whose *weight* is also large.
    # `wanda_only` mode (component ablation §C-1) zeros out the dynamic
    # Q²×K²×α term entirely and uses Wanda alone as the channel score.
    _use_wanda = ('wanda' in mode)
    _use_wanda_only = ('wanda_only' in mode)
    if _use_wanda:
        if k_proj_weight is None:
            raise ValueError("wanda mode requires k_proj_weight but got None")
        # k_proj_weight: [num_kv_heads*head_dim, hidden_size]
        hidden_size = k_proj_weight.shape[1]
        wk_per_head = k_proj_weight.view(num_kv_heads, head_dim, hidden_size)
        wk_row_norm = wk_per_head.float().norm(dim=-1)                       # [8, D]
        # Normalize per-head so that the multiplier doesn't dominate scale:
        # divide by row-norm mean → multiplier has mean≈1 within each group.
        wk_row_norm = wk_row_norm / wk_row_norm.mean(dim=-1, keepdim=True).clamp_min(1e-12)
        wk_row_norm = wk_row_norm.to(kv_states.dtype)
    else:
        wk_row_norm = None

    # [sensVar] replace E[K²] with Var(K) = E[K²] − E[K]². Removes the DC/mean
    # component of K so only per-channel *fluctuation* counts — approximates
    # Fisher-info sensitivity when attention weights are uniform.
    _use_sensvar = ('sensvar' in mode)
    if _use_sensvar:
        k_mean = kv_8.mean(dim=2).squeeze(0)                                 # [8, D]
        k_var_plain = kv_8.var(dim=2, unbiased=False).squeeze(0).clamp_min(0)   # [8, D]
        # k_var_plain has same units/scale as k_energy for ablation symmetry.
    else:
        k_var_plain = None

    # [sensAttn] attention-weighted Var(K). Second-order Taylor of
    # KL(softmax(QK) || softmax(QK − ΔS)) ≈ Σ_t Q²[t,d] · Var_s(K[·,d];
    # weights=p[t,·]). Aggregating over window t with the representative q per
    # kv head yields a per-(g, d) channel-sensitivity score.
    _use_sensattn = ('sensattn' in mode)
    if _use_sensattn:
        # One representative Q per kv_head (first of the 4 q heads in the group).
        q_rep = q_post[:, ::num_key_value_groups, :, :]                      # [1, 8, window, D]
        # Attention logits, softmax over past positions.
        _attn_logits = torch.matmul(q_rep, kv_8.transpose(2, 3)) / math.sqrt(head_dim)
        _attn_probs = torch.softmax(_attn_logits, dim=-1)                    # [1, 8, window, T]
        # Aggregate across window queries: equal-weight mean.
        p_agg = _attn_probs.mean(dim=2).squeeze(0)                           # [8, T]
        p_agg = p_agg / p_agg.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        kv_8_float = kv_8.squeeze(0).float()                                  # [8, T, D]
        p_agg_f = p_agg.float()
        mean_K_att = (p_agg_f.unsqueeze(-1) * kv_8_float).sum(dim=1)         # [8, D]
        meansq_K_att = (p_agg_f.unsqueeze(-1) * kv_8_float.pow(2)).sum(dim=1)  # [8, D]
        k_var_attn = (meansq_K_att - mean_K_att.pow(2)).clamp_min(0)          # [8, D]
        k_var_attn = k_var_attn.to(kv_states.dtype)
        del _attn_logits, _attn_probs, p_agg, p_agg_f, kv_8_float, mean_K_att, meansq_K_att
    else:
        k_var_attn = None

    # [2] 마스크 초기화: 처음엔 모든 채널 살림 → 루프 돌며 점점 쪼여들어감.
    current_mask = torch.ones((num_kv_heads, head_dim), dtype=torch.bool, device=kv_states.device)
    # step마다 attention 돌릴 때 쓸 Q window 크기 (score_window를 num_steps로 나눔).
    window = max(1, score_window // num_steps)

    # ────────────────────────────────────────────────────────────────────────
    # ABLATION_TODO_PAPER §C-2 — mask convergence diagnostic.
    # If GRIK_TRACE_CONVERGENCE=<path> is set, append one JSONL row per step
    # per layer reporting (a) mask Jaccard vs the previous step (averaged over
    # KV heads) and (b) the relative Frobenius change in masked Q² and K².
    # Zero overhead when the env var is not set.
    # ────────────────────────────────────────────────────────────────────────
    _trace_path = os.environ.get('GRIK_TRACE_CONVERGENCE')
    _prev_mask_for_trace = None
    _prev_eq_for_trace = None
    _prev_ek_for_trace = None

    # ========================================================================
    # [3] ★ 메인 루프: 32 step 점진 마스크 정제 ★
    # ========================================================================
    for step in range(num_steps):
        # step별 목표 keep 개수. step 0에서 큰 값 → step N-1에서 최종 target (=77).
        step_ratio = ratio * (step + 1) / num_steps
        step_keep = max(1, min(head_dim - int(round(head_dim * step_ratio)), head_dim - 1))

        # ---- 3a. 현재 M 상태에서 pruned attention 계산 (O_proj 영향력용) ----
        q_window = q_post[:, :, step * window:(step + 1) * window, :]            # [1, 32, window, D]
        # current_mask는 [8, D] → 32 head view로 broadcast.
        mask_32 = current_mask.repeat_interleave(num_key_value_groups, dim=0)    # [32, D]

        # mask 0인 채널은 값을 0으로 만들어서 dot product 기여 0 (= pruned과 수학적으로 동일).
        k_masked = kv_states * mask_32[None, :, None, :].to(kv_states.dtype)
        q_win_masked = q_window * mask_32[None, :, None, :].to(q_window.dtype)
        kept_dim = int(current_mask[0].sum().item())

        # sqrt(kept_dim) 주의: 여기 iterative loop **내부**는 sqrt(kept_dim) 맞음 (§3.4 근거).
        # decode 시점은 sqrt(head_dim=128) (§3.2, docs/SQRT_TEMPERATURE_CALIBRATION.md).
        attn_logits = torch.matmul(q_win_masked, k_masked.transpose(2, 3)) / math.sqrt(kept_dim)
        attn_probs = torch.softmax(attn_logits, dim=-1)
        head_out = torch.matmul(attn_probs, value_states)                        # [1, 32, window, D]
        # ---- 3b. O_proj 영향력 φ_h ----
        # → :65 _project_head_influence (위 정의 참조)
        proj_influence = _project_head_influence(head_out, o_proj_weight, num_heads, head_dim).squeeze(0)
        del attn_logits, attn_probs, head_out, k_masked, q_win_masked

        # ---- 3c-d. 각 group마다 채널 점수 계산 → 상위 step_keep개 선택 ----
        # 모드별 scoring:
        #   - 'k2only' in mode : K²만 (ablation; Q² drop)
        #   - 'qsnr' in mode   : Q²를 SNR-normalized Q² = Q²/(Var+ε)로 대체 (Q_noise down-weight)
        #   - default ('think'): Q²×K² + O_proj 가중합 (논문 §3.3)
        _use_k2only = ('k2only' in mode)
        # _use_qsnr, q_effective, _use_rope/wanda/sensvar/sensattn는 루프 밖에서 계산됨.
        for g in range(num_kv_heads):
            # group g에 속한 4개 query head 범위.
            qs, qe = g * num_key_value_groups, (g + 1) * num_key_value_groups

            # α_h = φ_h / Σ φ_h' (clamp로 음수 차단, 합 0이면 ε로 나눔 방지).
            raw = proj_influence[qs:qe].clamp_min(0)
            weights = raw / raw.sum().clamp_min(1e-12)

            if _use_wanda_only:
                # Component ablation: pure static Wanda K-norm score, no
                # Q²×K² × α aggregation. Mask is determined by the K-projection
                # row-norm alone, so it is identical at every iteration step.
                if wk_row_norm is None:
                    raise ValueError("wanda_only mode requires k_proj_weight")
                agg = wk_row_norm[g].clone()
            elif _use_k2only:
                # K²만: 어차피 group 공유, weights sum=1이니 α 곱 → K² 그대로.
                agg = k_energy[g]
            elif _use_sensvar:
                # K² 대신 Var(K) 사용 (DC 성분 제거). Q²×Var(K) + α_h.
                score_per_q = q_post_norm[qs:qe] * k_var_plain[g].unsqueeze(0)  # [4, D]
                agg = (weights[:, None] * score_per_q).sum(dim=0)
            elif _use_sensattn:
                # K² 대신 attention-가중 Var(K). Fisher-info 근사.
                score_per_q = q_post_norm[qs:qe] * k_var_attn[g].unsqueeze(0)   # [4, D]
                agg = (weights[:, None] * score_per_q).sum(dim=0)
            elif _use_qsnr:
                # Q² 대신 Q_effective = Q²/(Var+ε) 사용. 다른 건 동일.
                score_per_q = q_effective[qs:qe] * k_energy[g].unsqueeze(0)     # [4, D]
                agg = (weights[:, None] * score_per_q).sum(dim=0)                # [D]
            else:
                # 메인 공식: s[d] = Σ_h α_h · Q²[h,d] · K²[g,d]
                score_per_q = q_post_norm[qs:qe] * k_energy[g].unsqueeze(0)   # [4, D]
                agg = (weights[:, None] * score_per_q).sum(dim=0)              # [D]

            # 추가 multiplier들 (직교 — 다른 variant와 조합 가능).
            # wanda_only는 위에서 이미 wk_row_norm[g]으로 직접 set 했으므로
            # 여기 wanda multiplier 분기는 건드리지 않음 (double-apply 방지).
            if _use_rope:
                agg = agg * rope_weight                                          # [D]
            if _use_wanda and not _use_wanda_only:
                agg = agg * wk_row_norm[g]                                       # [D]

            # 상위 step_keep개 채널 선택 → 마스크 덮어쓰기.
            # (마스크는 monotonic shrink 아님: α가 step마다 바뀌므로 drop된 채널이 부활 가능)
            top_idx = agg.topk(step_keep).indices
            new_mask = torch.zeros(head_dim, dtype=torch.bool, device=kv_states.device)
            new_mask[top_idx] = True
            current_mask[g] = new_mask

        # ── §C-2 trace hook (no-op when env var unset) ──────────────────
        if _trace_path is not None:
            with torch.no_grad():
                _mask32 = current_mask.repeat_interleave(num_key_value_groups, dim=0).to(q_post_norm.dtype)
                _eq_curr = q_post_norm * _mask32
                _ek_curr = k_energy * current_mask.to(k_energy.dtype)
                if _prev_mask_for_trace is None:
                    jaccard = float('nan')
                    eq_rel = float('nan')
                    ek_rel = float('nan')
                else:
                    inter = (current_mask & _prev_mask_for_trace).sum(dim=-1).float()
                    union = (current_mask | _prev_mask_for_trace).sum(dim=-1).float()
                    jaccard = (inter / union.clamp_min(1.0)).mean().item()
                    eq_rel = ((_eq_curr - _prev_eq_for_trace).norm()
                              / _eq_curr.norm().clamp_min(1e-12)).item()
                    ek_rel = ((_ek_curr - _prev_ek_for_trace).norm()
                              / _ek_curr.norm().clamp_min(1e-12)).item()
                rec = {
                    "layer_idx": int(layer_idx) if layer_idx is not None else None,
                    "step": int(step),
                    "num_steps": int(num_steps),
                    "mask_jaccard_prev_to_curr": jaccard,
                    "eq_relative_change_frob": eq_rel,
                    "ek_relative_change_frob": ek_rel,
                }
                with open(_trace_path, "a") as _tf:
                    _tf.write(json.dumps(rec) + "\n")
                _prev_mask_for_trace = current_mask.clone()
                _prev_eq_for_trace = _eq_curr.detach().clone()
                _prev_ek_for_trace = _ek_curr.detach().clone()

    # ========================================================================
    # [4] 최종 mask 적용 → 8 head native 저장 형태로 자른다.
    # ========================================================================
    keep_mask = current_mask.unsqueeze(0)              # [1, 8, D]
    keep_dim = int(keep_mask[0, 0].sum().item())        # 77 for ratio=0.4
    past_len = seqlen - recent_size                     # 오래된 토큰 구간 길이

    # 오래된 토큰은 채널도 잘라서 저장: [1, 8, past_len, kept=77]
    past_mask = keep_mask.unsqueeze(2).expand(-1, -1, past_len, -1)
    kv_pruned = kv_8[:, :, :past_len, :][past_mask].reshape(1, num_kv_heads, past_len, keep_dim)

    # 최근 window는 채널 자르지 않음 (다음 decode에서 생성될 token이 full context 필요).
    # kv_8에서 slice → 8 head 유지. 이게 메모리 절약의 핵심 (ThinK는 32h로 저장).
    kv_recent = kv_8[:, :, past_len:, :]                # [1, 8, recent=32, 128]

    _write_pruning_monitor(
        'iterative_prune',
        mode=str(mode),
        ratio=float(ratio),
        num_steps=int(num_steps),
        head_dim=int(head_dim),
        keep_dim=int(keep_dim),
        seqlen=int(seqlen),
        pruned_shape=list(kv_pruned.shape),
        gqa_pruned=True,
    )
    return kv_pruned, kv_recent, keep_mask


def key_pruner_iterative_fast(kv_states, q_states, value_states, o_proj_weight,
                              num_key_value_groups, recent_size=128, ratio=0.4,
                              num_steps=None, mode='gqa_iter32_think_fast',
                              layer_idx=None, score_window=None):
    """B7 alternative: iterative pruning with *actual* channel indexing instead of
    multiply-by-mask (which keeps full dim FLOPs despite the name "pruned attention").

    Behaviour matches key_pruner_iterative semantically; only the per-step pruned
    attention computation differs. Intended for efficiency-measurement scripts —
    not for the main pipeline.
    """
    if num_steps is None:
        # Parse explicit N from `gqa_iter{N}_*` (covers iter2/8/16/32/64 sweep).
        # Fall back: wanda_only → 1 step (mask is purely static), else 4.
        _m_iter = re.search(r'iter(\d+)_', mode)
        if _m_iter is not None:
            num_steps = int(_m_iter.group(1))
        elif 'wanda_only' in mode:
            num_steps = 1
        else:
            num_steps = 4

    bsz, num_heads, seqlen, head_dim = kv_states.shape
    assert bsz == 1
    num_kv_heads = num_heads // num_key_value_groups

    recent_size = min(int(recent_size), int(seqlen))

    kv_8 = kv_states[:, ::num_key_value_groups, :, :]
    k_energy = kv_8.pow(2).mean(dim=2).squeeze(0)  # [8, D]

    if score_window is None:
        score_window = int(recent_size)
    score_window = max(num_steps, min(score_window, q_states.shape[-2]))
    q_post = q_states[..., -score_window:, :]
    q_post_norm = q_post.pow(2).mean(dim=2).squeeze(0)  # [32, D]

    current_mask = torch.ones((num_kv_heads, head_dim), dtype=torch.bool, device=kv_states.device)
    window = max(1, score_window // num_steps)

    _use_k2only = ('k2only' in mode)

    for step in range(num_steps):
        step_ratio = ratio * (step + 1) / num_steps
        step_keep = max(1, min(head_dim - int(round(head_dim * step_ratio)), head_dim - 1))

        # ----- actual channel indexing (not multiply-by-mask) -----
        mask_32 = current_mask.repeat_interleave(num_key_value_groups, dim=0)  # [32, D]
        kept_dim = int(current_mask[0].sum().item())

        # build index tensor [32, kept_dim]
        kept_idx = mask_32.nonzero(as_tuple=False)
        # reshape per-head indices: group rows per head
        kept_idx_per_head = kept_idx[:, 1].view(num_heads, kept_dim)  # [32, kept_dim]
        gather_index = kept_idx_per_head[None, :, None, :].expand(1, num_heads, seqlen, kept_dim)
        k_gathered = torch.gather(kv_states, dim=-1, index=gather_index)  # [1, 32, T, kept_dim]

        q_window = q_post[:, :, step * window:(step + 1) * window, :]
        gather_q = kept_idx_per_head[None, :, None, :].expand(1, num_heads, q_window.shape[2], kept_dim)
        q_win_kept = torch.gather(q_window, dim=-1, index=gather_q)

        attn_logits = torch.matmul(q_win_kept, k_gathered.transpose(2, 3)) / math.sqrt(kept_dim)
        attn_probs = torch.softmax(attn_logits, dim=-1)
        head_out = torch.matmul(attn_probs, value_states)
        proj_influence = _project_head_influence(head_out, o_proj_weight, num_heads, head_dim).squeeze(0)
        del attn_logits, attn_probs, head_out, k_gathered, q_win_kept, gather_index, gather_q

        # ----- update mask (identical to main pruner) -----
        for g in range(num_kv_heads):
            qs, qe = g * num_key_value_groups, (g + 1) * num_key_value_groups
            raw = proj_influence[qs:qe].clamp_min(0)
            weights = raw / raw.sum().clamp_min(1e-12)
            if _use_k2only:
                agg = k_energy[g]
            else:
                score_per_q = q_post_norm[qs:qe] * k_energy[g].unsqueeze(0)
                agg = (weights[:, None] * score_per_q).sum(dim=0)
            top_idx = agg.topk(step_keep).indices
            new_mask = torch.zeros(head_dim, dtype=torch.bool, device=kv_states.device)
            new_mask[top_idx] = True
            current_mask[g] = new_mask

    keep_mask = current_mask.unsqueeze(0)
    keep_dim = int(keep_mask[0, 0].sum().item())
    past_len = seqlen - recent_size
    past_mask = keep_mask.unsqueeze(2).expand(-1, -1, past_len, -1)
    kv_pruned = kv_8[:, :, :past_len, :][past_mask].reshape(1, num_kv_heads, past_len, keep_dim)
    kv_recent = kv_8[:, :, past_len:, :]
    return kv_pruned, kv_recent, keep_mask


def key_pruner_segmented(kv_states, q_states, value_states, o_proj_weight, num_key_value_groups, recent_size=128, ratio=0.3, refresh_interval=500, layer_idx=None):
    """Segment-based pruning: each segment gets its own optimal mask."""
    bsz, num_heads, seqlen, head_dim = kv_states.shape
    assert bsz == 1
    num_kv_heads = num_heads // num_key_value_groups
    keep_dim = max(1, min(head_dim - int(round(head_dim * ratio)), head_dim - 1))
    past_len = seqlen - recent_size

    # Q head importance via proj_influence
    q_post = q_states[..., -32:, :]
    attn_logits = torch.matmul(q_post, kv_states.transpose(2, 3)) / math.sqrt(head_dim)
    attn_probs = torch.softmax(attn_logits, dim=-1)
    head_out = torch.matmul(attn_probs, value_states)
    proj_influence = _project_head_influence(head_out, o_proj_weight, num_heads, head_dim).squeeze(0)

    segments = []
    for start in range(0, past_len, refresh_interval):
        end = min(start + refresh_interval, past_len)
        seg_k = kv_states[:, :, start:end, :]

        # K energy for this segment
        k_energy = seg_k.pow(2).mean(dim=2).squeeze(0)  # [num_heads, head_dim]

        # Build mask per KV head group
        seg_mask = torch.zeros((1, num_heads, head_dim), dtype=torch.bool, device=kv_states.device)
        for kv_h in range(num_kv_heads):
            q_start = kv_h * num_key_value_groups
            q_end = (kv_h + 1) * num_key_value_groups
            raw = proj_influence[q_start:q_end].clamp_min(0)
            denom = raw.sum().clamp_min(1e-12)
            weights = raw / denom
            group_energy = k_energy[q_start:q_end]
            chosen_dim = (weights[:, None] * group_energy).sum(dim=0)
            top_idx = torch.topk(chosen_dim, keep_dim, largest=True).indices
            dim_keep = torch.zeros(head_dim, dtype=torch.bool, device=kv_states.device)
            dim_keep[top_idx] = True
            seg_mask[:, q_start:q_end, :] = dim_keep.unsqueeze(0).unsqueeze(0).expand(1, q_end - q_start, head_dim)

        # Prune this segment
        seg_past_mask = seg_mask.unsqueeze(2).expand(-1, -1, end - start, -1)
        seg_pruned = seg_k[seg_past_mask].reshape(1, num_heads, end - start, keep_dim)
        segments.append((seg_pruned, seg_mask))

    kv_recent = kv_states[:, :, past_len:, :]

    _write_pruning_monitor(
        'segmented_prune',
        ratio=float(ratio),
        head_dim=int(head_dim),
        keep_dim=int(keep_dim),
        num_segments=len(segments),
        refresh_interval=int(refresh_interval),
        segment_sizes=[int(sk.shape[2]) for sk, _ in segments],
        recent_size=int(recent_size),
        seqlen=int(seqlen),
    )
    return segments, kv_recent


class SnapKVCluster():
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', recent_size = 32, ratio =  0.4):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.ratio = ratio
        self.recent_size = recent_size

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', recent_size = 32, ratio =  0.4):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.ratio = ratio
        self.recent_size = recent_size

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states
        
    def update_thinkv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups, query_states_pre=None, key_states_pre=None, o_proj_weight=None, k_proj_weight=None, cos=None, sin=None, pruner_mode='think', tau=1.0, layer_idx=None):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        if q_len < self.max_capacity_prompt:
            if pruner_mode == 'think':
                kv_pruned, kv_recent, mask = key_pruner_query_driven(key_states, query_states, self.recent_size, self.ratio, layer_idx=layer_idx)
            elif pruner_mode in ['think_gqa_mean', 'think_gqa_max', 'think_gqa_argmax']:
                kv_pruned, kv_recent, mask = key_pruner_query_driven_gqa(key_states, query_states, num_key_value_groups, self.recent_size, self.ratio, mode=pruner_mode, layer_idx=layer_idx)
            elif pruner_mode == 'grik':
                # Single-method build: GRIK iterative scorer (N=32, Wanda+dynamic).
                kv_pruned, kv_recent, mask = key_pruner_iterative(key_states, query_states, value_states, o_proj_weight=o_proj_weight, num_key_value_groups=num_key_value_groups, recent_size=self.recent_size, ratio=self.ratio, mode=pruner_mode, layer_idx=layer_idx, k_proj_weight=k_proj_weight)
            elif pruner_mode.startswith('gqa_'):
                kv_pruned, kv_recent, mask = key_pruner_gqa_aware(key_states, query_states, value_states, o_proj_weight=o_proj_weight, num_key_value_groups=num_key_value_groups, recent_size=self.recent_size, ratio=self.ratio, mode=pruner_mode, layer_idx=layer_idx)
            elif pruner_mode.startswith('seg_'):
                refresh_interval = int(getattr(self, 'refresh_interval', 0)) or int(os.environ.get('THINK_REFRESH_INTERVAL', 500))
                segments, kv_recent = key_pruner_segmented(key_states, query_states, value_states, o_proj_weight=o_proj_weight, num_key_value_groups=num_key_value_groups, recent_size=self.recent_size, ratio=self.ratio, refresh_interval=refresh_interval, layer_idx=layer_idx)
                return segments, kv_recent, None, value_states
            else:
                kv_pruned, kv_recent, mask = key_pruner_query_driven_ours(key_states, query_states, query_states_pre, key_states_pre, value_states, o_proj_weight=o_proj_weight, k_proj_weight=k_proj_weight, num_key_value_groups=num_key_value_groups, recent_size=self.recent_size, ratio=self.ratio, mode=pruner_mode, tau=tau, layer_idx=layer_idx, cos=cos, sin=sin)
            # GQA: return V at 8 KV heads to match K storage (parity with H2OKVCluster)
            if pruner_mode.startswith('gqa_'):
                value_states = value_states[:, ::num_key_value_groups, :, :].contiguous()
            return kv_pruned, kv_recent, mask, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')

            if pruner_mode.startswith('gqa_'):
                # GQA-aware eviction: O_proj weighted attention across Q heads in each group
                bsz_e = attn_cache.shape[0]
                num_kv_h = num_heads // num_key_value_groups

                # Compute O_proj head influence for eviction weighting
                # Use only recent window to save memory
                _v_window = value_states[:, :, -self.window_size:, :]
                _attn_window = attn_weights[:, :, :, -self.window_size:]
                _attn_window = _attn_window / _attn_window.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                head_out_evict = torch.matmul(_attn_window, _v_window)  # [B, 32, window, D] — small
                _evict_influence = _project_head_influence(head_out_evict, o_proj_weight, num_heads, head_dim).squeeze(0)
                del head_out_evict, _v_window, _attn_window

                # Per-group O_proj weights: [8, 4]
                _evict_inf_grouped = _evict_influence.view(num_kv_h, num_key_value_groups)
                _evict_weights = _evict_inf_grouped.clamp_min(0)
                _evict_weights = _evict_weights / _evict_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)  # [8, 4]

                # attn_cache: [B, 32, past_len] → O_proj weighted → [B, 8, past_len]
                attn_reshaped = attn_cache.view(bsz_e, num_kv_h, num_key_value_groups, -1)  # [B, 8, 4, past_len]
                attn_grouped = (_evict_weights[None, :, :, None] * attn_reshaped).sum(dim=2)  # [B, 8, past_len]
                indices_8h = attn_grouped.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices  # [B, 8, top_k]
                indices_8h_exp = indices_8h.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                # Extract 8 KV heads from key_states (undo repeat_kv)
                key_8h = key_states[:, ::num_key_value_groups, :, :]
                val_8h = value_states[:, ::num_key_value_groups, :, :]
                k_past_compress = key_8h[:, :, :-self.window_size, :].gather(dim=2, index=indices_8h_exp)
                v_past_compress = val_8h[:, :, :-self.window_size, :].gather(dim=2, index=indices_8h_exp)
                k_cur = key_8h[:, :, -self.window_size:, :]
                v_cur = val_8h[:, :, -self.window_size:, :]
                key_states_evicted = torch.cat([k_past_compress, k_cur], dim=2)  # [B, 8, compressed, D]
                value_states_evicted = torch.cat([v_past_compress, v_cur], dim=2)
                # Re-expand to 32 for pruner functions that expect it
                from transformers.models.llama.modeling_llama import repeat_kv as _repeat_kv
                key_states = _repeat_kv(key_states_evicted, num_key_value_groups)
                value_states = _repeat_kv(value_states_evicted, num_key_value_groups)
            else:
                # Original per-head eviction (32 heads)
                indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
                indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
                v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
                k_cur = key_states[:, :, -self.window_size:, :]
                v_cur = value_states[:, :, -self.window_size:, :]
                key_states = torch.cat([k_past_compress, k_cur], dim = 2)
                value_states = torch.cat([v_past_compress, v_cur], dim = 2)

            if pruner_mode == 'think':
                kv_pruned, kv_recent, mask = key_pruner_query_driven(key_states, query_states, self.recent_size, self.ratio, layer_idx=layer_idx)
            elif pruner_mode in ['think_gqa_mean', 'think_gqa_max', 'think_gqa_argmax']:
                kv_pruned, kv_recent, mask = key_pruner_query_driven_gqa(key_states, query_states, num_key_value_groups, self.recent_size, self.ratio, mode=pruner_mode, layer_idx=layer_idx)
            elif pruner_mode == 'grik':
                # Single-method build: GRIK iterative scorer (N=32, Wanda+dynamic).
                kv_pruned, kv_recent, mask = key_pruner_iterative(key_states, query_states, value_states, o_proj_weight=o_proj_weight, num_key_value_groups=num_key_value_groups, recent_size=self.recent_size, ratio=self.ratio, mode=pruner_mode, layer_idx=layer_idx, k_proj_weight=k_proj_weight)
            elif pruner_mode.startswith('gqa_'):
                kv_pruned, kv_recent, mask = key_pruner_gqa_aware(key_states, query_states, value_states, o_proj_weight=o_proj_weight, num_key_value_groups=num_key_value_groups, recent_size=self.recent_size, ratio=self.ratio, mode=pruner_mode, layer_idx=layer_idx)
            elif pruner_mode.startswith('seg_'):
                refresh_interval = int(getattr(self, 'refresh_interval', 0)) or int(os.environ.get('THINK_REFRESH_INTERVAL', 500))
                segments, kv_recent = key_pruner_segmented(key_states, query_states, value_states, o_proj_weight=o_proj_weight, num_key_value_groups=num_key_value_groups, recent_size=self.recent_size, ratio=self.ratio, refresh_interval=refresh_interval, layer_idx=layer_idx)
                return segments, kv_recent, None, value_states
            else:
                kv_pruned, kv_recent, mask = key_pruner_query_driven_ours(key_states, query_states, query_states_pre, key_states_pre, value_states, o_proj_weight=o_proj_weight, k_proj_weight=k_proj_weight, num_key_value_groups=num_key_value_groups, recent_size=self.recent_size, ratio=self.ratio, mode=pruner_mode, tau=tau, layer_idx=layer_idx, cos=cos, sin=sin)
            # GQA: return V at 8 KV heads to match K storage
            if pruner_mode.startswith('gqa_'):
                value_states = value_states[:, ::num_key_value_groups, :, :].contiguous()
            return kv_pruned, kv_recent, mask, value_states


class H2OKVCluster():
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', recent_size = 32, ratio =  0.4):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.ratio = ratio
        self.recent_size = recent_size

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', recent_size = 32, ratio =  0.4):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.ratio = ratio
        self.recent_size = recent_size

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, :, : -self.window_size].sum(dim = -2)
            # if self.pooling == 'avgpool':
            #     attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            # elif self.pooling == 'maxpool':
            #     attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            # else:
            #     raise ValueError('Pooling method not supported')
            attn_cache = attn_weights_sum
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states
        
    def update_thinkv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups, query_states_pre=None, key_states_pre=None, o_proj_weight=None, k_proj_weight=None, cos=None, sin=None, pruner_mode='think', tau=1.0, layer_idx=None):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        if q_len < self.max_capacity_prompt:
            if pruner_mode == 'think':
                kv_pruned, kv_recent, mask = key_pruner_query_driven(key_states, query_states, self.recent_size, self.ratio, layer_idx=layer_idx)
            elif pruner_mode in ['think_gqa_mean', 'think_gqa_max', 'think_gqa_argmax']:
                kv_pruned, kv_recent, mask = key_pruner_query_driven_gqa(key_states, query_states, num_key_value_groups, self.recent_size, self.ratio, mode=pruner_mode, layer_idx=layer_idx)
            elif pruner_mode == 'grik':
                # Single-method build: GRIK iterative scorer (N=32, Wanda+dynamic).
                kv_pruned, kv_recent, mask = key_pruner_iterative(key_states, query_states, value_states, o_proj_weight=o_proj_weight, num_key_value_groups=num_key_value_groups, recent_size=self.recent_size, ratio=self.ratio, mode=pruner_mode, layer_idx=layer_idx, k_proj_weight=k_proj_weight)
            elif pruner_mode.startswith('gqa_'):
                kv_pruned, kv_recent, mask = key_pruner_gqa_aware(key_states, query_states, value_states, o_proj_weight=o_proj_weight, num_key_value_groups=num_key_value_groups, recent_size=self.recent_size, ratio=self.ratio, mode=pruner_mode, layer_idx=layer_idx)
            elif pruner_mode.startswith('seg_'):
                refresh_interval = int(getattr(self, 'refresh_interval', 0)) or int(os.environ.get('THINK_REFRESH_INTERVAL', 500))
                segments, kv_recent = key_pruner_segmented(key_states, query_states, value_states, o_proj_weight=o_proj_weight, num_key_value_groups=num_key_value_groups, recent_size=self.recent_size, ratio=self.ratio, refresh_interval=refresh_interval, layer_idx=layer_idx)
                return segments, kv_recent, None, value_states
            else:
                kv_pruned, kv_recent, mask = key_pruner_query_driven_ours(key_states, query_states, query_states_pre, key_states_pre, value_states, o_proj_weight=o_proj_weight, k_proj_weight=k_proj_weight, num_key_value_groups=num_key_value_groups, recent_size=self.recent_size, ratio=self.ratio, mode=pruner_mode, tau=tau, layer_idx=layer_idx, cos=cos, sin=sin)
            # GQA: return V at 8 KV heads to match K storage
            if pruner_mode.startswith('gqa_'):
                value_states = value_states[:, ::num_key_value_groups, :, :].contiguous()
            return kv_pruned, kv_recent, mask, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, :, : -self.window_size].sum(dim = -2)
            # if self.pooling == 'avgpool':
            #     attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            # elif self.pooling == 'maxpool':
            #     attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            # else:
            #     raise ValueError('Pooling method not supported')
            attn_cache = attn_weights_sum

            if pruner_mode.startswith('gqa_'):
                # GQA-aware eviction → 8 heads (rectangular). Aggregation depends on suffix:
                #   - default:     O_proj influence weighted (ours)
                #   - _adakv:      GQA-mean (Ada-KV non-adaptive fallback, SnapKV-GQA style)
                #   - _adakvmax:   GQA-max (LAVa-style conservative)
                #   - _adakvada:   Ada-KV adaptive budget + floor_alpha safeguard (rectangular padding)
                bsz_e = attn_cache.shape[0]
                num_kv_h = num_heads // num_key_value_groups

                attn_reshaped = attn_cache.view(bsz_e, num_kv_h, num_key_value_groups, -1)  # [B, 8, 4, past_len]

                _is_adakv = ('_adakv' in pruner_mode) or pruner_mode.endswith('_adakv')
                _is_adakv_max = '_adakvmax' in pruner_mode
                _is_adakv_ada = '_adakvada' in pruner_mode

                if _is_adakv_max:
                    attn_grouped = attn_reshaped.max(dim=2).values  # [B, 8, past_len]
                elif _is_adakv:
                    # plain mean (Ada-KV GQA-mean aggregation, no adaptive budget unless _adakvada)
                    attn_grouped = attn_reshaped.mean(dim=2)  # [B, 8, past_len]
                else:
                    # default: O_proj influence weighted aggregation (ours)
                    _v_window = value_states[:, :, -self.window_size:, :]
                    _attn_window = attn_weights[:, :, :, -self.window_size:]
                    _attn_window = _attn_window / _attn_window.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    head_out_evict = torch.matmul(_attn_window, _v_window)
                    _evict_influence = _project_head_influence(head_out_evict, o_proj_weight, num_heads, head_dim).squeeze(0)
                    del head_out_evict, _v_window, _attn_window
                    _evict_inf_grouped = _evict_influence.view(num_kv_h, num_key_value_groups)
                    _evict_weights = _evict_inf_grouped.clamp_min(0)
                    _evict_weights = _evict_weights / _evict_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
                    attn_grouped = (_evict_weights[None, :, :, None] * attn_reshaped).sum(dim=2)

                base_cap = self.max_capacity_prompt - self.window_size

                if _is_adakv_ada:
                    # Ada-KV adaptive budget allocation across groups (Feng et al., NeurIPS 2025)
                    floor_alpha = float(os.environ.get('ADAKV_FLOOR_ALPHA', '0.2'))
                    floor_cap = int(base_cap * floor_alpha)
                    past_len = attn_grouped.shape[-1]
                    total_budget = num_kv_h * base_cap

                    # Global top-K across concatenated groups → count per group
                    flat_scores = attn_grouped.reshape(bsz_e, num_kv_h * past_len)
                    top_idx = flat_scores.topk(min(total_budget, flat_scores.shape[-1]), dim=-1).indices
                    group_ids = top_idx // past_len  # [B, total_budget]
                    group_counts = torch.zeros((bsz_e, num_kv_h), dtype=torch.long, device=attn_cache.device)
                    group_counts.scatter_add_(-1, group_ids, torch.ones_like(group_ids))

                    # Safeguard blend: B_i = round(count_i * (1 - alpha) + floor_cap)
                    head_budget = torch.round(group_counts.float() * (1 - floor_alpha) + floor_cap).long()
                    head_budget = head_budget.clamp(min=1, max=past_len)  # [B, 8]
                    max_budget = int(head_budget.max().item())

                    # Sort by score per group, keep top max_budget for rectangular storage.
                    # Groups with B_i < max_budget will have the extra slots filled with their
                    # next-best tokens — this slightly over-caches for small-budget groups but
                    # keeps the layout rectangular for downstream channel pruning. The stored
                    # total is therefore 8 × max_budget (≥ Ada-KV's true 8 × avg_budget).
                    sort_idx = attn_grouped.sort(dim=-1, descending=True).indices  # [B, 8, past_len]
                    indices_8h = sort_idx[:, :, :max_budget]  # [B, 8, max_budget]
                else:
                    # Uniform budget (Ada-KV's floor_alpha=1.0 or simple SnapKV-GQA)
                    indices_8h = attn_grouped.topk(base_cap, dim=-1).indices

                indices_8h_exp = indices_8h.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                key_8h = key_states[:, ::num_key_value_groups, :, :]
                val_8h = value_states[:, ::num_key_value_groups, :, :]
                k_past_compress = key_8h[:, :, :-self.window_size, :].gather(dim=2, index=indices_8h_exp)
                v_past_compress = val_8h[:, :, :-self.window_size, :].gather(dim=2, index=indices_8h_exp)
                k_cur = key_8h[:, :, -self.window_size:, :]
                v_cur = val_8h[:, :, -self.window_size:, :]
                key_states_evicted = torch.cat([k_past_compress, k_cur], dim=2)
                value_states_evicted = torch.cat([v_past_compress, v_cur], dim=2)
                from transformers.models.llama.modeling_llama import repeat_kv as _repeat_kv
                key_states = _repeat_kv(key_states_evicted, num_key_value_groups)
                value_states = _repeat_kv(value_states_evicted, num_key_value_groups)
            else:
                indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
                indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
                v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
                k_cur = key_states[:, :, -self.window_size:, :]
                v_cur = value_states[:, :, -self.window_size:, :]
                key_states = torch.cat([k_past_compress, k_cur], dim = 2)
                value_states = torch.cat([v_past_compress, v_cur], dim = 2)

            if pruner_mode == 'think':
                kv_pruned, kv_recent, mask = key_pruner_query_driven(key_states, query_states, self.recent_size, self.ratio, layer_idx=layer_idx)
            elif pruner_mode in ['think_gqa_mean', 'think_gqa_max', 'think_gqa_argmax']:
                kv_pruned, kv_recent, mask = key_pruner_query_driven_gqa(key_states, query_states, num_key_value_groups, self.recent_size, self.ratio, mode=pruner_mode, layer_idx=layer_idx)
            elif pruner_mode == 'grik':
                # Single-method build: GRIK iterative scorer (N=32, Wanda+dynamic).
                kv_pruned, kv_recent, mask = key_pruner_iterative(key_states, query_states, value_states, o_proj_weight=o_proj_weight, num_key_value_groups=num_key_value_groups, recent_size=self.recent_size, ratio=self.ratio, mode=pruner_mode, layer_idx=layer_idx, k_proj_weight=k_proj_weight)
            elif pruner_mode.startswith('gqa_'):
                kv_pruned, kv_recent, mask = key_pruner_gqa_aware(key_states, query_states, value_states, o_proj_weight=o_proj_weight, num_key_value_groups=num_key_value_groups, recent_size=self.recent_size, ratio=self.ratio, mode=pruner_mode, layer_idx=layer_idx)
            elif pruner_mode.startswith('seg_'):
                refresh_interval = int(getattr(self, 'refresh_interval', 0)) or int(os.environ.get('THINK_REFRESH_INTERVAL', 500))
                segments, kv_recent = key_pruner_segmented(key_states, query_states, value_states, o_proj_weight=o_proj_weight, num_key_value_groups=num_key_value_groups, recent_size=self.recent_size, ratio=self.ratio, refresh_interval=refresh_interval, layer_idx=layer_idx)
                return segments, kv_recent, None, value_states
            else:
                kv_pruned, kv_recent, mask = key_pruner_query_driven_ours(key_states, query_states, query_states_pre, key_states_pre, value_states, o_proj_weight=o_proj_weight, k_proj_weight=k_proj_weight, num_key_value_groups=num_key_value_groups, recent_size=self.recent_size, ratio=self.ratio, mode=pruner_mode, tau=tau, layer_idx=layer_idx, cos=cos, sin=sin)
            # GQA: return V at 8 KV heads to match K storage
            if pruner_mode.startswith('gqa_'):
                value_states = value_states[:, ::num_key_value_groups, :, :].contiguous()
            return kv_pruned, kv_recent, mask, value_states

    # ------------------------------------------------------------------
    # LeanK-fair prefill path.
    #   Separate method (does NOT touch update_thinkv) so the default 2-way
    #   [pruned_past | recent] flow is preserved byte-for-byte. Only methods
    #   whose pruner_mode ends with '_leankfair' route here.
    # ------------------------------------------------------------------
    def update_thinkv_leankfair(
        self,
        key_states,
        query_states,
        value_states,
        attention_mask,
        num_key_value_groups,
        query_states_pre=None,
        key_states_pre=None,
        o_proj_weight=None,
        k_proj_weight=None,
        cos=None,
        sin=None,
        pruner_mode='gqa_iter32_think_leankfair',
        tau=1.0,
        layer_idx=None,
        sink_size=4,
    ):
        """Prefill pruning with sink+recent exemption (LeanK token-position policy).

        Layout of the stored cache for this layer:
          [0, sink_size)                    → sink_K at full d_head
          [sink_size, T - recent_size)      → middle_K, channel-pruned to kept_dim
          [T - recent_size, T)              → recent_K at full d_head (decode appends here)

        V is stored as a single 8h tensor [1, 8, T, head_dim] covering the entire prefill
        range — V ordering already matches [sink | middle | recent] so no V-side split
        needed. Decode-time attention concat([w_sink, w_middle, w_recent]) lines up.

        Returns a 5-tuple: (key_sink, kv_pruned, kv_recent, mask, value_states_8h).
        The extra `key_sink` (vs update_thinkv's 4-tuple) is what the caller routes into
        `DynamicCache.update_think_leankfair`.
        """
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        assert bsz == 1

        # Apply H2O-style token eviction if prompt overshoots budget, same as update_thinkv.
        # We reuse the existing eviction path (GQA-aware O_proj weighting) and only diverge
        # when computing the channel-prune split.
        if q_len >= self.max_capacity_prompt:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            _mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            _mask_cond = torch.arange(_mask.size(-1), device=attn_weights.device)
            _mask.masked_fill_(_mask_cond < (_mask_cond + 1).view(_mask.size(-1), 1), 0)
            attn_weights[:, :, -self.window_size:, -self.window_size:] += _mask[None, None, :, :].to(attn_weights.device)
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_cache = attn_weights[:, :, :, : -self.window_size].sum(dim=-2)

            num_kv_h = num_heads // num_key_value_groups
            attn_reshaped = attn_cache.view(attn_cache.shape[0], num_kv_h, num_key_value_groups, -1)
            # O_proj-weighted GQA aggregation (ours).
            _v_window = value_states[:, :, -self.window_size:, :]
            _attn_window = attn_weights[:, :, :, -self.window_size:]
            _attn_window = _attn_window / _attn_window.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            _head_out = torch.matmul(_attn_window, _v_window)
            _evict_influence = _project_head_influence(_head_out, o_proj_weight, num_heads, head_dim).squeeze(0)
            del _head_out, _v_window, _attn_window
            _evict_grouped = _evict_influence.view(num_kv_h, num_key_value_groups).clamp_min(0)
            _evict_weights = _evict_grouped / _evict_grouped.sum(dim=1, keepdim=True).clamp_min(1e-12)
            attn_grouped = (_evict_weights[None, :, :, None] * attn_reshaped).sum(dim=2)

            base_cap = self.max_capacity_prompt - self.window_size
            indices_8h = attn_grouped.topk(base_cap, dim=-1).indices
            indices_8h_exp = indices_8h.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            key_8h = key_states[:, ::num_key_value_groups, :, :]
            val_8h = value_states[:, ::num_key_value_groups, :, :]
            k_past_compress = key_8h[:, :, :-self.window_size, :].gather(dim=2, index=indices_8h_exp)
            v_past_compress = val_8h[:, :, :-self.window_size, :].gather(dim=2, index=indices_8h_exp)
            k_cur = key_8h[:, :, -self.window_size:, :]
            v_cur = val_8h[:, :, -self.window_size:, :]
            key_states_evicted = torch.cat([k_past_compress, k_cur], dim=2)
            value_states_evicted = torch.cat([v_past_compress, v_cur], dim=2)
            from transformers.models.llama.modeling_llama import repeat_kv as _repeat_kv
            key_states = _repeat_kv(key_states_evicted, num_key_value_groups)
            value_states = _repeat_kv(value_states_evicted, num_key_value_groups)
            q_len = key_states.shape[-2]

        # Guard: degenerate prompt where sink+recent would leave no middle to prune.
        # Fall back to identity (no channel compression) so we never crash short prompts.
        effective_sink = int(max(0, sink_size))
        if q_len <= effective_sink + int(self.recent_size) + 1:
            # Treat the whole prompt as sink — nothing to prune. Emit an all-True mask
            # so downstream decode concat is well-formed.
            kv_8 = key_states[:, ::num_key_value_groups, :, :].contiguous()
            num_kv_h = num_heads // num_key_value_groups
            full_mask = torch.ones((1, num_kv_h, head_dim), dtype=torch.bool, device=key_states.device)
            # Split so the three buffers sum back to q_len. Put everything except the last
            # recent_size tokens into sink, leaving pruned empty.
            rec = min(int(self.recent_size), q_len)
            kv_sink = kv_8[:, :, :q_len - rec, :].contiguous()
            kv_pruned_empty = kv_8[:, :, q_len - rec:q_len - rec, :].contiguous()  # [1, H, 0, D]
            kv_recent = kv_8[:, :, q_len - rec:, :].contiguous()
            v_stored = value_states[:, ::num_key_value_groups, :, :].contiguous()
            return kv_sink, kv_pruned_empty, kv_recent, full_mask, v_stored

        # Slice the sink tokens off the FRONT, leave the rest for the standard iterative
        # pruner. Passing the sliced K/Q/V keeps the Q² and K² statistics free of sink-token
        # outliers (attention sinks are typically high-norm and would otherwise bias channel
        # selection toward preserving sink-friendly channels).
        kv_8_full = key_states[:, ::num_key_value_groups, :, :]
        kv_sink = kv_8_full[:, :, :effective_sink, :].contiguous()   # full d_head, 8h

        key_states_mid = key_states[:, :, effective_sink:, :]         # 32h view
        query_states_mid = query_states[:, :, effective_sink:, :]     # all 32 q heads, trimmed
        value_states_mid = value_states[:, :, effective_sink:, :]     # 32h view

        # The pruner_mode string is passed through unchanged so the inner iter32/Q²·K²
        # scoring logic in key_pruner_iterative stays identical. All 'iter32_think*',
        # 'iter32_rope', 'iter32_wanda', etc. variants work automatically — the sink
        # exemption is additive and orthogonal to the channel-scoring mechanism.
        kv_pruned, kv_recent, mask = key_pruner_iterative(
            key_states_mid,
            query_states_mid,
            value_states_mid,
            o_proj_weight=o_proj_weight,
            num_key_value_groups=num_key_value_groups,
            recent_size=self.recent_size,
            ratio=self.ratio,
            mode=pruner_mode,
            layer_idx=layer_idx,
            k_proj_weight=k_proj_weight,
        )

        # Store V at 8 KV heads covering the full prefill range [0, q_len).
        value_states_8h = value_states[:, ::num_key_value_groups, :, :].contiguous()
        return kv_sink, kv_pruned, kv_recent, mask, value_states_8h


def init_snapkv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 4096
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'recent_size'):
            self.config.recent_size = 32
        if not hasattr(self.config, 'ratio'):
            self.config.ratio = 0.4
    
    
    self.kv_cluster = SnapKVCluster( 
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        recent_size = self.config.recent_size,
        ratio = self.config.ratio
        )


def init_H2O(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'recent_size'):
            self.config.recent_size = 32
        if not hasattr(self.config, 'ratio'):
            self.config.ratio = 0.4
    
    
    self.kv_cluster = H2OKVCluster(
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        recent_size = self.config.recent_size,
        ratio = self.config.ratio
        )
