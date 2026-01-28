import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, deque
from tqdm import tqdm


# ============================================================
# Algorithm : Training & Evaluation Pipeline
# Inputs:
#   (1) model_name, model_size, top_k
#   (2) scorer_type ∈ {CECAR, FlashMoE}
#   (3) epochs, LFU window, dedup flag
# Outputs:
#   (1) trained scorer checkpoints
#   (2) hit-rate metrics and logs
# ============================================================

# ============================================================
# Config
# ============================================================

CACHE_SIZES = [8, 12, 16, 20, 24, 28, 32]


MODEL_CONFIGS = {
    "OLMoE_1B_7B_0125_Instruct": {
        "num_experts": 64,
        "belady_topk": 48,
        "default_top_k": 8,
        "train_dataset": "triviaqa",
        "ood_datasets": ["triviaqa", "mbpp", "math", "gpqa", "humaneval"],
    },
    "Qwen3_30B_A3B": {
        "num_experts": 128,
        "belady_topk": 98,
        "default_top_k": 8,
        "train_dataset": "triviaqa",
        "ood_datasets": ["triviaqa", "mbpp", "math", "gpqa", "humaneval"],
    },
    "DeepSeek_v2_Lite_Chat": {
        "num_experts": 64,
        "num_shared_experts": 2,
        "belady_topk": 48,
        "default_top_k": 6,
        "train_dataset": "triviaqa",
        "ood_datasets": ["triviaqa", "mbpp", "math", "gpqa", "humaneval"],
    },
}

MODEL_SIZE_CONFIGS = {
    "small": {"hidden_dim": 16, "num_layers": 2},
    "base": {"hidden_dim": 32, "num_layers": 3},
    "large": {"hidden_dim": 64, "num_layers": 4},
}

ML_CECAR = "ML_CECAR"          
ML_FlashMoE = "ML_FlashMoE"    

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
ROUTING_DATA_DIR = os.path.join(BASE_DIR, "src", "Routing_history")
PRETRAIN_ROOT = os.path.join(BASE_DIR, "Pre_trained_FFN")

# ============================================================
# Timing utilities
# ============================================================
def _sync_if_cuda(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    def __init__(self, device="cuda", enabled=True):
        self.device = device
        self.enabled = enabled
        self.t0 = None
        self.dt = 0.0

    def __enter__(self):
        if self.enabled:
            _sync_if_cuda(self.device)
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled:
            _sync_if_cuda(self.device)
            self.dt = time.perf_counter() - self.t0


def fmt_sec(s: float) -> str:
    if s < 1.0:
        return f"{s*1000:.2f} ms"
    return f"{s:.3f} s"


# ============================================================
# Routing PT loader (decode-only payload)
# ============================================================
def _extract_tensor_from_dict(d: dict, pt_path: str):
    preferred_keys = ["tensor", "routing", "routing_logits", "router_logits", "logits", "scores", "data", "x"]
    for k in preferred_keys:
        if k in d and torch.is_tensor(d[k]):
            return d[k]

    tensor_items = [(k, v) for k, v in d.items() if torch.is_tensor(v)]
    if len(tensor_items) == 1:
        return tensor_items[0][1]

    if len(tensor_items) > 1:
        tensor_items.sort(key=lambda kv: kv[1].numel(), reverse=True)
        return tensor_items[0][1]

    raise TypeError(f"{pt_path}: dict element has no tensor values. keys={list(d.keys())}")


def load_decode_routing_as_TBLE(pt_path: str, device: str = "cuda"):
    """
    Returns x as (T, B, L, E) on device.
    Accepts saved payload shapes:
      - (B, T, 1, L, E) or (T, B, L, E) or (B, T, L, E) etc.
    """
    payload = torch.load(pt_path, map_location="cpu")

    if isinstance(payload, dict):
        if "routing_decode" not in payload:
            keys = list(payload.keys())
            raise KeyError(f"{pt_path} missing key 'routing_decode'. Available keys: {keys}")

        routing_list = payload["routing_decode"]
        if not isinstance(routing_list, list) or len(routing_list) == 0:
            raise ValueError(f"{pt_path} payload['routing_decode'] is empty or not a list.")

        if isinstance(routing_list[0], dict):
            routing_list = [_extract_tensor_from_dict(d, pt_path) for d in routing_list]

        x = torch.stack(routing_list, dim=0)

    elif torch.is_tensor(payload):
        x = payload
    else:
        raise TypeError(f"{pt_path} must be dict or tensor. Got type={type(payload)}")

    if x.dim() == 5:
        B, T, b1, L, E = x.shape
        if b1 != 1:
            raise RuntimeError(f"{pt_path} expects inner batch dim=1. Got {b1}")
        x = x.squeeze(2).permute(1, 0, 2, 3)

    elif x.dim() == 4:
        s0, s1, s2, s3 = x.shape
        if s0 <= 64 and s1 > 64:
            x = x.permute(1, 0, 2, 3)

    else:
        raise RuntimeError(f"{pt_path} routing tensor must be 4D or 5D. Got shape={tuple(x.shape)}")

    return x.contiguous().to(device)


# ============================================================
# CPU-side feature/target build
# ============================================================
def get_topk_sequence(data, top_k=8):
    B, T, E = data.shape
    seq = [[[] for _ in range(T)] for _ in range(B)]
    for b in range(B):
        for t in range(T):
            seq[b][t] = torch.topk(data[b, t], top_k).indices.tolist()
    return seq


def build_belady_distance(sequence, num_experts: int):
    """
    Returns dist_tensor: (B, T, E) where dist = next_use_t - t, inf if never used again.
    """
    B, T = len(sequence), len(sequence[0])
    dist_tensor = torch.full((B, T, num_experts), float("inf"), dtype=torch.float32)

    for b in range(B):
        next_use = defaultdict(list)
        for t in range(T):
            for e in sequence[b][t]:
                next_use[e].append(t)
        for e in next_use:
            next_use[e].sort(reverse=True)

        for t in range(T):
            dist_list = []
            for e in range(num_experts):
                while next_use[e] and next_use[e][-1] <= t:
                    next_use[e].pop()
                d = next_use[e][-1] - t if next_use[e] else float("inf")
                dist_list.append(d)
            dist_tensor[b, t] = torch.tensor(dist_list, dtype=torch.float32)

    return dist_tensor


def belady_target_from_dist(dist_tensor: torch.Tensor, belady_topk: int):
    """
    dist_tensor: (B, T, E), inf allowed.
    Output target: (B, T, E) normalized to [0,1].
    """
    target = dist_tensor.clone()
    target[torch.isinf(target)] = float(belady_topk)
    target = target.clamp(min=1.0, max=float(belady_topk))
    target = (target - 1.0) / (float(belady_topk) - 1.0)
    return target


def build_lru_lfu_scores(sequence, num_experts=64, window_size=0):
    B, T = len(sequence), len(sequence[0])
    if window_size == 0:
        window_size = T

    lru_scores = torch.full((B, T, num_experts), fill_value=float("inf"), dtype=torch.float32)
    lfu_scores = torch.zeros((B, T, num_experts), dtype=torch.float32)

    for b in range(B):
        last_used = [-1] * num_experts
        recent_usage = [deque(maxlen=window_size) for _ in range(num_experts)]

        for t in range(T):
            current_experts = set(sequence[b][t])

            for e in range(num_experts):
                recent_usage[e].append(1 if e in current_experts else 0)
                if e in current_experts:
                    last_used[e] = t

            for e in range(num_experts):
                lru_scores[b, t, e] = (t - last_used[e]) if last_used[e] != -1 else float("inf")
                lfu_scores[b, t, e] = sum(recent_usage[e])

    lru_scores = lru_scores + 1.0
    return lru_scores, lfu_scores


# ============================================================
# Models
# ============================================================
class MultiEvictionScorer(nn.Module):
    """
    ML_CECAR:
      input  : x (B*T, E, 2)
      output : (score_short, score_long) each (B*T, E)
    """
    def __init__(self, input_dim=2, hidden_dim=32, num_layers=3):
        super().__init__()

        def make_mlp():
            layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
            for _ in range(num_layers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
            layers.append(nn.Linear(hidden_dim, 1))
            return nn.Sequential(*layers)

        self.ffn_short = make_mlp()
        self.ffn_long = make_mlp()

    def forward(self, x):
        BTE, E, F_dim = x.shape
        x_flat = x.view(BTE * E, F_dim)
        score_short = self.ffn_short(x_flat)
        score_long = self.ffn_long(x_flat)
        return torch.abs(score_short.view(BTE, E)), torch.abs(score_long.view(BTE, E))


class EvictionScorerMLCache(nn.Module):
    """
    ML_FlashMoE:
      input  : x (B*T, E, 2) or (B*T, 2E)
      output : scores (B*T, E)
    """
    def __init__(self, num_experts: int, hidden_dim=64, num_layers=3):
        super().__init__()
        input_dim = 2 * num_experts
        output_dim = num_experts

        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, output_dim)]
        self.ffn = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 3:
            x = x.reshape(x.size(0), -1) 
        elif x.dim() == 2:
            pass
        else:
            raise ValueError(f"Unexpected x shape: {tuple(x.shape)}")
        return torch.abs(self.ffn(x))


# ============================================================
# Loss
# ============================================================
def distance_regression_loss(pred, target):
    mask = ~torch.isinf(target)
    return F.mse_loss(pred[mask], target[mask])


# ============================================================
# GPU cache eval (tensorized) — supports evaluating ML_CECAR, ML_FlashMoE, or both
# ============================================================
@torch.no_grad()
def evaluate_model_multiple_caches_gpu(
    X_eval,
    model_cecar,
    model_flash,
    routing_weights_eval,
    dist_tensor_eval,
    top_k=8,
    num_samples=None,
    seq_len=None,
    cache_sizes=CACHE_SIZES,
    device="cuda",
    dedup_per_token=True,
):
    """
    Always computes: LRU/LFU/LIFO/Belady.
    Optionally computes:
      - ML_CECAR if model_cecar is not None
      - ML_FlashMoE if model_flash is not None
    """
    assert device.startswith("cuda"), "GPU eval requires cuda device"

    total_len, E = routing_weights_eval.shape
    if num_samples is None or seq_len is None:
        B = 1
        T = total_len
    else:
        assert num_samples * seq_len == total_len, f"Expected total_len==B*T. Got {total_len} vs {num_samples}*{seq_len}"
        B = num_samples
        T = seq_len

    X_eval = X_eval.to(device, non_blocking=True)
    routing_weights_eval = routing_weights_eval.to(device, non_blocking=True)
    dist_tensor_eval = dist_tensor_eval.to(device, non_blocking=True)

    # ------------------------------------------------
    # Score inference (only for selected models)
    # ------------------------------------------------
    score1 = score2 = None
    if model_cecar is not None:
        model_cecar.eval()
        s1, s2 = model_cecar(X_eval)          
        score1 = s1.view(B, T, E)
        score2 = s2.view(B, T, E)

    score_flash = None
    if model_flash is not None:
        model_flash.eval()
        s = model_flash(X_eval)              
        score_flash = s.view(B, T, E)

    dist_bt = dist_tensor_eval.view(B, T, E)

    # ------------------------------------------------
    # Precompute requests
    # ------------------------------------------------
    topk_idx = torch.topk(routing_weights_eval, k=top_k, dim=1).indices.view(B, T, top_k)

    if dedup_per_token:
        topk_sorted, _ = topk_idx.sort(dim=2)
        uniq_mask = torch.ones((B, T, top_k), device=device, dtype=torch.bool)
        uniq_mask[:, :, 1:] = topk_sorted[:, :, 1:] != topk_sorted[:, :, :-1]
        topk_idx_eff = topk_sorted
        valid_mask = uniq_mask
    else:
        topk_idx_eff = topk_idx
        valid_mask = torch.ones((B, T, top_k), device=device, dtype=torch.bool)

    cache_sizes_t = torch.tensor(cache_sizes, device=device, dtype=torch.int32)
    C = cache_sizes_t.numel()

    if model_cecar is not None:
        cs_f = cache_sizes_t.to(torch.float32)
        w2 = torch.where(cs_f >= 36.0, torch.full_like(cs_f, 0.5), 0.5 * (cs_f / 36.0))
        w1 = torch.where(cs_f >= 36.0, torch.full_like(cs_f, 0.5), 1.0 - w2)

    # ------------------------------------------------
    # Cache states
    # ------------------------------------------------
    in_cache_cecar = torch.zeros((B, C, E), device=device, dtype=torch.bool)   
    in_cache_flash = torch.zeros((B, C, E), device=device, dtype=torch.bool)   
    in_cache_bd = torch.zeros((B, C, E), device=device, dtype=torch.bool)
    in_cache_lru = torch.zeros((B, C, E), device=device, dtype=torch.bool)
    in_cache_lfu = torch.zeros((B, C, E), device=device, dtype=torch.bool)
    in_cache_lifo = torch.zeros((B, C, E), device=device, dtype=torch.bool)

    count_cecar = torch.zeros((B, C), device=device, dtype=torch.int32)
    count_flash = torch.zeros((B, C), device=device, dtype=torch.int32)
    count_bd = torch.zeros((B, C), device=device, dtype=torch.int32)
    count_lru = torch.zeros((B, C), device=device, dtype=torch.int32)
    count_lfu = torch.zeros((B, C), device=device, dtype=torch.int32)
    count_lifo = torch.zeros((B, C), device=device, dtype=torch.int32)

    INF_INT = 1_000_000_000
    NEG_INT = -1

    # LRU metadata
    last_used = torch.full((B, C, E), INF_INT, device=device, dtype=torch.int32)

    # LFU metadata
    freq = torch.full((B, C, E), INF_INT, device=device, dtype=torch.int32)

    # LIFO metadata
    insert_time = torch.full((B, C, E), NEG_INT, device=device, dtype=torch.int32)

    # ------------------------------------------------
    # Hit counters
    # ------------------------------------------------
    cecar_hits = torch.zeros((C,), device=device, dtype=torch.int64)
    flash_hits = torch.zeros((C,), device=device, dtype=torch.int64)
    bd_hits = torch.zeros((C,), device=device, dtype=torch.int64)
    lru_hits = torch.zeros((C,), device=device, dtype=torch.int64)
    lfu_hits = torch.zeros((C,), device=device, dtype=torch.int64)
    lifo_hits = torch.zeros((C,), device=device, dtype=torch.int64)
    total_requests = torch.zeros((), device=device, dtype=torch.int64)

    # ------------------------------------------------
    # Small helpers
    # ------------------------------------------------
    def _masked_evict_and_insert(in_cache, need_mask, evict_idx_bc, new_e_bc):
        if need_mask.any():
            bc = torch.nonzero(need_mask, as_tuple=False)
            b_ids = bc[:, 0]
            c_ids = bc[:, 1]
            ev = evict_idx_bc[b_ids, c_ids].to(torch.long)
            ne = new_e_bc[b_ids, c_ids].to(torch.long)
            in_cache[b_ids, c_ids, ev] = False
            in_cache[b_ids, c_ids, ne] = True

    def _masked_insert_only(in_cache, ins_mask, new_e_bc):
        if ins_mask.any():
            bc = torch.nonzero(ins_mask, as_tuple=False)
            b_ids = bc[:, 0]
            c_ids = bc[:, 1]
            ne = new_e_bc[b_ids, c_ids].to(torch.long)
            in_cache[b_ids, c_ids, ne] = True

    # ------------------------------------------------
    # Main loop
    # ------------------------------------------------
    for t in range(T):
        d = dist_bt[:, t, :]  

        if model_cecar is not None:
            s1 = score1[:, t, :]
            s2 = score2[:, t, :]
            total_score_cecar = (s1[:, None, :] * w1[None, :, None]) + (s2[:, None, :] * w2[None, :, None])

        for k in range(top_k):
            e_req = topk_idx_eff[:, t, k]        
            v = valid_mask[:, t, k]              
            total_requests += v.sum().to(torch.int64)

            req_time = t * top_k + k
            v_bc = v[:, None].expand(B, C)

            # ------------------------
            # ML_CECAR cache
            # ------------------------
            if model_cecar is not None:
                in_cecar = in_cache_cecar.gather(2, e_req[:, None, None].expand(B, C, 1)).squeeze(2)
                cecar_hits += (in_cecar & v_bc).sum(dim=0).to(torch.int64)

                miss = (~in_cecar) & v_bc
                if miss.any():
                    not_full = (count_cecar < cache_sizes_t[None, :]) & miss
                    need_evict = (count_cecar >= cache_sizes_t[None, :]) & miss

                    _masked_insert_only(in_cache_cecar, not_full, e_req[:, None].expand(B, C))
                    count_cecar = count_cecar + not_full.to(torch.int32)

                    if need_evict.any():
                        masked_score = total_score_cecar.masked_fill(~in_cache_cecar, -1e30)
                        evict_idx = masked_score.argmax(dim=2) 
                        _masked_evict_and_insert(in_cache_cecar, need_evict, evict_idx, e_req[:, None].expand(B, C))

            # ------------------------
            # ML_FlashMoE cache
            # ------------------------
            if model_flash is not None:
                s_flash = score_flash[:, t, :] 
                s_flash_bcE = s_flash[:, None, :].expand(B, C, E)

                in_flash = in_cache_flash.gather(2, e_req[:, None, None].expand(B, C, 1)).squeeze(2)
                flash_hits += (in_flash & v_bc).sum(dim=0).to(torch.int64)

                miss = (~in_flash) & v_bc
                if miss.any():
                    not_full = (count_flash < cache_sizes_t[None, :]) & miss
                    need_evict = (count_flash >= cache_sizes_t[None, :]) & miss

                    _masked_insert_only(in_cache_flash, not_full, e_req[:, None].expand(B, C))
                    count_flash = count_flash + not_full.to(torch.int32)

                    if need_evict.any():
                        masked_score = s_flash_bcE.masked_fill(~in_cache_flash, -1e30)
                        evict_idx = masked_score.argmax(dim=2)  
                        _masked_evict_and_insert(in_cache_flash, need_evict, evict_idx, e_req[:, None].expand(B, C))

            # ------------------------
            # LRU
            # ------------------------
            in_lru = in_cache_lru.gather(2, e_req[:, None, None].expand(B, C, 1)).squeeze(2)
            lru_hits += (in_lru & v_bc).sum(dim=0).to(torch.int64)

            hit = in_lru & v_bc
            if hit.any():
                bc = torch.nonzero(hit, as_tuple=False)
                b_ids, c_ids = bc[:, 0], bc[:, 1]
                last_used[b_ids, c_ids, e_req[b_ids].to(torch.long)] = int(req_time)

            miss = (~in_lru) & v_bc
            if miss.any():
                not_full = (count_lru < cache_sizes_t[None, :]) & miss
                need_evict = (count_lru >= cache_sizes_t[None, :]) & miss

                _masked_insert_only(in_cache_lru, not_full, e_req[:, None].expand(B, C))
                if not_full.any():
                    bc = torch.nonzero(not_full, as_tuple=False)
                    b_ids, c_ids = bc[:, 0], bc[:, 1]
                    last_used[b_ids, c_ids, e_req[b_ids].to(torch.long)] = int(req_time)
                count_lru = count_lru + not_full.to(torch.int32)

                if need_evict.any():
                    lu_masked = last_used.clone()
                    lu_masked[~in_cache_lru] = INF_INT
                    evict_idx = lu_masked.argmin(dim=2)

                    bc = torch.nonzero(need_evict, as_tuple=False)
                    b_ids, c_ids = bc[:, 0], bc[:, 1]
                    ev = evict_idx[b_ids, c_ids].to(torch.long)
                    ne = e_req[b_ids].to(torch.long)

                    in_cache_lru[b_ids, c_ids, ev] = False
                    last_used[b_ids, c_ids, ev] = INF_INT
                    in_cache_lru[b_ids, c_ids, ne] = True
                    last_used[b_ids, c_ids, ne] = int(req_time)

            # ------------------------
            # LFU
            # ------------------------
            in_lfu = in_cache_lfu.gather(2, e_req[:, None, None].expand(B, C, 1)).squeeze(2)
            lfu_hits += (in_lfu & v_bc).sum(dim=0).to(torch.int64)

            hit = in_lfu & v_bc
            if hit.any():
                bc = torch.nonzero(hit, as_tuple=False)
                b_ids, c_ids = bc[:, 0], bc[:, 1]
                e_long = e_req[b_ids].to(torch.long)
                freq[b_ids, c_ids, e_long] += 1

            miss = (~in_lfu) & v_bc
            if miss.any():
                not_full = (count_lfu < cache_sizes_t[None, :]) & miss
                need_evict = (count_lfu >= cache_sizes_t[None, :]) & miss

                if not_full.any():
                    bc = torch.nonzero(not_full, as_tuple=False)
                    b_ids, c_ids = bc[:, 0], bc[:, 1]
                    ne = e_req[b_ids].to(torch.long)
                    in_cache_lfu[b_ids, c_ids, ne] = True
                    freq[b_ids, c_ids, ne] = 1
                    count_lfu = count_lfu + not_full.to(torch.int32)

                if need_evict.any():
                    freq_masked = freq.clone()
                    freq_masked[~in_cache_lfu] = INF_INT
                    evict_idx = freq_masked.argmin(dim=2)

                    bc = torch.nonzero(need_evict, as_tuple=False)
                    b_ids, c_ids = bc[:, 0], bc[:, 1]
                    ev = evict_idx[b_ids, c_ids].to(torch.long)
                    ne = e_req[b_ids].to(torch.long)

                    in_cache_lfu[b_ids, c_ids, ev] = False
                    freq[b_ids, c_ids, ev] = INF_INT
                    in_cache_lfu[b_ids, c_ids, ne] = True
                    freq[b_ids, c_ids, ne] = 1

            # ------------------------
            # LIFO
            # ------------------------
            in_lifo = in_cache_lifo.gather(2, e_req[:, None, None].expand(B, C, 1)).squeeze(2)
            lifo_hits += (in_lifo & v_bc).sum(dim=0).to(torch.int64)

            miss = (~in_lifo) & v_bc
            if miss.any():
                not_full = (count_lifo < cache_sizes_t[None, :]) & miss
                need_evict = (count_lifo >= cache_sizes_t[None, :]) & miss

                _masked_insert_only(in_cache_lifo, not_full, e_req[:, None].expand(B, C))
                if not_full.any():
                    bc = torch.nonzero(not_full, as_tuple=False)
                    b_ids, c_ids = bc[:, 0], bc[:, 1]
                    insert_time[b_ids, c_ids, e_req[b_ids].to(torch.long)] = int(req_time)
                count_lifo = count_lifo + not_full.to(torch.int32)

                if need_evict.any():
                    it_masked = insert_time.clone().to(torch.int64)
                    it_masked[~in_cache_lifo] = -10**18
                    evict_idx = it_masked.argmax(dim=2)

                    bc = torch.nonzero(need_evict, as_tuple=False)
                    b_ids, c_ids = bc[:, 0], bc[:, 1]
                    ev = evict_idx[b_ids, c_ids].to(torch.long)
                    ne = e_req[b_ids].to(torch.long)

                    in_cache_lifo[b_ids, c_ids, ev] = False
                    insert_time[b_ids, c_ids, ev] = NEG_INT
                    in_cache_lifo[b_ids, c_ids, ne] = True
                    insert_time[b_ids, c_ids, ne] = int(req_time)

            # ------------------------
            # Belady
            # ------------------------
            in_bd = in_cache_bd.gather(2, e_req[:, None, None].expand(B, C, 1)).squeeze(2)
            bd_hits += (in_bd & v_bc).sum(dim=0).to(torch.int64)

            miss = (~in_bd) & v_bc
            if miss.any():
                not_full = (count_bd < cache_sizes_t[None, :]) & miss
                need_evict = (count_bd >= cache_sizes_t[None, :]) & miss

                _masked_insert_only(in_cache_bd, not_full, e_req[:, None].expand(B, C))
                count_bd = count_bd + not_full.to(torch.int32)

                if need_evict.any():
                    dist_expand = d[:, None, :].expand(B, C, E)
                    masked_dist = dist_expand.masked_fill(~in_cache_bd, -1e30)
                    evict_idx = masked_dist.argmax(dim=2)
                    _masked_evict_and_insert(in_cache_bd, need_evict, evict_idx, e_req[:, None].expand(B, C))

    total_requests_f = total_requests.to(torch.float32).clamp_min(1.0)

    out = {}
    for i, cs in enumerate(cache_sizes):
        out[int(cs)] = {
            "LRU": float((lru_hits[i].float() / total_requests_f).item()),
            "LFU": float((lfu_hits[i].float() / total_requests_f).item()),
            "LIFO": float((lifo_hits[i].float() / total_requests_f).item()),
            "Belady": float((bd_hits[i].float() / total_requests_f).item()),
            ML_CECAR: (float((cecar_hits[i].float() / total_requests_f).item()) if model_cecar is not None else None),
            ML_FlashMoE: (float((flash_hits[i].float() / total_requests_f).item()) if model_flash is not None else None),
        }
    return out


def evaluate_model_multiple_caches(
    X_eval,
    model_cecar,
    model_flash,
    routing_weights_eval,
    dist_tensor_eval,
    top_k=8,
    num_samples=None,
    seq_len=None,
    device="cuda",
    use_gpu_eval=True,
    dedup_per_token=True,
):
    if use_gpu_eval:
        return evaluate_model_multiple_caches_gpu(
            X_eval=X_eval,
            model_cecar=model_cecar,
            model_flash=model_flash,
            routing_weights_eval=routing_weights_eval,
            dist_tensor_eval=dist_tensor_eval,
            top_k=top_k,
            num_samples=num_samples,
            seq_len=seq_len,
            cache_sizes=CACHE_SIZES,
            device=device,
            dedup_per_token=dedup_per_token,
        )

    raise RuntimeError("CPU eval path removed for cleanliness (GPU eval only). Use --cpu_eval=False.")


# ============================================================
# Training (selectable)
# ============================================================
def train_ffn(
    train_ffn_model: str,
    X,              
    y_short,        
    y_long,          
    hidden_dim=32,
    num_layers=3,
    epochs=30,
    batch_size=256,
    device="cuda",
    profile=False,
):
    """
    train_ffn_model:
      - ML_CECAR     -> trains MultiEvictionScorer on (y_short, y_long)
      - ML_FlashMoE  -> trains EvictionScorerMLCache on (y_long)  (single-head)
    """
    X = X.to(device)
    y_short = y_short.to(device)
    y_long = y_long.to(device)

    total_len, E, _ = X.shape

    # simple split
    eval_len = min(total_len // 10, 20000)
    eval_len = max(eval_len, min(2048, total_len // 5))
    eval_len = min(eval_len, total_len - 1)

    X_train, X_eval = X[:-eval_len], X[-eval_len:]
    yS_train, yS_eval = y_short[:-eval_len], y_short[-eval_len:]
    yL_train, yL_eval = y_long[:-eval_len], y_long[-eval_len:]

    if train_ffn_model == ML_CECAR:
        model = MultiEvictionScorer(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)

        best_loss = float("inf")
        best_state = None
        patience, wait = 5, 0

        with Timer(device=device, enabled=profile) as t_train:
            for epoch in range(epochs):
                model.train()
                perm = torch.randperm(X_train.size(0), device=device)
                Xb = X_train[perm]
                ySb = yS_train[perm]
                yLb = yL_train[perm]

                n = Xb.size(0)
                step_loss = 0.0
                for i in range(0, n, batch_size):
                    xb = Xb[i:i+batch_size]
                    ys = ySb[i:i+batch_size]
                    yl = yLb[i:i+batch_size]

                    pred_s, pred_l = model(xb)
                    loss = distance_regression_loss(pred_s, ys) + distance_regression_loss(pred_l, yl)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    step_loss += float(loss.item())

                step_loss /= max(1, (n + batch_size - 1) // batch_size)

                if step_loss < best_loss:
                    best_loss = step_loss
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        break

        if best_state is not None:
            model.load_state_dict(best_state, strict=True)
        model.eval()

        if profile:
            print(f"    [{ML_CECAR}] train_total={fmt_sec(t_train.dt)} best_loss={best_loss:.6f}")

        return model

    elif train_ffn_model == ML_FlashMoE:
        model = EvictionScorerMLCache(num_experts=E, hidden_dim=64, num_layers=3).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)

        best_loss = float("inf")
        best_state = None
        patience, wait = 5, 0

        with Timer(device=device, enabled=profile) as t_train:
            for epoch in range(epochs):
                model.train()
                perm = torch.randperm(X_train.size(0), device=device)
                Xb = X_train[perm]
                yb = yL_train[perm]   

                n = Xb.size(0)
                step_loss = 0.0
                for i in range(0, n, batch_size):
                    xb = Xb[i:i+batch_size]
                    yl = yb[i:i+batch_size]

                    pred = model(xb)  
                    loss = distance_regression_loss(pred, yl)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    step_loss += float(loss.item())

                step_loss /= max(1, (n + batch_size - 1) // batch_size)

                if step_loss < best_loss:
                    best_loss = step_loss
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        break

        if best_state is not None:
            model.load_state_dict(best_state, strict=True)
        model.eval()

        if profile:
            print(f"    [{ML_FlashMoE}] train_total={fmt_sec(t_train.dt)} best_loss={best_loss:.6f}")

        return model

    else:
        raise ValueError(f"Unknown train_ffn_model={train_ffn_model}")


# ============================================================
# Pipeline
# ============================================================
def run_pipeline(
    model_name="OLMoE_1B_7B_0125_Instruct",
    model_size="base",
    top_k=None,
    epochs=200,
    save_dir="Trained_FFN",
    eval_out_dir="Hitrate_Evaluation", 
    lfu_window_size=0,
    profile=True,
    device="cuda",
    use_gpu_eval=True,
    dedup_per_token=True,
    mode_eval=False, 
    train_only=False,
    CECAR_ckpt_dir=None,           
    FlashMoE_ckpt_dir=None,      
    train_ffn_model=ML_CECAR, 
    eval_model="both",       
):
    cfg = MODEL_CONFIGS[model_name]
    belady_topk = int(cfg["belady_topk"])
    train_dataset = cfg["train_dataset"]
    ood_datasets = cfg["ood_datasets"]

    if top_k is None:
        top_k = int(cfg["default_top_k"])

    size_cfg = MODEL_SIZE_CONFIGS[model_size]
    hidden_dim = int(size_cfg["hidden_dim"])
    num_layers = int(size_cfg["num_layers"])

    os.makedirs(save_dir, exist_ok=True)
    model_root = os.path.join(save_dir, model_name)
    cecar_dir  = os.path.join(model_root, ML_CECAR)
    flash_dir  = os.path.join(model_root, ML_FlashMoE)
    os.makedirs(cecar_dir, exist_ok=True)
    os.makedirs(flash_dir, exist_ok=True)
    if not train_only:
        eval_out_dir = os.path.join(eval_out_dir, model_name, eval_model)
        os.makedirs(eval_out_dir, exist_ok=True)
    else:
        eval_out_dir = None
    
    if mode_eval:
        if CECAR_ckpt_dir is None:
            CECAR_ckpt_dir = os.path.join(PRETRAIN_ROOT, model_name, ML_CECAR)
        if FlashMoE_ckpt_dir is None:
            FlashMoE_ckpt_dir = os.path.join(PRETRAIN_ROOT, model_name, ML_FlashMoE)
    else:
        if CECAR_ckpt_dir is None:
            CECAR_ckpt_dir = cecar_dir
        if FlashMoE_ckpt_dir is None:
            FlashMoE_ckpt_dir = flash_dir
        
    # --------------------------------------------
    # Load TRAIN routing (for per-layer training)
    # --------------------------------------------
    train_pt = os.path.join(ROUTING_DATA_DIR, model_name, f"{model_name}_{train_dataset}.pt")
    if not os.path.exists(train_pt):
        raise FileNotFoundError(f"Missing train PT: {train_pt}")

    with Timer(device=device, enabled=profile) as t_load_train:
        x_train = load_decode_routing_as_TBLE(train_pt, device=device)    
        x_train = x_train.permute(2, 1, 0, 3).contiguous()               
    if profile:
        print(f"[time] load TRAIN PT {os.path.basename(train_pt)} -> CUDA: {fmt_sec(t_load_train.dt)}")

    L, B_tr, T_tr, E = x_train.shape


    ood_data_list, ood_names, ood_filenames = [], [], []
    if train_only:
        if profile:
            print("[mode] train_only: skip OOD loading")
    else:
        # --------------------------------------------
        # Load OOD routing list
        # --------------------------------------------
        ood_paths = [os.path.join(ROUTING_DATA_DIR, model_name, f"{model_name}_{ds}.pt") for ds in ood_datasets]
        for p in ood_paths:
            if not os.path.exists(p):
                if profile:
                    print(f"[skip] missing OOD PT: {p}")
                continue
            with Timer(device=device, enabled=profile) as t_load_ood:
                od = load_decode_routing_as_TBLE(p, device=device)             
                od = od.permute(2, 1, 0, 3).contiguous()                       
            if profile:
                print(f"[time] load OOD PT {os.path.basename(p)} -> CUDA: {fmt_sec(t_load_ood.dt)}")
            ood_data_list.append(od)
            ood_names.append(os.path.splitext(os.path.basename(p))[0])

        if len(ood_data_list) == 0:
            raise FileNotFoundError("No OOD PTs found.")

        # --------------------------------------------
        # Output text files
        # --------------------------------------------
        ood_filenames = [name + "_Evaluation.txt" for name in ood_names]
        for txt in ood_filenames:
            path = os.path.join(eval_out_dir, txt)
            if os.path.exists(path):
                os.remove(path)
            with open(path, "w") as f:
                f.write(f"Layer-wise Cache Hit Rates ({txt.replace('_Evaluation.txt','')})\n\n")

    layer_range = range(L)
    if model_name in ["DeepSeek_v2_Lite_Chat"]:
        layer_range = range(1, L)

    # Belady target horizons for ML_CECAR (short/long)
    belady_topk_short = max(8, belady_topk // 2)
    belady_topk_long = belady_topk

    for l in tqdm(layer_range, desc="Layers"):
        layer_total_t0 = time.perf_counter()


        # --------------------------------------------
        # Prepare TRAIN features/targets for this layer
        # --------------------------------------------
        with Timer(device=device, enabled=profile) as t_prep_train:
            layer_tr = x_train[l].float()                  
            layer_tr = F.softmax(layer_tr, dim=-1)

            seq_tr = get_topk_sequence(layer_tr, top_k)
            dist_tr = build_belady_distance(seq_tr, num_experts=E)        

            y_long_bt = belady_target_from_dist(dist_tr, belady_topk_long)
            y_short_bt = belady_target_from_dist(dist_tr, belady_topk_short)

            lru_bt, lfu_bt = build_lru_lfu_scores(seq_tr, num_experts=E, window_size=lfu_window_size)
            lru_bt = 1.0 / lru_bt + 1e-6
            lfu_bt = lfu_bt / (lfu_bt.max(dim=2, keepdim=True).values + 1e-6)

            X_bt = torch.stack([lru_bt, lfu_bt], dim=-1)    

            X_layer = X_bt.view(-1, E, 2).contiguous()
            yS_layer = y_short_bt.view(-1, E).contiguous()
            yL_layer = y_long_bt.view(-1, E).contiguous()

        # --------------------------------------------
        # Train or load model(s) for this layer
        # --------------------------------------------
        model_cecar = None
        model_flash = None

        cecar_ckpt = os.path.join(CECAR_ckpt_dir, f"{model_size}_layer{l}.pt")
        flash_ckpt = os.path.join(FlashMoE_ckpt_dir, f"{model_size}_layer{l}.pt")

        if mode_eval:
            if eval_model in ["both", ML_CECAR]:
                if os.path.exists(cecar_ckpt):
                    model_cecar = MultiEvictionScorer(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
                    state = torch.load(cecar_ckpt, map_location="cpu")
                    model_cecar.load_state_dict(state, strict=True)
                    model_cecar.eval()
                else:
                    if profile:
                        print(f"[warn] missing {ML_CECAR} ckpt: {cecar_ckpt}")

            if eval_model in ["both", ML_FlashMoE]:
                if os.path.exists(flash_ckpt):
                    model_flash = EvictionScorerMLCache(num_experts=E, hidden_dim=64, num_layers=3).to(device)
                    state = torch.load(flash_ckpt, map_location="cpu")
                    model_flash.load_state_dict(state, strict=True)
                    model_flash.eval()
                else:
                    if profile:
                        print(f"[warn] missing {ML_FlashMoE} ckpt: {flash_ckpt}")

        else:
            if train_ffn_model == ML_CECAR:
                model_cecar = train_ffn(
                    train_ffn_model=ML_CECAR,
                    X=X_layer,
                    y_short=yS_layer,
                    y_long=yL_layer,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    epochs=epochs,
                    batch_size=256,
                    device=device,
                    profile=profile,
                )
                torch.save({k: v.detach().cpu() for k, v in model_cecar.state_dict().items()}, cecar_ckpt)

                if eval_model == "both":
                    if os.path.exists(flash_ckpt):
                        model_flash = EvictionScorerMLCache(num_experts=E, hidden_dim=64, num_layers=3).to(device)
                        state = torch.load(flash_ckpt, map_location="cpu")
                        model_flash.load_state_dict(state, strict=True)
                        model_flash.eval()

            elif train_ffn_model == ML_FlashMoE:
                model_flash = train_ffn(
                    train_ffn_model=ML_FlashMoE,
                    X=X_layer,
                    y_short=yS_layer,  
                    y_long=yL_layer,
                    hidden_dim=64,
                    num_layers=3,
                    epochs=epochs,
                    batch_size=256,
                    device=device,
                    profile=profile,
                )
                torch.save({k: v.detach().cpu() for k, v in model_flash.state_dict().items()}, flash_ckpt)

                if eval_model == "both":
                    if os.path.exists(cecar_ckpt):
                        model_cecar = MultiEvictionScorer(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
                        state = torch.load(cecar_ckpt, map_location="cpu")
                        model_cecar.load_state_dict(state, strict=True)
                        model_cecar.eval()
            else:
                raise ValueError(f"train_ffn_model must be {ML_CECAR} or {ML_FlashMoE}. Got {train_ffn_model}")

        if eval_model == ML_CECAR:
            model_flash = None
        elif eval_model == ML_FlashMoE:
            model_cecar = None

        if train_only:
            if profile:
                print(f"[train-only] layer={l} done (skip eval)")
            layer_total = time.perf_counter() - layer_total_t0
            if profile:
                print(f"[layer done] layer={l} total={fmt_sec(layer_total)}")
            continue
        # --------------------------------------------
        # Evaluate on each OOD dataset for this layer
        # --------------------------------------------
        for od, name, out_txt in zip(ood_data_list, ood_names, ood_filenames):
            with Timer(device=device, enabled=profile) as t_prep_ood:
                layer_od = od[l].float()                     
                layer_od = F.softmax(layer_od, dim=-1)

                seq_od = get_topk_sequence(layer_od, top_k)
                dist_od = build_belady_distance(seq_od, num_experts=E)    

                B_od, T_od, _ = layer_od.shape
                routing_flat = layer_od.view(-1, E).contiguous()
                dist_flat = dist_od.view(-1, E).contiguous()

                lru_od, lfu_od = build_lru_lfu_scores(seq_od, num_experts=E, window_size=lfu_window_size)
                lru_od = 1.0 / lru_od + 1e-6
                lfu_od = lfu_od / (lfu_od.max(dim=2, keepdim=True).values + 1e-6)
                X_od = torch.stack([lru_od, lfu_od], dim=-1).view(-1, E, 2).contiguous()

            if profile:
                print(f"[time] prep OOD {name} layer{l}: {fmt_sec(t_prep_ood.dt)}")

            with Timer(device=device, enabled=profile) as t_eval:
                results = evaluate_model_multiple_caches(
                    X_eval=X_od,
                    model_cecar=model_cecar,
                    model_flash=model_flash,
                    routing_weights_eval=routing_flat,
                    dist_tensor_eval=dist_flat,
                    top_k=top_k,
                    num_samples=B_od,
                    seq_len=T_od,
                    device=device,
                    use_gpu_eval=use_gpu_eval,
                    dedup_per_token=dedup_per_token,
                )

            if profile:
                print(f"[time] eval OOD {name} layer{l}: {fmt_sec(t_eval.dt)}")

            line = f"Layer {l:02d} | \n"
            for cs in CACHE_SIZES:
                r = results[int(cs)]
                line += (f"C{cs:02d} "
                         f"LRU={r['LRU']:.4f} "
                         f"LFU={r['LFU']:.4f} "
                         f"LIFO={r['LIFO']:.4f} "
                         f"BD={r['Belady']:.4f} ")
                if r.get(ML_CECAR) is not None:
                    line += f"{ML_CECAR}={r[ML_CECAR]:.4f} "
                if r.get(ML_FlashMoE) is not None:
                    line += f"{ML_FlashMoE}={r[ML_FlashMoE]:.4f} "
                line += "| \n"

            with open(os.path.join(eval_out_dir, out_txt), "a") as f:
                f.write(line + "\n")

        layer_total = time.perf_counter() - layer_total_t0
        if profile:
            print(f"[layer done] layer={l} total={fmt_sec(layer_total)}")

    if profile:
        print("[done] All layers evaluated.")
    return True


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="OLMoE_1B_7B_0125_Instruct", choices=list(MODEL_CONFIGS.keys()))
    p.add_argument("--model_size", type=str, default="base", choices=list(MODEL_SIZE_CONFIGS.keys()))
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--save_dir", type=str, default="Trained_FFN")
    p.add_argument("--eval_out_dir", type=str, default="Hitrate_Evaluation")
    p.add_argument("--lfu_window_size", type=int, default=0)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--cpu_eval", action="store_true")  
    p.add_argument("--no_dedup", action="store_true")

    # eval-only & ckpt dirs
    p.add_argument("--mode", type=str, default="train_eval", choices=["train_eval", "train_only", "eval_only"],
                   help="train_eval: train+eval, train_only: train only, eval_only: eval only (load ckpts)")
    p.add_argument("--CECAR_ckpt_dir", type=str, default=None,
                   help="Default (mode_eval): <script_dir>/Pre_trained_FFN/{model_name}/ML_CECAR")
    p.add_argument("--FlashMoE_ckpt_dir", type=str, default=None,
                   help="Default (mode_eval): <script_dir>/Pre_trained_FFN/{model_name}/ML_FlashMoE")


    # selectable train/eval
    p.add_argument("--train_ffn_model", type=str, default=ML_CECAR, choices=[ML_CECAR, ML_FlashMoE])
    p.add_argument("--eval_model", type=str, default="both", choices=["both", ML_CECAR, ML_FlashMoE])

    return p.parse_args()


def main():
    args = parse_args()
    mode_eval = (args.mode == "eval_only")
    train_only = (args.mode == "train_only")
    run_pipeline(
        model_name=args.model_name,
        model_size=args.model_size,
        top_k=args.top_k,
        epochs=args.epochs,
        save_dir=args.save_dir,
        eval_out_dir=args.eval_out_dir,
        lfu_window_size=args.lfu_window_size,
        profile=args.profile,
        device=args.device,
        use_gpu_eval=(not args.cpu_eval),
        dedup_per_token=(not args.no_dedup),
        mode_eval=mode_eval,
        train_only=train_only,
        CECAR_ckpt_dir=args.CECAR_ckpt_dir,
        FlashMoE_ckpt_dir=args.FlashMoE_ckpt_dir,
        train_ffn_model=args.train_ffn_model,
        eval_model=args.eval_model,
    )


if __name__ == "__main__":
    main()



