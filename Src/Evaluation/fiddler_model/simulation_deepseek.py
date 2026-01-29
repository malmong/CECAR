"""
SimulationDeepseek (All-GPU MoE + VirtualCache) — Clean / Paper+GitHub reference

Requirements (as per user request):
- Keep ODP/DES logic as in this Option1 (already code2-style DES/ODP).
- Do NOT add per-step prints.
- Align everything else to the "second code" semantics:
  (1) self_attn return parsing + KV cache update semantics
  (2) MoE execution semantics (decode filter) + cache stats counting
  (3) get_cache_stats / remove_experts accumulation semantics
  (4) LocalMoEGate topk behavior (sorted=True, greedy only)
- Naming:
  - "caer" -> "cecar"
  - "caer_const" -> "constant"
  - Decode filter triggers on bonus_strategy in {"cecar","constant"}.

Notes:
- LocalMoEGate assumes batch_size=1 (as in your original).
- SimulationDeepseek forward path supports batch_size>=1 for expert execution,
  but gating remains batch_size=1 safe-guard.
"""

from __future__ import annotations

import os
import json
import time
import warnings
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers.cache_utils import DynamicCache

from .utils_cache import VirtualCache


# -----------------------------------------------------------------------------
# Logging / warnings
# -----------------------------------------------------------------------------
transformers.utils.logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=r"The module name .* is not a valid Python identifier.*")
warnings.filterwarnings("ignore", message=r"Some weights of .* were not initialized from the model checkpoint.*")
warnings.filterwarnings("ignore", message=r"`torch_dtype` is deprecated! Use `dtype` instead!")


# -----------------------------------------------------------------------------
# DeepSeek remote-code compatibility patch
# -----------------------------------------------------------------------------
if not hasattr(DynamicCache, "get_usable_length"):
    def _get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)
    DynamicCache.get_usable_length = _get_usable_length


# -----------------------------------------------------------------------------
# Attention mask helper (2D attn -> 4D additive causal mask)
# -----------------------------------------------------------------------------
def build_4d_additive_causal_mask(
    attn2d: torch.Tensor,
    q_len: int,
    kv_len: int,
    device,
    dtype,
) -> torch.Tensor:
    """
    attn2d: (B, kv_len) with 1 for valid tokens, 0 for pads
    returns: (B, 1, q_len, kv_len) additive mask with {0, -inf}
    """
    bsz = attn2d.size(0)
    neg_inf = torch.finfo(dtype).min

    if q_len == 1:
        causal = torch.ones((1, kv_len), device=device, dtype=torch.bool)
    else:
        causal = torch.tril(torch.ones((q_len, kv_len), device=device, dtype=torch.bool), diagonal=0)

    key_ok = attn2d[:, None, None, :kv_len].to(torch.bool)
    allow = causal[None, None, :, :] & key_ok

    mask = torch.zeros((bsz, 1, q_len, kv_len), device=device, dtype=dtype)
    mask = mask.masked_fill(~allow, neg_inf)
    return mask


# =============================================================================
# Local MoE Gate (bonus on logits, decode-only)  [batch_size=1 assumed]
# - Aligned to second-code behavior:
#   * topk_method == "greedy" only
#   * topk on SELECT uses sorted=True
# =============================================================================
class LocalMoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.top_k = int(getattr(config, "num_experts_per_tok", 1))
        self.n_routed_experts = int(getattr(config, "n_routed_experts", 0))
        self.routed_scaling_factor = float(getattr(config, "routed_scaling_factor", 1.0))
        self.scoring_func = getattr(config, "scoring_func", "softmax")
        self.norm_topk_prob = bool(getattr(config, "norm_topk_prob", False))

        self.topk_method = getattr(config, "topk_method", "greedy")
        self.n_group = int(getattr(config, "n_group", 1))
        self.topk_group = int(getattr(config, "topk_group", 1))
        self.gating_dim = int(getattr(config, "hidden_size", 0))

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

        # Runtime context injected by SimulationDeepseek.deepseek_forward()
        self._ctx_layer_id = None
        self._ctx_is_prefill = True
        self._ctx_cache = None
        self._ctx_bonus_strategy = "none"  
        self._ctx_lambda_cache = 0.0
        self._ctx_top_J = 1
        self._ctx_cache_delta_tracker = None
        self._ctx_stats = None
        self._ctx_lru_counter = None  
        self._ctx_lfu_counter = None  

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor):
        """
        hidden_states: (B, S, H)
        returns:
          topk_idx:    (T, K) where T=B*S (here B==1)
          topk_weight: (T, K) RAW-gathered + normalized (second-code style)
          aux: None
        """
        bsz, seq_len, h = hidden_states.shape
        assert bsz == 1, "LocalMoEGate assumes batch_size=1."

        x2d = hidden_states.view(-1, h)  

        logits = F.linear(
            x2d.to(torch.float32),
            self.weight.to(torch.float32),
            None
        ) 

        if self.scoring_func != "softmax":
            raise NotImplementedError(f"Unsupported scoring_func: {self.scoring_func}")

        logits_select = logits.clone()

        layer_id = self._ctx_layer_id
        is_prefill = self._ctx_is_prefill
        cache_obj = self._ctx_cache
        bs = self._ctx_bonus_strategy
        stats = self._ctx_stats

        # -------------------------
        # Decode-only bonus (logit-level)
        # -------------------------
        if (cache_obj is not None) and (layer_id is not None) and (not is_prefill) and (bs != "none"):
            b = 0
            E = int(logits.shape[-1])
            token_row = seq_len - 1  

            z = logits[token_row:token_row + 1] 

            cached = cache_obj.get_cached_expert_ids(b, layer_id) or []
            cached = [int(ce) for ce in cached if 0 <= int(ce) < E]

            if bs == "mocce":
                tracker_list = self._ctx_cache_delta_tracker
                if tracker_list is None:
                    raise RuntimeError("cache_delta_tracker not injected for mocce.")
                tracker = tracker_list[layer_id]

                cur_range = (z.max() - z.min()).item()
                if not tracker["init"]:
                    tracker["avg"] = cur_range
                    tracker["init"] = True
                else:
                    tracker["avg"] = 0.99 * tracker["avg"] + 0.01 * cur_range
                delta_avg = tracker["avg"]

                mask = torch.zeros_like(z)  
                topJ = min(int(self._ctx_top_J), E)
                _, topJ_idx = torch.topk(z, k=topJ, dim=1)
                mask.scatter_(1, topJ_idx, 1.0)
                if cached:
                    mask[0, cached] = 1.0

                lam = float(self._ctx_lambda_cache)
                logits_select[token_row:token_row + 1] = z + lam * delta_avg * mask

            elif bs in ("cecar", "constant"):
                k_gap = min(int(self.top_k), E)
                topk_logits, _ = torch.topk(z.view(-1), k=k_gap)
                delta = topk_logits[0].item() - topk_logits[-1].item()

                if cached:
                    scores = [cache_obj.get_eviction_score(b, layer_id, ce) for ce in cached]
                    max_score = max(scores) if scores else 1.0
                    for ce, score in zip(cached, scores):
                        if bs == "constant":
                            alpha = 0.4858016637499412
                        else:
                            alpha = 1.0 - float(score) / (float(max_score) + 1e-5)
                        if stats is not None and hasattr(stats, "alpha_records"):
                            stats.alpha_records[b].append(float(alpha))
                        logits_select[token_row, ce] += delta * alpha

            elif bs in ("random", "lru", "lfu"):
                if cached:
                    z_cache = z[0, cached]
                    rw_mean = z_cache.mean().item()
                    rw_std = z_cache.std(unbiased=False).item()
                    cv_cache = rw_std / (rw_mean + 1e-5)
                    scale = 1.0 / (1.0 + cv_cache)

                    if stats is not None and hasattr(stats, "scale_records"):
                        stats.scale_records[b].append(float(scale))

                    k_gap = min(int(self.top_k), E)
                    topk_logits, _ = torch.topk(z.view(-1), k=k_gap)
                    delta = topk_logits[0].item() - topk_logits[-1].item()

                    for ce in cached:
                        if bs == "random":
                            alpha = float(torch.rand((), device=z.device).item())
                        elif bs == "lru":
                            ctr = self._ctx_lru_counter
                            if ctr is None:
                                raise RuntimeError("lru_counter not injected.")
                            lru_val = float(ctr[b][layer_id][ce])
                            max_lru = float(max(ctr[b][layer_id])) if max(ctr[b][layer_id]) > 0 else 1.0
                            alpha = lru_val / max_lru
                        elif bs == "lfu":
                            ctr = self._ctx_lfu_counter
                            if ctr is None:
                                raise RuntimeError("lfu_counter not injected.")
                            lfu_val = float(ctr[b][layer_id][ce])
                            max_lfu = float(max(ctr[b][layer_id])) if max(ctr[b][layer_id]) > 0 else 1.0
                            alpha = lfu_val / max_lfu
                        else:
                            alpha = 0.0

                        if stats is not None and hasattr(stats, "alpha_records"):
                            stats.alpha_records[b].append(float(alpha))

                        logits_select[token_row, ce] += float(delta) * float(scale) * float(alpha)

        scores_raw = logits.softmax(dim=-1, dtype=torch.float32)
        scores_sel = logits_select.softmax(dim=-1, dtype=torch.float32)

        # top-k on SELECT (greedy + sorted=True)
        if str(self.topk_method).lower() == "greedy":
            _, topk_idx = torch.topk(scores_sel, k=self.top_k, dim=-1, sorted=True)
        else:
            raise NotImplementedError(f"Unknown topk_method: {self.topk_method}")

        # gather RAW weights + normalize (second-code style)
        topk_weight = torch.gather(scores_raw, 1, topk_idx)
        if self.top_k > 1 and self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        else:
            topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight, None


def replace_moe_gates_with_local(base_model):
    layers = getattr(base_model, "layers", [])
    if not layers:
        raise RuntimeError("base_model.layers not found or empty; cannot replace gates.")

    cfg = getattr(base_model, "config", None)
    if cfg is None:
        raise RuntimeError("base_model.config not found; cannot construct LocalMoEGate.")

    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        old_gate = getattr(mlp, "gate", None)
        if old_gate is None:
            continue
        if not hasattr(old_gate, "weight"):
            continue

        new_gate = LocalMoEGate(cfg)
        dev = old_gate.weight.device
        new_gate.to(dev)

        with torch.no_grad():
            if new_gate.weight.shape != old_gate.weight.shape:
                raise RuntimeError(
                    f"Gate weight shape mismatch: new={tuple(new_gate.weight.shape)} "
                    f"old={tuple(old_gate.weight.shape)}"
                )
            new_gate.weight.copy_(old_gate.weight)

        mlp.gate = new_gate


# =============================================================================
# Stats
# =============================================================================
class SimulationDeepseekStats:
    def __init__(self, n_layer: int, batch_size: int = 1):
        self.n_layer = int(n_layer)
        self.batch_size = int(batch_size)
        self.reset()

    def reset(self):
        self.prefill_time = 0.0
        self.decode_time = 0.0
        self.prefill_tokens = 0
        self.decode_tokens = 0

        self.cnt_expert_hit_by_layer = [[0] * self.n_layer for _ in range(self.batch_size)]
        self.cnt_expert_all_by_layer = [[0] * self.n_layer for _ in range(self.batch_size)]

        self.scale_records = [[] for _ in range(self.batch_size)]
        self.alpha_records = [[] for _ in range(self.batch_size)]

    @property
    def prefill_speed(self):
        return self.prefill_tokens / self.prefill_time if self.prefill_time > 0 else 0.0

    @property
    def decode_speed(self):
        return self.decode_tokens / self.decode_time if self.decode_time > 0 else 0.0

    def summary(self):
        all_scales = [s for batch_scales in self.scale_records for s in batch_scales]
        all_alphas = [a for batch_alphas in self.alpha_records for a in batch_alphas]
        return {
            "prefill_time": self.prefill_time,
            "decode_time": self.decode_time,
            "prefill_speed": self.prefill_speed,
            "decode_speed": self.decode_speed,
            "avg_scale": sum(all_scales) / len(all_scales) if all_scales else 0.0,
            "avg_alpha": sum(all_alphas) / len(all_alphas) if all_alphas else 0.0,
        }


# =============================================================================
# SimulationDeepseek
# =============================================================================
class SimulationDeepseek:
    """
    SimulationDeepseek:
    - ODP/DES block kept in code2-style as provided.
    - Everything else aligned to second-code semantics.
    """

    def __init__(self, args):
        self.args = args
        self.dtype = torch.bfloat16

        non_expert_model = getattr(args, "non_expert_model", None)
        device_map = getattr(args, "device_map", "auto")

        kwargs = dict(device_map=device_map, trust_remote_code=True, use_cache=True)
        try:
            full_model = transformers.AutoModelForCausalLM.from_pretrained(
                non_expert_model, dtype=self.dtype, **kwargs
            )
        except TypeError:
            full_model = transformers.AutoModelForCausalLM.from_pretrained(
                non_expert_model, torch_dtype=self.dtype, **kwargs
            )

        self.lm_head = getattr(full_model, "lm_head", None)
        self.model = getattr(full_model, "model", None)
        if self.model is None or self.lm_head is None:
            raise RuntimeError("DeepSeek model/lm_head not found from AutoModelForCausalLM.")

        self.dev = next(self.model.parameters()).device
        self.dense_first_layer = True

        # Replace gates with LocalMoEGate
        replace_moe_gates_with_local(self.model)

        # Tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(non_expert_model, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id

        # Paths
        self.expert_path = getattr(args, "expert_path", None)
        self.shared_expert_path = getattr(args, "shared_expert_path", None)

        # Model config
        self.n_layer = len(self.model.layers)
        self.hidden_dim = int(self.model.config.hidden_size)
        self.n_routing_expert = int(getattr(self.model.config, "num_experts_per_tok", 0))
        self.n_shared_experts = int(getattr(self.model.config, "n_shared_experts", 0) or 0)
        self.n_expert = self._infer_num_experts_from_gates()

        # Cache params
        self.cache_size = int(getattr(args, "cache_size", 12))
        self.cache_policy = getattr(args, "cache_policy", "lru")
        self.batch_size = int(getattr(args, "batch_size", 1))

        # LRU/LFU counters
        self.lru_counter = None
        self.lfu_counter = None
        self.global_step = None

        # Unified args
        self.mode = getattr(args, "mode", "none")
        self.bonus_strategy = getattr(args, "bonus_strategy", "none")
        if self.mode != "none":
            self.bonus_strategy = "none"

        # DES/ODP flags
        self.des_enabled = bool(getattr(args, "enable_des", False))
        self.protection_enabled = bool(getattr(args, "enable_protection", False))
        self.protection_top_ratio = float(getattr(args, "protection_top_ratio", 0.02))
        self.mcmoe_threshold_path = getattr(args, "mcmoe_threshold_path", None)

        # MOCCE / bonus params (second-code defaults)
        self.lambda_cache = float(getattr(args, "lambda_cache", 0.2))
        self.top_J = int(getattr(args, "top_J", 2))
        self.cache_delta_tracker = [{"init": False, "avg": 0.0} for _ in range(self.n_layer)]

        # Decode-filter params (second-code defaults)
        self.compute_k = int(getattr(args, "compute_k", self.n_routing_expert))
        self.topk_threshold = int(getattr(args, "topk_threshold", 3))

        # DES thresholds (layerwise)
        self.des_mu_4_5_by_layer: Dict[int, float] = {}
        self.des_mu_5_6_by_layer: Dict[int, float] = {}
        
        if self.des_enabled:
            self._init_des_thresholds()

        # Stats/cache
        self.stats = SimulationDeepseekStats(self.n_layer, self.batch_size)
        self.virtual_cache = None
        self.past_key_values = DynamicCache()
        self.ffn_model_path = getattr(args, "ffn_model_path", None)

        # Accumulated stats (second-code style)
        self._accumulated_total_count = 0
        self._accumulated_hit_count = 0
        self._accumulated_total_count_by_layer = [0] * self.n_layer
        self._accumulated_hit_count_by_layer   = [0] * self.n_layer

        # Preload experts
        self.all_experts = self._preload_all_experts()
        self.shared_expert_modules = self._preload_shared_experts()

    # ---------------------------------------------------------------------
    # DES threshold loader (schema compatible)
    # ---------------------------------------------------------------------
    @staticmethod
    def _load_thresholds_layerwise(threshold_path: str, *, strict_method: bool = True):
        if not threshold_path or (not os.path.exists(threshold_path)):
            raise ValueError(f"threshold_path not found: {threshold_path}")

        with open(threshold_path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if not isinstance(obj, dict):
            raise ValueError("threshold file must be a JSON object.")

        if strict_method:
            method = obj.get("method", None)
            if method != "des_top4_6_layerwise":
                raise ValueError(
                    f"threshold file method mismatch: expected 'des_top4_6_layerwise', got {method!r}"
                )

        min_k = obj.get("min_k", None)
        max_k = obj.get("max_k", None)
        if int(min_k) != 4 or int(max_k) != 6:
            raise ValueError(f"threshold file must have min_k=4 and max_k=6, got min_k={min_k}, max_k={max_k}")

        layers = obj.get("layers", None)
        if not isinstance(layers, list) or len(layers) == 0:
            raise ValueError("threshold file must contain a non-empty 'layers' list.")

        mu1, mu2, seen = {}, {}, set()
        for it in layers:
            if not isinstance(it, dict):
                raise ValueError("each element in 'layers' must be an object.")
            if "layer" not in it or "mu_4_5" not in it or "mu_5_6" not in it:
                raise ValueError(f"layer entry must contain 'layer','mu_4_5','mu_5_6'. got keys={list(it.keys())}")

            lid = int(it["layer"])
            if lid in seen:
                raise ValueError(f"duplicate layer id in threshold file: layer={lid}")
            seen.add(lid)

            mu1[lid] = float(it["mu_4_5"])
            mu2[lid] = float(it["mu_5_6"])

        return mu1, mu2

    def _init_des_thresholds(self):
        # force gate K=6 for DES
        self.n_routing_expert = 6
        try:
            self.model.config.num_experts_per_tok = 6
        except Exception:
            pass

        for i_layer, layer in enumerate(self.model.layers):
            if self._layer_is_dense0(i_layer) or (not self._layer_has_moe_gate(layer)):
                continue
            gate = layer.mlp.gate
            if isinstance(gate, LocalMoEGate):
                gate.top_k = 6

        base_dir = self.mcmoe_threshold_path
        if not base_dir or (not os.path.isdir(base_dir)):
            raise ValueError(f"mcmoe_threshold_path must be a directory, got: {base_dir}")

        task_raw = getattr(self.args, "tasks", None)
        task = (task_raw or "").strip().lower().replace("-", "_")
        task_to_stem = {"humaneval": "mmlu_ccsc", "mbpp": "mmlu_ccsc", "gpqa": "arc_challenge", "math500": "mathqa"}
        stem = task_to_stem.get(task, None)
        if stem is None:
            raise ValueError(f"Unsupported task for DES threshold selection: {task_raw!r}")

        threshold_file = f"DeepSeek_v2_Lite_Chat_{stem}_top4_6_thresholds.json"
        threshold_path = os.path.join(base_dir, threshold_file)
        if not os.path.exists(threshold_path):
            raise ValueError(
                f"DES threshold file not found for task={task_raw!r}:\n"
                f"  expected: {threshold_path}"
            )

        mu1, mu2 = self._load_thresholds_layerwise(threshold_path, strict_method=True)
        self.des_mu_4_5_by_layer = mu1
        self.des_mu_5_6_by_layer = mu2
        print(f"[DES] Loaded thresholds for task={task_raw} from {os.path.abspath(threshold_path)}")

    # ---------------------------------------------------------------------
    # ODP/DES helpers (UNCHANGED)
    # ---------------------------------------------------------------------
    def _extract_attn_weights(self, attn_out):
        if isinstance(attn_out, (tuple, list)):
            for x in attn_out:
                if torch.is_tensor(x) and x.dim() in (3, 4):
                    return x
            return None
        if isinstance(attn_out, dict):
            for k in ["attn_weights", "attentions", "attention_probs", "attn_probs"]:
                v = attn_out.get(k, None)
                if torch.is_tensor(v):
                    return v
            return None
        return None

    def _compute_protection_mask_from_attn(self, t_states: torch.Tensor, attn_weights: torch.Tensor, top_ratio: float):
        if t_states.dim() != 3:
            return torch.zeros((t_states.shape[0], t_states.shape[1]), device=t_states.device, dtype=torch.bool)

        B, L, _ = t_states.shape
        if attn_weights is None or (not torch.is_tensor(attn_weights)):
            return torch.zeros((B, L), device=t_states.device, dtype=torch.bool)

        A = attn_weights
        if A.dim() == 3:
            if A.shape[0] == B:
                A = A.unsqueeze(1)
            else:
                A = A.unsqueeze(0)
        elif A.dim() != 4:
            return torch.zeros((B, L), device=t_states.device, dtype=torch.bool)

        Q = int(A.shape[-2])
        K = int(A.shape[-1])
        if Q != L:
            return torch.zeros((B, L), device=t_states.device, dtype=torch.bool)

        K_use = min(K, L)
        A = A[..., :K_use]
        A_mean = A.float().mean(dim=1) 

        tril = torch.tril(torch.ones((L, K_use), device=A_mean.device, dtype=A_mean.dtype), diagonal=0)
        past_sum = (A_mean * tril).sum(dim=2) 

        denom = (torch.arange(L, device=A_mean.device) + 1).clamp(min=1).to(A_mean.dtype)
        past_avg = past_sum / denom.unsqueeze(0) 

        l1 = t_states.float().abs().sum(dim=-1) 
        importance = l1 * past_avg

        k = max(1, int(math.ceil(L * float(top_ratio))))
        top_idx = torch.topk(importance, k=k, dim=1, largest=True).indices

        mask = torch.zeros((B, L), device=t_states.device, dtype=torch.bool)
        mask.scatter_(1, top_idx, True)
        return mask

    # ---------------------------------------------------------------------
    # Model helpers
    # ---------------------------------------------------------------------
    def _layer_is_dense0(self, layer_id: int) -> bool:
        return bool(self.dense_first_layer) and int(layer_id) == 0

    def _layer_has_moe_gate(self, layer) -> bool:
        return hasattr(layer, "mlp") and hasattr(layer.mlp, "gate")

    def _infer_num_experts_from_gates(self) -> int:
        for i_layer, layer in enumerate(self.model.layers):
            if self._layer_is_dense0(i_layer):
                continue
            if not self._layer_has_moe_gate(layer):
                continue
            gate = layer.mlp.gate
            if hasattr(gate, "weight") and gate.weight is not None:
                return int(gate.weight.shape[0])
        return int(getattr(self.model.config, "n_routed_experts", 0))

    # ---------------------------------------------------------------------
    # Expert preload (second-code style: warn on missing)
    # ---------------------------------------------------------------------
    def _preload_all_experts(self):
        experts: Dict[int, Dict[int, object]] = {}
        if not self.expert_path:
            print("Warning: expert_path is empty; no routed experts will be loaded.")
            return experts

        print("Pre-loading routed experts for gate-available layers...")
        loaded_layers = 0
        for i_layer, layer in enumerate(self.model.layers):
            if self._layer_is_dense0(i_layer) or (not self._layer_has_moe_gate(layer)):
                continue

            experts[i_layer] = {}
            for i_expert in range(self.n_expert):
                path = f"{self.expert_path}/layer{i_layer}_expert{i_expert}.pt"
                try:
                    experts[i_layer][i_expert] = torch.load(path, map_location=self.dev, weights_only=False)
                except FileNotFoundError:
                    print(f"Warning: Expert not found at {path}")
            loaded_layers += 1

        print(f"Loaded routed experts for {loaded_layers} layers (gate-present).")
        return experts

    def _preload_shared_experts(self):
        if (not self.shared_expert_path) or (not self.n_shared_experts):
            return {}
        print("Pre-loading shared experts for gate-available layers...")
        shared: Dict[int, object] = {}
        loaded = 0
        for i_layer, layer in enumerate(self.model.layers):
            if self._layer_is_dense0(i_layer) or (not self._layer_has_moe_gate(layer)):
                continue
            path = f"{self.shared_expert_path}/layer{i_layer}_shared_expert.pt"
            try:
                shared[i_layer] = torch.load(path, map_location=self.dev, weights_only=False)
                loaded += 1
            except FileNotFoundError:
                print(f"Warning: Shared expert not found at {path}")
        print(f"Loaded shared experts for {loaded} layers.")
        return shared

    # ---------------------------------------------------------------------
    # Generation state
    # ---------------------------------------------------------------------
    def _init_generation_state(self, batch_size: int):
        self.batch_size = int(batch_size)

        self.virtual_cache = VirtualCache(
            n_layers=self.n_layer,
            num_experts=self.n_expert,
            cache_size=self.cache_size,
            cache_policy=self.cache_policy,
            batch_size=self.batch_size,
            ffn_model_path=self.ffn_model_path,
            dense_first_layer=self.dense_first_layer,
            model="DeepSeek_v2_Lite_Chat",
        )

        # tracker resets per generation (mocce)
        self.cache_delta_tracker = [{"init": False, "avg": 0.0} for _ in range(self.n_layer)]

        # KV cache
        self.past_key_values = DynamicCache()

        # stats per generation
        self.stats = SimulationDeepseekStats(self.n_layer, self.batch_size)

        # LRU/LFU per batch (same pattern as your other sims)
        self.lru_counter = [[[0] * self.n_expert for _ in range(self.n_layer)] for _ in range(self.batch_size)]
        self.lfu_counter = [[[0] * self.n_expert for _ in range(self.n_layer)] for _ in range(self.batch_size)]
        self.global_step = [0] * self.batch_size

    # ---------------------------------------------------------------------
    # Generate (aligned to second code; no per-step print)
    # ---------------------------------------------------------------------
    def generate(self, input_ids, attention_mask=None, max_length=2048, max_new_tokens=None, **kwargs):
        input_ids = input_ids.to(self.dev)
        batch_size = int(input_ids.shape[0])
        input_length = int(input_ids.shape[1])

        if max_new_tokens is not None:
            output_token = int(max_new_tokens)
        else:
            output_token = int(max_length - input_length)

        self._init_generation_state(batch_size)

        all_token_ids = input_ids.clone()
        finished = [False] * batch_size

        if attention_mask is None:
            attention_mask_2d = torch.ones_like(input_ids, dtype=torch.long, device=self.dev)
        else:
            attention_mask_2d = attention_mask.to(self.dev)

        tick = time.time()
        is_prefill = True
        self.stats.prefill_tokens = int(input_length * batch_size)

        for _ in range(output_token):
            if all(finished):
                break

            if is_prefill:
                position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.long, device=self.dev)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            else:
                cur_pos = self.past_key_values.get_seq_length(0)
                position_ids = torch.full((batch_size, 1), cur_pos, dtype=torch.long, device=self.dev)

            logits = self.deepseek_forward(input_ids, position_ids, attention_mask_2d, is_prefill)

            if is_prefill:
                self.stats.prefill_time = time.time() - tick
                tick = time.time()
                is_prefill = False

            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            all_token_ids = torch.cat([all_token_ids, next_token], dim=-1)
            input_ids = next_token

            attention_mask_2d = torch.ones((batch_size, 1), device=self.dev, dtype=torch.long)

            eos_id = int(self.tokenizer.eos_token_id)
            for b in range(batch_size):
                if int(next_token[b, 0].item()) == eos_id:
                    finished[b] = True

            self.stats.decode_tokens += int(batch_size - sum(finished))

        self.stats.decode_time = time.time() - tick
        return all_token_ids

    # =============================================================================
    # deepseek_forward
    # - ODP/DES block is preserved (code2-style) except only formatting cleanup.
    # - Non-ODP/DES parts aligned to second-code:
    #   * self_attn parsing + KV cache update
    #   * decode filter triggers on {"cecar","constant"}
    #   * cache hit/all stats counted at expert execution time
    # =============================================================================
    @torch.no_grad()
    def deepseek_forward(self, input_ids, position_ids, attention_mask_2d, is_prefill: bool):
        hidden_states = self.model.embed_tokens(input_ids.to(self.dev)).to(self.dtype)

        use_des = bool(self.des_enabled and self.des_mu_4_5_by_layer and self.des_mu_5_6_by_layer)
        use_protect = bool(use_des and is_prefill and self.protection_enabled and float(self.protection_top_ratio) > 0.0)

        for i_layer, layer in enumerate(self.model.layers):
            # ----------------------------
            # Self-attention (second-code parsing + KV update)
            # ----------------------------
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            bsz, q_len, _ = hidden_states.shape

            if is_prefill:
                kv_len = q_len
                attn4d = build_4d_additive_causal_mask(
                    attention_mask_2d, q_len=q_len, kv_len=kv_len,
                    device=self.dev, dtype=self.dtype
                )
            else:
                past_len = self.past_key_values.get_seq_length(i_layer)
                kv_len = int(past_len + q_len)
                attn2d_decode = torch.ones((bsz, kv_len), device=self.dev, dtype=torch.long)
                attn4d = build_4d_additive_causal_mask(
                    attn2d_decode, q_len=q_len, kv_len=kv_len,
                    device=self.dev, dtype=self.dtype
                )

            # Request attentions only when ODP uses them (prefill-only)
            try:
                attn_out = layer.self_attn(
                    hidden_states,
                    attention_mask=attn4d,
                    position_ids=position_ids,
                    past_key_value=self.past_key_values,
                    use_cache=True,
                    output_attentions=bool(is_prefill and use_protect),
                )
            except TypeError:
                attn_out = layer.self_attn(
                    hidden_states,
                    attention_mask=attn4d,
                    position_ids=position_ids,
                    past_key_value=self.past_key_values,
                    use_cache=True,
                )

            attn_weights = self._extract_attn_weights(attn_out) if (is_prefill and use_protect) else None

            if isinstance(attn_out, (tuple, list)) and len(attn_out) >= 3:
                attn_hidden, present = attn_out[0], attn_out[2]
            elif isinstance(attn_out, dict):
                if "hidden_states" not in attn_out:
                    raise RuntimeError("layer.self_attn dict missing 'hidden_states'.")
                attn_hidden = attn_out["hidden_states"]
                present = attn_out.get("past_key_value", None)
                if present is None:
                    present = attn_out.get("past_key_values", None)
            else:
                raise RuntimeError("layer.self_attn returned unparsable output; cannot maintain KV cache.")

            if present is not None:
                self.past_key_values = present

            hidden_states = residual + attn_hidden

            # ----------------------------
            # Post-attn norm
            # ----------------------------
            residual = hidden_states
            moe_in = layer.post_attention_layernorm(hidden_states)

            if self._layer_is_dense0(i_layer) or (not self._layer_has_moe_gate(layer)):
                hidden_states = residual + layer.mlp(moe_in)
                continue

            # ----------------------------
            # MoE gate
            # ----------------------------
            gate = layer.mlp.gate
            if isinstance(gate, LocalMoEGate):
                gate._ctx_layer_id = i_layer
                gate._ctx_is_prefill = is_prefill
                gate._ctx_cache = self.virtual_cache
                gate._ctx_bonus_strategy = self.bonus_strategy
                gate._ctx_lambda_cache = self.lambda_cache
                gate._ctx_top_J = self.top_J
                gate._ctx_cache_delta_tracker = self.cache_delta_tracker
                gate._ctx_stats = self.stats
                gate._ctx_lru_counter = self.lru_counter
                gate._ctx_lfu_counter = self.lfu_counter

            topk_idx_2d, topk_weight_2d, _ = gate(moe_in)

            # ----------------------------
            # DES / ODP  (UNCHANGED logic, only formatting)
            # ----------------------------
            des_keep_mask_2d = None
            if use_des:
                if int(topk_weight_2d.shape[1]) != 6:
                    raise RuntimeError(f"DES Top4-6 requires gate topk=6, got K={int(topk_weight_2d.shape[1])}")

                is_protected = torch.zeros((topk_weight_2d.shape[0],), device=self.dev, dtype=torch.bool)
                if use_protect:
                    try:
                        p_mask = self._compute_protection_mask_from_attn(moe_in, attn_weights, self.protection_top_ratio)
                        is_protected = p_mask.view(-1)
                    except Exception:
                        is_protected = torch.zeros_like(is_protected)

                order = torch.argsort(topk_weight_2d, dim=-1, descending=True)
                w_s = torch.gather(topk_weight_2d, 1, order)

                ratio1 = w_s[:, 3] / (w_s[:, 4] + 1e-6)
                ratio2 = w_s[:, 4] / (w_s[:, 5] + 1e-6)

                mu1_l = self.des_mu_4_5_by_layer.get(int(i_layer), None)
                mu2_l = self.des_mu_5_6_by_layer.get(int(i_layer), None)

                keep_sorted = torch.ones_like(w_s, dtype=torch.bool, device=w_s.device)

                cond1 = torch.zeros_like(ratio1, dtype=torch.bool)
                if mu1_l is not None:
                    cond1 = (ratio1 > float(mu1_l)) & (~is_protected)
                keep_sorted[cond1, 4] = False
                keep_sorted[cond1, 5] = False

                cond2 = torch.zeros_like(ratio2, dtype=torch.bool)
                if mu2_l is not None:
                    cond2 = (~cond1) & (ratio2 > float(mu2_l)) & (~is_protected)
                keep_sorted[cond2, 5] = False

                keep_mask = torch.zeros_like(keep_sorted, dtype=torch.bool, device=w_s.device)
                keep_mask.scatter_(1, order, keep_sorted)

                topk_weight_2d = topk_weight_2d * keep_mask.to(topk_weight_2d.dtype)
                des_keep_mask_2d = keep_mask   


            if is_prefill and (self.virtual_cache is not None):
                # shapes: topk_idx_2d = (T, K) where T=bsz*seq, K=topk
                bsz, seq, _ = moe_in.shape
                K = int(topk_idx_2d.shape[-1])
            
                topk_idx_flat = topk_idx_2d.view(bsz * seq, K)
            
                keep_mask_flat = None
                if use_des and (des_keep_mask_2d is not None):
                    keep_mask_flat = des_keep_mask_2d.view(bsz * seq, K)
            
                for b in range(bsz):
                    last_token_idx = b * seq + (seq - 1)
            
                    if keep_mask_flat is not None:
                        kept_pos = keep_mask_flat[last_token_idx].nonzero(as_tuple=True)[0]
                        warm_experts = (
                            topk_idx_flat[last_token_idx, kept_pos].tolist() if kept_pos.numel() > 0 else []
                        )
                    else:
                        warm_experts = topk_idx_flat[last_token_idx].tolist()
            
                    # keep only valid expert ids
                    warm_experts = [int(e) for e in warm_experts if 0 <= int(e) < self.n_expert]
            
                    # safety fallback: warm at least one expert
                    if len(warm_experts) == 0:
                        e0 = int(topk_idx_flat[last_token_idx, 0].item())
                        if 0 <= e0 < self.n_expert:
                            warm_experts = [e0]
            
                    # Warm cache: prefer explicit add_to_cache; fallback to access()
                    if hasattr(self.virtual_cache, "add_to_cache"):
                        for e in warm_experts:
                            self.virtual_cache.add_to_cache(b, i_layer, int(e))
                    else:
                        for e in warm_experts:
                            self.virtual_cache.access(b, i_layer, int(e))             


            # ----------------------------
            # Expert execution (second-code semantics)
            # - decode filter triggers on bonus_strategy in {"cecar","constant"}
            # - cache stats are counted at expert execution time
            # ----------------------------
            bsz, seq, _ = moe_in.shape
            N = int(bsz * seq)
            topk = int(topk_idx_2d.shape[-1])

            x2d = moe_in.view(N, self.hidden_dim).to(self.dev, dtype=self.dtype)
            topk_idx = topk_idx_2d.view(N, topk)
            topk_w = topk_weight_2d.view(N, topk).to(self.dev, dtype=self.dtype)

            y2d = torch.zeros((N, self.hidden_dim), device=self.dev, dtype=self.dtype)

            is_decode = (not is_prefill)
            apply_decode_filter = is_decode and (self.bonus_strategy in ("cecar", "constant"))

            compute_k = min(int(self.compute_k), topk)
            topk_threshold = min(int(self.topk_threshold), compute_k)

            for n in range(N):
                b = n // seq
                x_row = x2d[n:n + 1]
            
                experts_topk = [int(e) for e in topk_idx[n].tolist()]
                weights_topk = [float(w) for w in topk_w[n].tolist()]
            
                # -------------------------
                # 1) Apply DES keep_mask to execution list (NO renorm)
                # -------------------------
                if use_des and (des_keep_mask_2d is not None):
                    keep_row = des_keep_mask_2d.view(N, topk)[n]  # (K,) bool
                    experts_topk = [e for j, e in enumerate(experts_topk) if bool(keep_row[j])]
                    weights_topk = [w for j, w in enumerate(weights_topk) if bool(keep_row[j])]
            
                    # safety: if everything pruned, fallback to first original expert
                    if len(experts_topk) == 0:
                        # fallback: take argmax among original topk_w (already masked earlier, but just in case)
                        # here we fallback to first expert in original ordering
                        experts_topk = [int(topk_idx[n, 0].item())]
                        weights_topk = [float(topk_w[n, 0].item())]
            
                # also skip zero-weight experts (important: avoid access on pruned experts)
                tmp_experts = []
                tmp_weights = []
                for e, w in zip(experts_topk, weights_topk):
                    if (e < 0) or (e >= self.n_expert):
                        continue
                    if w <= 0.0:
                        continue
                    tmp_experts.append(e)
                    tmp_weights.append(w)
                experts_topk, weights_topk = tmp_experts, tmp_weights
            
                if len(experts_topk) == 0:
                    # final safety fallback
                    e0 = int(topk_idx[n, 0].item())
                    w0 = float(topk_w[n, 0].item())
                    if 0 <= e0 < self.n_expert and w0 > 0:
                        experts_topk, weights_topk = [e0], [w0]
            
                # -------------------------
                # 2) Decode filtering (cecar/constant) — choose final exec_experts (NO renorm)
                # -------------------------
                if apply_decode_filter:
                    experts_cut = experts_topk[:compute_k]
                    weights_cut = weights_topk[:compute_k]
            
                    cached = self.virtual_cache.get_cached_expert_ids(b, i_layer) or []
                    cached_set = set(int(eid) for eid in cached)
            
                    kept_experts: List[int] = []
                    kept_weights: List[float] = []
                    for k_idx, e in enumerate(experts_cut):
                        if (k_idx < topk_threshold) or (int(e) in cached_set):
                            kept_experts.append(int(e))
                            kept_weights.append(float(weights_cut[k_idx]))
            
                    if len(kept_experts) == 0:
                        kept_experts = [int(experts_cut[0])]
                        kept_weights = [float(weights_cut[0])]
            
                    exec_experts = kept_experts
                    exec_weights = kept_weights
                else:
                    exec_experts = experts_topk
                    exec_weights = weights_topk
            
                # final skip-zero check (after filter)
                tmp_experts = []
                tmp_weights = []
                for e, w in zip(exec_experts, exec_weights):
                    if (e < 0) or (e >= self.n_expert):
                        continue
                    if w <= 0.0:
                        continue
                    tmp_experts.append(int(e))
                    tmp_weights.append(float(w))
                exec_experts, exec_weights = tmp_experts, tmp_weights
            
                if len(exec_experts) == 0:
                    # safety fallback
                    e0 = int(topk_idx[n, 0].item())
                    w0 = float(topk_w[n, 0].item())
                    if 0 <= e0 < self.n_expert and w0 > 0:
                        exec_experts, exec_weights = [e0], [w0]
            
                # -------------------------
                # 3) Cache access + stats (decode only) — access FIRST
                # -------------------------
                if is_decode:
                    # access for each expert used (exactly once)
                    for e in exec_experts:
                        hit = self.virtual_cache.access(b, i_layer, int(e))
            
                        self.stats.cnt_expert_all_by_layer[b][i_layer] += 1
                        if hit:
                            self.stats.cnt_expert_hit_by_layer[b][i_layer] += 1
            
                    # update_arithmetic ONCE per token, AFTER access
                    self.virtual_cache.update_arithmetic(b, i_layer, exec_experts)
            
                # -------------------------
                # 4) Expert compute (unchanged math, but uses exec_experts/exec_weights)
                # -------------------------
                for e, w in zip(exec_experts, exec_weights):
                    expert_layer = self.all_experts.get(i_layer, {}).get(int(e), None)
                    if expert_layer is None:
                        continue
                    y2d[n] += (expert_layer(x_row) * float(w)).squeeze(0)
            
                # -------------------------
                # 5) LRU/LFU step update (same as A, but based on exec_experts)
                # -------------------------
                if (not is_prefill) and (self.bonus_strategy in ("lru", "lfu")):
                    self.global_step[b] += 1
                    for exp in exec_experts:
                        exp = int(exp)
                        if exp < 0 or exp >= self.n_expert:
                            continue
                        if self.bonus_strategy == "lru":
                            self.lru_counter[b][i_layer][exp] = self.global_step[b]
                        if self.bonus_strategy == "lfu":
                            self.lfu_counter[b][i_layer][exp] += 1


            hidden_after = y2d.view(bsz, seq, self.hidden_dim)

            # Shared expert add (if any)
            if self.n_shared_experts and self.shared_expert_path:
                shared_layer = self.shared_expert_modules.get(i_layer, None)
                if shared_layer is not None:
                    try:
                        shared_out = shared_layer(moe_in)
                    except Exception:
                        shared_out = shared_layer(moe_in.view(-1, self.hidden_dim)).view(bsz, seq, self.hidden_dim)
                    hidden_after = hidden_after + shared_out.to(self.dev, self.dtype)

            hidden_states = residual + hidden_after

        hidden_states = self.model.norm(hidden_states)
        return self.lm_head(hidden_states)

    # ---------------------------------------------------------------------
    # Cache stats (second-code accumulation style)
    # ---------------------------------------------------------------------
    def get_cache_stats(self):
        if self.virtual_cache is None:
            return {}
    
        current_total = sum(sum(tc) for tc in self.virtual_cache.total_count)
        current_hits  = sum(sum(hc) for hc in self.virtual_cache.hit_count)
    
        total = int(current_total + getattr(self, "_accumulated_total_count", 0))
        hits  = int(current_hits  + getattr(self, "_accumulated_hit_count", 0))
    
        B = self.batch_size
        layers = list(self.virtual_cache._layer_range())
    
        cur_total_by_layer = [0] * self.n_layer
        cur_hit_by_layer   = [0] * self.n_layer
        for l in layers:
            cur_total_by_layer[l] = sum(self.virtual_cache.total_count[b][l] for b in range(B))
            cur_hit_by_layer[l]   = sum(self.virtual_cache.hit_count[b][l]   for b in range(B))
    
        accT = getattr(self, "_accumulated_total_count_by_layer", [0] * self.n_layer)
        accH = getattr(self, "_accumulated_hit_count_by_layer",   [0] * self.n_layer)
        if len(accT) != self.n_layer: accT = [0] * self.n_layer
        if len(accH) != self.n_layer: accH = [0] * self.n_layer
    
        total_by_layer = [cur_total_by_layer[l] + accT[l] for l in range(self.n_layer)]
        hit_by_layer   = [cur_hit_by_layer[l]   + accH[l] for l in range(self.n_layer)]
    
        hit_rate_by_layer = [
            (hit_by_layer[l] / total_by_layer[l]) if total_by_layer[l] > 0 else 0.0
            for l in range(self.n_layer)
        ]
    
        return {
            "hit_rate": hits / total if total > 0 else 0.0,
            "hit_rate_by_layer": hit_rate_by_layer,
            "hit_rate_by_batch": [self.virtual_cache.get_hit_rate(batch_idx=b) for b in range(self.batch_size)],
        }


    # ---------------------------------------------------------------------
    # remove_experts (second-code: accumulate then reset)
    # ---------------------------------------------------------------------
    def remove_experts(self):
        if self.virtual_cache is not None:
            self._accumulated_total_count += sum(sum(tc) for tc in self.virtual_cache.total_count)
            self._accumulated_hit_count   += sum(sum(hc) for hc in self.virtual_cache.hit_count)

            B = self.batch_size
            layers = list(self.virtual_cache._layer_range())  
    
            if len(self._accumulated_total_count_by_layer) != self.n_layer:
                self._accumulated_total_count_by_layer = [0] * self.n_layer
            if len(self._accumulated_hit_count_by_layer) != self.n_layer:
                self._accumulated_hit_count_by_layer = [0] * self.n_layer
    
            for l in layers:
                self._accumulated_total_count_by_layer[l] += sum(self.virtual_cache.total_count[b][l] for b in range(B))
                self._accumulated_hit_count_by_layer[l]   += sum(self.virtual_cache.hit_count[b][l]   for b in range(B))
    
            self.virtual_cache.reset()
    
        self.past_key_values = DynamicCache()
        self.past_key_values_length = [0] * self.batch_size
