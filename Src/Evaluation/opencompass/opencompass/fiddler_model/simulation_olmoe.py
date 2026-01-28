# ============================================================================
# SimulationOlmoe
# ============================================================================
# All-GPU MoE inference with virtual cache simulation for OLMoE.
#
# Key properties:
#   - All experts are preloaded on GPU (no offloading)
#   - VirtualCache simulates cache state (no real eviction)
#   - Batch-wise independent cache states
#   - Optional routing modification:
#       * none / mocce / cecar
#       * DES (top4/5/6 pruning) + ODP token protection (prefill-only)
#
# Notes:
#   - Keeps your original OLMoE semantics:
#       * If DES is disabled, routing weights for selected experts are NOT renormalized
#         (gather RAW weights directly, as in your original code).
#   - Designed as a paper-submission algorithm reference + GitHub artifact.
# ============================================================================

from __future__ import annotations

import os
import json
import time
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import transformers

from .utils_cache import VirtualCache


# ============================================================================
# Stats
# ============================================================================

class SimulationOlmoeStats:
    """Statistics container for SimulationOlmoe inference."""

    def __init__(self, n_layers: int = 16, batch_size: int = 1):
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self.prefill_time = 0.0
        self.decode_time = 0.0
        self.prefill_tokens = 0
        self.decode_tokens = 0
        self.scale_records = [[] for _ in range(self.batch_size)]
        self.alpha_records = [[] for _ in range(self.batch_size)]

    @property
    def prefill_speed(self) -> float:
        return 0.0 if self.prefill_time == 0 else float(self.prefill_tokens) / float(self.prefill_time)

    @property
    def decode_speed(self) -> float:
        return 0.0 if self.decode_time == 0 else float(self.decode_tokens) / float(self.decode_time)

    def summary(self) -> Dict[str, float]:
        scales = [s for xs in self.scale_records for s in xs]
        alphas = [a for xs in self.alpha_records for a in xs]
        return {
            "prefill_time": float(self.prefill_time),
            "decode_time": float(self.decode_time),
            "prefill_speed": float(self.prefill_speed),
            "decode_speed": float(self.decode_speed),
            "avg_scale": float(sum(scales) / len(scales)) if scales else 0.0,
            "avg_alpha": float(sum(alphas) / len(alphas)) if alphas else 0.0,
        }


# ============================================================================
# Model
# ============================================================================

class SimulationOlmoe:
    """
    Simulation MoE model with all experts on GPU and virtual cache for OLMoE.
    Supports batch processing with independent cache states.
    """

    def __init__(self, args):
        self.args = args
        self.dtype = torch.bfloat16

        # ------------------------------------------------------------
        # Load base model (non-expert parts)
        # ------------------------------------------------------------
        self.non_expert_model_path = args.non_expert_model
        device_map = getattr(args, "device_map", "auto")

        full_model = transformers.OlmoeForCausalLM.from_pretrained(
            args.non_expert_model,
            torch_dtype=self.dtype,
            device_map=device_map,
            use_cache=True,
        )

        self.lm_head = full_model.lm_head
        self.model = full_model.model

        self.expert_path = args.expert_path
        self.dev = next(self.model.parameters()).device
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.non_expert_model)
        self.dense_first_layer = False

        # ------------------------------------------------------------
        # Model dimensions
        # ------------------------------------------------------------
        self.n_layer = len(self.model.layers)
        self.n_expert = len(self.model.layers[0].mlp.experts)
        self.n_routing_expert = int(self.model.config.num_experts_per_tok)
        self.hidden_dim = int(self.model.config.hidden_size)

        # ------------------------------------------------------------
        # Cache configuration
        # ------------------------------------------------------------
        self.cache_size = int(getattr(args, "cache_size", 12))
        self.cache_policy = str(getattr(args, "cache_policy", "lru"))
        self.batch_size = int(getattr(args, "batch_size", 1))

        # ------------------------------------------------------------
        # Bonus strategy
        # ------------------------------------------------------------
        self.mode = str(getattr(args, "mode", "none"))
        self.bonus_strategy = str(getattr(args, "bonus_strategy", "none"))
        self.bonus_scale = 0.05

        # ------------------------------------------------------------
        # MOCCE parameters
        # ------------------------------------------------------------
        self.lambda_cache = float(getattr(args, "lambda_cache", 0.2))
        self.top_J = int(getattr(args, "top_J", 2))
        self.cache_delta_tracker = [{"init": False, "avg": 0.0} for _ in range(self.n_layer)]

        # ------------------------------------------------------------
        # DES / ODP (MCMOE) parameters
        # ------------------------------------------------------------
        self.des_enabled = bool(getattr(args, "enable_des", False))
        self.protection_enabled = bool(getattr(args, "enable_protection", False))
        self.protection_top_ratio = float(getattr(args, "protection_top_ratio", 0.02))
        self.mcmoe_threshold_path = getattr(args, "mcmoe_threshold_path", None)

        self.des_mu_4_5_by_layer: Dict[int, float] = {}
        self.des_mu_5_6_by_layer: Dict[int, float] = {}
        if self.mode != "none":
            self.bonus_strategy = "none"
        if self.des_enabled:
            self.n_routing_expert = 6
            self._init_des_thresholds()

        # ------------------------------------------------------------
        # Stats & routing record
        # ------------------------------------------------------------
        self.stats = SimulationOlmoeStats(self.n_layer, self.batch_size)
        self.virtual_cache = None
        self.routing_record: List[dict] = []

        # ------------------------------------------------------------
        # Decode filter knobs (for CECAR path)
        # ------------------------------------------------------------
        self.compute_k = int(getattr(args, "compute_k", self.n_routing_expert))
        self.topk_threshold = int(getattr(args, "topk_threshold", 4))

        # LRU/LFU counters
        self.lru_counter = None
        self.lfu_counter = None
        self.global_step = None
                
        # Accumulated stats (scalar + layerwise)
        self._accumulated_total_count = 0
        self._accumulated_hit_count = 0
        self._accumulated_total_count_by_layer = [0] * self.n_layer
        self._accumulated_hit_count_by_layer   = [0] * self.n_layer        

        # ------------------------------------------------------------
        # Preload all experts
        # ------------------------------------------------------------
        print(f"Loading all experts to GPU ({self.n_layer} layers x {self.n_expert} experts)...")
        self.all_experts = self._preload_all_experts()
        print(
            f"SimulationOlmoe ready. Cache policy: {self.cache_policy}, "
            f"Cache size: {self.cache_size}, Mode: {self.mode}, bonus_strategy: {self.bonus_strategy}"
        )

    # ---------------------------------------------------------------------
    # DES thresholds
    # ---------------------------------------------------------------------

    def _init_des_thresholds(self):
        base_dir = self.mcmoe_threshold_path
        if not base_dir or (not os.path.isdir(base_dir)):
            raise ValueError(f"mcmoe_threshold_path must be a directory, got: {base_dir}")

        task_raw = getattr(self.args, "tasks", None)
        task = (task_raw or "").strip().lower().replace("-", "_")

        task_to_stem = {
            "humaneval": "mmlu_ccsc",
            "mbpp": "mmlu_ccsc",
            "gpqa": "arc_challenge",
            "math500": "mathqa",
        }
        stem = task_to_stem.get(task, None)
        if stem is None:
            raise ValueError(f"Unsupported task for DES threshold selection: {task_raw!r}")

        threshold_file = f"OLMoE_1B_7B_0125_Instruct_{stem}_top4_6_thresholds.json"
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

        mu1: Dict[int, float] = {}
        mu2: Dict[int, float] = {}
        seen = set()

        for it in layers:
            if not isinstance(it, dict):
                raise ValueError("each element in 'layers' must be an object.")
            if "layer" not in it or "mu_4_5" not in it or "mu_5_6" not in it:
                raise ValueError(
                    f"each layer entry must contain 'layer', 'mu_4_5', 'mu_5_6'. got keys={list(it.keys())}"
                )
            lid = int(it["layer"])
            if lid in seen:
                raise ValueError(f"duplicate layer id in threshold file: layer={lid}")
            seen.add(lid)
            try:
                mu1[lid] = float(it["mu_4_5"])
                mu2[lid] = float(it["mu_5_6"])
            except Exception as e:
                raise ValueError(f"failed to parse mu_4_5/mu_5_6 as float at layer={lid}: {e}")
        if not mu1 or not mu2:
            raise ValueError("parsed thresholds but got empty mu1/mu2 dicts (unexpected).")

        return mu1, mu2

    # ---------------------------------------------------------------------
    # Experts
    # ---------------------------------------------------------------------

    def _preload_all_experts(self):
        """Load all expert weights to GPU at initialization."""
        experts = [[None for _ in range(self.n_expert)] for _ in range(self.n_layer)]
        for i_layer in range(self.n_layer):
            for i_expert in range(self.n_expert):
                path = f"{self.expert_path}/layer{i_layer}_expert{i_expert}.pt"
                experts[i_layer][i_expert] = torch.load(path, map_location=self.dev, weights_only=False)
            print(f"  Layer {i_layer + 1}/{self.n_layer} loaded", end="\r")
        print(f"  All {self.n_layer} layers loaded.           ")
        return experts

    # ---------------------------------------------------------------------
    # Per-generation state
    # ---------------------------------------------------------------------

    def _init_generation_state(self, batch_size: int):
        """Initialize per-generation state (cache, counters, KV cache, etc.)."""
        self.batch_size = int(batch_size)

        self.virtual_cache = VirtualCache(
            model="OLMoE_1B_7B_0125_Instruct",
            cache_policy=self.cache_policy,
            cache_size=self.cache_size,
            n_layers=self.n_layer,
            batch_size=self.batch_size,
            num_experts=self.n_expert,
            dense_first_layer=self.dense_first_layer,
            ffn_model_path=getattr(self.args, "ffn_model_path", None),
        )

        # Counters: [B][L][E]
        self.lru_counter = [[[0] * self.n_expert for _ in range(self.n_layer)] for _ in range(self.batch_size)]
        self.lfu_counter = [[[0] * self.n_expert for _ in range(self.n_layer)] for _ in range(self.batch_size)]
        self.global_step = [0] * self.batch_size

        self.stats = SimulationOlmoeStats(self.n_layer, self.batch_size)
        self.routing_record = []

        self.past_key_values = transformers.cache_utils.DynamicCache.from_legacy_cache(None)
        self.past_key_values_length = [0] * self.batch_size

    def reset_stats(self):
        self.stats.reset()

    # ---------------------------------------------------------------------
    # Generation (HF-like)
    # ---------------------------------------------------------------------

    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        max_length=2048,
        max_new_tokens=None,
        do_sample=False,
        temperature=0.7,
        top_p=0.95,
        top_k=0,
        num_return_sequences=1,
        stopping_criteria=None,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids must be provided")

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        input_ids = input_ids.to(self.dev)
        batch_size = int(input_ids.shape[0])
        input_length = int(input_ids.shape[1])

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(self.dev)
        self.attention_mask = attention_mask

        self._init_generation_state(batch_size)

        if max_new_tokens is not None:
            output_token = int(max_new_tokens)
        else:
            output_token = int(max(1, max_length - input_length))

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        position_ids = position_ids.to(self.dev)
        self.actual_seq_lengths = attention_mask.sum(dim=1).tolist()

        all_token_ids = input_ids.clone()
        finished = [False] * batch_size

        tick = time.time()
        is_prefill = True
        self.stats.prefill_tokens = int(input_length * batch_size)

        for _ in range(output_token):
            if all(finished):
                break

            if stopping_criteria is not None and stopping_criteria(all_token_ids, None):
                break

            logits = self.olmoe_forward(input_ids, position_ids, is_prefill)

            if is_prefill:
                self.stats.prefill_time = time.time() - tick
                tick = time.time()
                for b in range(batch_size):
                    self.past_key_values_length[b] = input_length
            else:
                for b in range(batch_size):
                    self.past_key_values_length[b] += 1

            is_prefill = False

            logits_last = logits[:, -1, :].to("cpu")

            if do_sample and temperature > 0:
                logits_last = logits_last / float(temperature)
                if top_k > 0:
                    kth = torch.topk(logits_last, int(top_k))[0][..., -1, None]
                    logits_last = logits_last.masked_fill(logits_last < kth, float("-inf"))
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits_last, descending=True)
                    probs = F.softmax(sorted_logits, dim=-1)
                    cum = torch.cumsum(probs, dim=-1)
                    sorted_remove = cum > float(top_p)
                    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
                    sorted_remove[..., 0] = 0
                    remove = sorted_remove.scatter(1, sorted_indices, sorted_remove)
                    logits_last = logits_last.masked_fill(remove, float("-inf"))
                probs = F.softmax(logits_last, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits_last, dim=-1, keepdim=True)

            next_token = next_token.to(self.dev)
            all_token_ids = torch.cat([all_token_ids, next_token], dim=1)

            self.attention_mask = torch.cat(
                [
                    self.attention_mask,
                    torch.ones((batch_size, 1), device=self.dev, dtype=self.attention_mask.dtype),
                ],
                dim=1,
            )

            eos_id = int(self.tokenizer.eos_token_id)
            for b in range(batch_size):
                if int(next_token[b, 0].item()) == eos_id:
                    finished[b] = True

            input_ids = next_token
            position_ids = torch.tensor([[self.actual_seq_lengths[b]] for b in range(batch_size)], device=self.dev)
            for b in range(batch_size):
                self.actual_seq_lengths[b] += 1

        self.stats.decode_time = time.time() - tick
        self.stats.decode_tokens = int((all_token_ids.shape[1] - input_length) * batch_size)
        return all_token_ids

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def olmoe_forward(self, input_ids, position_ids, is_prefill: bool):
        """
        Forward pass with virtual cache simulation (batch).

        Semantics:
          - Selection uses SELECT weights (bonus-applied).
          - Compute uses RAW weights gathered for selected experts.
          - If DES is OFF: NO renormalization over selected experts (preserve original OLMoE behavior).
          - If DES is ON: apply keep_mask.
          - ODP protection applies only during prefill when DES is active.
        """
        batch_size = int(input_ids.shape[0])
        seq_len = int(input_ids.shape[1])

        use_des = bool(self.des_enabled and self.des_mu_4_5_by_layer and self.des_mu_5_6_by_layer)
        use_protect = bool(use_des and is_prefill and self.protection_enabled and float(self.protection_top_ratio) > 0.0)
        K_select = 6 if use_des else int(self.n_routing_expert)

        hidden_states = self.model.embed_tokens(input_ids.to(self.dev))

        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

        past_key_value = self.past_key_values
        cache_position = position_ids[0] if position_ids.dim() == 2 else position_ids

        for i_layer, layer in enumerate(self.model.layers):
            original_hidden_states_shape = hidden_states.shape
            residual = hidden_states

            # -------------------------
            # Self-attention
            # -------------------------
            hidden_states = layer.input_layernorm(hidden_states)
            attn_in = hidden_states

            hidden_states, _, _ = layer.self_attn(
                hidden_states,
                position_embeddings=(position_embeddings[0].to(self.dtype), position_embeddings[1].to(self.dtype)),
                past_key_value=past_key_value,
                use_cache=True,
                attention_mask=None,
                output_attentions=False,
                cache_position=cache_position,
            )

            hidden_states = residual + hidden_states
            residual = hidden_states

            # -------------------------
            # Router
            # -------------------------
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = hidden_states.view(-1, self.hidden_dim) 

            router_logits = layer.mlp.gate(hidden_states)           
            router_logits_select = router_logits.clone()

            # -------------------------
            # MOCCE (decode only)
            # -------------------------
            if (not is_prefill) and (self.bonus_strategy == "mocce"):
                z = router_logits
                for b in range(batch_size):
                    token_idx = b
                    z_b = z[token_idx:token_idx + 1]

                    cur_range = (z_b.max() - z_b.min()).item()
                    tracker = self.cache_delta_tracker[i_layer]
                    if not tracker["init"]:
                        tracker["avg"] = cur_range
                        tracker["init"] = True
                    else:
                        tracker["avg"] = 0.99 * tracker["avg"] + 0.01 * cur_range
                    delta_avg = tracker["avg"]

                    mask = torch.zeros_like(z_b)
                    _, topJ = torch.topk(z_b, self.top_J, dim=1)
                    mask.scatter_(1, topJ, 1.0)

                    cached = self.virtual_cache.get_cached_expert_ids(b, i_layer)
                    if cached:
                        mask[0, cached] = 1.0

                    router_logits_select[token_idx:token_idx + 1] = z_b + self.lambda_cache * delta_avg * mask

            # -------------------------
            # CECAR (decode only)
            # -------------------------
            if (not is_prefill) and (self.bonus_strategy == "cecar"):
                for b in range(batch_size):
                    token_idx = b
                    z_b = router_logits[token_idx]

                    topk_logits, _ = torch.topk(z_b, k=int(self.n_routing_expert))
                    z_top1 = topk_logits[0].item()
                    z_topk = topk_logits[-1].item()
                    delta = z_top1 - z_topk

                    cached_experts = self.virtual_cache.get_cached_expert_ids(b, i_layer)
                    if cached_experts:
                        scores = [self.virtual_cache.get_eviction_score(b, i_layer, ce) for ce in cached_experts]
                        max_score = max(scores) if scores else 1.0

                        for ce, score in zip(cached_experts, scores):
                            alpha = 1.0 - score / (max_score + 1e-5)
                            self.stats.alpha_records[b].append(alpha)
                            router_logits_select[token_idx, ce] += delta * alpha

            routing_weights_raw = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights_select = F.softmax(router_logits_select, dim=1, dtype=torch.float)

            routing_weights_raw_batched = routing_weights_raw.view(batch_size, seq_len, -1)
            routing_weights_select_batched = routing_weights_select.view(batch_size, seq_len, -1)

            # -------------------------
            # Other bonus strategies on weights (decode only)
            # -------------------------
            if (not is_prefill) and (self.bonus_strategy not in ["none", "mocce", "cecar"]):
                for b in range(batch_size):
                    cached_experts = self.virtual_cache.get_cached_expert_ids(b, i_layer)
                    if cached_experts:
                        rw_cache = routing_weights_select_batched[b, 0, list(cached_experts)]
                        rw_cache_mean = rw_cache.mean().item()
                        rw_cache_std = rw_cache.std().item()
                        cv_cache = rw_cache_std / (rw_cache_mean + 1e-5)
                        scale = 1.0 / (1.0 + cv_cache)
                        self.stats.scale_records[b].append(scale)

                        for ce in cached_experts:
                            if self.bonus_strategy == "random":
                                import random
                                alpha = random.random()
                            elif self.bonus_strategy == "constant":
                                scale = 0.533446
                                alpha = 0.629355
                            elif self.bonus_strategy == "lru":
                                lru_val = self.lru_counter[b][i_layer][ce]
                                max_lru = max(self.lru_counter[b][i_layer]) if max(self.lru_counter[b][i_layer]) > 0 else 1
                                alpha = lru_val / max_lru
                            elif self.bonus_strategy == "lfu":
                                lfu_val = self.lfu_counter[b][i_layer][ce]
                                max_lfu = max(self.lfu_counter[b][i_layer]) if max(self.lfu_counter[b][i_layer]) > 0 else 1
                                alpha = lfu_val / max_lfu
                            else:
                                alpha = 0.0

                            self.stats.alpha_records[b].append(alpha)
                            routing_weights_select_batched[b, 0, ce] += self.bonus_scale * scale * alpha

            # -------------------------
            # Top-K selection (based on SELECT weights)
            # -------------------------
            _, selected_experts_batched = torch.topk(
                routing_weights_select_batched, k=K_select, dim=-1, largest=True, sorted=True
            )  

            selected_experts = selected_experts_batched.view(-1, K_select)

            # gather RAW weights for selected experts (for DES logic and/or compute)
            selected_raw_weights_batched = torch.gather(
                routing_weights_raw_batched, dim=2, index=selected_experts_batched
            ).to(torch.float)  

            # -------------------------
            # DES + ODP (optional)
            # -------------------------
            N = batch_size * seq_len
            keep_mask = torch.ones((N, K_select), dtype=torch.bool, device=self.dev)

            keep_mask_batched = None
            routing_weights_kept_batched = None

            if use_des:
                # Normalize across K_select first (RAW-based)
                selected_raw_norm_batched = selected_raw_weights_batched 
                selected_raw_norm = selected_raw_norm_batched.view(N, K_select) 

                # ODP protection (prefill only)
                is_protected_flat = torch.zeros((N,), dtype=torch.bool, device=self.dev)
                if use_protect:
                    l1_norm = torch.norm(hidden_states.to(torch.float), p=1, dim=1).view(batch_size, seq_len)
                    score_bs = l1_norm

                    attn_weights = None
                    try:
                        pos_dbg = self.model.rotary_emb(attn_in, position_ids)
                        attn_dbg = layer.self_attn(
                            attn_in,
                            position_embeddings=(pos_dbg[0].to(self.dtype), pos_dbg[1].to(self.dtype)),
                            past_key_value=None,
                            use_cache=False,
                            attention_mask=None,
                            output_attentions=True,
                            cache_position=cache_position,
                        )
                        if isinstance(attn_dbg, (tuple, list)):
                            for x in attn_dbg:
                                if torch.is_tensor(x) and x.dim() == 4:
                                    attn_weights = x
                                    break
                    except Exception:
                        attn_weights = None

                    if attn_weights is not None:
                        attn = attn_weights.to(torch.float)
                        A = attn.mean(dim=1)  
                        causal = torch.tril(torch.ones((seq_len, seq_len), device=A.device, dtype=A.dtype))
                        A = A * causal.unsqueeze(0)

                        incoming_sum = A.sum(dim=1)  
                        denom2 = torch.arange(seq_len, 0, -1, device=A.device, dtype=A.dtype)
                        incoming_avg = incoming_sum / denom2.unsqueeze(0)

                        score_bs = l1_norm * incoming_avg

                    k_protect = int(max(1, round(seq_len * float(self.protection_top_ratio))))
                    top_idx = torch.topk(score_bs, k_protect, dim=1, largest=True, sorted=False).indices
                    mask_bs = torch.zeros_like(score_bs, dtype=torch.bool)
                    mask_bs.scatter_(1, top_idx, True)
                    is_protected_flat = mask_bs.to(self.dev).reshape(-1)

                # DES pruning (ranked by weight)
                w = selected_raw_norm.to(torch.float) 
                order = torch.argsort(w, dim=-1, descending=True)
                w_sorted = torch.gather(w, 1, order)

                r4 = w_sorted[:, 3]
                r5 = w_sorted[:, 4]
                r6 = w_sorted[:, 5]
                ratio1 = r4 / (r5 + 1e-6)
                ratio2 = r5 / (r6 + 1e-6)

                mu1_layer = self.des_mu_4_5_by_layer.get(int(i_layer), None)
                mu2_layer = self.des_mu_5_6_by_layer.get(int(i_layer), None)

                keep_sorted = torch.ones_like(w_sorted, dtype=torch.bool, device=self.dev)

                cond1 = torch.zeros((N,), dtype=torch.bool, device=self.dev)
                if mu1_layer is not None:
                    cond1 = (ratio1 > float(mu1_layer)) & (~is_protected_flat)
                keep_sorted[cond1, 4] = False
                keep_sorted[cond1, 5] = False

                cond2 = torch.zeros((N,), dtype=torch.bool, device=self.dev)
                if mu2_layer is not None:
                    cond2 = (~cond1) & (ratio2 > float(mu2_layer)) & (~is_protected_flat)
                keep_sorted[cond2, 5] = False

                keep_mask = torch.zeros_like(keep_sorted, dtype=torch.bool, device=self.dev)
                keep_mask.scatter_(1, order, keep_sorted)

                w_kept = selected_raw_norm * keep_mask.to(selected_raw_norm.dtype)

                keep_mask_batched = keep_mask.view(batch_size, seq_len, K_select)
                routing_weights_kept_batched = w_kept.view(batch_size, seq_len, K_select)

            # -------------------------
            # Record routing (decode only)
            # -------------------------
            if not is_prefill:
                _, exact_experts_batched = torch.topk(
                    routing_weights_raw_batched, k=int(self.n_routing_expert), dim=-1
                )
                for b in range(batch_size):
                    sel = selected_experts_batched[b, 0].tolist()
                    selected_weights = routing_weights_select_batched[b, 0, sel].tolist()
                    exact = exact_experts_batched[b, 0].tolist()
                    exact_weights = routing_weights_raw_batched[b, 0, exact].tolist()
                    self.routing_record.append(
                        {
                            "batch": b,
                            "layer": i_layer,
                            "experts": sel,
                            "weights": selected_weights,
                            "exact_experts": exact,
                            "exact_weights": exact_weights,
                        }
                    )

            # -------------------------
            # Decode additional filtering (CECAR only, keep your behavior)
            # -------------------------
            apply_decode_filter = (not is_prefill) and (self.bonus_strategy == "cecar")

            filtered_experts_per_batch = None
            filtered_weights_per_batch = None

            if apply_decode_filter:
                filtered_experts_per_batch = [[] for _ in range(batch_size)]
                filtered_weights_per_batch = [[] for _ in range(batch_size)]

                compute_k = min(int(self.compute_k), int(self.n_routing_expert), K_select)
                threshold = min(int(self.topk_threshold), compute_k)

                for b in range(batch_size):
                    experts_topk = selected_experts_batched[b, 0].tolist()[:compute_k]
                    weights_topk = selected_raw_weights_batched[b, 0].tolist()[:compute_k]

                    cached_set = set(self.virtual_cache.get_cached_expert_ids(b, i_layer))

                    kept_experts = []
                    kept_weights = []
                    for topk_idx, e in enumerate(experts_topk):
                        if (topk_idx < threshold) or (e in cached_set):
                            kept_experts.append(e)
                            kept_weights.append(weights_topk[topk_idx])

                    if len(kept_experts) == 0:
                        kept_experts = [experts_topk[0]]
                        kept_weights = [weights_topk[0]]

                    topk_sum = float(sum(weights_topk))
                    kept_sum = float(sum(kept_weights))

                    filtered_experts_per_batch[b] = kept_experts
                    filtered_weights_per_batch[b] = kept_weights

            # -------------------------
            # Cache + LRU/LFU update (decode only)
            # -------------------------
            if not is_prefill:
                for b in range(batch_size):
                    self.global_step[b] += 1

                    if apply_decode_filter:
                        experts_this_batch = filtered_experts_per_batch[b]
                    else:
                        if use_des and (keep_mask_batched is not None):
                            kept_pos = keep_mask_batched[b, 0].nonzero(as_tuple=True)[0]
                            experts_this_batch = (
                                selected_experts_batched[b, 0, kept_pos].tolist() if kept_pos.numel() > 0 else []
                            )
                            if len(experts_this_batch) == 0:
                                experts_this_batch = [int(selected_experts_batched[b, 0, 0].item())]
                        else:
                            experts_this_batch = selected_experts_batched[b, 0].tolist()

                    for exp in experts_this_batch:
                        self.lru_counter[b][i_layer][exp] = self.global_step[b]
                        self.lfu_counter[b][i_layer][exp] += 1
                        self.virtual_cache.access(b, i_layer, exp)

            # -------------------------
            # Compute weights (OLMoE semantics)
            # -------------------------
            if use_des and (routing_weights_kept_batched is not None):
                routing_weights_for_compute = routing_weights_kept_batched.view(-1, K_select).to(self.dtype)
            else:
                routing_weights_for_compute = torch.gather(routing_weights_raw, 1, selected_experts)

            hidden_states_after_experts = torch.zeros_like(hidden_states, device=self.dev, dtype=self.dtype)

            # -------------------------
            # Prefill compute (vectorized by expert id)
            # -------------------------
            if is_prefill:
                for b in range(batch_size):
                    last_token_idx = b * seq_len + (seq_len - 1)

                    if use_des and (keep_mask_batched is not None):
                        kept_pos = keep_mask_batched.view(-1, K_select)[last_token_idx].nonzero(as_tuple=True)[0]
                        kept_experts = selected_experts[last_token_idx, kept_pos].tolist() if kept_pos.numel() > 0 else []
                        for exp in kept_experts:
                            self.virtual_cache.add_to_cache(b, i_layer, int(exp))
                    else:
                        for exp in selected_experts[last_token_idx].tolist():
                            self.virtual_cache.add_to_cache(b, i_layer, int(exp))

                keep_mask_2d = keep_mask_batched.view(-1, K_select) if (use_des and keep_mask_batched is not None) else None

                for i_expert in range(self.n_expert):
                    if keep_mask_2d is not None:
                        mask = (selected_experts == i_expert) & keep_mask_2d
                    else:
                        mask = selected_experts.eq(i_expert)

                    idxs = mask.nonzero(as_tuple=False)
                    if idxs.numel() == 0:
                        continue

                    expert_layer = self.all_experts[i_layer][i_expert]
                    token_idxs = idxs[:, 0]
                    topk_idxs = idxs[:, 1]

                    current_state = hidden_states[token_idxs].to(self.dev, dtype=self.dtype)
                    weight = routing_weights_for_compute[token_idxs, topk_idxs, None]
                    weighted_output = expert_layer(current_state) * weight.to(self.dev, dtype=self.dtype)

                    hidden_states_after_experts.index_add_(0, token_idxs.to(self.dev), weighted_output.to(self.dev))

            # -------------------------
            # Decode compute (1 token per batch)
            # -------------------------
            else:
                for b in range(batch_size):
                    token_idx = b

                    if apply_decode_filter:
                        experts_used = filtered_experts_per_batch[b]
                        weights_used = filtered_weights_per_batch[b]  
                        for j, i_expert in enumerate(experts_used):
                            expert_layer = self.all_experts[i_layer][i_expert]
                            routing_weight = float(weights_used[j])
                            hidden_states_after_experts[token_idx:token_idx + 1] += (
                                expert_layer(hidden_states[token_idx:token_idx + 1]) * routing_weight
                            )
                        try:
                            self.virtual_cache.update_arithmetic(token_idx, i_layer, experts_used)
                        except TypeError:
                            self.virtual_cache.update_arithmetic(b, i_layer, experts_used)

                    else:
                        unique_experts = selected_experts[token_idx].tolist()
                        for topk_idx, i_expert in enumerate(unique_experts):
                            routing_weight = routing_weights_for_compute[token_idx, topk_idx]
                            if use_des and float(routing_weight) <= 0.0:
                                continue
                            expert_layer = self.all_experts[i_layer][i_expert]
                            hidden_states_after_experts[token_idx:token_idx + 1] += (
                                expert_layer(hidden_states[token_idx:token_idx + 1]) * routing_weight
                            )

            hidden_states = residual + hidden_states_after_experts.reshape(original_hidden_states_shape)

        hidden_states = self.model.norm(hidden_states)
        lm_logits = self.lm_head(hidden_states)

        self.present_key_value = past_key_value
        return lm_logits

    # ---------------------------------------------------------------------
    # Cache stats / cleanup
    # ---------------------------------------------------------------------
    
    def get_cache_stats(self):
        if self.virtual_cache is None:
            return {}
    
        current_total = sum(sum(tc) for tc in self.virtual_cache.total_count)
        current_hits  = sum(sum(hc) for hc in self.virtual_cache.hit_count)
    
        accumulated_total = getattr(self, "_accumulated_total_count", 0)
        accumulated_hits  = getattr(self, "_accumulated_hit_count", 0)
    
        total = current_total + accumulated_total
        hits  = current_hits + accumulated_hits

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

    def remove_experts(self):
        """
        Compatibility method - simulation mode never removes experts.
        We reset virtual cache state but accumulate stats.
        """
        if self.virtual_cache is not None:
            if not hasattr(self, "_accumulated_total_count"):
                self._accumulated_total_count = 0
                self._accumulated_hit_count = 0
            if not hasattr(self, "_accumulated_total_count_by_layer"):
                self._accumulated_total_count_by_layer = [0] * self.n_layer
                self._accumulated_hit_count_by_layer   = [0] * self.n_layer
    
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
    
        self.past_key_values = transformers.cache_utils.DynamicCache.from_legacy_cache(None)
        self.past_key_values_length = [0] * self.batch_size
    
        if hasattr(self, "attention_mask"):
            del self.attention_mask

