import os
import json
import time
import warnings
import concurrent.futures
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import transformers

from utils_cache import CachePolicyWrapper

# -----------------------------------------------------------------------------
# Logging / warnings
# -----------------------------------------------------------------------------
transformers.utils.logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=r"The module name .* is not a valid Python identifier.*")
warnings.filterwarnings("ignore", message=r"Some weights of .* were not initialized from the model checkpoint.*")
warnings.filterwarnings("ignore", message=r"`torch_dtype` is deprecated! Use `dtype` instead!")


@dataclass
class SamplingConfig:
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    do_sample: bool = False


class FiddlerQwen3Moe:
    """
    Qwen3-MoE inference wrapper with:
      - Expert cache simulation/management
      - Multiple expert selection strategies:
          vanilla / cecar / mocce / des / odp
      - Prefill/Decode profiling and cache hit statistics
    """

    # -----------------------------
    # Init
    # -----------------------------
    def __init__(self, args):
        self.dtype = torch.bfloat16
        self.dev = torch.device("cuda")

        # HF model
        base = transformers.Qwen3MoeForCausalLM.from_pretrained(
            args.model,
            torch_dtype=self.dtype,
            device_map="cuda",
            use_cache=True,
        )
        self.lm_head = base.lm_head
        self.model = base.model

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
        self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
        self.past_key_values_length = 0

        # Model dims
        self.n_layer = len(self.model.layers)
        self.n_expert = len(self.model.layers[0].mlp.experts)
        self.n_routing_expert = self.model.config.num_experts_per_tok
        self.hidden_dim = self.model.config.hidden_size

        # Args
        self.expert_path = args.expert_path
        self.beam_width = getattr(args, "beam_width", 1)
        self.compute_k = getattr(args, "compute_k", 10)

        # Cache
        self.expert_cache = CachePolicyWrapper(
            cache_size=args.cache_size,
            cache_policy=args.cache_policy,
            n_layers=self.n_layer,
            expert_path=self.expert_path,
            ffn_model_path=getattr(args, "ffn_model_path", ""),
            num_experts=self.n_expert,
        )

        # Thread (cache arithmetic update)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        # Stats
        self._reset_stats()

        # Cache-bonus stats
        self.alpha_records = []
        self.delta_records = []

        # MOCCE params
        self.cache_delta_tracker = [{"avg": 0.0, "init": False} for _ in range(self.n_layer)]
        self.lambda_cache = 0.2
        self.top_J = 2

        # CECAR params
        self.margin_topk_idx = getattr(args, "margin_topk_idx", 8)
        self.topk_threshold = getattr(args, "topk_threshold", 4)
        self.do_renormalize = getattr(args, "do_renormalize", False)

        # Strategy mode
        self.bonus_strategy = getattr(args, "bonus_strategy", "none")  # cecar/mocce/none
        self.mode = getattr(args, "mode", "none")                      # des/odp/none
        if self.bonus_strategy in ("des", "odp"):
            self.mode = self.bonus_strategy
            self.bonus_strategy = "none"

        # DES thresholds
        self.des_mu1 = getattr(args, "des_mu1", 1.5)
        self.des_mu2 = getattr(args, "des_mu2", 1.5)
        self.des_layerwise = False
        self.des_mu_4_5 = None
        self.des_mu_5_6 = None
        self.des_thresholds_path = getattr(args, "des_thresholds_path", None)
        if self.des_thresholds_path:
            self._load_thresholds(path=self.des_thresholds_path, kind="des")
        self.des_k_hist = {4: 0, 5: 0, 6: 0}

        # ODP thresholds
        self.odp_mu1 = getattr(args, "odp_mu1", None) or self.des_mu1
        self.odp_mu2 = getattr(args, "odp_mu2", None) or self.des_mu2
        self.odp_layerwise = False
        self.odp_mu_4_5 = None
        self.odp_mu_5_6 = None
        self.odp_thresholds_path = getattr(args, "odp_thresholds_path", None) or self.des_thresholds_path
        if self.odp_thresholds_path and self.mode == "odp":
            self._load_thresholds(path=self.odp_thresholds_path, kind="odp")
        self.odp_k_hist = {4: 0, 5: 0, 6: 0}

        # Sampling config
        self.sampling_config = SamplingConfig(
            temperature=getattr(args, "temperature", 0.7),
            top_k=getattr(args, "top_k", 0),
            top_p=getattr(args, "top_p", 0.95),
            do_sample=getattr(args, "do_sample", False),
        )

        print("Model is ready.")
        if self.mode == "des":
            print(f"  - mode=DES, thresholds={'layerwise' if self.des_layerwise else 'global'}")
        elif self.mode == "odp":
            print(f"  - mode=ODP, thresholds={'layerwise' if self.odp_layerwise else 'global'}")
        else:
            print(f"  - mode=none, bonus_strategy={self.bonus_strategy}")

    def _reset_stats(self):
        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0
        self.cnt_expert_miss = 0
        self.cnt_expert_hit_by_layer = [0] * self.n_layer
        self.cnt_expert_all_by_layer = [0] * self.n_layer
        self.cnt_expert_miss_by_layer = [0] * self.n_layer
        self.routed_expert_counts = []

    # -----------------------------
    # Threshold loading (DES/ODP)
    # -----------------------------
    def _load_thresholds(self, path: str, kind: str):
        """
        kind: 'des' or 'odp'
        Supports:
          - global: {"mu1": x, "mu2": y}
          - layerwise: {"layers":[{"layer":0,"mu_4_5":x,"mu_5_6":y}, ...]}
        Also supports wrapper key: {"odp": {...}}.
        """
        if not path or not os.path.exists(path):
            print(f"Warning: {kind.upper()} thresholds file not found: {path}")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception as e:
            print(f"Warning: failed to read {kind} thresholds file={path}: {e}")
            return

        obj2 = obj.get("odp") if ("odp" in obj and isinstance(obj["odp"], dict)) else obj

        # global
        if isinstance(obj2, dict) and ("mu1" in obj2) and ("mu2" in obj2):
            mu1, mu2 = float(obj2["mu1"]), float(obj2["mu2"])
            if kind == "des":
                self.des_mu1, self.des_mu2 = mu1, mu2
                self.des_layerwise = False
            else:
                self.odp_mu1, self.odp_mu2 = mu1, mu2
                self.odp_layerwise = False
            return

        # layerwise
        if isinstance(obj, dict) and isinstance(obj.get("layers"), list):
            mu_4_5, mu_5_6 = {}, {}
            for it in obj["layers"]:
                if not isinstance(it, dict) or "layer" not in it:
                    continue
                lid = int(it["layer"])
                if "mu_4_5" in it and "mu_5_6" in it:
                    mu_4_5[lid] = float(it["mu_4_5"])
                    mu_5_6[lid] = float(it["mu_5_6"])

            if mu_4_5 and mu_5_6:
                if kind == "des":
                    self.des_mu_4_5, self.des_mu_5_6 = mu_4_5, mu_5_6
                    self.des_layerwise = True
                else:
                    self.odp_mu_4_5, self.odp_mu_5_6 = mu_4_5, mu_5_6
                    self.odp_layerwise = True

    def _get_mu(self, layer_id: int, kind: str):
        if kind == "des":
            if self.des_layerwise and self.des_mu_4_5 and self.des_mu_5_6:
                return (
                    self.des_mu_4_5.get(layer_id, self.des_mu1),
                    self.des_mu_5_6.get(layer_id, self.des_mu2),
                )
            return self.des_mu1, self.des_mu2

        # odp
        if self.odp_layerwise and self.odp_mu_4_5 and self.odp_mu_5_6:
            return (
                self.odp_mu_4_5.get(layer_id, self.odp_mu1),
                self.odp_mu_5_6.get(layer_id, self.odp_mu2),
            )
        return self.odp_mu1, self.odp_mu2

    # -----------------------------
    # Tokenization / sampling
    # -----------------------------
    def tokenize(self, text: str):
        enc = self.tokenizer(text, return_tensors="pt")
        input_id = enc.input_ids.to(self.dev)

        input_ids = [input_id[0] for _ in range(self.beam_width)]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.dev)

        position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.long, device=self.dev)
        position_ids = position_ids.unsqueeze(0).view(-1, input_ids.shape[-1])
        return input_ids, position_ids

    def _sample_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        cfg = self.sampling_config
        if not cfg.do_sample:
            return torch.argmax(logits, dim=-1)

        if cfg.temperature and cfg.temperature != 1.0:
            logits = logits / cfg.temperature

        if cfg.top_k > 0:
            top_k = min(cfg.top_k, logits.size(-1))
            thr = torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
            logits = logits.masked_fill(logits < thr, float("-inf"))

        if cfg.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            probs = F.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(probs, dim=-1)

            remove = cum > cfg.top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False

            remove = remove.scatter(dim=-1, index=sorted_idx, src=remove)
            logits = logits.masked_fill(remove, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    # -----------------------------
    # Expert load / execute helpers
    # -----------------------------
    def _load_expert(self, layer_id: int, expert_id: int):
        path = f"{self.expert_path}/layer{layer_id}_expert{expert_id}.pt"
        return torch.load(path, map_location="cuda", weights_only=False)

    def _run_self_attention(self, layer, hidden_states, position_ids, position_embeddings=None, cache_position=None, output_attentions=False, use_cache=True):
        if position_embeddings is None:
            cos, sin = self.model.rotary_emb(hidden_states, position_ids)
            position_embeddings = (cos.to(self.dtype), sin.to(self.dtype))

        out = layer.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            past_key_value=self.past_key_value if use_cache else None,
            use_cache=use_cache,
            attention_mask=None,
            output_attentions=output_attentions,
            cache_position=cache_position if cache_position is not None else position_ids,
        )
        return out

    def _execute_prefill(self, layer_id, hidden_states_flat, selected_experts, selected_weights):
        """
        vectorized token-expert dispatch (prefill):
          hidden_states_flat: (N_tokens, hidden_dim)
          selected_experts:   (N_tokens, K)
          selected_weights:   (N_tokens, K)  already normalized
        """
        out = torch.zeros_like(hidden_states_flat, device=self.dev, dtype=self.dtype)

        for e in range(self.n_expert):
            mask = (selected_experts == e)
            idxs = mask.nonzero(as_tuple=False)
            if idxs.numel() == 0:
                continue

            expert_layer = self._load_expert(layer_id, e)

            if e in selected_experts[-1]:
                self.expert_cache.add(layer_id, e, expert_layer)

            token_idxs = idxs[:, 0]
            topk_idxs = idxs[:, 1]
            x = hidden_states_flat[token_idxs].to(self.dev, dtype=self.dtype)
            w = selected_weights[token_idxs, topk_idxs, None].to(self.dev, dtype=self.dtype)
            y = expert_layer(x) * w
            out.index_add_(0, token_idxs.to(self.dev, non_blocking=True), y.to(self.dev, non_blocking=True))

        torch.cuda.empty_cache()
        return out

    def _execute_decode(self, layer_id, hidden_states_flat, experts_list, weights_list):
        """
        decode stage: one token (or a few) → sequential experts
        experts_list: List[int]
        weights_list: List[float] or 1D tensor
        """
        out = torch.zeros_like(hidden_states_flat, device=self.dev, dtype=self.dtype)

        for e, w in zip(experts_list, weights_list):
            self.cnt_expert_all += 1
            self.cnt_expert_all_by_layer[layer_id] += 1

            expert_layer = self.expert_cache.get(layer_id, e)
            if expert_layer is None:
                expert_layer = self._load_expert(layer_id, e)
                if self.expert_cache.is_cache_full(layer_id):
                    self.expert_cache.replace(layer_id, e, expert_layer)
                else:
                    self.expert_cache.add(layer_id, e, expert_layer)
                self.cnt_expert_miss += 1
                self.cnt_expert_miss_by_layer[layer_id] += 1
            else:
                self.cnt_expert_hit += 1
                self.cnt_expert_hit_by_layer[layer_id] += 1

            out = out + expert_layer(hidden_states_flat) * torch.as_tensor(w, device=self.dev, dtype=self.dtype)

        return out

    # -----------------------------
    # Strategy: expert selection
    # -----------------------------
    def _select_vanilla(self, layer_id, router_logits, is_prefill):
        w = F.softmax(router_logits, dim=1, dtype=torch.float)
        topw, tope = torch.topk(w, k=self.n_routing_expert, dim=-1)
        topw = topw / (topw.sum(dim=-1, keepdim=True) + 1e-20)
        return tope, topw, None  # (selected_experts, selected_weights, meta)

    def _select_des(self, layer_id, router_logits, is_prefill):
        """
        DES: compute top-6 then prune to {4,5,6} by ratio thresholds.
        """
        w = F.softmax(router_logits, dim=1, dtype=torch.float)
        topk = 6
        topw, tope = torch.topk(w, k=topk, dim=-1)
        topw = topw / (topw.sum(dim=-1, keepdim=True) + 1e-20)

        # sort by weight ranks
        order = torch.argsort(topw, dim=-1, descending=True)
        w_sorted = torch.gather(topw, 1, order)

        r4, r5, r6 = w_sorted[:, 3], w_sorted[:, 4], w_sorted[:, 5]
        ratio1 = r4 / (r5 + 1e-6)
        ratio2 = r5 / (r6 + 1e-6)

        mu1, mu2 = self._get_mu(layer_id, "des")

        keep_sorted = torch.ones_like(w_sorted, dtype=torch.bool)
        cond1 = ratio1 > float(mu1)
        keep_sorted[cond1, 4] = False
        keep_sorted[cond1, 5] = False

        cond2 = (~cond1) & (ratio2 > float(mu2))
        keep_sorted[cond2, 5] = False

        keep_mask = torch.zeros_like(keep_sorted, dtype=torch.bool)
        keep_mask.scatter_(1, order, keep_sorted)

        masked = topw * keep_mask.to(topw.dtype)
        den = masked.sum(dim=-1, keepdim=True)
        masked = torch.where(den <= 0, topw, masked)
        masked = masked / (masked.sum(dim=-1, keepdim=True) + 1e-20)

        kept_k = keep_mask.sum(dim=-1)
        self.des_k_hist[4] += int((kept_k == 4).sum().item())
        self.des_k_hist[5] += int((kept_k == 5).sum().item())
        self.des_k_hist[6] += int((kept_k == 6).sum().item())

        return tope, masked, keep_mask

    def _select_mocce(self, layer_id, router_logits, is_prefill):
        """
        MOCCE: decode-only cache-prior logit shaping (Top-J guarantee + cached promotion).
        Prefill: vanilla top-k.
        """
        z = router_logits
        z_select = z

        if not is_prefill:
            # Δ_avg tracking (range)
            cur_range = (z.max() - z.min()).item()
            tr = self.cache_delta_tracker[layer_id]
            if not tr["init"]:
                tr["avg"] = cur_range
                tr["init"] = True
            else:
                tr["avg"] = 0.99 * tr["avg"] + 0.01 * cur_range
            delta_avg = tr["avg"]

            mask = torch.zeros_like(z)

            # Top-J guarantee
            _, topJ = torch.topk(z, self.top_J, dim=1)
            mask.scatter_(1, topJ, 1.0)

            # Cached promotion
            cached = self.expert_cache.get_cached_expert_ids(layer_id)
            if cached:
                mask[:, cached] = 1.0

            z_select = z + self.lambda_cache * delta_avg * mask

        w_raw = F.softmax(z, dim=1, dtype=torch.float)
        w_sel = F.softmax(z_select, dim=1, dtype=torch.float)

        _, tope = torch.topk(w_sel, k=self.n_routing_expert, dim=-1)

        # decode: async arithmetic update
        if not is_prefill:
            self.executor.submit(self.expert_cache.update_arithmetic, layer_id, tope.tolist())

        topw = torch.gather(w_raw, 1, tope)
        topw = topw / (topw.sum(dim=-1, keepdim=True) + 1e-20)
        return tope, topw, None

    def _select_cecar(self, layer_id, router_logits, is_prefill):
        """
        CECAR (your code): decode-only cache-aware ranking with eviction-score bonus.
        Prefill: vanilla top-k.
        """
        w_raw = F.softmax(router_logits, dim=1, dtype=torch.float)

        if is_prefill:
            tope, topw, _ = self._select_vanilla(layer_id, router_logits, is_prefill=True)
            return tope, topw, None

        # delta margin
        topk_logits, _ = torch.topk(router_logits[0], self.n_expert)
        z_top1 = topk_logits[0].item()
        z_topk = topk_logits[self.margin_topk_idx].item()
        delta = z_top1 - z_topk
        self.delta_records.append(delta)

        z_select = router_logits.clone()
        cached_tensor = self.expert_cache.get_cached_expert_ids_tensor(layer_id)
        if cached_tensor.numel() > 0:
            scores = self.expert_cache.get_eviction_scores_batch(layer_id, cached_tensor)
            max_score = scores.max()
            alpha = 1.0 - scores / (max_score + 1e-5)
            bonus = delta * alpha
            z_select[0, cached_tensor] += bonus
            self.alpha_records.extend(alpha.tolist())

        ranking = torch.argsort(z_select[0], descending=True).tolist()
        ranking_w = w_raw[0, torch.as_tensor(ranking, device=w_raw.device)].tolist()

        cached_set = set(self.expert_cache.get_cached_expert_ids(layer_id))
        kept_e, kept_w = [], []

        # rule: rank < topk_threshold always; otherwise only cached; stop at 8
        for r, e in enumerate(ranking):
            if r < self.topk_threshold or (e in cached_set):
                kept_e.append(e)
                kept_w.append(ranking_w[r])
            if len(kept_e) == 8:
                break

        w = torch.tensor(kept_w, device=self.dev, dtype=self.dtype)
        w = w / (w.sum() + 1e-12)

        self.routed_expert_counts.append(len(kept_e))
        self.executor.submit(self.expert_cache.update_arithmetic, layer_id, [kept_e])

        # return in "topk-style" format for unified executor
        tope = torch.tensor([kept_e], device=self.dev, dtype=torch.long)
        topw = w.view(1, -1).to(torch.float)
        return tope, topw, None

    def _select_odp(self, layer_id, router_logits, is_prefill, protection_mask=None):
        """
        ODP: top-6 ratio pruning with protection_mask (True = protected, do not prune).
        protection_mask is token-level (N_tokens,) for prefill.
        """
        w = F.softmax(router_logits, dim=1, dtype=torch.float)
        topk = 6
        topw, tope = torch.topk(w, k=topk, dim=-1)
        topw = topw / (topw.sum(dim=-1, keepdim=True) + 1e-20)

        keep_mask = torch.ones_like(topw, dtype=torch.bool)
        mu1, mu2 = self._get_mu(layer_id, "odp")

        r4, r5, r6 = topw[:, 3], topw[:, 4], topw[:, 5]
        ratio1 = r4 / (r5 + 1e-6)
        ratio2 = r5 / (r6 + 1e-6)

        if protection_mask is None:
            protection_mask = torch.zeros((topw.shape[0],), dtype=torch.bool, device=topw.device)

        cond1 = (ratio1 > float(mu1)) & (~protection_mask)
        keep_mask[cond1, 4] = False
        keep_mask[cond1, 5] = False

        cond2 = (~cond1) & (ratio2 > float(mu2)) & (~protection_mask)
        keep_mask[cond2, 5] = False

        masked = topw * keep_mask.to(topw.dtype)
        masked = masked / (masked.sum(dim=-1, keepdim=True) + 1e-20)

        # decode stat
        if not is_prefill:
            kept_k = keep_mask.sum(dim=-1)
            for kk in (4, 5, 6):
                self.odp_k_hist[kk] += int((kept_k == kk).sum().item())

        return tope, masked, keep_mask

    # -----------------------------
    # Forward core (unified)
    # -----------------------------
    @torch.no_grad()
    def forward_step(self, input_ids, position_ids, is_prefill: bool):
        """
        One forward pass over all layers for current token(s).
        Strategy chosen by (mode, bonus_strategy).
        Returns: lm_logits (B, T, vocab)
        """
        hidden_states = self.model.embed_tokens(input_ids.to(self.dev)).to(self.dtype)

        for layer_id, layer in enumerate(self.model.layers):
            orig_shape = hidden_states.shape
            residual = hidden_states

            # ---- Self-attn ----
            hidden_states = layer.input_layernorm(hidden_states)

            attn_out, _attn_dbg = None, None
            if self.mode == "odp" and is_prefill:
                try:
                    attn_out_dbg = self._run_self_attention(
                        layer, hidden_states, position_ids,
                        output_attentions=True, use_cache=False
                    )
                    attn_out, attn_w = attn_out_dbg[0], attn_out_dbg[1]
                    _attn_dbg = attn_w if (attn_w is not None and attn_w.dim() == 4) else None
                except Exception:
                    _attn_dbg = None

            attn_out_main = self._run_self_attention(
                layer, hidden_states, position_ids,
                output_attentions=False, use_cache=True
            )
            hidden_states = attn_out_main[0]
            hidden_states = residual + hidden_states

            # ---- MoE ----
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states_flat = hidden_states.view(-1, self.hidden_dim)

            router_logits = layer.mlp.gate(hidden_states_flat)

            # select experts based on current strategy
            if self.mode == "des":
                selected_experts, selected_weights, keep_mask = self._select_des(layer_id, router_logits, is_prefill)
            elif self.mode == "odp":
                protection_mask = None
                if is_prefill:
                    protection_mask = self._build_odp_protection_mask(hidden_states_flat, orig_shape, _attn_dbg)
                selected_experts, selected_weights, keep_mask = self._select_odp(layer_id, router_logits, is_prefill, protection_mask)
            else:
                if self.bonus_strategy == "mocce":
                    selected_experts, selected_weights, keep_mask = self._select_mocce(layer_id, router_logits, is_prefill)
                elif self.bonus_strategy == "cecar":
                    selected_experts, selected_weights, keep_mask = self._select_cecar(layer_id, router_logits, is_prefill)
                else:
                    selected_experts, selected_weights, keep_mask = self._select_vanilla(layer_id, router_logits, is_prefill)

            if (self.mode == "none") and (self.bonus_strategy == "none") and (not is_prefill):
                fut = self.executor.submit(self.expert_cache.update_arithmetic, layer_id, selected_experts.tolist())
                _ = fut.result()

            if is_prefill:
                # keep_mask exists for DES/ODP: mask out dropped experts
                if keep_mask is not None:
                    selected_weights = selected_weights * keep_mask.to(selected_weights.dtype)
                    selected_weights = selected_weights / (selected_weights.sum(dim=-1, keepdim=True) + 1e-20)

                hs_after = self._execute_prefill(layer_id, hidden_states_flat, selected_experts, selected_weights)
            else:
                # decode: (assume batch=1, token=1)
                if keep_mask is not None:
                    kept_idx = keep_mask[0].nonzero(as_tuple=True)[0]
                    experts = selected_experts[0, kept_idx].tolist()
                    weights = selected_weights[0, kept_idx].tolist()
                else:
                    experts = selected_experts[0].tolist()
                    weights = selected_weights[0].tolist()

                self.routed_expert_counts.append(len(experts))
                hs_after = self._execute_decode(layer_id, hidden_states_flat, experts, weights)

            hidden_states = residual + hs_after.reshape(orig_shape)

        hidden_states = self.model.norm(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        return lm_logits

    def _build_odp_protection_mask(self, hidden_states_flat, orig_shape, attn_weights_4d):
        """
        Your ODP protection (McMoE-like):
          score_j = ||t_j||_1 * attention_importance(j)
        Return:
          protection_mask (N_tokens,) boolean
        """
        batch, seq_len = orig_shape[0], orig_shape[1]
        l1_norm = torch.norm(hidden_states_flat, p=1, dim=1).to(torch.float).view(batch, seq_len)

        score = l1_norm
        if attn_weights_4d is not None:
            attn = attn_weights_4d.to(torch.float)  # [B,H,L,L]
            A = attn.mean(dim=1)                    # [B,L,L]
            causal = torch.tril(torch.ones((seq_len, seq_len), device=A.device, dtype=A.dtype))
            A = A * causal.unsqueeze(0)
            incoming_sum = A.sum(dim=1)             # [B,L]
            denom = torch.arange(seq_len, 0, -1, device=A.device, dtype=A.dtype)
            incoming_avg = incoming_sum / denom.unsqueeze(0)
            score = l1_norm * incoming_avg

        protection_ratio = 0.02
        k = max(1, int(seq_len * protection_ratio))
        top_idx = torch.topk(score, k, dim=1, largest=True, sorted=False).indices
        mask = torch.zeros_like(score, dtype=torch.bool)
        mask.scatter_(1, top_idx, True)
        return mask.view(-1)

    # -----------------------------
    # Generate loop
    # -----------------------------
    def generate(self, text=None, output_token=2048, input_token=None):
        self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
        self.past_key_values_length = 0
        self._reset_stats()

        input_ids, position_ids = self.tokenize(text)
        if input_token is not None:
            input_ids = input_ids[:, :input_token]
            position_ids = position_ids[:, :input_token]

        len_token = input_ids.shape[1]
        token_log_f = open("Results/generated_tokens.txt", "a", encoding="utf-8")

        tick = time.time()
        is_prefill = True
        prefill_time = 0.0

        for i_token in range(output_token):
            if self.beam_width == 1:
                token_str = self.tokenizer.decode(input_ids[0])
                print(token_str, end="", flush=True)
                token_log_f.write(token_str)
                token_log_f.flush()

            logits = self.forward_step(input_ids, position_ids, is_prefill=is_prefill)
            is_prefill = False

            logits_cpu = logits.to("cpu")
            next_id = self._sample_from_logits(logits_cpu[:, -1, :])

            stop_ids = {151645, 151643}  # <|im_end|>, <|endoftext|> 
            if all(int(t) in stop_ids for t in next_id.tolist()):
                if self.beam_width == 1:
                    eos_str = self.tokenizer.decode(next_id[0].view(1))
                    print(eos_str, end="", flush=True)
                    token_log_f.write(eos_str)
                    token_log_f.flush()
                    print("\n")
                break

            self.past_key_values_length += logits.shape[1]
            input_ids = next_id.view(-1, 1).to(self.dev)

            position_ids = torch.arange(
                self.past_key_values_length,
                self.past_key_values_length + 1,
                device=self.dev
            ).unsqueeze(0).view(-1, 1)

            if i_token == 0:
                prefill_time += time.time() - tick
                tick = time.time()

        decode_time = time.time() - tick
        token_log_f.write("\n\n\nNext Prompt : \n")
        token_log_f.close()

        ratios = [
            (hit / all_ if all_ != 0 else 0)
            for hit, all_ in zip(self.cnt_expert_hit_by_layer, self.cnt_expert_all_by_layer)
        ]

        avg_delta = sum(self.delta_records) / len(self.delta_records) if self.delta_records else 0.0
        avg_alpha = sum(self.alpha_records) / len(self.alpha_records) if self.alpha_records else 0.0

        print("\n================ Cache Bonus Statistics ================")
        print(f"Average delta: {avg_delta:.6f}")
        print(f"Average alpha: {avg_alpha:.6f}")
        print("========================================================\n")

        return (
            prefill_time,
            decode_time,
            (len_token / prefill_time) if prefill_time > 0 else 0.0,
            (i_token / decode_time) if decode_time > 0 else 0.0,
            (self.cnt_expert_hit / self.cnt_expert_all) if self.cnt_expert_all > 0 else 0.0,
            ratios,
        )
