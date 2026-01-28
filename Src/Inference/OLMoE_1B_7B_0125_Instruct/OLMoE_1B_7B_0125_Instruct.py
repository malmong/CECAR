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


class FiddlerOlmoe:
    """
    OLMoE inference wrapper with:
      - Expert cache (LRU/LFU/ML policies via CachePolicyWrapper)
      - Expert selection strategies:
          vanilla / cecar / mocce / des / odp
      - Prefill/Decode timing + cache hit/miss statistics

    Note:
      - OLMoE in your implementation uses *unnormalized* routing weights for MoE combine
        (i.e., do NOT force sum-to-1 after top-k).
    """

    # -----------------------------
    # Init
    # -----------------------------
    def __init__(self, args):
        self.dtype = torch.bfloat16
        self.dev = torch.device("cuda")

        base = transformers.OlmoeForCausalLM.from_pretrained(
            args.model,
            torch_dtype=self.dtype,
            device_map="cuda",
            use_cache=True,
        )
        self.lm_head = base.lm_head
        self.model = base.model

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)

        self.expert_path = args.expert_path
        self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
        self.past_key_values_length = 0

        self.n_layer = len(self.model.layers)
        self.n_expert = len(self.model.layers[0].mlp.experts)
        self.n_routing_expert = self.model.config.num_experts_per_tok
        self.hidden_dim = self.model.config.hidden_size

        self.beam_width = getattr(args, "beam_width", 1)

        # Cache wrapper
        self.expert_cache = CachePolicyWrapper(
            cache_size=args.cache_size,
            cache_policy=args.cache_policy,
            n_layers=self.n_layer,
            expert_path=self.expert_path,
            ffn_model_path=getattr(args, "ffn_model_path", ""),
            num_experts=self.n_expert,
        )

        # Thread executor for cache arithmetic update
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        # Stats
        self._reset_stats()
        self.cnt_hitrate_by_time_and_layer = [[] for _ in range(self.n_layer)]

        # Bonus stats
        self.alpha_records = []
        self.delta_records = []

        # MOCCE params
        self.lambda_cache = 0.2
        self.top_J = 2
        self.cache_delta_tracker = [{"avg": 0.0, "init": False} for _ in range(self.n_layer)]

        # CECAR params
        self.margin_topk_idx = getattr(args, "margin_topk_idx", 8)
        self.topk_threshold = getattr(args, "topk_threshold", 4)  

        # Strategy controls
        self.bonus_strategy = getattr(args, "bonus_strategy", "none")  # cecar/mocce/none
        self.mode = getattr(args, "mode", "none")  # des/odp/none
        if self.mode in ("des", "odp"):
            self.bonus_strategy = "none"

        # DES thresholds
        self.des_mu1 = getattr(args, "des_mu1", 1.5)
        self.des_mu2 = getattr(args, "des_mu2", 1.5)
        self.des_layerwise = False
        self.des_mu_4_5 = None
        self.des_mu_5_6 = None
        self.des_thresholds_path = getattr(args, "des_thresholds_path", None)
        if self.des_thresholds_path:
            self._load_thresholds(self.des_thresholds_path, kind="des")
        self.des_k_hist = {4: 0, 5: 0, 6: 0}

        # ODP thresholds
        self.odp_mu1 = getattr(args, "odp_mu1", None) or self.des_mu1
        self.odp_mu2 = getattr(args, "odp_mu2", None) or self.des_mu2
        self.odp_layerwise = False
        self.odp_mu_4_5 = None
        self.odp_mu_5_6 = None
        self.odp_thresholds_path = getattr(args, "odp_thresholds_path", None) or self.des_thresholds_path
        if self.odp_thresholds_path and self.mode == "odp":
            self._load_thresholds(self.odp_thresholds_path, kind="odp")
        self.odp_k_hist = {4: 0, 5: 0, 6: 0}

        # Sampling
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
    # Threshold I/O
    # -----------------------------
    def _load_thresholds(self, path: str, kind: str):
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

        # global: {"mu1": x, "mu2": y}
        if isinstance(obj2, dict) and ("mu1" in obj2) and ("mu2" in obj2):
            mu1, mu2 = float(obj2["mu1"]), float(obj2["mu2"])
            if kind == "des":
                self.des_mu1, self.des_mu2 = mu1, mu2
                self.des_layerwise = False
            else:
                self.odp_mu1, self.odp_mu2 = mu1, mu2
                self.odp_layerwise = False
            return

        # layerwise: {"layers":[{"layer":0,"mu_4_5":x,"mu_5_6":y}, ...]}
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
            cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            remove = cum > cfg.top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False

            remove = remove.scatter(dim=-1, index=sorted_idx, src=remove)
            logits = logits.masked_fill(remove, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    # -----------------------------
    # Expert load / execute
    # -----------------------------
    def _load_expert(self, layer_id: int, expert_id: int):
        path = f"{self.expert_path}/layer{layer_id}_expert{expert_id}.pt"
        return torch.load(path, map_location="cuda", weights_only=False)

    def _execute_prefill(self, layer_id, hidden_flat, selected_experts, selected_weights, keep_mask=None):
        """
        Prefill: group by expert and run expert_layer(x) for all tokens routed to it.
        OLMoE: do NOT normalize selected_weights.
        """
        out = torch.zeros_like(hidden_flat, device=self.dev, dtype=self.dtype)

        if keep_mask is not None:
            selected_weights = selected_weights * keep_mask.to(selected_weights.dtype)

        for e in range(self.n_expert):
            idxs = ((selected_experts == e) & (keep_mask if keep_mask is not None else True)).nonzero(as_tuple=False)
            if idxs.numel() == 0:
                continue

            expert_layer = self._load_expert(layer_id, e)

            if e in selected_experts[-1]:
                self.expert_cache.add(layer_id, e, expert_layer)

            token_idxs = idxs[:, 0]
            topk_idxs = idxs[:, 1]

            x = hidden_flat[token_idxs].to(self.dev, dtype=self.dtype)
            w = selected_weights[token_idxs, topk_idxs, None].to(self.dev, dtype=self.dtype)
            y = expert_layer(x) * w
            out.index_add_(0, token_idxs.to(self.dev, non_blocking=True), y.to(self.dev, non_blocking=True))

        torch.cuda.empty_cache()
        return out

    def _execute_decode(self, layer_id, hidden_flat, experts, weights):
        out = torch.zeros_like(hidden_flat, device=self.dev, dtype=self.dtype)

        cnt_local_hit = 0
        for e, w in zip(experts, weights):
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
                cnt_local_hit += 1

            out = out + expert_layer(hidden_flat) * torch.as_tensor(w, device=self.dev, dtype=self.dtype)

        self.cnt_hitrate_by_time_and_layer[layer_id].append(cnt_local_hit)
        return out

    # -----------------------------
    # Strategy: expert selection
    # -----------------------------
    def _select_vanilla(self, layer_id, router_logits, is_prefill):
        w = F.softmax(router_logits, dim=1, dtype=torch.float)
        topw, tope = torch.topk(w, k=self.n_routing_expert, dim=-1)
        return tope, topw, None

    def _select_mocce(self, layer_id, router_logits, is_prefill):
        z = router_logits
        z_sel = z

        if not is_prefill:
            cur_range = (z.max() - z.min()).item()
            tr = self.cache_delta_tracker[layer_id]
            if not tr["init"]:
                tr["avg"] = cur_range
                tr["init"] = True
            else:
                tr["avg"] = 0.99 * tr["avg"] + 0.01 * cur_range
            delta_avg = tr["avg"]

            mask = torch.zeros_like(z)
            _, topJ = torch.topk(z, self.top_J, dim=1)
            mask.scatter_(1, topJ, 1.0)

            cached = self.expert_cache.get_cached_expert_ids(layer_id)
            if cached:
                mask[:, cached] = 1.0

            z_sel = z + self.lambda_cache * delta_avg * mask

        w_raw = F.softmax(z, dim=1, dtype=torch.float)
        w_sel = F.softmax(z_sel, dim=1, dtype=torch.float)

        _, tope = torch.topk(w_sel, k=self.n_routing_expert, dim=-1)
        if not is_prefill:
            self.executor.submit(self.expert_cache.update_arithmetic, layer_id, tope.tolist())

        topw = torch.gather(w_raw, 1, tope)
        return tope, topw, None

    def _select_cecar(self, layer_id, router_logits, is_prefill):
        """
        cecar:
          - ranking uses bonus-adjusted logits (decode)
          - enforce: topk_threshold always kept; rest kept only if cached; cap=8
          - weights: raw softmax values (no normalize)
        """
        z = router_logits
        z_sel = z.clone()

        # delta
        topk_logits, _ = torch.topk(z[0], self.n_expert)
        delta = (topk_logits[0] - topk_logits[self.margin_topk_idx]).item()
        self.delta_records.append(delta)

        w_raw = F.softmax(z, dim=1, dtype=torch.float)

        if is_prefill:
            _, tope = torch.topk(w_raw, k=self.n_routing_expert, dim=-1)
            topw = torch.gather(w_raw, 1, tope)
            return tope, topw, None

        cached_tensor = self.expert_cache.get_cached_expert_ids_tensor(layer_id)
        if cached_tensor.numel() > 0:
            scores = self.expert_cache.get_eviction_scores_batch(layer_id, cached_tensor)
            max_score = scores.max()
            alpha = 1.0 - scores / (max_score + 1e-5)
            z_sel[0, cached_tensor] += delta * alpha
            self.alpha_records.extend(alpha.tolist())

        ranking = torch.argsort(z_sel[0], descending=True).tolist()
        cached_set = set(self.expert_cache.get_cached_expert_ids(layer_id))

        kept_e, kept_w = [], []
        for rank, e in enumerate(ranking):
            if rank < self.topk_threshold or (e in cached_set):
                kept_e.append(e)
                kept_w.append(w_raw[0, e].item())
            if len(kept_e) == 8:
                break

        self.routed_expert_counts.append(len(kept_e))
        self.executor.submit(self.expert_cache.update_arithmetic, layer_id, [kept_e])

        tope = torch.tensor([kept_e], device=self.dev, dtype=torch.long)
        topw = torch.tensor([kept_w], device=self.dev, dtype=torch.float)
        return tope, topw, None

    def _select_des(self, layer_id, router_logits, is_prefill):
        """
        DES: top-6 + ratio pruning to {4,5,6}.
        weights: top-6 normalized (per your code), then masked + renorm.
        """
        w = F.softmax(router_logits, dim=1, dtype=torch.float)
        topk = 6
        topw, tope = torch.topk(w, k=topk, dim=-1)

        mu1, mu2 = self._get_mu(layer_id, "des")
        order = torch.argsort(topw, dim=-1, descending=True)
        w_sorted = torch.gather(topw, 1, order)

        r4, r5, r6 = w_sorted[:, 3], w_sorted[:, 4], w_sorted[:, 5]
        ratio1 = r4 / (r5 + 1e-6)
        ratio2 = r5 / (r6 + 1e-6)

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


        kept_k = keep_mask.sum(dim=-1)
        self.des_k_hist[4] += int((kept_k == 4).sum().item())
        self.des_k_hist[5] += int((kept_k == 5).sum().item())
        self.des_k_hist[6] += int((kept_k == 6).sum().item())

        return tope, masked, keep_mask

    def _build_odp_protection_mask(self, hidden_flat, orig_shape, attn_weights_4d):
        bsz, seq_len = orig_shape[0], orig_shape[1]
        l1 = torch.norm(hidden_flat, p=1, dim=1).to(torch.float).view(bsz, seq_len)

        score = l1
        if attn_weights_4d is not None:
            attn = attn_weights_4d.to(torch.float)  # [B,H,L,L]
            A = attn.mean(dim=1)                    # [B,L,L]
            causal = torch.tril(torch.ones((seq_len, seq_len), device=A.device, dtype=A.dtype))
            A = A * causal.unsqueeze(0)
            incoming_sum = A.sum(dim=1)             # [B,L]
            denom = torch.arange(seq_len, 0, -1, device=A.device, dtype=A.dtype)
            incoming_avg = incoming_sum / denom.unsqueeze(0)
            score = l1 * incoming_avg

        ratio = 0.02
        k = max(1, int(seq_len * ratio))
        top_idx = torch.topk(score, k, dim=1, largest=True, sorted=False).indices
        mask = torch.zeros_like(score, dtype=torch.bool)
        mask.scatter_(1, top_idx, True)
        return mask.view(-1)

    def _select_odp(self, layer_id, router_logits, is_prefill, protection_mask=None):
        """
        ODP: top-6 + ratio pruning, but skip pruning for protected tokens in prefill.
        weights: top-6 normalized then masked + renorm (your current behavior).
        """
        w = F.softmax(router_logits, dim=1, dtype=torch.float)
        topk = 6
        topw, tope = torch.topk(w, k=topk, dim=-1)

        keep_mask = torch.ones_like(topw, dtype=torch.bool)

        if protection_mask is None:
            protection_mask = torch.zeros((topw.shape[0],), dtype=torch.bool, device=topw.device)

        mu1, mu2 = self._get_mu(layer_id, "odp")

        r4, r5, r6 = topw[:, 3], topw[:, 4], topw[:, 5]
        ratio1 = r4 / (r5 + 1e-6)
        ratio2 = r5 / (r6 + 1e-6)

        cond1 = (ratio1 > float(mu1)) & (~protection_mask)
        keep_mask[cond1, 4] = False
        keep_mask[cond1, 5] = False

        cond2 = (~cond1) & (ratio2 > float(mu2)) & (~protection_mask)
        keep_mask[cond2, 5] = False

        masked = topw * keep_mask.to(topw.dtype)
        den = masked.sum(dim=-1, keepdim=True)
        masked = torch.where(den <= 0, topw, masked)

        if not is_prefill:
            kept_k = keep_mask.sum(dim=-1)
            for kk in (4, 5, 6):
                self.odp_k_hist[kk] += int((kept_k == kk).sum().item())

        return tope, masked, keep_mask

    # -----------------------------
    # Unified forward (core)
    # -----------------------------
    @torch.no_grad()
    def forward_step(self, input_ids, position_ids, is_prefill: bool):
        hidden_states = self.model.embed_tokens(input_ids.to(self.dev))
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

        for layer_id, layer in enumerate(self.model.layers):
            orig_shape = hidden_states.shape
            residual = hidden_states

            hidden_states = layer.input_layernorm(hidden_states)

            # ODP: capture attention weights only in prefill
            attn_weights = None
            if self.mode == "odp" and is_prefill:
                try:
                    dbg = layer.self_attn(
                        hidden_states,
                        position_embeddings=position_embeddings,
                        past_key_value=None,
                        use_cache=False,
                        output_attentions=True,
                    )
                    if isinstance(dbg, tuple) and len(dbg) >= 2 and dbg[1] is not None and dbg[1].dim() == 4:
                        attn_weights = dbg[1]
                except Exception:
                    attn_weights = None

            hidden_states, _, present_kv = layer.self_attn(
                hidden_states,
                position_embeddings=position_embeddings,
                past_key_value=self.past_key_value,
                use_cache=True,
            )

            hidden_states = residual + hidden_states
            residual = hidden_states

            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_flat = hidden_states.view(-1, self.hidden_dim)

            router_logits = layer.mlp.gate(hidden_flat)

            # ---- Select experts ----
            if self.mode == "des":
                tope, topw, keep_mask = self._select_des(layer_id, router_logits, is_prefill)
            elif self.mode == "odp":
                pmask = None
                if is_prefill:
                    pmask = self._build_odp_protection_mask(hidden_flat, orig_shape, attn_weights)
                tope, topw, keep_mask = self._select_odp(layer_id, router_logits, is_prefill, pmask)
            else:
                if self.bonus_strategy == "mocce":
                    tope, topw, keep_mask = self._select_mocce(layer_id, router_logits, is_prefill)
                elif self.bonus_strategy == "cecar":
                    tope, topw, keep_mask = self._select_cecar(layer_id, router_logits, is_prefill)
                else:
                    tope, topw, keep_mask = self._select_vanilla(layer_id, router_logits, is_prefill)

            # vanilla decode
            if self.mode == "none" and self.bonus_strategy == "none" and (not is_prefill):
                self.executor.submit(self.expert_cache.update_arithmetic, layer_id, tope.tolist())

            # ---- Execute experts ----
            hs_after = torch.zeros_like(hidden_flat, device=self.dev, dtype=self.dtype)

            if is_prefill:
                hs_after = self._execute_prefill(layer_id, hidden_flat, tope, topw, keep_mask)
            else:
                if keep_mask is not None:
                    kept_idx = keep_mask[0].nonzero(as_tuple=True)[0]
                    experts = tope[0, kept_idx].tolist()
                    weights = topw[0, kept_idx].tolist()
                else:
                    experts = tope[0].tolist()
                    weights = topw[0].tolist()

                self.routed_expert_counts.append(len(experts))
                if self.mode in ("des", "odp") or self.bonus_strategy in ("cecar",):
                    self.executor.submit(self.expert_cache.update_arithmetic, layer_id, [experts])

                hs_after = self._execute_decode(layer_id, hidden_flat, experts, weights)

            hidden_states = residual + hs_after.reshape(orig_shape)

        hidden_states = self.model.norm(hidden_states)
        lm_logits = self.lm_head(hidden_states)

        self.present_key_value = present_kv
        return lm_logits

    # -----------------------------
    # Generate
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

        os.makedirs("Results", exist_ok=True)
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

            stop_ids = {50279}  # |||IP_ADDRESS|||
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
                device=self.dev,
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
