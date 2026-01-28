"""
FiddlerDeepseek: DeepSeek MoE inference with expert offload + cache policy simulation.

This file is organized for paper/code submission:
- minimal duplication
- explicit strategy definitions
- deterministic and reproducible behavior (as much as possible)
- clear separation between:
    (1) attention + residual path
    (2) dense MLP path
    (3) MoE gating + expert execution path
    (4) routing strategies (CECAR / MOCCE / DES / ODP)

Assumptions:
- batch_size is usually 1 in your DeepSeek pipeline (but code is written for bsz>=1 where possible)
- experts are stored on disk as:
    {expert_path}/layer{L}_expert{E}.pt
  and shared expert:
    {shared_expert_path}/layer{L}_shared_expert.pt
"""

from __future__ import annotations

import os
import json
import time
import warnings
import concurrent.futures
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import transformers
from transformers.cache_utils import DynamicCache

from utils_cache import CachePolicyWrapper


# -----------------------------------------------------------------------------
# Logging / warnings cleanup
# -----------------------------------------------------------------------------
transformers.utils.logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=r"The module name .* is not a valid Python identifier.*")
warnings.filterwarnings("ignore", message=r"Some weights of .* were not initialized from the model checkpoint.*")
warnings.filterwarnings("ignore", message=r"`torch_dtype` is deprecated! Use `dtype` instead!")


# -----------------------------------------------------------------------------
# DeepSeek remote code compatibility patch (DynamicCache API)
# -----------------------------------------------------------------------------
if not hasattr(DynamicCache, "get_usable_length"):
    def _get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)
    DynamicCache.get_usable_length = _get_usable_length


# -----------------------------------------------------------------------------
# Small dataclasses
# -----------------------------------------------------------------------------
@dataclass
class SamplingConfig:
    """Sampling configuration for text generation."""
    temperature: float = 0.7
    top_k: int = 0
    top_p: float = 0.95
    do_sample: bool = False


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def disable_internal_experts(base_model: torch.nn.Module) -> None:
    """
    DeepSeek remote code often includes internal expert modules in the model graph.
    This fiddler loads experts from disk and uses an external cache policy, so we
    disable any internal expert containers to avoid GPU memory overhead.
    """
    for layer in getattr(base_model, "layers", []):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        if hasattr(mlp, "experts"):
            mlp.experts = torch.nn.ModuleList()
        if hasattr(mlp, "shared_experts"):
            mlp.shared_experts = torch.nn.Identity()
    torch.cuda.empty_cache()


def build_4d_additive_causal_mask(
    attn2d: torch.Tensor,
    q_len: int,
    kv_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    attn2d: (B, kv_len) with 1 for valid tokens, 0 for pads
    returns: (B, 1, q_len, kv_len) additive mask (0 or -inf)
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


# -----------------------------------------------------------------------------
# Main class
# -----------------------------------------------------------------------------
class FiddlerDeepseek:
    """
    Supported routing modes:
      - mode == 'des' : DES forward (dynamic k=4/5/6)
      - mode == 'odp' : ODP forward (DES + protection in prefill)
      - else bonus_strategy in {'cecar','mocce','none'}:
          * cecar: cache-aware bonus scoring (v2)
          * mocce: cache promotion mask (lambda_cache * delta_avg)
          * none : vanilla gate output
    """

    # -------------------------
    # Init
    # -------------------------
    def __init__(self, args: Any):
        self.dtype = torch.bfloat16
        self.dev = torch.device("cuda")

        # ---- load model
        kwargs = dict(device_map="cuda", trust_remote_code=True, use_cache=True)
        try:
            full_model = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=self.dtype, **kwargs)
        except TypeError:
            full_model = transformers.AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=self.dtype, **kwargs)

        self.lm_head = getattr(full_model, "lm_head", None)
        self.model = getattr(full_model, "model", None)

        disable_internal_experts(self.model)

        # ---- tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id

        # ---- paths
        self.expert_path = args.expert_path
        self.shared_expert_path = args.shared_expert_path

        # ---- kv cache
        self.past_key_value = DynamicCache()
        self.past_key_values_length = 0

        # ---- model meta
        self.n_layer = len(self.model.layers)
        self.hidden_dim = self.model.config.hidden_size
        self.n_expert = int(getattr(self.model.config, "n_routed_experts", 0))
        self.n_routing_expert = int(getattr(self.model.config, "num_experts_per_tok", 0))
        self.n_shared_experts = int(getattr(self.model.config, "n_shared_experts", 0) or 0)

        self.beam_width = int(getattr(args, "beam_width", 1))

        # ---- stats
        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0
        self.cnt_expert_miss = 0
        self.cnt_expert_hit_by_layer = [0 for _ in range(self.n_layer)]
        self.cnt_expert_all_by_layer = [0 for _ in range(self.n_layer)]
        self.cnt_expert_miss_by_layer = [0 for _ in range(self.n_layer)]

        # Routed expert count tracking
        self.routed_expert_counts: List[int] = []

        # ---- cache policy wrapper
        self.expert_cache = CachePolicyWrapper(
            cache_size=getattr(args, "cache_size", 16),
            cache_policy=getattr(args, "cache_policy", "lru"),
            n_layers=self.n_layer,
            expert_path=getattr(args, "eviction_path", ""),
            ffn_model_path=getattr(args, "ffn_model_path", ""),
            num_experts=self.n_expert,
        )

        # ---- shared expert modules cache (GPU-resident after first load)
        self.shared_expert_modules: Dict[int, torch.nn.Module] = {}

        # disable Linear.reset_parameters for speed/stability in some remote codes
        torch.nn.Linear.reset_parameters = lambda self: None

        # thread executor for async cache arithmetic
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        # Records for paper analysis
        self.routing_record: List[Any] = []
        self.alpha_records: List[float] = []
        self.delta_records: List[float] = []

        # MOCCE params
        self.lambda_cache = float(getattr(args, "lambda_cache", 0.2))
        self.top_J = int(getattr(args, "top_J", 2))
        self.cache_delta_tracker = [{"avg": 0.0, "init": False} for _ in range(self.n_layer)]

        # CECAR params
        self.margin_topk_idx = int(getattr(args, "margin_topk_idx", 6))
        self.topk_threshold = int(getattr(args, "topk_threshold", 3))  

        # Strategy selection
        self.bonus_strategy = getattr(args, "bonus_strategy", "none")
        self.mode = getattr(args, "mode", "none")
        if self.bonus_strategy in ("des", "odp"):
            self.mode = self.bonus_strategy
            self.bonus_strategy = "none"

        # DES params
        self.des_mu1 = float(getattr(args, "des_mu1", 1.5))
        self.des_mu2 = float(getattr(args, "des_mu2", 1.5))
        self.des_layerwise = False
        self.des_mu_4_5 = None
        self.des_mu_5_6 = None
        self.des_thresholds_path = getattr(args, "des_thresholds_path", None)
        if self.des_thresholds_path:
            self._load_des_thresholds_from_file(self.des_thresholds_path)
        self.des_k_hist = {4: 0, 5: 0, 6: 0}

        # ODP params
        self.odp_mu1 = float(getattr(args, "odp_mu1", self.des_mu1))
        self.odp_mu2 = float(getattr(args, "odp_mu2", self.des_mu2))
        self.odp_layerwise = False
        self.odp_mu_4_5 = None
        self.odp_mu_5_6 = None
        self.odp_thresholds_path = getattr(args, "odp_thresholds_path", None) or self.des_thresholds_path
        if self.odp_thresholds_path and self.mode == "odp":
            self._load_odp_thresholds_from_file(self.odp_thresholds_path)
        self.odp_k_hist = {4: 0, 5: 0, 6: 0}

        # Sampling
        self.sampling_config = SamplingConfig(
            temperature=float(getattr(args, "temperature", 0.7)),
            top_k=int(getattr(args, "top_k", 0)),
            top_p=float(getattr(args, "top_p", 0.95)),
            do_sample=bool(getattr(args, "do_sample", False)),
        )

        print("Model is ready.")
        if self.mode == "des":
            if self.des_layerwise:
                print(f"  - DES thresholds: layerwise (loaded from {self.des_thresholds_path})")
            else:
                print(f"  - DES thresholds: mu1={self.des_mu1}, mu2={self.des_mu2}")
        elif self.mode == "odp":
            if self.odp_layerwise:
                print(f"  - ODP thresholds: layerwise (loaded from {self.odp_thresholds_path})")
            else:
                print(f"  - ODP thresholds: mu1={self.odp_mu1}, mu2={self.odp_mu2}")
        else:
            print(f"  - bonus_strategy={self.bonus_strategy}")

    # -------------------------
    # Disk loaders
    # -------------------------
    def _load_routed_expert_layer(self, layer_idx: int, expert_idx: int) -> torch.nn.Module:
        path = f"{self.expert_path}/layer{layer_idx}_expert{expert_idx}.pt"
        return torch.load(path, map_location="cuda", weights_only=False)

    def _load_shared_expert_layer(self, layer_idx: int) -> torch.nn.Module:
        path = f"{self.shared_expert_path}/layer{layer_idx}_shared_expert.pt"
        return torch.load(path, map_location="cuda", weights_only=False)

    # -------------------------
    # Tokenize + sampling
    # -------------------------
    def tokenize(self, text: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encodings = self.tokenizer(text, return_tensors="pt")
        input_id = encodings.input_ids.to(self.dev)

        # beam replication (kept)
        input_ids = [input_id[0] for _ in range(self.beam_width)]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.dev)

        attention_mask_2d = (input_ids != self.tokenizer.pad_token_id).long()
        position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.long, device=self.dev)
        position_ids = position_ids.unsqueeze(0).expand(input_ids.size(0), -1)
        return input_ids, position_ids, attention_mask_2d

    def _sample_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        cfg = self.sampling_config
        if not cfg.do_sample:
            return torch.argmax(logits, dim=-1)

        if cfg.temperature != 1.0 and cfg.temperature > 0:
            logits = logits / cfg.temperature

        if cfg.top_k > 0:
            top_k = min(cfg.top_k, logits.size(-1))
            cutoff = torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
            logits = logits.masked_fill(logits < cutoff, float("-inf"))

        if cfg.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > cfg.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    # -------------------------
    # Public generate
    # -------------------------
    def generate(self, text: Optional[str] = None, output_token: int = 2048, input_token: Optional[int] = None):
        self.past_key_value = DynamicCache()
        self.past_key_values_length = 0

        input_ids, position_ids, attention_mask_2d = self.tokenize(text or "")
        token_log_f = open("Results/generated_tokens.txt", "a", encoding="utf-8")

        if input_token is not None:
            input_ids = input_ids[:, :input_token]
            position_ids = position_ids[:, :input_token]
            attention_mask_2d = attention_mask_2d[:, :input_token]

        len_token = int(input_ids.shape[1])
        print(len_token, "tokens in input")

        tick = time.time()
        is_decode = False
        is_prefill = True
        prefill_time = 0.0

        for i_token in range(output_token):
            if self.beam_width == 1:
                token_str = self.tokenizer.decode(input_ids[0])
                print(token_str, end="", flush=True)
                token_log_f.write(token_str)
                token_log_f.flush()

            # forward
            if self.mode == "des":
                logits = self.deepseek_forward_des(input_ids, position_ids, attention_mask_2d, is_prefill=is_prefill)
            elif self.mode == "odp":
                logits = self.deepseek_forward_odp(input_ids, position_ids, attention_mask_2d, is_prefill=is_prefill)
            else:
                if self.bonus_strategy == "cecar":
                    logits = self.deepseek_forward_cecar(input_ids, position_ids, attention_mask_2d, is_prefill=is_prefill)
                elif self.bonus_strategy == "mocce":
                    logits = self.deepseek_forward_mocce(input_ids, position_ids, attention_mask_2d, is_prefill=is_prefill)
                else:
                    logits = self.deepseek_forward_vanilla(input_ids, position_ids, attention_mask_2d, is_prefill=is_prefill)

            is_prefill = False

            logits_cpu = logits.to("cpu")
            next_token = self._sample_from_logits(logits_cpu[:, -1, :])

            eos_id = self.tokenizer.eos_token_id
            if eos_id is not None and torch.all(next_token == eos_id):
                if self.beam_width == 1:
                    eos_str = self.tokenizer.decode(next_token[0].view(1))
                    print(eos_str, end="", flush=True)
                    token_log_f.write(eos_str)
                    token_log_f.flush()
                    print("\n")
                break
    
            self.past_key_values_length += logits_cpu.shape[1]

            input_ids = next_token.view(-1, 1).to(self.dev)
            position_ids = torch.arange(
                self.past_key_values_length,
                self.past_key_values_length + 1,
                device=self.dev,
            ).unsqueeze(0).view(-1, 1)
            attention_mask_2d = torch.ones((input_ids.size(0), 1), device=self.dev, dtype=torch.long)

            if not is_decode:
                prefill_time += time.time() - tick
                tick = time.time()
            is_decode = True

        decode_time = time.time() - tick
        token_log_f.write("\n\n\nNext Prompt : \n")
        token_log_f.close()

        ratios = [
            (hit / all_ if all_ != 0 else 0.0)
            for hit, all_ in zip(self.cnt_expert_hit_by_layer, self.cnt_expert_all_by_layer)
        ]
        hit_rate = (self.cnt_expert_hit / self.cnt_expert_all) if self.cnt_expert_all != 0 else 0.0

        return (
            prefill_time,
            decode_time,
            len_token / prefill_time if prefill_time > 0 else 0.0,
            i_token / decode_time if decode_time > 0 else 0.0,
            hit_rate,
            ratios,
        )

    # -------------------------
    # Common subroutines
    # -------------------------
    def _make_attn4d(self, layer_idx: int, q_len: int, attention_mask_2d: torch.Tensor, is_prefill: bool) -> Tuple[torch.Tensor, int]:
        bsz = attention_mask_2d.size(0)
        if is_prefill:
            kv_len = q_len
            attn4d = build_4d_additive_causal_mask(attention_mask_2d, q_len=q_len, kv_len=kv_len, device=self.dev, dtype=self.dtype)
        else:
            past_len = self.past_key_value.get_seq_length(layer_idx)
            kv_len = past_len + q_len
            attn2d_decode = torch.ones((bsz, kv_len), device=self.dev, dtype=torch.long)
            attn4d = build_4d_additive_causal_mask(attn2d_decode, q_len=q_len, kv_len=kv_len, device=self.dev, dtype=self.dtype)
        return attn4d, kv_len

    def _is_moe_layer(self, layer: Any, hidden_states: torch.Tensor) -> Tuple[bool, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Return (is_moe, topk_idx, topk_weight) from gate output.
        Many DeepSeek MoE gates return a tuple/list like (idx, weight, ...).
        """
        if not (hasattr(layer, "mlp") and hasattr(layer.mlp, "gate")):
            return False, None, None
        try:
            gate_out = layer.mlp.gate(hidden_states)
            if isinstance(gate_out, (tuple, list)) and len(gate_out) >= 2:
                topk_idx, topk_weight = gate_out[0], gate_out[1]
                if topk_idx is not None and topk_weight is not None and topk_idx.numel() > 0:
                    return True, topk_idx, topk_weight
        except Exception:
            pass
        return False, None, None

    def _maybe_add_shared_expert(self, layer_idx: int, hidden_states_3d: torch.Tensor, moe_out_3d: torch.Tensor) -> torch.Tensor:
        if not (self.n_shared_experts and self.shared_expert_path):
            return moe_out_3d
        if layer_idx not in self.shared_expert_modules:
            self.shared_expert_modules[layer_idx] = self._load_shared_expert_layer(layer_idx)
        shared_layer = self.shared_expert_modules[layer_idx]
        bsz, seq, _ = hidden_states_3d.shape
        try:
            shared_out = shared_layer(hidden_states_3d)
        except Exception:
            shared_out = shared_layer(hidden_states_3d.view(-1, self.hidden_dim)).view(bsz, seq, self.hidden_dim)
        return moe_out_3d + shared_out.to(device=self.dev, dtype=self.dtype)

    # -------------------------
    # Strategy: Vanilla
    # -------------------------
    @torch.no_grad()
    def deepseek_forward_vanilla(self, input_ids, position_ids, attention_mask_2d, is_prefill: bool):
        hidden_states = self.model.embed_tokens(input_ids.to(self.dev))

        for i_layer, layer in enumerate(self.model.layers):
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            bsz, q_len, _ = hidden_states.shape
            attn4d, _ = self._make_attn4d(i_layer, q_len, attention_mask_2d, is_prefill)

            hidden_states, _, present_key_value = layer.self_attn(
                hidden_states,
                attention_mask=attn4d,
                position_ids=position_ids,
                past_key_value=self.past_key_value,
                use_cache=True,
            )
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)

            is_moe, topk_idx, topk_weight = self._is_moe_layer(layer, hidden_states)

            # Dense MLP
            if not is_moe:
                hidden_states = residual + layer.mlp(hidden_states)
                continue

            # MoE path
            bsz, seq, _ = hidden_states.shape
            hidden_states_2d = hidden_states.view(-1, self.hidden_dim)
            out_2d = torch.zeros_like(hidden_states_2d, device=self.dev, dtype=self.dtype)

            if is_prefill:
                # gate already gives top-6 (DeepSeek)
                for e in range(self.n_expert):
                    mask = (topk_idx == e)
                    idxs = mask.nonzero(as_tuple=False)
                    if idxs.numel() == 0:
                        continue
                    expert_layer = self._load_routed_expert_layer(i_layer, e)
                    if (e in topk_idx[-1]):
                        self.expert_cache.add(i_layer, e, expert_layer)

                    token_idxs = idxs[:, 0]
                    topk_idxs = idxs[:, 1]
                    current_state = hidden_states_2d[token_idxs].to(self.dev, dtype=self.dtype)
                    weighted_output = expert_layer(current_state) * topk_weight[token_idxs, topk_idxs, None].to(self.dev, dtype=self.dtype)
                    out_2d.index_add_(0, token_idxs, weighted_output)

                torch.cuda.empty_cache()

            else:
                routed_flat = topk_idx.reshape(-1).tolist()
                fut = self.executor.submit(self.expert_cache.update_arithmetic, i_layer, routed_flat)

                unique_experts = topk_idx[0].tolist() 
                for topk_i, e in enumerate(unique_experts):
                    e = int(e)
                    self.cnt_expert_all += 1
                    self.cnt_expert_all_by_layer[i_layer] += 1

                    expert_layer = self.expert_cache.get(i_layer, e)
                    if expert_layer is None:
                        expert_layer = self._load_routed_expert_layer(i_layer, e)
                        if self.expert_cache.is_cache_full(i_layer):
                            fut.result()
                            self.expert_cache.replace(i_layer, e, expert_layer)
                        else:
                            self.expert_cache.add(i_layer, e, expert_layer)
                        self.cnt_expert_miss += 1
                        self.cnt_expert_miss_by_layer[i_layer] += 1
                    else:
                        self.cnt_expert_hit += 1
                        self.cnt_expert_hit_by_layer[i_layer] += 1

                    out_2d += expert_layer(hidden_states_2d) * topk_weight[0, topk_i]

            out_3d = out_2d.view(bsz, seq, self.hidden_dim)
            out_3d = self._maybe_add_shared_expert(i_layer, hidden_states, out_3d)

            hidden_states = residual + out_3d

        hidden_states = self.model.norm(hidden_states)
        lm_logits = self.lm_head(hidden_states)

        self.present_key_value = present_key_value
        return lm_logits

    # -------------------------
    # Strategy: CECAR
    # -------------------------
    @torch.no_grad()
    def deepseek_forward_cecar(self, input_ids, position_ids, attention_mask_2d, is_prefill: bool):
        """
        CECAR (decode):
          - compute router logits
          - delta = z_top1 - z_topk (margin_topk_idx)
          - for cached experts: bonus = delta * alpha(ex: 1 - score/max_score)
          - ranking by (logits+bonus), but weights from raw softmax(logits)
          - keep topk_threshold unconditionally, fill remaining from cached only, until 6 experts
          - DeepSeek does NOT renormalize kept weights in this implementation
        """
        hidden_states = self.model.embed_tokens(input_ids.to(self.dev))

        for i_layer, layer in enumerate(self.model.layers):
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            bsz, q_len, _ = hidden_states.shape
            attn4d, _ = self._make_attn4d(i_layer, q_len, attention_mask_2d, is_prefill)

            hidden_states, _, present_key_value = layer.self_attn(
                hidden_states,
                attention_mask=attn4d,
                position_ids=position_ids,
                past_key_value=self.past_key_value,
                use_cache=True,
            )
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)

            is_moe, _, _ = self._is_moe_layer(layer, hidden_states)

            # Dense MLP
            if not is_moe:
                hidden_states = residual + layer.mlp(hidden_states)
                continue

            bsz, seq, _ = hidden_states.shape
            hidden_states_2d = hidden_states.view(-1, self.hidden_dim)
            out_2d = torch.zeros_like(hidden_states_2d, device=self.dev, dtype=self.dtype)

            if is_prefill:
                # Prefill: behave like "compute logits -> topk" 
                router_logits = (layer.mlp.gate.weight @ hidden_states_2d.T).T
                routing_weights_raw = F.softmax(router_logits, dim=1, dtype=torch.float)
                selected_weights, selected_experts = torch.topk(routing_weights_raw, k=self.n_routing_expert, dim=-1)

                for e in range(self.n_expert):
                    mask = (selected_experts == e)
                    idxs = mask.nonzero(as_tuple=False)
                    if idxs.numel() == 0:
                        continue
                    expert_layer = self._load_routed_expert_layer(i_layer, e)
                    if (e in selected_experts[-1]):
                        self.expert_cache.add(i_layer, e, expert_layer)

                    token_idxs = idxs[:, 0]
                    topk_idxs = idxs[:, 1]
                    current_state = hidden_states_2d[token_idxs].to(self.dev, dtype=self.dtype)
                    weighted_output = expert_layer(current_state) * selected_weights[token_idxs, topk_idxs, None].to(self.dev, dtype=self.dtype)
                    out_2d.index_add_(0, token_idxs, weighted_output)

                torch.cuda.empty_cache()

            else:
                # Decode: CECAR
                router_logits = (layer.mlp.gate.weight @ hidden_states_2d.T).T 
                router_logits_select = router_logits.clone()

                topk_logits, _ = torch.topk(router_logits[0], self.n_expert)
                z_top1 = float(topk_logits[0].item())
                z_topk = float(topk_logits[self.margin_topk_idx].item())
                delta = z_top1 - z_topk
                self.delta_records.append(delta)

                cached_tensor = self.expert_cache.get_cached_expert_ids_tensor(i_layer)
                if cached_tensor.numel() > 0:
                    scores_tensor = self.expert_cache.get_eviction_scores_batch(i_layer, cached_tensor)
                    max_score = scores_tensor.max()
                    alpha_tensor = 1.0 - scores_tensor / (max_score + 1e-5)
                    bonus_tensor = delta * alpha_tensor
                    router_logits_select[0, cached_tensor] += bonus_tensor
                    self.alpha_records.extend(alpha_tensor.tolist())

                # ranking by modified logits
                raw_ranking = torch.argsort(router_logits_select[0], descending=True)
                routing_weights_raw = F.softmax(router_logits, dim=1, dtype=torch.float)
                raw_weights_list = routing_weights_raw[0, raw_ranking].tolist()

                cached_set = set(self.expert_cache.get_cached_expert_ids(i_layer))

                kept_experts: List[int] = []
                kept_weights: List[float] = []

                for rank, e in enumerate(raw_ranking.tolist()):
                    if rank < self.topk_threshold:
                        kept_experts.append(int(e))
                        kept_weights.append(float(raw_weights_list[rank]))
                    else:
                        if int(e) in cached_set:
                            kept_experts.append(int(e))
                            kept_weights.append(float(raw_weights_list[rank]))
                    if len(kept_experts) == 6:
                        break

                if len(kept_experts) != 6:
                    raise RuntimeError(f"[CECAR] Expected 6 experts, got {len(kept_experts)} at layer {i_layer}")

                self.routed_expert_counts.append(len(kept_experts))
                kept_w = torch.tensor(kept_weights, device=self.dev, dtype=self.dtype)

                self.executor.submit(self.expert_cache.update_arithmetic, i_layer, [kept_experts])

                for e, w in zip(kept_experts, kept_w):
                    self.cnt_expert_all += 1
                    self.cnt_expert_all_by_layer[i_layer] += 1

                    expert_layer = self.expert_cache.get(i_layer, e)
                    if expert_layer is None:
                        expert_layer = self._load_routed_expert_layer(i_layer, e)
                        if self.expert_cache.is_cache_full(i_layer):
                            self.expert_cache.replace(i_layer, e, expert_layer)
                        else:
                            self.expert_cache.add(i_layer, e, expert_layer)
                        self.cnt_expert_miss += 1
                        self.cnt_expert_miss_by_layer[i_layer] += 1
                    else:
                        self.cnt_expert_hit += 1
                        self.cnt_expert_hit_by_layer[i_layer] += 1

                    out_2d += expert_layer(hidden_states_2d) * w

            out_3d = out_2d.view(bsz, seq, self.hidden_dim)
            out_3d = self._maybe_add_shared_expert(i_layer, hidden_states, out_3d)
            hidden_states = residual + out_3d

        hidden_states = self.model.norm(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        self.present_key_value = present_key_value
        return lm_logits

    # -------------------------
    # Strategy: MOCCE
    # -------------------------
    @torch.no_grad()
    def deepseek_forward_mocce(self, input_ids, position_ids, attention_mask_2d, is_prefill: bool):
        """
        MOCCE (decode):
          - build mask: Top-J always + cached experts
          - router_logits_select = z + lambda_cache * delta_avg * mask
          - select experts by softmax(router_logits_select), but weights from raw softmax(z)
        """
        hidden_states = self.model.embed_tokens(input_ids.to(self.dev))

        for i_layer, layer in enumerate(self.model.layers):
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            bsz, q_len, _ = hidden_states.shape
            attn4d, _ = self._make_attn4d(i_layer, q_len, attention_mask_2d, is_prefill)

            hidden_states, _, present_key_value = layer.self_attn(
                hidden_states,
                attention_mask=attn4d,
                position_ids=position_ids,
                past_key_value=self.past_key_value,
                use_cache=True,
            )
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)

            is_moe, _, _ = self._is_moe_layer(layer, hidden_states)

            if not is_moe:
                hidden_states = residual + layer.mlp(hidden_states)
                continue

            bsz, seq, _ = hidden_states.shape
            hidden_states_2d = hidden_states.view(-1, self.hidden_dim)
            out_2d = torch.zeros_like(hidden_states_2d, device=self.dev, dtype=self.dtype)

            router_logits = (layer.mlp.gate.weight @ hidden_states_2d.T).T 
            routing_weights_raw = F.softmax(router_logits, dim=1, dtype=torch.float)

            if not is_prefill:
                z = router_logits
                cur_range = (z.max() - z.min()).item()
                tracker = self.cache_delta_tracker[i_layer]
                if not tracker["init"]:
                    tracker["avg"] = cur_range
                    tracker["init"] = True
                else:
                    tracker["avg"] = 0.99 * tracker["avg"] + 0.01 * cur_range
                delta_avg = tracker["avg"]

                mask = torch.zeros_like(z)

                _, topJ = torch.topk(z, self.top_J, dim=1)
                mask.scatter_(1, topJ, 1.0)

                cached = self.expert_cache.get_cached_expert_ids(i_layer)
                if cached:
                    mask[:, cached] = 1.0

                router_logits_select = z + self.lambda_cache * delta_avg * mask
                routing_weights_select = F.softmax(router_logits_select, dim=1, dtype=torch.float)

                _, selected_experts = torch.topk(routing_weights_select, k=self.n_routing_expert, dim=-1)
                selected_weights = torch.gather(routing_weights_raw, 1, selected_experts)
            else:
                selected_weights, selected_experts = torch.topk(routing_weights_raw, k=self.n_routing_expert, dim=-1)

            if is_prefill:
                for e in range(self.n_expert):
                    mask_e = (selected_experts == e)
                    idxs = mask_e.nonzero(as_tuple=False)
                    if idxs.numel() == 0:
                        continue
                    expert_layer = self._load_routed_expert_layer(i_layer, e)
                    if (e in selected_experts[-1]):
                        self.expert_cache.add(i_layer, e, expert_layer)

                    token_idxs = idxs[:, 0]
                    topk_idxs = idxs[:, 1]
                    current_state = hidden_states_2d[token_idxs].to(self.dev, dtype=self.dtype)
                    weighted_output = expert_layer(current_state) * selected_weights[token_idxs, topk_idxs, None].to(self.dev, dtype=self.dtype)
                    out_2d.index_add_(0, token_idxs, weighted_output)

                torch.cuda.empty_cache()

            else:
                unique_experts = selected_experts[0].tolist()
                self.executor.submit(self.expert_cache.update_arithmetic, i_layer, unique_experts)
                self.routed_expert_counts.append(len(unique_experts))

                for idx, e in enumerate(unique_experts):
                    e = int(e)
                    w = selected_weights[0, idx]

                    self.cnt_expert_all += 1
                    self.cnt_expert_all_by_layer[i_layer] += 1

                    expert_layer = self.expert_cache.get(i_layer, e)
                    if expert_layer is None:
                        expert_layer = self._load_routed_expert_layer(i_layer, e)
                        if self.expert_cache.is_cache_full(i_layer):
                            self.expert_cache.replace(i_layer, e, expert_layer)
                        else:
                            self.expert_cache.add(i_layer, e, expert_layer)
                        self.cnt_expert_miss += 1
                        self.cnt_expert_miss_by_layer[i_layer] += 1
                    else:
                        self.cnt_expert_hit += 1
                        self.cnt_expert_hit_by_layer[i_layer] += 1

                    out_2d += expert_layer(hidden_states_2d) * w

            out_3d = out_2d.view(bsz, seq, self.hidden_dim)
            out_3d = self._maybe_add_shared_expert(i_layer, hidden_states, out_3d)
            hidden_states = residual + out_3d

        hidden_states = self.model.norm(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        self.present_key_value = present_key_value
        return lm_logits

    # -------------------------
    # Threshold loaders (DES/ODP)
    # -------------------------
    def _load_des_thresholds_from_file(self, path: str) -> None:
        if not path or (not os.path.exists(path)):
            print(f"Warning: DES thresholds file not found: {path}")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception as e:
            print(f"Warning: failed to read DES thresholds file={path}: {e}")
            return

        obj2 = obj["odp"] if ("odp" in obj and isinstance(obj["odp"], dict)) else obj

        if isinstance(obj2, dict) and ("mu1" in obj2) and ("mu2" in obj2):
            try:
                self.des_mu1 = float(obj2["mu1"])
                self.des_mu2 = float(obj2["mu2"])
                self.des_layerwise = False
                print(f"Loaded global DES thresholds: mu1={self.des_mu1}, mu2={self.des_mu2}")
            except Exception:
                pass
            return

        if isinstance(obj, dict) and isinstance(obj.get("layers", None), list):
            mu_4_5, mu_5_6 = {}, {}
            for it in obj["layers"]:
                if not isinstance(it, dict) or ("layer" not in it):
                    continue
                lid = int(it["layer"])
                if ("mu_4_5" in it) and ("mu_5_6" in it):
                    mu_4_5[lid] = float(it["mu_4_5"])
                    mu_5_6[lid] = float(it["mu_5_6"])
            if mu_4_5 and mu_5_6:
                self.des_mu_4_5 = mu_4_5
                self.des_mu_5_6 = mu_5_6
                self.des_layerwise = True
                print(f"Loaded layerwise DES thresholds for {len(mu_4_5)} layers")

    def _get_des_thresholds_for_layer(self, layer_idx: int) -> Tuple[float, float]:
        if self.des_layerwise and (self.des_mu_4_5 is not None) and (self.des_mu_5_6 is not None):
            mu1 = self.des_mu_4_5.get(layer_idx, self.des_mu1)
            mu2 = self.des_mu_5_6.get(layer_idx, self.des_mu2)
            return float(mu1), float(mu2)
        return self.des_mu1, self.des_mu2

    def _load_odp_thresholds_from_file(self, path: str) -> None:
        if not path or (not os.path.exists(path)):
            print(f"Warning: ODP thresholds file not found: {path}")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception as e:
            print(f"Warning: failed to read ODP thresholds file={path}: {e}")
            return

        obj2 = obj["odp"] if ("odp" in obj and isinstance(obj["odp"], dict)) else obj

        if isinstance(obj2, dict) and ("mu1" in obj2) and ("mu2" in obj2):
            try:
                self.odp_mu1 = float(obj2["mu1"])
                self.odp_mu2 = float(obj2["mu2"])
                self.odp_layerwise = False
                print(f"Loaded global ODP thresholds: mu1={self.odp_mu1}, mu2={self.odp_mu2}")
            except Exception:
                pass
            return

        if isinstance(obj, dict) and isinstance(obj.get("layers", None), list):
            mu_4_5, mu_5_6 = {}, {}
            for it in obj["layers"]:
                if not isinstance(it, dict) or ("layer" not in it):
                    continue
                lid = int(it["layer"])
                if ("mu_4_5" in it) and ("mu_5_6" in it):
                    mu_4_5[lid] = float(it["mu_4_5"])
                    mu_5_6[lid] = float(it["mu_5_6"])
            if mu_4_5 and mu_5_6:
                self.odp_mu_4_5 = mu_4_5
                self.odp_mu_5_6 = mu_5_6
                self.odp_layerwise = True
                print(f"Loaded layerwise ODP thresholds for {len(mu_4_5)} layers")

    def _get_odp_thresholds_for_layer(self, layer_idx: int) -> Tuple[float, float]:
        if self.odp_layerwise and (self.odp_mu_4_5 is not None) and (self.odp_mu_5_6 is not None):
            mu1 = self.odp_mu_4_5.get(layer_idx, self.odp_mu1)
            mu2 = self.odp_mu_5_6.get(layer_idx, self.odp_mu2)
            return float(mu1), float(mu2)
        return self.odp_mu1, self.odp_mu2

    # -------------------------
    # Strategy: DES
    # -------------------------
    @torch.no_grad()
    def deepseek_forward_des(self, input_ids, position_ids, attention_mask_2d, is_prefill: bool = False):
        """
        DES: gate gives top-6; choose k in {4,5,6} based on ratios:
            ratio1 = w4/w5, ratio2 = w5/w6
          if ratio1 > mu1 -> k=4
          elif ratio2 > mu2 -> k=5
          else k=6
        """
        hidden_states = self.model.embed_tokens(input_ids.to(self.dev))

        for i_layer, layer in enumerate(self.model.layers):
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            bsz, q_len, _ = hidden_states.shape
            attn4d, _ = self._make_attn4d(i_layer, q_len, attention_mask_2d, is_prefill)

            hidden_states, _, present_key_value = layer.self_attn(
                hidden_states,
                attention_mask=attn4d,
                position_ids=position_ids,
                past_key_value=self.past_key_value,
                use_cache=True,
            )
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)

            is_moe, topk_idx, topk_weight = self._is_moe_layer(layer, hidden_states)

            if not is_moe:
                hidden_states = residual + layer.mlp(hidden_states)
                continue

            bsz, seq, _ = hidden_states.shape
            topk_idx = topk_idx.view(bsz, seq, -1)
            topk_weight = topk_weight.view(bsz, seq, -1)

            mu1, mu2 = self._get_des_thresholds_for_layer(i_layer)

            hidden_states_2d = hidden_states.view(-1, self.hidden_dim)
            out_2d = torch.zeros_like(hidden_states_2d, device=self.dev, dtype=self.dtype)

            if is_prefill:
                # prefill over all tokens
                for token_idx in range(bsz * seq):
                    b = token_idx // seq
                    s = token_idx % seq
                    w = topk_weight[b, s]  
                    idx = topk_idx[b, s]   

                    r4, r5, r6 = w[3].item(), w[4].item(), w[5].item()
                    ratio1 = r4 / (r5 + 1e-6)
                    ratio2 = r5 / (r6 + 1e-6)

                    if ratio1 > mu1:
                        k_actual = 4; self.des_k_hist[4] += 1
                    elif ratio2 > mu2:
                        k_actual = 5; self.des_k_hist[5] += 1
                    else:
                        k_actual = 6; self.des_k_hist[6] += 1

                    w_actual = w[:k_actual]
                    idx_actual = idx[:k_actual]

                    for topk_i in range(k_actual):
                        e = int(idx_actual[topk_i].item())
                        expert_layer = self._load_routed_expert_layer(i_layer, e)

                        if token_idx == bsz * seq - 1:
                            self.expert_cache.add(i_layer, e, expert_layer)

                        cur = hidden_states_2d[token_idx:token_idx + 1].to(self.dev, dtype=self.dtype)
                        out_2d[token_idx:token_idx + 1] += expert_layer(cur) * w_actual[topk_i]
                        del expert_layer

                torch.cuda.empty_cache()

            else:
                # decode: seq=1
                w = topk_weight[0, 0]
                idx = topk_idx[0, 0]

                r4, r5, r6 = w[3].item(), w[4].item(), w[5].item()
                ratio1 = r4 / (r5 + 1e-6)
                ratio2 = r5 / (r6 + 1e-6)

                if ratio1 > mu1:
                    k_actual = 4; self.des_k_hist[4] += 1
                elif ratio2 > mu2:
                    k_actual = 5; self.des_k_hist[5] += 1
                else:
                    k_actual = 6; self.des_k_hist[6] += 1

                w_actual = w[:k_actual]
                idx_actual = idx[:k_actual].tolist()

                self.executor.submit(self.expert_cache.update_arithmetic, i_layer, idx_actual)

                for topk_i in range(k_actual):
                    e = int(idx_actual[topk_i])
                    self.cnt_expert_all += 1
                    self.cnt_expert_all_by_layer[i_layer] += 1

                    expert_layer = self.expert_cache.get(i_layer, e)
                    if expert_layer is None:
                        expert_layer = self._load_routed_expert_layer(i_layer, e)
                        if self.expert_cache.is_cache_full(i_layer):
                            self.expert_cache.replace(i_layer, e, expert_layer)
                        else:
                            self.expert_cache.add(i_layer, e, expert_layer)
                        self.cnt_expert_miss += 1
                        self.cnt_expert_miss_by_layer[i_layer] += 1
                    else:
                        self.cnt_expert_hit += 1
                        self.cnt_expert_hit_by_layer[i_layer] += 1

                    out_2d += expert_layer(hidden_states_2d) * w_actual[topk_i]

            out_3d = out_2d.view(bsz, seq, self.hidden_dim)
            out_3d = self._maybe_add_shared_expert(i_layer, hidden_states, out_3d)
            hidden_states = residual + out_3d

        hidden_states = self.model.norm(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        self.present_key_value = present_key_value
        return lm_logits

    # -------------------------
    # Strategy: ODP
    # -------------------------
    @torch.no_grad()
    def deepseek_forward_odp(self, input_ids, position_ids, attention_mask_2d, is_prefill: bool = False):
        """
        ODP = DES pruning + prefill protection.
        Protection score:
            score_j = ||t_j||_1 * attention_importance(j)
        selects top (ratio=0.02) tokens as protected -> keep k=6 regardless of ratios.
        """
        hidden_states = self.model.embed_tokens(input_ids.to(self.dev))

        for i_layer, layer in enumerate(self.model.layers):
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            bsz, q_len, _ = hidden_states.shape

            # capture attention weights for protection (prefill only)
            attn_weights = None
            if is_prefill:
                try:
                    attn4d_dbg = build_4d_additive_causal_mask(attention_mask_2d, q_len=q_len, kv_len=q_len, device=self.dev, dtype=self.dtype)
                    out_dbg = layer.self_attn(
                        hidden_states,
                        attention_mask=attn4d_dbg,
                        position_ids=position_ids,
                        past_key_value=None,
                        use_cache=False,
                        output_attentions=True,
                    )
                    if isinstance(out_dbg, tuple) and len(out_dbg) >= 2:
                        w_dbg = out_dbg[1]
                        if w_dbg is not None and w_dbg.dim() == 4:
                            attn_weights = w_dbg
                except Exception:
                    attn_weights = None

            attn4d, _ = self._make_attn4d(i_layer, q_len, attention_mask_2d, is_prefill)

            hidden_states, _, present_key_value = layer.self_attn(
                hidden_states,
                attention_mask=attn4d,
                position_ids=position_ids,
                past_key_value=self.past_key_value,
                use_cache=True,
            )
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)

            is_moe, topk_idx, topk_weight = self._is_moe_layer(layer, hidden_states)

            if not is_moe:
                hidden_states = residual + layer.mlp(hidden_states)
                continue

            bsz, seq, _ = hidden_states.shape
            topk_idx = topk_idx.view(bsz, seq, -1)
            topk_weight = topk_weight.view(bsz, seq, -1)

            mu1, mu2 = self._get_odp_thresholds_for_layer(i_layer)

            hidden_states_2d = hidden_states.view(-1, self.hidden_dim)
            out_2d = torch.zeros_like(hidden_states_2d, device=self.dev, dtype=self.dtype)

            # protection set (prefill only)
            is_protected: Dict[int, bool] = {}
            if is_prefill:
                l1_norm = torch.norm(hidden_states_2d, p=1, dim=1).to(torch.float)
                l1_bs = l1_norm.view(bsz, seq)
                score_bs = l1_bs

                if attn_weights is not None:
                    attn = attn_weights.to(torch.float)  # [B, H, L, L]
                    A = attn.mean(dim=1)                 # [B, L, L]
                    causal = torch.tril(torch.ones((seq, seq), device=A.device, dtype=A.dtype))
                    A = A * causal.unsqueeze(0)
                    incoming_sum = A.sum(dim=1)          # [B, L]
                    denom = torch.arange(seq, 0, -1, device=A.device, dtype=A.dtype)
                    incoming_avg = incoming_sum / denom.unsqueeze(0)
                    score_bs = l1_bs * incoming_avg

                protection_ratio = 0.02
                k_protect = max(1, int(seq * protection_ratio))
                top_idx = torch.topk(score_bs, k_protect, dim=1, largest=True, sorted=False).indices
                mask_bs = torch.zeros_like(score_bs, dtype=torch.bool)
                mask_bs.scatter_(1, top_idx, True)

                mask_flat = mask_bs.view(-1)
                for idx in range(bsz * seq):
                    if mask_flat[idx].item():
                        is_protected[idx] = True

            if is_prefill:
                for token_idx in range(bsz * seq):
                    b = token_idx // seq
                    s = token_idx % seq
                    w = topk_weight[b, s]
                    idx = topk_idx[b, s]

                    if token_idx in is_protected:
                        k_actual = 6; self.odp_k_hist[6] += 1
                    else:
                        r4, r5, r6 = w[3].item(), w[4].item(), w[5].item()
                        ratio1 = r4 / (r5 + 1e-6)
                        ratio2 = r5 / (r6 + 1e-6)

                        if ratio1 > mu1:
                            k_actual = 4; self.odp_k_hist[4] += 1
                        elif ratio2 > mu2:
                            k_actual = 5; self.odp_k_hist[5] += 1
                        else:
                            k_actual = 6; self.odp_k_hist[6] += 1

                    w_actual = w[:k_actual]
                    idx_actual = idx[:k_actual]

                    for topk_i in range(k_actual):
                        e = int(idx_actual[topk_i].item())
                        expert_layer = self._load_routed_expert_layer(i_layer, e)

                        if token_idx == bsz * seq - 1:
                            self.expert_cache.add(i_layer, e, expert_layer)

                        cur = hidden_states_2d[token_idx:token_idx + 1].to(self.dev, dtype=self.dtype)
                        out_2d[token_idx:token_idx + 1] += expert_layer(cur) * w_actual[topk_i]
                        del expert_layer

                torch.cuda.empty_cache()

            else:
                # decode: no protection
                w = topk_weight[0, 0]
                idx = topk_idx[0, 0]

                r4, r5, r6 = w[3].item(), w[4].item(), w[5].item()
                ratio1 = r4 / (r5 + 1e-6)
                ratio2 = r5 / (r6 + 1e-6)

                if ratio1 > mu1:
                    k_actual = 4; self.odp_k_hist[4] += 1
                elif ratio2 > mu2:
                    k_actual = 5; self.odp_k_hist[5] += 1
                else:
                    k_actual = 6; self.odp_k_hist[6] += 1

                w_actual = w[:k_actual]
                idx_actual = idx[:k_actual].tolist()

                self.executor.submit(self.expert_cache.update_arithmetic, i_layer, [idx_actual])
                self.routed_expert_counts.append(k_actual)

                for topk_i in range(k_actual):
                    e = int(idx_actual[topk_i])

                    self.cnt_expert_all += 1
                    self.cnt_expert_all_by_layer[i_layer] += 1

                    expert_layer = self.expert_cache.get(i_layer, e)
                    if expert_layer is None:
                        expert_layer = self._load_routed_expert_layer(i_layer, e)
                        if self.expert_cache.is_cache_full(i_layer):
                            self.expert_cache.replace(i_layer, e, expert_layer)
                        else:
                            self.expert_cache.add(i_layer, e, expert_layer)
                        self.cnt_expert_miss += 1
                        self.cnt_expert_miss_by_layer[i_layer] += 1
                    else:
                        self.cnt_expert_hit += 1
                        self.cnt_expert_hit_by_layer[i_layer] += 1

                    out_2d += expert_layer(hidden_states_2d) * w_actual[topk_i]

            out_3d = out_2d.view(bsz, seq, self.hidden_dim)
            out_3d = self._maybe_add_shared_expert(i_layer, hidden_states, out_3d)
            hidden_states = residual + out_3d

        hidden_states = self.model.norm(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        self.present_key_value = present_key_value
        return lm_logits
