# ============================================================
# Algorithm:
#   Given a MoE LLM, run (prefill + fixed-length decode) on a dataset,
#   and record per-layer router logits (full logits or top-K) for each decode step.
# 
# Outputs:
#      .pt payload containing:
#     - prompt token ids (trimmed to non-pad length)
#     - generated token ids (fixed length, EOS stopping disabled)
#     - routing logs per sample:
#         * full_logits: (T,1,L,E) float
#         * or topk: idx/val (T,1,L,K)
# 
# Notes:
#   - For Qwen3_30B_A3B, we optionally disable SDPA by setting attn_implementation="eager"
#     to avoid mask shape issues observed in certain transformer versions.
#   - For DeepSeek_v2_Lite_Chat models, we patch DynamicCache for backward compatibility.
# ============================================================

import os
import argparse
from typing import Optional, Dict, Any, List, Tuple, Iterator
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache


# ============================================================
# 0) Compatibility patch (DeepSeek_v2_Lite_Chat + DynamicCache)
# ============================================================
def patch_dynamic_cache_for_deepseek() -> None:
    if not hasattr(DynamicCache, "seen_tokens"):

        @property
        def seen_tokens(self):
            try:
                return self.get_seq_length(0)
            except Exception:
                return 0

        DynamicCache.seen_tokens = seen_tokens

    if not hasattr(DynamicCache, "get_max_length"):

        def get_max_length(self):
            return None

        DynamicCache.get_max_length = get_max_length

    if not hasattr(DynamicCache, "get_usable_length"):

        def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
            try:
                return self.get_seq_length(layer_idx)
            except Exception:
                return 0

        DynamicCache.get_usable_length = get_usable_length


# ============================================================
# 1) Dataset registry
# ============================================================
DATASET_CFG = {
    "humaneval": {
        "path": "openai/openai_humaneval",
        "name": None,
        "text_fields": ["prompt"],
        "default_split": "test",
    },
    "mbpp": {
        "path": "mbpp",
        "name": None,
        "text_fields": ["text"],
        "default_split": "test",
    },
    "gpqa": {
        "path": "math-ai/gpqa",
        "name": None,
        "text_fields": ["problem"],
        "default_split": "test",
    },
    "triviaqa": {
        "path": "trivia_qa",
        "name": "rc",
        "text_fields": ["question"],
        "default_split": "test",
    },
    "math": {
        "path": "Maxwell-Jia/MATH",
        "name": None,
        "text_fields": ["problem"],
        "default_split": "test",
    },
}


def pick_text_field(ex: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for k in candidates:
        if k in ex and ex[k] is not None:
            s = str(ex[k]).strip()
            if s:
                return s
    return None


def load_questions(dataset_name: str, split: str, n: int) -> List[str]:
    cfg = DATASET_CFG[dataset_name]
    if cfg["name"] is None:
        ds = load_dataset(cfg["path"], split=split)
    else:
        ds = load_dataset(cfg["path"], cfg["name"], split=split)

    out: List[str] = []
    for ex in ds:
        t = pick_text_field(ex, cfg["text_fields"])
        if t is None:
            continue
        out.append(t)
        if len(out) >= n:
            break

    if not out:
        raise RuntimeError(f"No samples extracted: dataset={dataset_name}, split={split}")
    return out


def chunked(xs: List[str], bs: int) -> Iterator[List[str]]:
    for i in range(0, len(xs), bs):
        yield xs[i : i + bs]


# ============================================================
# 2) Model registry / dtype helpers
# ============================================================
def resolve_model_id(alias: str) -> str:
    if alias in ("Qwen3_30B_A3B"):
        return "Qwen/Qwen3-30B-A3B"
    if alias == "OLMoE_1B_7B_0125_Instruct":
        return "allenai/OLMoE-1B-7B-0125-Instruct"
    if alias in ("DeepSeek_v2_Lite_Chat"):
        return "deepseek-ai/DeepSeek-V2-Lite-Chat"
    raise ValueError(f"Unknown model alias: {alias}")


def resolve_dtype(name: str) -> torch.dtype:
    n = name.lower()
    if n in ("bf16", "bfloat16"):
        return torch.bfloat16
    if n in ("fp16", "float16", "half"):
        return torch.float16
    if n in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")


def any_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cpu")


# ============================================================
# 3) Gate discovery (router module per layer)
# ============================================================
def infer_num_experts_from_config(model) -> int:
    cfg = model.config
    for key in ("n_routed_experts", "num_experts", "num_experts_total", "moe_num_experts"):
        v = getattr(cfg, key, None)
        if v is not None:
            return int(v)
    raise RuntimeError("Cannot infer num_experts from model.config")


def find_gate_modules(model) -> Tuple[List[Optional[torch.nn.Module]], int]:
    base = getattr(model, "model", None)
    if base is None or not hasattr(base, "layers"):
        raise RuntimeError("Cannot locate model.model.layers")

    L = len(base.layers)
    gates: List[Optional[torch.nn.Module]] = [None] * L

    E: Optional[int] = None
    try:
        E = infer_num_experts_from_config(model)
    except Exception:
        E = None

    candidate_names = (
        "gate", "router", "gate_proj", "router_proj",
        "gate_linear", "router_linear",
        "router_gate", "routing", "router_logits",
    )

    for li, layer in enumerate(base.layers):
        gate = None

        # check layer.mlp.*
        if hasattr(layer, "mlp"):
            mlp = layer.mlp
            for name in candidate_names:
                if hasattr(mlp, name):
                    m = getattr(mlp, name)
                    if isinstance(m, torch.nn.Module):
                        gate = m
                        break

        # fallback: check layer.*
        if gate is None:
            for name in candidate_names:
                if hasattr(layer, name):
                    m = getattr(layer, name)
                    if isinstance(m, torch.nn.Module):
                        gate = m
                        break

        gates[li] = gate

        # infer E from gate shape if not found
        if E is None and gate is not None:
            if hasattr(gate, "out_features"):
                E = int(getattr(gate, "out_features"))
            elif hasattr(gate, "weight") and getattr(gate, "weight") is not None:
                w = gate.weight
                if torch.is_tensor(w) and w.dim() == 2:
                    E = int(w.shape[0])

    if E is None:
        E = infer_num_experts_from_config(model)

    return gates, int(E)


# ============================================================
# 4) Routing collector (stores per-step per-layer logits)
# ============================================================
def _first_tensor(x):
    if isinstance(x, (tuple, list)):
        return x[0]
    return x


class RoutingCollector:
    """
    Stores routing info during decode only.

    If store_full_logits:
      full[t,b,l,e]  (T,B,L,E)

    else:
      topk_idx[t,b,l,k], topk_val[t,b,l,k]  (T,B,L,K)
    """

    def __init__(
        self,
        n_layers: int,
        n_experts: int,
        max_steps: int,
        top_k: int,
        store_full_logits: bool,
        store_dtype: torch.dtype,
    ):
        self.n_layers = int(n_layers)
        self.n_experts = int(n_experts)
        self.max_steps = int(max_steps)
        self.top_k = int(top_k)
        self.store_full_logits = bool(store_full_logits)
        self.store_dtype = store_dtype

        self.enabled = False
        self.cur_step = 0
        self.batch_size: Optional[int] = None

        self.full: Optional[torch.Tensor] = None
        self.topk_idx: Optional[torch.Tensor] = None
        self.topk_val: Optional[torch.Tensor] = None

    def reset_for_batch(self, batch_size: int) -> None:
        self.batch_size = int(batch_size)
        self.cur_step = 0
        self.enabled = False

        if self.store_full_logits:
            self.full = torch.zeros(
                (self.max_steps, self.batch_size, self.n_layers, self.n_experts),
                dtype=self.store_dtype,
                device="cpu",
            )
            self.topk_idx, self.topk_val = None, None
        else:
            self.topk_idx = torch.zeros(
                (self.max_steps, self.batch_size, self.n_layers, self.top_k),
                dtype=torch.int32,
                device="cpu",
            )
            self.topk_val = torch.zeros(
                (self.max_steps, self.batch_size, self.n_layers, self.top_k),
                dtype=self.store_dtype,
                device="cpu",
            )
            self.full = None

    def start_decode(self) -> None:
        self.enabled = True
        self.cur_step = 0

    def stop(self) -> None:
        self.enabled = False

    def set_step(self, t: int) -> None:
        self.cur_step = int(t)

    def write_layer_logits(self, layer_idx: int, logits_2d: torch.Tensor) -> None:
        # guards
        if not self.enabled:
            return
        if self.cur_step < 0 or self.cur_step >= self.max_steps:
            return
        if self.batch_size is None:
            return

        logits_2d = _first_tensor(logits_2d)
        if (not torch.is_tensor(logits_2d)) or logits_2d.dim() != 2:
            return
        if logits_2d.size(-1) != self.n_experts:
            return

        B = self.batch_size
        if logits_2d.size(0) < B:
            return

        rows = logits_2d[-B:].detach()  # (B,E), decode token rows

        if self.store_full_logits:
            assert self.full is not None
            self.full[self.cur_step, :, layer_idx, :].copy_(rows.to(dtype=self.store_dtype, device="cpu"))
        else:
            assert self.topk_idx is not None and self.topk_val is not None
            k = min(self.top_k, rows.size(-1))
            vals, idx = torch.topk(rows, k=k, dim=-1)  # (B,k)
            self.topk_val[self.cur_step, :, layer_idx, :k].copy_(vals.to(dtype=self.store_dtype, device="cpu"))
            self.topk_idx[self.cur_step, :, layer_idx, :k].copy_(idx.to(dtype=torch.int32, device="cpu"))
            if k < self.top_k:
                self.topk_val[self.cur_step, :, layer_idx, k:].zero_()
                self.topk_idx[self.cur_step, :, layer_idx, k:].zero_()

    def export_per_sample(self, sample_idx: int) -> Dict[str, Any]:
        i = int(sample_idx)
        if self.store_full_logits:
            assert self.full is not None
            logits = self.full[:, i, :, :].unsqueeze(1).contiguous()  # (T,1,L,E)
            return {"type": "full_logits", "logits": logits}
        else:
            assert self.topk_idx is not None and self.topk_val is not None
            idx = self.topk_idx[:, i, :, :].unsqueeze(1).contiguous()  # (T,1,L,K)
            val = self.topk_val[:, i, :, :].unsqueeze(1).contiguous()  # (T,1,L,K)
            return {"type": "topk", "topk": int(self.top_k), "idx": idx, "val": val}


def register_gate_hooks(
    gates_per_layer: List[Optional[torch.nn.Module]],
    collector: RoutingCollector,
) -> List[Any]:
    hooks = []
    for li, gate in enumerate(gates_per_layer):
        if gate is None:
            continue

        layer_idx = int(li)

        def hook_fn(module, inp, out, layer_idx=layer_idx):
            collector.write_layer_logits(layer_idx, out)

        hooks.append(gate.register_forward_hook(hook_fn))
    return hooks


# ============================================================
# 5) Prefill + fixed-length decode (EOS stopping disabled)
# ============================================================
@torch.no_grad()
def prefill(model, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    return out.past_key_values


def infer_past_len(past_key_values) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        try:
            return int(past_key_values.get_seq_length(0))
        except Exception:
            pass
            
    # tuple/list fallback
    if isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
        layer0 = past_key_values[0]
        if isinstance(layer0, (tuple, list)) and len(layer0) > 0:
            k0 = layer0[0]
            if torch.is_tensor(k0) and k0.dim() >= 3:
            
                # common shapes: (B, H, T, Dh) or (B, T, H, Dh)
                if k0.size(2) >= 1:
                    return int(k0.size(2))
                if k0.size(1) >= 1:
                    return int(k0.size(1))
    return 0


def gather_last_nonpad(input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    lengths = attn_mask.long().sum(dim=1).clamp(min=1)  
    idx = (lengths - 1).view(-1, 1)  
    return input_ids.gather(1, idx)  


@torch.no_grad()
def decode_fixed_steps(
    model,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    collector: RoutingCollector,
) -> torch.Tensor:
    """
    Fixed-length decoding loop.
    - Keeps attention_mask 2D.
    - Does NOT pass cache_position/position_ids (robust for Qwen3_30B_A3B SDPA issues).
    """
    collector.stop()

    dev = any_model_device(model)
    prompt_input_ids = prompt_input_ids.to(dev)
    prompt_attention_mask = prompt_attention_mask.to(dev).to(torch.long)

    past_key_values = prefill(model, prompt_input_ids, prompt_attention_mask)

    B = prompt_input_ids.size(0)
    decode_ids = torch.empty((B, max_new_tokens), dtype=torch.long, device="cpu")

    next_input_ids = gather_last_nonpad(prompt_input_ids, prompt_attention_mask).clone() 
    _ = infer_past_len(past_key_values)  

    collector.start_decode()

    for t in range(max_new_tokens):
        collector.set_step(t)

        # attention_mask: [B, S + (t+1)]
        gen_mask = torch.ones((B, t + 1), dtype=torch.long, device=dev)
        attn_mask_2d = torch.cat([prompt_attention_mask, gen_mask], dim=1)

        out = model(
            input_ids=next_input_ids,          
            attention_mask=attn_mask_2d,       
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )

        past_key_values = out.past_key_values
        logits = out.logits[:, -1, :]  

        if do_sample:
            temp = max(float(temperature), 1e-5)
            probs = torch.softmax(logits / temp, dim=-1)

            if float(top_p) < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                cutoff = cumsum > float(top_p)
                cutoff[..., 0] = False
                sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                sampled = torch.multinomial(sorted_probs, 1)
                next_token = sorted_idx.gather(-1, sampled)
            else:
                next_token = torch.multinomial(probs, 1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        decode_ids[:, t] = next_token.squeeze(1).to("cpu")
        next_input_ids = next_token  

    collector.stop()
    return decode_ids


# ============================================================
# 6) CLI / main
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Collect routing logits during fixed-length decode.",
    )

    # model + dataset
    p.add_argument("--model", type=str, required=True,
                  choices=["Qwen3_30B_A3B", "OLMoE_1B_7B_0125_Instruct", "DeepSeek_v2_Lite_Chat"])
    p.add_argument("--dataset", type=str, required=True, choices=list(DATASET_CFG.keys()))
    p.add_argument("--split", type=str, default=None)

    # io
    p.add_argument("--save_root", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "Routing_history"),
               help="Root dir for routing history. Default: <script_dir>/Routing_history")
    p.add_argument("--save_name", type=str, default=None,
                  help="Optional file name override. Default: {model}_{dataset}.pt")

    # workload
    p.add_argument("--num_questions", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_prompt_tokens", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--top_k", type=int, default=8, help="routing top-k. Qwen3_30B_A3B, OLMoE_1B_7B_0125_Instruct to 8. DeepSeek_v2_Lite_Chat to 6")

    # generation (EOS stopping disabled by design)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)

    # routing storage
    p.add_argument("--store_full_logits", action="store_true",
                  help="Store full logits (T,1,L,E). If off: store top-k idx/val.")
    p.add_argument("--routing_store_dtype", type=str, default="float16",
                  choices=["float16", "bfloat16", "float32"])

    # model load
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--model_dtype", type=str, default="bfloat16",
                  choices=["bfloat16", "float16", "float32"])

    return p.parse_args()


def main():
    args = parse_args()

    # ----------------------------
    # Step 1: enforce paper settings
    # ----------------------------
    model_id = resolve_model_id(args.model)
    model_dtype = resolve_dtype(args.model_dtype)
    routing_dtype = resolve_dtype(args.routing_store_dtype)

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_SAVE_ROOT = os.path.join(SCRIPT_DIR, "Routing_history")

    model_dir = os.path.join(args.save_root, args.model)
    os.makedirs(model_dir, exist_ok=True)
    save_name = args.save_name or f"{args.model}_{args.dataset}.pt"
    save_path = os.path.join(
        model_dir,
        f"{args.model}_{args.dataset}.pt"
    )

    if args.model in ("DeepSeek_v2_Lite_Chat"):
        patch_dynamic_cache_for_deepseek()

    # ----------------------------
    # Step 2: load tokenizer
    # ----------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # ----------------------------
    # Step 3: load model (Qwen3_30B_A3B: disable SDPA if needed)
    # ----------------------------
    model_kwargs = dict(
        torch_dtype=model_dtype,
        device_map=args.device_map,
        trust_remote_code=True,
        use_cache=True,
    )
    if args.model in ("Qwen3_30B_A3B"):
        model_kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()

    eos_id = tokenizer.eos_token_id or getattr(model.config, "eos_token_id", None)
    if eos_id is None:
        raise ValueError("EOS token id not found.")

    # ----------------------------
    # Step 4: discover router gates & install hooks
    # ----------------------------
    base = getattr(model, "model", None)
    if base is None or not hasattr(base, "layers"):
        raise RuntimeError("model.model.layers not found.")
    n_layers = len(base.layers)

    gates_per_layer, E = find_gate_modules(model)
    collector = RoutingCollector(
        n_layers=n_layers,
        n_experts=E,
        max_steps=args.max_new_tokens,
        top_k=args.top_k,
        store_full_logits=args.store_full_logits,
        store_dtype=routing_dtype,
    )
    hooks = register_gate_hooks(gates_per_layer, collector)

    # ----------------------------
    # Step 5: load dataset
    # ----------------------------
    cfg = DATASET_CFG[args.dataset]
    split = args.split if args.split is not None else cfg["default_split"]
    questions = load_questions(args.dataset, split, args.num_questions)

    # ----------------------------
    # Step 6: run (prefill + fixed decode), collect routing logs
    # ----------------------------
    prompt_len_list: List[int] = []
    input_ids_prompt_list: List[torch.Tensor] = []
    gen_ids_decode_list: List[torch.Tensor] = []
    routing_decode_list: List[Dict[str, Any]] = []
    eos_pos_list: List[int] = []

    print(f"[INFO] model_id={model_id}")
    print(f"[INFO] device_map={args.device_map}")
    print(f"[INFO] layers={n_layers} experts(E)={E}")
    if args.model in ("Qwen3_30B_A3B"):
        print("[INFO] attn_implementation=eager (SDPA disabled)")
    print(f"[INFO] batch_size={args.batch_size} prompt_max={args.max_prompt_tokens} decode_T={args.max_new_tokens}")
    print(f"[INFO] routing_store={'full_logits' if args.store_full_logits else 'topk'} (k={args.top_k}) dtype={routing_dtype}")

    for batch_questions in tqdm(list(chunked(questions, args.batch_size)), desc="Routing collection"):
        enc = tokenizer(
            batch_questions,
            add_special_tokens=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_prompt_tokens,
        )
        prompt_ids = enc.input_ids                       
        attn_mask = enc.attention_mask.to(torch.long)   

        B = prompt_ids.size(0)
        prompt_lens = attn_mask.sum(dim=1).tolist()

        collector.reset_for_batch(B)

        decode_ids = decode_fixed_steps(
            model=model,
            prompt_input_ids=prompt_ids,
            prompt_attention_mask=attn_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            collector=collector,
        )

        for i in range(B):
            p_len = int(prompt_lens[i])
            p_ids_trim = prompt_ids[i : i + 1, :p_len].clone().to("cpu")  
            d_ids = decode_ids[i : i + 1, :].clone().to("cpu")          

            eos_pos = -1
            pos = (d_ids[0] == eos_id).nonzero(as_tuple=False)
            if pos.numel() > 0:
                eos_pos = int(pos[0].item())

            routing_pack = collector.export_per_sample(i)

            prompt_len_list.append(p_len)
            input_ids_prompt_list.append(p_ids_trim)
            gen_ids_decode_list.append(d_ids)
            routing_decode_list.append(routing_pack)
            eos_pos_list.append(eos_pos)

    # ----------------------------
    # Step 7: serialize payload
    # ----------------------------
    payload = {
        # identity
        "model_alias": args.model,
        "hf_model_id": model_id,
        "dataset_name": args.dataset,
        "dataset_id": cfg["path"],
        "dataset_config": cfg["name"],
        "dataset_split": split,

        # workload config
        "num_questions": len(prompt_len_list),
        "batch_size": args.batch_size,
        "max_prompt_tokens": args.max_prompt_tokens,
        "fixed_decode_tokens": args.max_new_tokens,

        # generation config (EOS stopping disabled)
        "do_sample": bool(args.do_sample),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),

        # tokens
        "eos_token_id": int(eos_id),
        "pad_token_id": int(tokenizer.pad_token_id),

        # schema notes
        "generation_note": "EOS stopping disabled; decode length fixed.",
        "routing_layout_note": (
            "routing_decode[i] is dict. "
            "type=='full_logits': logits (T,1,L,E). "
            "type=='topk': idx/val (T,1,L,K)."
        ),

        # per-sample data
        "prompt_len": prompt_len_list,
        "input_ids_prompt": input_ids_prompt_list,  
        "gen_ids_decode": gen_ids_decode_list,      
        "eos_pos_in_decode": eos_pos_list,
        "routing_decode": routing_decode_list,
    }

    torch.save(payload, save_path)
    print(f"[DONE] saved: {save_path}")

    # cleanup hooks
    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass


if __name__ == "__main__":
    main()
