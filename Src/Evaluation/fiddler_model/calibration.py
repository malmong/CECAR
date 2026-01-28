"""
DES Calibration (Top4/5/6, layer-wise)

This script:
  1) Loads a non-expert base model (HF AutoModelForCausalLM)
  2) Injects MoE experts + (optional) shared experts
  3) Collects router statistics and computes layer-wise DES thresholds
  4) Saves results to JSON and prints an inference-only summary
"""

import argparse
import json
import os
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig
import transformers

from calibration_util import get_loaders


# =========================
# Logging / warnings
# =========================
transformers.utils.logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=r"The module name .* is not a valid Python identifier.*")
warnings.filterwarnings("ignore", message=r"Some weights of .* were not initialized from the model checkpoint.*")
warnings.filterwarnings("ignore", message=r"`torch_dtype` is deprecated! Use `dtype` instead!")


# =========================
# Model loading
# =========================
def get_model(model_path: str, dtype: str = "bf16"):
    print(f"Loading model from: {os.path.abspath(model_path)}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    if dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp32":
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float16

    # Disable KV cache for calibration forward passes
    if hasattr(config, "use_cache"):
        config.use_cache = False

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    if hasattr(model, "config"):
        model.config.use_cache = False

    # Standardize seqlen field (used for truncation in calibration)
    if not hasattr(model, "seqlen"):
        seqlen = getattr(getattr(model, "config", None), "max_position_embeddings", None)
        model.seqlen = int(seqlen) if seqlen is not None else 1024

    model.eval()
    return model


def _get_input_device(model) -> torch.device:
    """Prefer embedding device if available (better with device_map=auto)."""
    try:
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            return model.model.embed_tokens.weight.device
    except Exception:
        pass
    return next(model.parameters()).device


# =========================
# Expert discovery / injection
# =========================
def _find_expert_container(layer):
    """
    Locate expert container within a transformer layer.
    Returns: (experts_container, name_str) or (None, None)
    """
    if hasattr(layer, "mlp"):
        mlp = layer.mlp
        if hasattr(mlp, "experts"):
            return mlp.experts, "mlp.experts"
        for attr in ["moe", "block_sparse_moe", "sparse_moe", "ffn", "mlp"]:
            if hasattr(mlp, attr):
                sub = getattr(mlp, attr)
                if hasattr(sub, "experts"):
                    return sub.experts, f"mlp.{attr}.experts"

    for attr in ["moe", "block_sparse_moe", "sparse_moe", "ffn"]:
        if hasattr(layer, attr):
            sub = getattr(layer, attr)
            if hasattr(sub, "experts"):
                return sub.experts, f"layer.{attr}.experts"

    return None, None


def inject_experts(model, expert_path: str):
    """
    Inject per-expert weights from:
        {expert_path}/layer{i}_expert{j}.pt
    into the experts container discovered from each layer.

    This function is a no-op for layers without experts.
    """
    base = getattr(model, "model", None)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise RuntimeError("Model has no model.layers")

    expert_path = os.path.abspath(expert_path)
    if not os.path.isdir(expert_path):
        raise FileNotFoundError(f"expert_path not found: {expert_path}")

    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    n_layer = len(layers)

    # find first experts container to infer n_expert
    first_idx = None
    experts0 = None
    experts0_name = None
    for i in range(n_layer):
        exp, exp_name = _find_expert_container(layers[i])
        if exp is not None:
            first_idx = i
            experts0 = exp
            experts0_name = exp_name
            break

    if experts0 is None:
        raise RuntimeError("Could not find any experts container in model layers.")

    n_expert = len(experts0)

    print(
        f"Injecting experts from {expert_path} "
        f"({n_layer} layers, {n_expert} experts, first_moe_layer={first_idx}, container={experts0_name}) ..."
    )

    loaded_layers = 0
    loaded_experts = 0

    for i in range(n_layer):
        experts_i, _ = _find_expert_container(layers[i])
        if experts_i is None:
            continue

        layer_loaded_any = False
        for j in range(n_expert):
            path = os.path.join(expert_path, f"layer{i}_expert{j}.pt")
            if not os.path.exists(path):
                continue

            obj = torch.load(path, map_location=dev, weights_only=False)

            if isinstance(obj, dict):
                experts_i[j].load_state_dict(obj, strict=True)
                experts_i[j].to(device=dev, dtype=dtype)
                experts_i[j].eval()
            else:
                m = obj.to(device=dev, dtype=dtype)
                m.eval()
                experts_i[j] = m

            loaded_experts += 1
            layer_loaded_any = True

        if layer_loaded_any:
            loaded_layers += 1
        print(f"  layer {i+1}/{n_layer}", end="\r")

    print(f"\nDone injecting experts. loaded_layers={loaded_layers}, loaded_experts={loaded_experts}")


def inject_shared_experts(model, shared_expert_path: Optional[str]):
    """
    Optional: inject per-layer shared expert from:
        {shared_expert_path}/layer{i}_shared_expert.pt

    This function only injects if the model defines a shared-expert slot
    under layer.mlp (e.g., mlp.shared_expert / shared_experts / expert_shared).
    """
    if not shared_expert_path:
        print("shared_expert_path not provided; skip injecting shared experts.")
        return

    shared_expert_path = os.path.abspath(shared_expert_path)
    if not os.path.isdir(shared_expert_path):
        print(f"shared_expert_path does not exist; skip: {shared_expert_path}")
        return

    base = getattr(model, "model", None)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise RuntimeError("Model has no model.layers")

    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    n_layer = len(layers)
    print(f"Injecting shared experts from {shared_expert_path} ({n_layer} layers) ...")

    loaded = 0
    for i in range(n_layer):
        path = os.path.join(shared_expert_path, f"layer{i}_shared_expert.pt")
        if not os.path.exists(path):
            continue

        obj = torch.load(path, map_location=dev, weights_only=False)

        mlp = getattr(layers[i], "mlp", None)
        if mlp is None:
            continue

        cand = None
        cand_name = None
        for name in ["shared_expert", "shared_experts", "expert_shared"]:
            if hasattr(mlp, name):
                cand = getattr(mlp, name)
                cand_name = name
                break

        if cand is None:
            continue

        try:
            if isinstance(obj, dict):
                cand.load_state_dict(obj, strict=True)
                cand.to(device=dev, dtype=dtype)
                cand.eval()
            else:
                m = obj.to(device=dev, dtype=dtype)
                m.eval()
                setattr(mlp, cand_name, m)
            loaded += 1
        except Exception as e:
            print(f"\nWarning: failed to inject shared expert for layer {i}: {e}")

        print(f"  layer {i+1}/{n_layer}", end="\r")

    print(f"\nDone injecting shared experts. loaded={loaded}")


# =========================
# Router module discovery
# =========================
def find_router_modules_with_layers(model) -> List[Tuple[int, torch.nn.Module]]:
    """
    Returns list of (layer_idx, gate_module) from model.model.layers.
    """
    base = getattr(model, "model", None)
    layers = getattr(base, "layers", None)
    if layers is None:
        return []

    out: List[Tuple[int, torch.nn.Module]] = []
    for i, layer in enumerate(layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
            out.append((i, layer.mlp.gate))
            continue

        for attr in ["moe", "block_sparse_moe", "sparse_moe", "ffn", "mlp"]:
            if hasattr(layer, attr):
                sub = getattr(layer, attr)
                if hasattr(sub, "gate"):
                    out.append((i, sub.gate))
                    break

    uniq = []
    seen = set()
    for li, m in out:
        key = (li, id(m))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((li, m))
    return uniq


# =========================
# DES calibration (Top4/5/6)
# =========================
def _collect_ratios_top4_6(
    model,
    dataloader,
    num_samples: int,
    ratio_eps: float = 1e-6,
):
    """
    Collect layer-wise ratio statistics:
        r45 = w4 / w5
        r56 = w5 / w6
    where w4,w5,w6 are the 4th/5th/6th largest routing weights.

    Returns:
        ratios_4_5, ratios_5_6, meta
    """
    router = find_router_modules_with_layers(model)
    if len(router) == 0:
        raise RuntimeError("No MoE gate/router modules found in model.model.layers.")

    input_device = _get_input_device(model)

    ratios_4_5: Dict[int, List[torch.Tensor]] = {}
    ratios_5_6: Dict[int, List[torch.Tensor]] = {}
    dbg_once = {"done": False}

    def make_gate_hook(layer_idx: int):
        def gate_hook(module, inputs, output):
            hs = inputs[0]
            hs_flat = hs.detach().reshape(-1, hs.shape[-1])

            w: Optional[torch.Tensor] = None

            # 1) gate returns tuple/list and 2nd element is weights (common MoE)
            if isinstance(output, (tuple, list)) and len(output) >= 2 and torch.is_tensor(output[1]):
                topk_weight = output[1].detach().float()
                if topk_weight.dim() == 3:
                    topk_weight = topk_weight.reshape(-1, topk_weight.shape[-1])
                elif topk_weight.dim() != 2:
                    return
                w = topk_weight

                if not dbg_once["done"]:
                    try:
                        print(f"[DBG] gate type: {type(module)}")
                        print(f"[DBG] gate output tuple len: {len(output)}")
                        if torch.is_tensor(output[0]):
                            print(f"[DBG] out[0] shape: {tuple(output[0].shape)} dtype={output[0].dtype}")
                        print(f"[DBG] out[1] shape: {tuple(output[1].shape)} dtype={output[1].dtype}")
                    except Exception:
                        pass
                    dbg_once["done"] = True

            # 2) gate returns logits tensor
            elif torch.is_tensor(output):
                logits = output.detach().float()
                if logits.dim() == 3:
                    logits = logits.reshape(-1, logits.shape[-1])
                if logits.size(-1) >= 6:
                    w = torch.softmax(logits, dim=-1)

            # 3) fallback: linear(h, W)
            else:
                if hasattr(module, "weight") and module.weight is not None:
                    logits = F.linear(hs_flat.float(), module.weight.detach().float())
                    if logits.size(-1) >= 6:
                        w = torch.softmax(logits, dim=-1)

            if w is None or w.size(-1) < 6:
                return

            w_sorted = torch.sort(w, dim=-1, descending=True).values
            r45 = (w_sorted[:, 3] / (w_sorted[:, 4] + ratio_eps)).reshape(-1).cpu()
            r56 = (w_sorted[:, 4] / (w_sorted[:, 5] + ratio_eps)).reshape(-1).cpu()

            ratios_4_5.setdefault(layer_idx, []).append(r45)
            ratios_5_6.setdefault(layer_idx, []).append(r56)

        return gate_hook

    handles = [gate.register_forward_hook(make_gate_hook(layer_idx)) for layer_idx, gate in router]

    used_batches = 0
    seq_lens: List[int] = []

    cfg = getattr(model, "config", None)
    pad_id = getattr(cfg, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(cfg, "eos_token_id", 0)
    pad_id = int(pad_id) if pad_id is not None else 0
    vocab_size = int(getattr(cfg, "vocab_size", 0)) if cfg is not None else 0

    for i, batch in enumerate(tqdm(dataloader, desc="Calibrating")):
        if i >= num_samples:
            break

        inp = batch[0] if isinstance(batch, (tuple, list)) else batch
        if inp.dim() == 1:
            inp = inp.unsqueeze(0)

        if inp.shape[1] > model.seqlen:
            inp = inp[:, : model.seqlen]

        bad = (inp < 0)
        if vocab_size > 0:
            bad = bad | (inp >= vocab_size)

        if bool(bad.any().item()):
            inp = inp.clone()
            inp[bad] = pad_id

        attn = (~bad) & inp.ne(pad_id)
        if attn.dim() == 2:
            all_pad = attn.sum(dim=1) == 0
            if all_pad.any():
                attn = attn.clone()
                attn[all_pad] = True

        inp = inp.to(input_device)
        attn = attn.to(device=input_device, dtype=torch.bool)

        seq_lens.append(int(inp.shape[1]))
        used_batches += 1

        _ = model(input_ids=inp, attention_mask=attn)

    for h in handles:
        h.remove()

    layers_present = sorted(set(ratios_4_5.keys()) & set(ratios_5_6.keys()))
    if not layers_present:
        raise RuntimeError("No ratio stats collected. Check gate hook output parsing and model MoE structure.")

    meta = {
        "used_batches": int(used_batches),
        "avg_seq_len": float(np.mean(seq_lens)) if seq_lens else 0.0,
        "n_router_modules": int(len(router)),
        "effective_seqlen": int(model.seqlen),
        "layers_present": layers_present,
    }
    return ratios_4_5, ratios_5_6, meta


def calibrate_des_top4_6_layerwise(
    ratios_4_5: Dict[int, List[torch.Tensor]],
    ratios_5_6: Dict[int, List[torch.Tensor]],
):
    """
    Compute layer-wise thresholds:
        mu_4_5 = median(w4/w5)
        mu_5_6 = median(w5/w6)

    Returns:
        layer_results: list of dicts with thresholds + debug stats
    """
    layers_present = sorted(set(ratios_4_5.keys()) & set(ratios_5_6.keys()))
    layer_results = []

    for li in layers_present:
        r45 = torch.cat(ratios_4_5[li], dim=0).numpy().astype(np.float64)
        r56 = torch.cat(ratios_5_6[li], dim=0).numpy().astype(np.float64)

        mu1 = float(np.median(r45))
        mu2 = float(np.median(r56))

        top4 = (r45 > mu1)
        top5 = (~top4) & (r56 > mu2)
        top6 = (~top4) & (~top5)

        layer_results.append(
            {
                "layer": int(li),
                "mu_4_5": mu1,
                "mu_5_6": mu2,
                "n_tokens": int(r45.shape[0]),
                "top4_rate": float(top4.mean()),
                "top5_rate": float(top5.mean()),
                "top6_rate": float(top6.mean()),
                "r45_p10": float(np.quantile(r45, 0.10)),
                "r45_p50": float(np.quantile(r45, 0.50)),
                "r45_p90": float(np.quantile(r45, 0.90)),
                "r56_p10": float(np.quantile(r56, 0.10)),
                "r56_p50": float(np.quantile(r56, 0.50)),
                "r56_p90": float(np.quantile(r56, 0.90)),
            }
        )

    return layer_results


def _make_console_summary(method: str, min_k: int, max_k: int, meta: dict, layer_results: list):
    """Compact, inference-only summary (thresholds only)."""
    return {
        "method": method,
        "min_k": min_k,
        "max_k": max_k,
        "n_router_modules": meta["n_router_modules"],
        "effective_seqlen": meta["effective_seqlen"],
        "layers": [{"layer": x["layer"], "mu_4_5": x["mu_4_5"], "mu_5_6": x["mu_5_6"]} for x in layer_results],
    }


def _save_json(obj: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)
    print(f"Saved: {os.path.abspath(output_path)}")


@torch.no_grad()
def run_calibration_and_save(
    model,
    dataloader,
    model_name: str,
    dataset_name: str,
    output_name: str,
    num_samples: int,
    out_dir: str,
    ratio_eps: float = 1e-6,
):
    """
    End-to-end pipeline:
      1) collect ratios
      2) compute thresholds
      3) print inference-only summary
      4) save full debug JSON
    """
    print(">>> Start Calibration (DES: Top4/5/6, layer-wise)")

    ratios_4_5, ratios_5_6, meta = _collect_ratios_top4_6(
        model=model,
        dataloader=dataloader,
        num_samples=num_samples,
        ratio_eps=ratio_eps,
    )

    layer_results = calibrate_des_top4_6_layerwise(ratios_4_5, ratios_5_6)

    results = {
        "method": "des_top4_6_layerwise",
        "min_k": 4,
        "max_k": 6,
        **meta,
        "layers": layer_results,
    }

    console = _make_console_summary(
        method=results["method"],
        min_k=results["min_k"],
        max_k=results["max_k"],
        meta=meta,
        layer_results=layer_results,
    )

    print("\n" + "=" * 60)
    print("Calibration Result (console: inference-only)")
    print(json.dumps(console, indent=4))
    print("=" * 60 + "\n")

    safe_model = model_name.replace("/", "_")
    safe_data = dataset_name.replace("/", "_")
    out_path = os.path.join(out_dir, f"{safe_model}_{safe_data}_{output_name}")
    _save_json(results, out_path)


# =========================
# CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="DES calibration for MoE routers (Top4/5/6, layer-wise).")

    p.add_argument("--model_name", type=str, required=True,
                   help="Identifier used for output naming (e.g., Qwen3_30B_A3B, DeepSeek_v2_Lite_Chat).")
    p.add_argument("--non_expert_path", type=str, required=True, help="Path to non-expert base model.")
    p.add_argument("--expert_path", type=str, required=True, help="Directory containing expert checkpoints.")
    p.add_argument("--shared_expert_path", type=str, default=None,
                   help="Optional directory containing shared expert checkpoints (model-dependent).")

    p.add_argument("--dataset", type=str, required=True, choices=["mmlu_ccsc", "arc_challenge", "mathqa"],
                   help="Calibration dataset name.")
    p.add_argument("--nsamples", type=int, default=32, help="Number of calibration batches.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--seqlen", type=int, default=512, help="Sequence length used for calibration.")
    p.add_argument("--output_name", type=str, default="top4_6_thresholds_layerwise.json",
                   help="Output JSON filename suffix.")
    p.add_argument("--out_dir", type=str, default="./MCMOE", help="Output directory.")
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"], help="Model dtype.")

    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = get_model(args.non_expert_path, dtype=args.dtype)
    print("[CHECK] model param dtype:", next(model.parameters()).dtype)

    inject_experts(model, args.expert_path)
    inject_shared_experts(model, args.shared_expert_path)

    model.seqlen = int(args.seqlen)

    dataloader, _ = get_loaders(
        model_id=args.model_name,
        dataset_name=args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model_path=args.non_expert_path,
        seqlen=args.seqlen,
    )

    run_calibration_and_save(
        model=model,
        dataloader=dataloader,
        model_name=args.model_name,
        dataset_name=args.dataset,
        output_name=args.output_name,
        num_samples=args.nsamples,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
