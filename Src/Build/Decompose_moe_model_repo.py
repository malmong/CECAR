import os
import shutil
import argparse
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download


# ============================================================
# Algorithm: MoE Model Decomposition
#   Input : HF repo_id (MoE model), model family selector
#   Output: (1) non-expert "skeleton" model dir (config+tokenizer+code)
#           (2) expert shards: per-layer routed experts (and shared expert if exists)
# ============================================================


def get_deepseek_dense_prefix(model_config) -> int:
    """
    DeepSeek-V2-Lite has a dense prefix (layer 0..k-1) which is non-MoE.
    first_k_dense_replace is used by DeepSeek configs to indicate this.
    """
    k = getattr(model_config, "first_k_dense_replace", 0)
    return int(k or 0)


def copy_repo_runtime_files(repo_snapshot_dir: str, dst_dir: str):
    """
    Copy remote-code (modeling_*.py etc.) and related runtime files into dst_dir
    so that `trust_remote_code=True` works purely from local dir.
    """
    os.makedirs(dst_dir, exist_ok=True)

    # copy all python files
    for root, _, files in os.walk(repo_snapshot_dir):
        for fn in files:
            if fn.endswith(".py"):
                src = os.path.join(root, fn)
                rel = os.path.relpath(src, repo_snapshot_dir)
                dst = os.path.join(dst_dir, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)

    # copy additional runtime files (only if missing)
    extra_ext = {".json", ".txt", ".model", ".tiktoken", ".pyi"}
    for root, _, files in os.walk(repo_snapshot_dir):
        for fn in files:
            ext = os.path.splitext(fn)[1]
            if ext in extra_ext:
                src = os.path.join(root, fn)
                rel = os.path.relpath(src, repo_snapshot_dir)
                dst = os.path.join(dst_dir, rel)
                if not os.path.exists(dst):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)


def resolve_paths(model_name: str):
    """
    Centralized path definition to keep the script easy to read and modify.
    """
    if model_name == "Qwen3_30B_A3B":
        return dict(
            repo_id="Qwen/Qwen3-30B-A3B",
            save_expert_dir="./models/Qwen3_30B_A3B/experts",
            save_non_expert_dir="./models/Qwen3_30B_A3B/non_expert",
            save_shared_expert_dir=None,
        )
    if model_name == "OLMoE_1B_7B_0125_Instruct":
        return dict(
            repo_id="allenai/OLMoE-1B-7B-0125-Instruct",
            save_expert_dir="./models/OLMoE_1B_7B_0125_Instruct/experts",
            save_non_expert_dir="./models/OLMoE_1B_7B_0125_Instruct/non_expert",
            save_shared_expert_dir=None,
        )
    if model_name == "DeepSeek_v2_Lite_Chat":
        return dict(
            repo_id="deepseek-ai/DeepSeek-V2-Lite-Chat",
            save_expert_dir="./models/DeepSeek_v2_Lite_Chat/experts",
            save_non_expert_dir="./models/DeepSeek_v2_Lite_Chat/non_expert",
            save_shared_expert_dir="./models/DeepSeek_v2_Lite_Chat/shared_expert",
        )
    raise ValueError(f"Unknown model_name: {model_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Decompose MoE model into (non-expert skeleton) + (expert shards)."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["Qwen3_30B_A3B", "OLMoE_1B_7B_0125_Instruct", "DeepSeek_v2_Lite_Chat"],
        required=True,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional HF revision/commit hash for reproducibility.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional HF cache dir for snapshot_download.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="If set, do not fetch from internet; use local cache only.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------
    # 0) Resolve model repo_id and output directories
    # ------------------------------------------------------------
    paths = resolve_paths(args.model_name)
    repo_id = paths["repo_id"]
    save_expert_dir = paths["save_expert_dir"]
    save_non_expert_dir = paths["save_non_expert_dir"]
    save_shared_expert_dir = paths["save_shared_expert_dir"]

    os.makedirs(save_non_expert_dir, exist_ok=True)
    os.makedirs(save_expert_dir, exist_ok=True)
    if save_shared_expert_dir is not None:
        os.makedirs(save_shared_expert_dir, exist_ok=True)

    # ------------------------------------------------------------
    # 1) Snapshot download (includes remote modeling code)
    # ------------------------------------------------------------
    repo_snapshot_dir = snapshot_download(
        repo_id=repo_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_dir=None,
        local_dir_use_symlinks=False,
        local_files_only=args.local_files_only,
    )

    # ------------------------------------------------------------
    # 2) Load model/tokenizer from snapshot dir (trust_remote_code)
    # ------------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        repo_snapshot_dir,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=True,
        local_files_only=True,  
    )
    tokenizer = AutoTokenizer.from_pretrained(
        repo_snapshot_dir,
        trust_remote_code=True,
        local_files_only=True,
    )

    tokenizer.save_pretrained(save_non_expert_dir)
    print(f"[OK] Saved tokenizer -> {save_non_expert_dir}")

    # ------------------------------------------------------------
    # 3) Extract expert modules and detach from model
    #    (family-specific)
    # ------------------------------------------------------------
    layers = model.model.layers

    if args.model_name == "DeepSeek_v2_Lite_Chat":
        dense_prefix = get_deepseek_dense_prefix(model.config)

        n_layers = int(model.config.num_hidden_layers)
        n_routed = int(getattr(model.config, "n_routed_experts", 0))
        n_shared = int(getattr(model.config, "n_shared_experts", 0))

        for layer_idx in range(dense_prefix, n_layers):
            mlp = layers[layer_idx].mlp

            # shared expert
            if (
                n_shared > 0
                and hasattr(mlp, "shared_experts")
                and mlp.shared_experts is not None
            ):
                shared_expert = mlp.shared_experts.cpu()
                out_path = os.path.join(
                    save_shared_expert_dir, f"layer{layer_idx}_shared_expert.pt"
                )
                torch.save(shared_expert, out_path)
                mlp.shared_experts = None

            # routed experts
            if n_routed > 0 and hasattr(mlp, "experts") and mlp.experts is not None:
                for expert_idx in range(n_routed):
                    expert = mlp.experts[expert_idx].cpu()
                    out_path = os.path.join(
                        save_expert_dir, f"layer{layer_idx}_expert{expert_idx}.pt"
                    )
                    torch.save(expert, out_path)
                    mlp.experts[expert_idx] = None

    else:
        # Qwen/OLMoE family (routed experts only)
        n_layers = int(getattr(model.config, "num_hidden_layers", len(layers)))

        num_experts = getattr(model.config, "num_experts", None)
        if num_experts is None:
            num_experts = getattr(model.config, "n_routed_experts", None)
        if num_experts is None:
            raise ValueError("Cannot find number of experts in config for this model.")
        num_experts = int(num_experts)

        for layer_idx in range(n_layers):
            mlp = layers[layer_idx].mlp
            if not hasattr(mlp, "experts") or mlp.experts is None:
                continue

            for expert_idx in range(num_experts):
                expert = mlp.experts[expert_idx].cpu()
                out_path = os.path.join(
                    save_expert_dir, f"layer{layer_idx}_expert{expert_idx}.pt"
                )
                torch.save(expert, out_path)
                mlp.experts[expert_idx] = None

    # ------------------------------------------------------------
    # 4) Make skeleton config consistent (remove MoE sizes)
    # ------------------------------------------------------------
    if args.model_name == "DeepSeek_v2_Lite_Chat":
        if hasattr(model.config, "moe_intermediate_size"):
            model.config.moe_intermediate_size = 0
    elif args.model_name == "Qwen3_30B_A3B":
        if hasattr(model.config, "moe_intermediate_size"):
            model.config.moe_intermediate_size = 0
    elif args.model_name == "OLMoE_1B_7B_0125_Instruct":
        if hasattr(model.config, "intermediate_size"):
            model.config.intermediate_size = 0

    # Defensive: ensure no references remain
    for layer in model.model.layers:
        if hasattr(layer, "mlp"):
            if hasattr(layer.mlp, "shared_experts"):
                layer.mlp.shared_experts = None
            if hasattr(layer.mlp, "experts"):
                layer.mlp.experts = None

    # ------------------------------------------------------------
    # 5) Save skeleton model + config
    # ------------------------------------------------------------
    model.save_pretrained(save_non_expert_dir)
    model.config.save_pretrained(save_non_expert_dir)

    # ------------------------------------------------------------
    # 6) Copy remote-code runtime files into skeleton dir
    # ------------------------------------------------------------
    copy_repo_runtime_files(repo_snapshot_dir, save_non_expert_dir)

    print(f"[DONE] Saved non-expert skeleton -> {save_non_expert_dir}")
    print(f"[DONE] Saved expert shards          -> {save_expert_dir}")
    if save_shared_expert_dir is not None:
        print(f"[DONE] Saved shared expert shards   -> {save_shared_expert_dir}")


if __name__ == "__main__":
    main()
