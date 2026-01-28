import argparse
import os
import json
import time

from transformers import AutoTokenizer
from DeepSeek_v2_Lite_Chat import FiddlerDeepseek


# -----------------------------------------------------------------------------
# Prompt loading
# -----------------------------------------------------------------------------
def load_prompts_from_json(test_task: str) -> list[tuple[str, str]]:
    """
    Load 5 prompts from ../Prompt/{test_task}_prompts_0_4.json.

    Expected JSON:
      { "<task>_0": "...", ..., "<task>_4": "..." }

    Returns:
      [(key, prompt)] in deterministic order: task_0..task_4
    """
    prompt_path = os.path.join("..", "Prompt", f"{test_task}_prompts_0_4.json")
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict) or not obj:
        raise ValueError(f"Invalid prompt JSON (expected non-empty dict): {prompt_path}")

    items: list[tuple[str, str]] = []
    for i in range(5):
        k = f"{test_task}_{i}"
        if k not in obj:
            raise KeyError(f"Missing key '{k}' in {prompt_path}")
        v = obj[k]
        if not isinstance(v, str):
            raise ValueError(f"Prompt value for '{k}' must be a string.")
        items.append((k, v))
    return items


# -----------------------------------------------------------------------------
# Argparse
# -----------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Paths
    parser.add_argument(
        "--model",
        type=str,
        default="../../Build/models/DeepSeek_v2_Lite_Chat/non_expert",
        help="Base model path (non-expert weights).",
    )
    parser.add_argument(
        "--expert_path",
        type=str,
        default="../../Build/models/DeepSeek_v2_Lite_Chat/experts",
        help="Expert checkpoint path.",
    )
    parser.add_argument(
        "--shared_expert_path",
        type=str,
        default="../../Build/models/DeepSeek_v2_Lite_Chat/shared_expert",
        help="Shared-expert checkpoint path (DeepSeek dense/shared component).",
    )
    parser.add_argument(
        "--cpu-offload",
        type=int,
        default=1,
        choices=[0, 1],
        help="0: execute on GPU (baseline), 1: CPU offload.",
    )

    # Input selection
    parser.add_argument("--input", type=str, default="", help="Input text (ignored if --test-task set).")
    parser.add_argument(
        "--test-task",
        type=str,
        default=None,
        choices=["humaneval", "mbpp", "gpqa", "math500", "None"],
        help="If set, load prompts from ../Prompt/{test_task}_prompts_0_4.json (5 prompts).",
    )
    parser.add_argument(
        "--task-num",
        type=int,
        default=None,
        choices=[0, 1, 2, 3, 4],
        help="If set with --test-task, run only {test_task}_{task_num}.",
    )

    # Generation
    parser.add_argument("--n-token", type=int, default=2048, help="Number of tokens to generate.")
    parser.add_argument("--beam-width", type=int, default=1, help="Beam search width.")

    # Cache
    parser.add_argument("--cache-size", type=int, default=24, help="Expert cache size per layer.")
    parser.add_argument(
        "--cache-policy",
        type=str,
        default="ML_CECAR",
        help="Cache policy (lru, lfu, lifo, ML_CECAR, ML_FlashMoE, etc.)",
    )
    parser.add_argument(
        "--ffn-model-path",
        type=str,
        default="../../Train/Pre_trained_FFN/DeepSeek_v2_Lite_Chat/",
        help="Base path to FFN eviction models (for ML cache policies).",
    )

    # DES / ODP / bonus
    parser.add_argument("--margin-topk-idx", type=int, default=8)
    parser.add_argument("--mode", type=str, default="none", choices=["des", "odp", "none"])
    parser.add_argument("--bonus-strategy", type=str, default="none", choices=["cecar", "mocce", "none"])
    parser.add_argument("--des-mu1", type=float, default=1.5)
    parser.add_argument("--des-mu2", type=float, default=1.5)
    parser.add_argument(
        "--des-thresholds-path",
        type=str,
        default="../../Evaluation/fiddler_model/MCMOE/DeepSeek_v2_Lite_Chat_mmlu_ccsc_top4_6_thresholds.json",
        help="DES thresholds JSON (global or layerwise).",
    )
    parser.add_argument("--topk-threshold", type=int, default=3)
    parser.add_argument("--do-renormalize", action="store_true")

    parser.add_argument(
        "--use-chat-template",
        dest="use_chat_template",
        action="store_true",
        help="Apply chat template (needed for GPQA/MATH500, not for HumanEval/MBPP).",
    )

    # Sampling
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--do-sample", action="store_true")

    return parser


# -----------------------------------------------------------------------------
# Normalization / derived args
# -----------------------------------------------------------------------------
def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    """
    - Treat '--test-task None' as not using test-task.
    - Auto-select DES thresholds file based on test_task stem mapping.
    - For ML cache policies, ffn_model_path := <base>/<policy>.
    """
    if args.test_task == "None":
        args.test_task = None

    if args.test_task is not None:
        task_to_stem = {
            "humaneval": "mmlu_ccsc",
            "mbpp": "mmlu_ccsc",
            "gpqa": "arc_challenge",
            "math500": "mathqa",
        }
        stem = task_to_stem.get(args.test_task)
        if stem is None:
            raise ValueError(f"Unsupported test task for auto thresholds: {args.test_task}")

        args.des_thresholds_path = os.path.join(
            "..", "..", "Evaluation", "fiddler_model", "MCMOE",
            f"DeepSeek_v2_Lite_Chat_{stem}_top4_6_thresholds.json",
        )

    if args.cache_policy in ("ML_CECAR", "ML_FlashMoE"):
        base = args.ffn_model_path.rstrip("/")
        args.ffn_model_path = os.path.join(base, args.cache_policy)

    return args


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def prepare_results_dir() -> None:
    os.makedirs("Results", exist_ok=True)
    with open("Results/generated_tokens.txt", "w", encoding="utf-8") as f:
        f.write(f"# reset at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("# ========================================\n")


def build_prompt_items(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.test_task is not None:
        items = load_prompts_from_json(args.test_task)
        if args.task_num is not None:
            wanted = f"{args.test_task}_{args.task_num}"
            items = [(k, v) for (k, v) in items if k == wanted]
            if len(items) != 1:
                raise KeyError(f"Failed to select prompt key '{wanted}' from loaded prompts.")
        return items

    if not args.input:
        raise ValueError("Either --test-task must be set or --input must be non-empty.")
    return [("single_input", args.input)]


def maybe_apply_chat_template(tokenizer: AutoTokenizer, text: str, enabled: bool) -> str:
    if not enabled:
        return text
    messages = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    args = build_argparser().parse_args()
    args = normalize_args(args)

    tick = time.time()
    model = FiddlerDeepseek(args)
    print(f"Model loading time: {time.time() - tick} seconds")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    prepare_results_dir()
    prompt_items = build_prompt_items(args)

    all_results = []
    for idx, (key, prompt) in enumerate(prompt_items, start=1):
        cur_input = maybe_apply_chat_template(tokenizer, prompt, args.use_chat_template)

        print("\n" + "=" * 80)
        print(f"[{idx}/{len(prompt_items)}] Running: {key}")
        print("=" * 80)

        prefill_time, decode_time, prefill_speed, token_speed, hit_rate, hit_rate_by_layer = model.generate(
            cur_input,
            output_token=args.n_token,
        )

        print(
            f"prefill_time: {prefill_time}, decode_time: {decode_time}, "
            f"hit_rate: {hit_rate}, prefill speed: {prefill_speed}, decode speed: {token_speed}"
        )
        print("hit_rate by layer")
        for i, hr in enumerate(hit_rate_by_layer):
            print(f"\tlayer_{i}: {hr}")

        all_results.append({
            "key": key,
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "prefill_speed": prefill_speed,
            "decode_speed": token_speed,
            "hit_rate": hit_rate,
            "hit_rate_by_layer": hit_rate_by_layer,
        })

    out_path = "Results/run_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
