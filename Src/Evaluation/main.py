import os
import sys
import json
import warnings
import argparse
import subprocess

import datasets
import torch
import transformers
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, HfArgumentParser

from bigcode_eval.arguments import EvalArguments
from bigcode_eval.evaluator import Evaluator


# ----------------------------
# OpenCompass inline configs
# ----------------------------
MATH500_CONFIG = r'''
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MATHDataset, MATHEvaluator, math_postprocess_v2

math_reader_cfg = dict(
    input_columns=['problem'],
    output_column='solution',
)

math_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='{problem}\nPlease reason step by step, and put your final answer within \\boxed{}.',
                ),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

math_eval_cfg = dict(
    evaluator=dict(type=MATHEvaluator, version='v2'),
    pred_postprocessor=dict(type=math_postprocess_v2),
)

datasets = [
    dict(
        type=MATHDataset,
        abbr='math_prm800k_500',
        path='opencompass/math',
        file_name='test_prm800k_500.json',
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg,
    )
]
'''

GPQA_CONFIG = r'''
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GPQADataset, GPQAEvaluator
from opencompass.utils import first_option_postprocess

gpqa_reader_cfg = dict(
    input_columns=['question', 'A', 'B', 'C', 'D'],
    output_column='answer',
)

gpqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='What is the correct answer to this question: {question}\nChoices:\n'
                           '(A){A}\n'
                           '(B){B}\n'
                           '(C){C}\n'
                           '(D){D}\n'
                           'Format your response as follows: "The correct answer is (insert answer here)"',
                ),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

gpqa_eval_cfg = dict(
    evaluator=dict(type=GPQAEvaluator),
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
)

datasets = [
    dict(
        type=GPQADataset,
        abbr='GPQA_diamond',
        path='./data/gpqa/',
        name='gpqa_diamond.csv',
        reader_cfg=gpqa_reader_cfg,
        infer_cfg=gpqa_infer_cfg,
        eval_cfg=gpqa_eval_cfg,
    )
]
'''


ML_CACHE_POLICIES = {"ML_FlashMoE", "ML_CECAR"}
SUPPORTED_TASKS = {"humaneval", "mbpp", "gpqa", "math500"}
OC_TASKS = {"gpqa", "math500"}


def parse_args() -> argparse.Namespace:
    parser = HfArgumentParser(EvalArguments)

    # --- args ---
    parser.add_argument("--model", default=None)
    parser.add_argument("--modeltype", default="causal")
    parser.add_argument("--peft_model", type=str, default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--use_auth_token", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--tasks",
        type=str,
        required=True,
        help="Single task name (humaneval, mbpp, gpqa, math500)",
    )

    parser.add_argument("--instruction_tokens", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length_generation", type=int, default=2048)

    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=4096,
        help="Maximum sequence length (OpenCompass simulation path only).",
    )

    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--left_padding", action="store_true")

    # limit / offset
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit_start", type=int, default=0)

    parser.add_argument("--save_every_k_tasks", type=int, default=-1)
    parser.add_argument("--postprocess", action="store_false")
    parser.add_argument("--allow_code_execution", action="store_true")
    parser.add_argument("--generation_only", action="store_true")

    parser.add_argument("--load_generations_path", type=str, default=None)
    parser.add_argument("--load_data_path", type=str, default=None)
    parser.add_argument("--metric_output_path", type=str, default="evaluation_results.json")

    parser.add_argument("--save_generations", action="store_true")
    parser.add_argument("--load_generations_intermediate_paths", type=str, nargs="*")
    parser.add_argument("--save_generations_path", type=str, default="generations.json")
    parser.add_argument("--save_references", action="store_true")
    parser.add_argument("--save_references_path", type=str, default="references.json")
    parser.add_argument("--prompt", type=str, default="prompt")
    parser.add_argument("--max_memory_per_gpu", type=str, default=None)
    parser.add_argument("--check_references", action="store_true")

    # Simulation model arguments (shared)
    parser.add_argument("--expert_path", type=str, default=None)
    parser.add_argument("--cache_size", type=int, default=24)
    parser.add_argument("--cache_policy", type=str, default="ML_CECAR")
    parser.add_argument(
        "--bonus_strategy",
        type=str,
        default="none",
        choices=["none", "random", "constant", "lfu", "lru", "mocce", "const", "cecar"],
    )
    parser.add_argument("--mode", type=str, default="none", choices=["none", "des", "odp"])
    parser.add_argument("--lambda_cache", type=float, default=0.2)
    parser.add_argument("--top_J", type=int, default=2)

    parser.add_argument("--simulation_stats_path", type=str, default="simulation_stats.json")
    parser.add_argument("--enable_thinking", action="store_true")

    parser.add_argument(
        "--simulation_model",
        type=str,
        default=None,
        choices=["Qwen3_30B_A3B", "OLMoE_1B_7B_0125_Instruct", "DeepSeek_v2_Lite_Chat"],
    )
    parser.add_argument("--shared_expert_path", type=str, default=None)
    parser.add_argument("--question_batch_size", type=int, default=1)
    parser.add_argument("--non_expert_model", type=str, default=None)
    parser.add_argument("--simulation_device_map", type=str, default="auto", choices=["auto", "cuda"])
    parser.add_argument("--ffn_model_path", type=str, default=None)
    parser.add_argument("--topk_threshold", type=int, default=4)

    parser.add_argument(
        "--mcmoe_threshold_path",
        type=str,
        default="./fiddler_model/MCMOE",
        help="Threshold JSON dir/path (required when mode is des/odp). Parsing is handled in fiddler_model.",
    )

    return parser.parse_args()


def normalize_task_name(task: str) -> str:
    return (task or "").strip().lower()


def require_exists(path: str, name: str) -> None:
    if path is None or str(path).strip() == "":
        raise ValueError(f"{name} must be provided")
    if not os.path.exists(path):
        raise ValueError(f"{name} not found: {path}")


def is_des_on(args: argparse.Namespace) -> bool:
    return args.mode in ("des", "odp")


def is_protection_on(args: argparse.Namespace) -> bool:
    return args.mode == "odp"


def resolve_ffn_subdir(args: argparse.Namespace) -> str | None:
    """
    If cache_policy is ML_*, require ffn_model_path and append subdir <ffn_model_path>/<cache_policy>.
    Else return None (do not pass ffn_model_path).
    """
    if args.cache_policy not in ML_CACHE_POLICIES:
        return None
    if args.ffn_model_path is None:
        raise ValueError("--ffn_model_path must be provided when using --cache_policy ML_FlashMoE or ML_CECAR")
    return os.path.join(args.ffn_model_path, args.cache_policy)


def maybe_write_simulation_config_json(simulation_model: str, cache_policy: str) -> None:
    """
    Replicates your config write (n_layers, num_experts) for ML cache policies.
    """
    if cache_policy not in ML_CACHE_POLICIES:
        return

    cfg_dir = "./fiddler_model/Config"
    os.makedirs(cfg_dir, exist_ok=True)

    if simulation_model == "Qwen3_30B_A3B":
        cfg = {"n_layers": 48, "num_experts": 128}
    elif simulation_model == "OLMoE_1B_7B_0125_Instruct":
        cfg = {"n_layers": 16, "num_experts": 64}
    elif simulation_model == "DeepSeek_v2_Lite_Chat":
        cfg = {"n_layers": 27, "num_experts": 64}
    else:
        return

    config_path = os.path.join(cfg_dir, f"{simulation_model}_config.json")
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[Simulation Config] Saved to {os.path.abspath(config_path)}")


def validate_common_args(args: argparse.Namespace) -> None:
    task = normalize_task_name(args.tasks)
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"tasks must be one of: {sorted(SUPPORTED_TASKS)}")

    # des/odp requires threshold path exists
    if is_des_on(args):
        require_exists(args.mcmoe_threshold_path, "--mcmoe_threshold_path")


def validate_opencompass_args(args: argparse.Namespace) -> None:
    if args.model is None:
        raise ValueError("--model must be provided for gpqa/math500 (it is the non_expert_model by mapping)")
    if args.simulation_model is None:
        raise ValueError("--simulation_model must be provided for gpqa/math500 (OpenCompass simulation)")
    require_exists(args.expert_path, "--expert_path")

    if args.simulation_model == "DeepSeek_v2_Lite_Chat":
        require_exists(args.shared_expert_path, "--shared_expert_path")


def validate_bigcode_simulation_args(args: argparse.Namespace) -> None:
    require_exists(args.expert_path, "--expert_path")
    require_exists(args.non_expert_model, "--non_expert_model")
    if args.simulation_model == "DeepSeek_v2_Lite_Chat":
        require_exists(args.shared_expert_path, "--shared_expert_path")


def get_gpus_max_memory(max_memory: str, num_gpus: int) -> dict:
    max_memory = {i: max_memory for i in range(num_gpus)}
    print("Loading model via these GPUs & max memories:", max_memory)
    return max_memory


# ----------------------------
# OpenCompass runner (gpqa/math500 only)
# ----------------------------
def build_opencompass_config_text(args: argparse.Namespace) -> tuple[str, str]:
    """
    Returns (cfg_text, abbr).
    """
    enable_des = bool(getattr(args, "enable_des", is_des_on(args)))
    enable_protection = bool(getattr(args, "enable_protection", is_protection_on(args)))
    protection_top_ratio = float(getattr(args, "protection_top_ratio", 0.02))

    sim_model_map = {
        "OLMoE_1B_7B_0125_Instruct": "olmoe",
        "DeepSeek_v2_Lite_Chat": "deepseek",
        "Qwen3_30B_A3B": "qwen3moe",
    }
    oc_sim = sim_model_map.get(args.simulation_model)
    if oc_sim is None:
        raise ValueError(f"Unsupported simulation_model for OpenCompass: {args.simulation_model}")

    mode_suffix = f"-{args.mode}" if args.mode in ("des", "odp") else ""
    abbr = f"simulation-{oc_sim}-{args.bonus_strategy}-{args.cache_policy}{mode_suffix}"

    ffn_subdir = resolve_ffn_subdir(args)

    config_parts: list[str] = []
    config_parts.append('"""Auto-generated config for simulation evaluation."""\n')
    config_parts.append("from opencompass.models import SimulationOlmoe, SimulationDeepseek, SimulationQwen3Moe\n\n")
    config_parts.append(
        "MODEL_TYPES = {\n"
        "    'olmoe': SimulationOlmoe,\n"
        "    'deepseek': SimulationDeepseek,\n"
        "    'qwen3moe': SimulationQwen3Moe,\n"
        "}\n\n"
    )

    config_parts.append(f"""
model_type = MODEL_TYPES['{oc_sim}']
abbr = '{abbr}'

model_config = dict(
    type=model_type,
    abbr=abbr,

    non_expert_model='{args.model}',              
    expert_path='{args.expert_path}',
    cache_size={args.cache_size},
    cache_policy='{args.cache_policy}',
    mode='{args.mode}',
    bonus_strategy='{args.bonus_strategy}',
    batch_size={args.batch_size},
    question_batch_size={args.question_batch_size},
    device_map='{args.simulation_device_map}',
    ffn_model_path={repr(ffn_subdir)},

    lambda_cache={args.lambda_cache},
    top_J={args.top_J},
    topk_threshold={args.topk_threshold},

    enable_des={str(enable_des)},
    enable_protection={str(enable_protection)},
    protection_top_ratio={protection_top_ratio},

    mcmoe_threshold_path='{args.mcmoe_threshold_path}',
    tasks='{args.tasks}',

    max_seq_len={args.max_seq_len},
    max_out_len={args.max_length_generation},
    run_cfg=dict(num_gpus=1),
    
)
""")

    if oc_sim == "deepseek":
        config_parts.append(f"model_config['shared_expert_path'] = '{args.shared_expert_path}'\n")
    if ffn_subdir is not None:
        config_parts.append(f"model_config['ffn_model_path'] = '{ffn_subdir}'\n")

    config_parts.append("\nmodels = [model_config]\n\n")

    if normalize_task_name(args.tasks) == "math500":
        config_parts.append(MATH500_CONFIG)
    elif normalize_task_name(args.tasks) == "gpqa":
        config_parts.append(GPQA_CONFIG)
    else:
        raise ValueError(f"OpenCompass path only supports tasks=gpqa|math500, got {args.tasks}")

    if args.limit is not None:
        start = int(args.limit_start or 0)
        end = start + int(args.limit)
        config_parts.append(f"""
# Apply sample range
for ds in datasets:
    ds.setdefault('reader_cfg', {{}})
    ds['reader_cfg']['test_range'] = '[{start}:{end}]'
    ds['abbr'] = ds.get('abbr', 'dataset') + '-{start}to{end}'
""")

    return "".join(config_parts), abbr


def run_opencompass_for_gpqa_math500(args: argparse.Namespace) -> None:
    des_on = is_des_on(args)                 
    protection_on = is_protection_on(args)   
    protection_top_ratio = 0.02
    
    args.enable_des = bool(des_on)
    args.enable_protection = bool(protection_on)
    args.protection_top_ratio = float(protection_top_ratio)    
    
    cfg_text, abbr = build_opencompass_config_text(args)

    os.makedirs("./opencompass", exist_ok=True)
    cfg_path = "./opencompass/_temp_simulation_config.py"
    with open(cfg_path, "w") as f:
        f.write(cfg_text)

    print(f"[OpenCompass] Generated config: {os.path.abspath(cfg_path)}")
    print(f"[OpenCompass] abbr: {abbr}")
    print(f"[OpenCompass] Running tasks={args.tasks} with simulation_model={args.simulation_model}")

    run_py = os.path.join(".", "opencompass", "run.py")
    cmd = [sys.executable, run_py, cfg_path]
    print(f"[OpenCompass] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=False)


# ----------------------------
# Bigcode path helpers
# ----------------------------
def load_base_model_and_tokenizer(args: argparse.Namespace, accelerator: Accelerator):
    dict_precisions = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if args.precision not in dict_precisions:
        raise ValueError(f"Non valid precision {args.precision}, choose from: fp16, fp32, bf16")

    model_kwargs = {
        "revision": args.revision,
        "trust_remote_code": args.trust_remote_code,
        "token": args.use_auth_token,
    }

    if args.load_in_8bit:
        print("Loading model in 8bit")
        model_kwargs["load_in_8bit"] = True
        model_kwargs["device_map"] = {"": accelerator.process_index}
    elif args.load_in_4bit:
        print("Loading model in 4bit")
        model_kwargs["load_in_4bit"] = True
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["bnb_4bit_compute_dtype"] = torch.float16
        model_kwargs["device_map"] = {"": accelerator.process_index}
    else:
        print(f"Loading model in {args.precision}")
        model_kwargs["torch_dtype"] = dict_precisions[args.precision]

        if args.max_memory_per_gpu:
            if args.max_memory_per_gpu != "auto":
                model_kwargs["max_memory"] = get_gpus_max_memory(args.max_memory_per_gpu, accelerator.num_processes)
                model_kwargs["offload_folder"] = "offload"
            else:
                model_kwargs["device_map"] = "auto"
                print("Loading model in auto mode")

    if args.modeltype == "causal":
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    elif args.modeltype == "seq2seq":
        warnings.warn("Seq2Seq models have only been tested for HumanEvalPack & CodeT5+ models.")
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model, **model_kwargs)
    else:
        raise ValueError("Non valid modeltype, choose from: causal, seq2seq")

    if args.peft_model:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.peft_model)
        print("Loaded PEFT model. Merging...")
        model.merge_and_unload()
        print("Merge complete.")

    if args.left_padding:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            revision=args.revision,
            trust_remote_code=args.trust_remote_code,
            token=args.use_auth_token,
            padding_side="left",
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            revision=args.revision,
            trust_remote_code=args.trust_remote_code,
            token=args.use_auth_token,
            truncation_side="left",
            padding_side="right",
        )

    if not tokenizer.eos_token:
        if tokenizer.bos_token:
            tokenizer.eos_token = tokenizer.bos_token
            print("bos_token used as eos_token")
        else:
            raise ValueError("No eos_token or bos_token found")

    try:
        tokenizer.pad_token = tokenizer.eos_token
    except AttributeError:
        print("Not setting pad_token to eos_token (read-only pad_token)")
        pass

    WIZARD_LLAMA_MODELS = [
        "WizardLM/WizardCoder-Python-34B-V1.0",
        "WizardLM/WizardCoder-34B-V1.0",
        "WizardLM/WizardCoder-Python-13B-V1.0",
    ]
    if args.model is not None and args.model in WIZARD_LLAMA_MODELS:
        tokenizer.bos_token = "<s>"
        tokenizer.bos_token_id = 1
        print("Changing bos_token to <s>")

    return model, tokenizer


def build_simulation_model(args: argparse.Namespace, accelerator: Accelerator):
    """
    Returns (model, tokenizer, simulation_model_ref)
    simulation_model_ref is used to dump stats later.
    """
    validate_bigcode_simulation_args(args)

    des_on = is_des_on(args)
    protection_on = is_protection_on(args)
    protection_top_ratio = 0.02
    ffn_subdir = resolve_ffn_subdir(args)

    maybe_write_simulation_config_json(args.simulation_model, args.cache_policy)

    if args.simulation_model == "Qwen3_30B_A3B":
        from fiddler_model import SimulationQwen3Moe

        if accelerator.is_main_process:
            print("Loading SimulationQwen3Moe model")
            print(f"Non-expert model: {os.path.abspath(args.non_expert_model)}")
            print(f"Expert path: {os.path.abspath(args.expert_path)}")
            print(f"Cache size: {args.cache_size}, Cache policy: {args.cache_policy}")
            if ffn_subdir is not None:
                print(f"FFN model path: {os.path.abspath(ffn_subdir)}")
            print(f"Bonus Strategy: {args.bonus_strategy}")
            print(f"Expert Mode: {args.mode}")
            print(f"Batch size: {args.batch_size}")
            print(f"Question batch size: {args.question_batch_size}")

        sim_args = argparse.Namespace(
            non_expert_model=args.non_expert_model,
            expert_path=args.expert_path,
            cache_size=args.cache_size,
            cache_policy=args.cache_policy,
            mode=args.mode,
            bonus_strategy=args.bonus_strategy,
            batch_size=args.batch_size,
            question_batch_size=args.question_batch_size,
            device_map=args.simulation_device_map,
            ffn_model_path=ffn_subdir,
            lambda_cache=args.lambda_cache,
            top_J=args.top_J,
            topk_threshold=args.topk_threshold,
            enable_des=bool(des_on),
            enable_protection=bool(protection_on),
            protection_top_ratio=float(protection_top_ratio),
            mcmoe_threshold_path=args.mcmoe_threshold_path,
            tasks=args.tasks,
        )

        model = SimulationQwen3Moe(sim_args)
        return model, model.tokenizer, model

    if args.simulation_model == "OLMoE_1B_7B_0125_Instruct":
        from fiddler_model import SimulationOlmoe

        if accelerator.is_main_process:
            print("Loading SimulationOlmoe model")
            print(f"Non-expert model: {os.path.abspath(args.non_expert_model)}")
            print(f"Expert path: {os.path.abspath(args.expert_path)}")
            print(f"Cache size: {args.cache_size}, Cache policy: {args.cache_policy}")
            if ffn_subdir is not None:
                print(f"FFN model path: {os.path.abspath(ffn_subdir)}")
            print(f"Bonus Strategy: {args.bonus_strategy}")
            print(f"Expert Mode: {args.mode}")
            print(f"Batch size: {args.batch_size}")
            print(f"Question batch size: {args.question_batch_size}")

        sim_args = argparse.Namespace(
            non_expert_model=args.non_expert_model,
            expert_path=args.expert_path,
            cache_size=args.cache_size,
            cache_policy=args.cache_policy,
            mode=args.mode,
            bonus_strategy=args.bonus_strategy,
            batch_size=args.batch_size,
            question_batch_size=args.question_batch_size,
            device_map=args.simulation_device_map,
            ffn_model_path=ffn_subdir,
            lambda_cache=args.lambda_cache,
            top_J=args.top_J,
            topk_threshold=args.topk_threshold,
            enable_des=bool(des_on),
            enable_protection=bool(protection_on),
            protection_top_ratio=float(protection_top_ratio),
            mcmoe_threshold_path=args.mcmoe_threshold_path,
            tasks=args.tasks,
        )

        model = SimulationOlmoe(sim_args)
        return model, model.tokenizer, model

    if args.simulation_model == "DeepSeek_v2_Lite_Chat":
        from fiddler_model import SimulationDeepseek

        if accelerator.is_main_process:
            print("Loading SimulationDeepseek model")
            print(f"Non-expert model: {os.path.abspath(args.non_expert_model)}")
            print(f"Expert path: {os.path.abspath(args.expert_path)}")
            print(f"Shared expert path: {os.path.abspath(args.shared_expert_path)}")
            print(f"Cache size: {args.cache_size}, Cache policy: {args.cache_policy}")
            if ffn_subdir is not None:
                print(f"FFN model path: {os.path.abspath(ffn_subdir)}")
            print(f"Bonus Strategy: {args.bonus_strategy}")
            print(f"Expert Mode: {args.mode}")
            print(f"Batch size: {args.batch_size}")
            print(f"Question batch size: {args.question_batch_size}")

        sim_args = argparse.Namespace(
            non_expert_model=args.non_expert_model,
            expert_path=args.expert_path,
            shared_expert_path=args.shared_expert_path,
            cache_size=args.cache_size,
            cache_policy=args.cache_policy,
            mode=args.mode,
            bonus_strategy=args.bonus_strategy,
            batch_size=args.batch_size,
            question_batch_size=args.question_batch_size,
            device_map=args.simulation_device_map,
            ffn_model_path=ffn_subdir,
            lambda_cache=args.lambda_cache,
            top_J=args.top_J,
            topk_threshold=args.topk_threshold,
            enable_des=bool(des_on),
            enable_protection=bool(protection_on),
            protection_top_ratio=float(protection_top_ratio),
            mcmoe_threshold_path=args.mcmoe_threshold_path,
            tasks=args.tasks,
        )

        model = SimulationDeepseek(sim_args)
        return model, model.tokenizer, model

    raise ValueError(f"Unsupported simulation_model: {args.simulation_model}")


def main():
    args = parse_args()
    transformers.logging.set_verbosity_error()
    datasets.logging.set_verbosity_error()

    validate_common_args(args)
    task = normalize_task_name(args.tasks)

    # ----------------------------
    # OpenCompass path (gpqa/math500)
    # ----------------------------
    if task in OC_TASKS:
        validate_opencompass_args(args)
        _ = resolve_ffn_subdir(args)  
        run_opencompass_for_gpqa_math500(args)
        return

    # ----------------------------
    # Bigcode path (humaneval/mbpp)
    # ----------------------------
    if task not in {"humaneval", "mbpp"}:
        raise ValueError(f"tasks must be one of: {sorted(SUPPORTED_TASKS)}")

    accelerator = Accelerator()
    if accelerator.is_main_process:
        print(f"Selected Task: {task}")

    results = {}
    simulation_model_ref = None

    if args.load_generations_path:
        if accelerator.is_main_process:
            print("evaluation only mode")
        evaluator = Evaluator(accelerator, None, None, args)
        results[task] = evaluator.evaluate(task)
    else:
        if args.simulation_model is not None:
            model, tokenizer, simulation_model_ref = build_simulation_model(args, accelerator)
        else:
            if args.model is None:
                raise ValueError("--model must be provided when not using --simulation_model")
            model, tokenizer = load_base_model_and_tokenizer(args, accelerator)

        evaluator = Evaluator(accelerator, model, tokenizer, args)

        intermediate_generations = None
        if args.load_generations_intermediate_paths:
            if len(args.load_generations_intermediate_paths) != 1:
                raise ValueError("--load_generations_intermediate_paths must have exactly 1 file for single-task mode")
            with open(args.load_generations_intermediate_paths[0], "r") as f_in:
                intermediate_generations = json.load(f_in)

        if args.generation_only:
            if accelerator.is_main_process:
                print("generation mode only")
            generations, references = evaluator.generate_text(task, intermediate_generations=intermediate_generations)
            if accelerator.is_main_process:
                save_generations_path = f"{os.path.splitext(args.save_generations_path)[0]}_{task}.json"
                save_references_path = f"references_{task}.json"
                evaluator.save_json_files(generations, references, save_generations_path, save_references_path)
        else:
            results[task] = evaluator.evaluate(task, intermediate_generations=intermediate_generations)

    results["config"] = vars(args)
    if not args.generation_only:
        dumped = json.dumps(results, indent=2)
        if accelerator.is_main_process:
            print(dumped)
        with open(args.metric_output_path, "w") as f:
            f.write(dumped)

    # Save simulation stats (only bigcode path simulation models)
    if simulation_model_ref is not None and accelerator.is_main_process:
        sim_stats = simulation_model_ref.stats.summary()
        cache_stats = simulation_model_ref.get_cache_stats()
        sim_stats["cache_size"] = args.cache_size
        sim_stats["cache_policy"] = args.cache_policy
        sim_stats["batch_size"] = args.batch_size
        sim_stats["cache_hit_rate"] = cache_stats.get("hit_rate", 0.0)
        sim_stats["cache_hit_rate_by_layer"] = cache_stats.get("hit_rate_by_layer", [])

        with open(args.simulation_stats_path, "w") as f:
            json.dump(sim_stats, f, indent=2)
        print(f"Simulation stats saved to: {os.path.abspath(args.simulation_stats_path)}")


if __name__ == "__main__":
    main()
