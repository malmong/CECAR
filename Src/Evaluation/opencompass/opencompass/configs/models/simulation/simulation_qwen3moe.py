from opencompass.models import SimulationQwen3Moe

# Qwen3-MoE Simulation Model - CECAR + ML_CECAR
models = [
    dict(
        type=SimulationQwen3Moe,
        abbr='simulation-qwen3moe-caer-mlv2',
        non_expert_model='../Build/models/Qwen3_30B_A3B/non_expert',
        expert_path='../Build/models/Qwen3_30B_A3B/experts',
        ffn_model_path='../Train/Pre_trained_FFN/Qwen3_30B_A3B/ML_CECAR/',
        cache_size=12,
        cache_policy='ML_CECAR',
        bonus_strategy='cecar',
        batch_size=1,
        max_seq_len=4096,
        max_out_len=512,
        run_cfg=dict(num_gpus=1),
    ),
]
