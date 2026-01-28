from opencompass.models import SimulationOlmoe

# OLMoE Simulation Model - CECAR + ML_CECAR
models = [
    dict(
        type=SimulationOlmoe,
        abbr='simulation-olmoe-caer-mlv2',
        non_expert_model='../Build/models/OLMoE_1B_7B_0125_Instruct/non_expert',
        expert_path='../Build/models/OLMoE_1B_7B_0125_Instruct/experts',
        ffn_model_path='../Train/Pre_trained_FFN/OLMoE_1B_7B_0125_Instruct/ML_CECAR/',
        cache_size=12,
        cache_policy='ML_CECAR',
        bonus_strategy='cecar',
        batch_size=1,
        max_seq_len=4096,
        max_out_len=512,
        run_cfg=dict(num_gpus=1),
    ),
]
