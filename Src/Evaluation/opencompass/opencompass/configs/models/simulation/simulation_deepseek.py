from opencompass.models import SimulationDeepseek

# DeepSeek V2 Simulation Model - CECAR + ML_CECAR
models = [
    dict(
        type=SimulationDeepseek,
        abbr='simulation-deepseek-caer-mlv2',
        non_expert_model='../Build/models/DeepSeek_v2_Lite_Chat/non_expert',
        expert_path='../Build/models/DeepSeek_v2_Lite_Chat/experts',
        shared_expert_path='../Build/models/DeepSeek_v2_Lite_Chat/shared_expert/',
        ffn_model_path='../Train/Pre_trained_FFN/DeepSeek_v2_Lite_Chat/ML_CECAR/',
        cache_size=12,
        cache_policy='mlv2',
        bonus_strategy='caer',
        batch_size=1,
        max_seq_len=4096,
        max_out_len=512,
        run_cfg=dict(num_gpus=1),
    ),
]
