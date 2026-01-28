"""Auto-generated config for simulation evaluation."""
from opencompass.models import SimulationOlmoe, SimulationDeepseek, SimulationQwen3Moe

MODEL_TYPES = {
    'olmoe': SimulationOlmoe,
    'deepseek': SimulationDeepseek,
    'qwen3moe': SimulationQwen3Moe,
}


model_type = MODEL_TYPES['deepseek']
abbr = 'simulation-deepseek-cecar-ML_CECAR'

model_config = dict(
    type=model_type,
    abbr=abbr,
    non_expert_model='../Build/models/DeepSeek_v2_Lite_Chat/non_expert',
    expert_path='../Build/models/DeepSeek_v2_Lite_Chat/experts',
    cache_size=24,
    cache_policy='ML_CECAR',
    bonus_strategy='cecar',
    mode='none',
    batch_size=1,
    max_seq_len=4096,
    max_out_len=1024,
    lambda_cache=0.5,
    top_J=2,
    topk_threshold=4,
    mcmoe_threshold_path='./fiddler_model/MCMOE',
    run_cfg=dict(num_gpus=1),
)
model_config['shared_expert_path'] = '../Build/models/DeepSeek_v2_Lite_Chat/shared_expert'
model_config['ffn_model_path'] = '../Train/Pre_trained_FFN/DeepSeek_v2_Lite_Chat/ML_CECAR'

models = [model_config]


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

# Apply sample range
for ds in datasets:
    ds.setdefault('reader_cfg', {})
    ds['reader_cfg']['test_range'] = '[10:11]'
    ds['abbr'] = ds.get('abbr', 'dataset') + '-10to11'
