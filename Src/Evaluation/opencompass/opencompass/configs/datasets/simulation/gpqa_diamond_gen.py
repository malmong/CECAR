from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GPQADataset, GPQAEvaluator
from opencompass.utils import first_option_postprocess

# GPQA Diamond dataset
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
