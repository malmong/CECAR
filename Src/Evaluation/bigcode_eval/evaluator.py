import inspect
import json
import os
import warnings

from tqdm import tqdm
from typing import List

import torch

from bigcode_eval import tasks
from bigcode_eval.generation import parallel_generations

_WARNING = """
################################################################################
                                  !!!WARNING!!!
################################################################################
The "code_eval"/"apps_metric" you are about to use, execute untrusted 
model-generated code in Python.
Although it is highly unlikely that model-generated code will do something
overtly malicious in response to this test suite, model-generated code may act
destructively due to a lack of model capability or alignment.
Users are strongly encouraged to sandbox this evaluation suite so that it
does not perform destructive actions on their host or network. For more
information on how OpenAI sandboxes its code, see the paper "Evaluating Large
Language Models Trained on Code" (https://arxiv.org/abs/2107.03374).
Once you have read this disclaimer and taken appropriate precautions, set the argument 
"allow_code_execution" to True.
################################################################################\
"""

class Evaluator:
    def __init__(self, accelerator, model, tokenizer, args):
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

        # setup arguments
        self.metric_output_path = args.metric_output_path

        # code evaluation permission
        self.allow_code_execution = args.allow_code_execution

    def generate_text(self, task_name, intermediate_generations=None):
        task = tasks.get_task(task_name, self.args)
        dataset = task.get_dataset()
        # if args.limit is None, use all samples
        # if args.limit is used, make sure args.limit_start + args.limit <= len(dataset)
        n_tasks = min(self.args.limit, len(dataset) - self.args.limit_start) if self.args.limit else len(dataset)
        # when args.limit is None
        # adjust n_tasks by args.limit_start to prevent out of bounds issues 
        if not self.args.limit:
            n_tasks -= self.args.limit_start
        references = [task.get_reference(dataset[i]) for i in range(self.args.limit_start, self.args.limit_start+n_tasks)]

        if self.args.check_references:
            if "get_solution" in inspect.signature(task.get_reference).parameters:
                solutions = [[task.get_reference(dataset[i], get_solution=True)] for i in range(self.args.limit_start, self.args.limit_start+n_tasks)]
            else:
                solutions = [[ref] for ref in references]
            return solutions, references

        curr_generations = []  # list[list[str | None] | None]
        if intermediate_generations:
            curr_generations = [gen for gen in intermediate_generations if gen]
            n_tasks -= len(curr_generations)
        intermediate_save_generations_path = f"{os.path.splitext(self.args.save_generations_path)[0]}_{task_name}_intermediate.json"
        curr_sample_idx = len(curr_generations)

        # Simulation model: batched generation (multiple questions at once)
        if getattr(self.args, "simulation_model", None) is not None:
            generations = self._generate_with_simulation(task, dataset, n_tasks)
        else:
            generations = parallel_generations(
                task,
                dataset,
                self.accelerator,
                self.model,
                self.tokenizer,
                n_tasks=n_tasks,
                args=self.args,
                curr_sample_idx=curr_sample_idx,  # curr_sample_idx will added to limit_start to fix indexing
                save_every_k_tasks=self.args.save_every_k_tasks,
                intermediate_generations=curr_generations,
                intermediate_save_generations_path=intermediate_save_generations_path,
            )

        if len(generations[0]) > self.args.n_samples:
            generations = [l[: self.args.n_samples] for l in generations]
            warnings.warn(
                f"Number of tasks wasn't proportional to number of devices, we removed extra predictions to only keep nsamples={self.args.n_samples}"
            )
        return generations, references

    def evaluate(self, task_name, intermediate_generations=None):
        task = tasks.get_task(task_name, self.args)
        if task.requires_execution and not self.allow_code_execution:
            raise ValueError(_WARNING)

        generations, references = self.generate_text(task_name, intermediate_generations=intermediate_generations)

        if self.accelerator.is_main_process:
            if not self.args.load_generations_path:
                save_generations_path = f"{os.path.splitext(self.args.save_generations_path)[0]}_{task_name}.json"
                self.save_json_files(generations, references, save_generations_path, f"references_{task_name}.json")

            # make sure tokenizer plays nice with multiprocessing
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            if self.allow_code_execution and task.requires_execution:
                os.environ["HF_ALLOW_CODE_EVAL"] = "1"
            print("Evaluating generations...")
            results = task.process_results(generations, references)
            return results

    def _generate_with_simulation(self, task, dataset, n_tasks):
        """Batched generation for SimulationQwen3Moe - processes multiple questions at once"""
        generations = []

        question_batch_size = getattr(self.args, 'question_batch_size', 1)

        # Intermediate save path
        save_path = f"{os.path.splitext(self.args.save_generations_path)[0]}_simulation_intermediate.json"

        # Process questions in batches
        total_batches = (n_tasks + question_batch_size - 1) // question_batch_size

        for batch_idx in tqdm(range(total_batches), desc="Generating (batched)"):
            batch_start = batch_idx * question_batch_size
            batch_end = min(batch_start + question_batch_size, n_tasks)
            actual_batch_size = batch_end - batch_start

            batch_indices = list(range(
                self.args.limit_start + batch_start,
                self.args.limit_start + batch_end
            ))

            try:
                # Collect prompts for this batch
                prompts = [task.get_prompt(dataset[i]) for i in batch_indices]
                original_prompts = prompts.copy()

                # Apply chat template if enabled
                if getattr(self.args, "use_chat_template", False):
                    processed_prompts = []
                    for prompt in prompts:
                        messages = [{"role": "user", "content": prompt}]
                        processed = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=getattr(self.args, "enable_thinking", False)
                        )
                        processed_prompts.append(processed)
                    prompts = processed_prompts

                # Tokenize with padding (left padding for generation)
                self.tokenizer.padding_side = "left"
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                inputs = self.tokenizer(
                    prompts,
                    padding=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.args.max_length_generation
                )

                input_ids = inputs.input_ids.to("cuda")
                attention_mask = inputs.attention_mask.to("cuda")

                # Generate (greedy decoding, top_k=0)
                with torch.no_grad():
                    output_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=self.args.max_length_generation,
                        do_sample=False,
                        top_k=0,
                    )

                # Process each result in the batch
                for b in range(actual_batch_size):
                    # Decode generated text (skip special tokens to remove pad/eos/bos)
                    generated_text = self.tokenizer.decode(
                        output_ids[b],
                        skip_special_tokens=True
                    )

                    # Remove prompt from generated text
                    prompt = prompts[b]
                    original_prompt = original_prompts[b]

                    # Also decode prompt without special tokens for comparison
                    prompt_clean = self.tokenizer.decode(
                        self.tokenizer.encode(prompt, add_special_tokens=False),
                        skip_special_tokens=True
                    )

                    if generated_text.startswith(prompt_clean):
                        generated_code = generated_text[len(prompt_clean):]
                    elif generated_text.startswith(prompt):
                        generated_code = generated_text[len(prompt):]
                    else:
                        # Fallback: just use the generated text
                        generated_code = generated_text

                    # Stop word processing
                    if task.stop_words:
                        for stop_word in task.stop_words:
                            if stop_word in generated_code:
                                generated_code = generated_code[:generated_code.index(stop_word)]

                    # Prepend original prompt for proper evaluation
                    final_text = original_prompt + generated_code

                    # n_samples=1 (greedy decoding)
                    generations.append([final_text])

                # Clear routing record to free memory (don't save it)
                if hasattr(self.model, 'routing_record'):
                    self.model.routing_record = []

                # Save intermediate results
                with open(save_path, "w") as fp:
                    json.dump(generations, fp)

                # Reset cache state for next batch
                self.model.remove_experts()
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\nOOM at batch {batch_idx}. Saving {len(generations)} generations to {save_path}")
                    with open(save_path, "w") as fp:
                        json.dump(generations, fp)
                    torch.cuda.empty_cache()
                raise e

        print(f"Intermediate generations saved to: {save_path}")
        return generations

    def save_json_files(
        self,
        generations: List[str],
        references: List[str],
        save_generations_path: str,
        save_references_path: str,
    ) -> None:
        if self.args.save_generations:
            with open(save_generations_path, "w") as fp:
                json.dump(generations, fp)
                print(f"generations were saved at {save_generations_path}")
        if self.args.save_references:
            with open(save_references_path, "w") as fp:
                json.dump(references, fp)
                print(f"references were saved at {save_references_path}")
