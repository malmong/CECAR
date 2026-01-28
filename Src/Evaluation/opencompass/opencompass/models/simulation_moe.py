"""Simulation MoE Model Wrappers for OpenCompass.

These wrappers provide OpenCompass-compatible interfaces for the simulation models
(OLMoE, Qwen3MoE, DeepSeek) that support virtual cache simulation.
"""
import os
from argparse import Namespace
from typing import Dict, List, Optional, Union

import torch

from opencompass.models.base import BaseModel
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]


@MODELS.register_module()
class SimulationOlmoe(BaseModel):
    """OpenCompass wrapper for SimulationOlmoe model.

    Args:
        non_expert_model (str): Path to non-expert model weights.
        expert_path (str): Path to expert weights directory.
        cache_size (int): Number of experts to cache per layer. Defaults to 12.
        cache_policy (str): Cache eviction policy. Defaults to 'lru'.
        bonus_strategy (str): Bonus strategy for routing. Defaults to 'none'.
        batch_size (int): Batch size for inference. Defaults to 1.
        max_seq_len (int): Maximum sequence length. Defaults to 2048.
        max_out_len (int): Maximum output length. Defaults to 512.
        device_map (str): Device map for model loading. Defaults to 'auto'.
        ffn_model_path (str): Path to FFN model for mlv2 policy. Defaults to None.
        lambda_cache (float): Lambda for cache policy. Defaults to 0.5.
        top_J (int): Top-J for MOCCE. Defaults to 4.
    """

    is_api: bool = False

    def __init__(
        self,
        non_expert_model: str,
        expert_path: str,
        cache_size: int = 12,
        cache_policy: str = 'lru',
        bonus_strategy: str = 'none',
        mode: str = 'none',               
        topk_threshold: int = 4,          
        batch_size: int = 1,
        max_seq_len: int = 2048,
        max_out_len: int = 512,
        device_map: str = 'auto',
        ffn_model_path: Optional[str] = None,
        lambda_cache: float = 0.5,
        top_J: int = 4,
        meta_template: Optional[Dict] = None,
        enable_des: bool = False,
        enable_protection: bool = False,
        protection_top_ratio: float = 0.02,
        mcmoe_threshold_path: str = './fiddler_model/MCMOE',
        tasks: str = '',
        **kwargs
    ):
        # Note: path is set to non_expert_model for compatibility
        super().__init__(
            path=non_expert_model,
            max_seq_len=max_seq_len,
            tokenizer_only=False,
            meta_template=meta_template,
        )
        self.logger = get_logger()
        self.max_out_len = max_out_len

        # Import and initialize the simulation model
        from opencompass.fiddler_model import SimulationOlmoe as SimOlmoe

        sim_args = Namespace(
            non_expert_model=non_expert_model,
            expert_path=expert_path,
            cache_size=cache_size,
            cache_policy=cache_policy,
            mode=mode,                               
            bonus_strategy=bonus_strategy,
            batch_size=batch_size,
            question_batch_size=batch_size,
            device_map=device_map,
            ffn_model_path=ffn_model_path,
            lambda_cache=lambda_cache,
            top_J=top_J,
            topk_threshold=topk_threshold,           
            enable_des=enable_des,
            enable_protection=enable_protection,
            protection_top_ratio=protection_top_ratio,
            mcmoe_threshold_path=mcmoe_threshold_path,
            tasks=tasks,
        )

        self.logger.info(f"Loading SimulationOlmoe model")
        self.logger.info(f"Non-expert model: {non_expert_model}")
        self.logger.info(f"Expert path: {expert_path}")
        self.logger.info(f"Cache size: {cache_size}, Cache policy: {cache_policy}")
        self.logger.info(f"Bonus strategy: {bonus_strategy}")
        self.logger.info(f"Mode: {mode}, topk_threshold: {topk_threshold}")        

        self.model = SimOlmoe(sim_args)
        self.tokenizer = self.model.tokenizer

    def generate(
        self,
        inputs: List[str],
        max_out_len: int,
        min_out_len: Optional[int] = None,
        stopping_criteria: List[str] = [],
        **kwargs
    ) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of prompt strings.
            max_out_len (int): The maximum length of the output.
            min_out_len (Optional[int]): The minimum length of the output.
            stopping_criteria (List[str]): List of stop strings.

        Returns:
            List[str]: A list of generated strings.
        """
        results = []
        for input_text in inputs:
            # Tokenize
            input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
            input_ids = input_ids.to(self.model.dev)

            # Generate
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_length=input_ids.shape[1] + max_out_len,
            )

            # Decode only the generated part
            generated_ids = output_ids[0, input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

            # Apply stopping criteria
            for stop_str in stopping_criteria:
                if stop_str in generated_text:
                    generated_text = generated_text.split(stop_str)[0]

            results.append(generated_text)

            # Reset for next input
            self.model.remove_experts()

        return results

    def get_ppl(
        self,
        inputs: List[str],
        mask_length: Optional[List[int]] = None
    ) -> List[float]:
        """Get perplexity scores (not implemented for simulation models)."""
        raise NotImplementedError(
            'SimulationOlmoe does not support ppl-based evaluation.'
        )

    def get_ppl_tokenwise(
        self,
        inputs: List[str],
        mask_length: Optional[List[int]] = None
    ) -> List[float]:
        """Get tokenwise perplexity scores (not implemented)."""
        raise NotImplementedError(
            'SimulationOlmoe does not support ppl-based evaluation.'
        )

    def encode(self, prompt: str) -> torch.Tensor:
        """Encode prompt to tokens."""
        return self.tokenizer.encode(prompt, return_tensors='pt')

    def decode(self, tokens: torch.Tensor) -> str:
        """Decode tokens to text."""
        return self.tokenizer.decode(tokens[0], skip_special_tokens=True)

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings."""
        return len(self.tokenizer.encode(prompt))

    def get_cache_stats(self) -> Dict:
        """Get virtual cache statistics."""
        return self.model.get_cache_stats()


@MODELS.register_module()
class SimulationQwen3Moe(BaseModel):
    """OpenCompass wrapper for SimulationQwen3Moe model.

    Args:
        non_expert_model (str): Path to non-expert model weights.
        expert_path (str): Path to expert weights directory.
        cache_size (int): Number of experts to cache per layer. Defaults to 12.
        cache_policy (str): Cache eviction policy. Defaults to 'lru'.
        bonus_strategy (str): Bonus strategy for routing. Defaults to 'none'.
        batch_size (int): Batch size for inference. Defaults to 1.
        max_seq_len (int): Maximum sequence length. Defaults to 2048.
        max_out_len (int): Maximum output length. Defaults to 512.
        device_map (str): Device map for model loading. Defaults to 'auto'.
        ffn_model_path (str): Path to FFN model for mlv2 policy. Defaults to None.
        lambda_cache (float): Lambda for cache policy. Defaults to 0.5.
        top_J (int): Top-J for MOCCE. Defaults to 4.
    """

    is_api: bool = False

    def __init__(
        self,
        non_expert_model: str,
        expert_path: str,
        cache_size: int = 12,
        cache_policy: str = 'lru',
        bonus_strategy: str = 'none',
        mode: str = 'none',               
        topk_threshold: int = 4,          
        batch_size: int = 1,
        max_seq_len: int = 2048,
        max_out_len: int = 512,
        device_map: str = 'auto',
        ffn_model_path: Optional[str] = None,
        lambda_cache: float = 0.5,
        top_J: int = 4,
        meta_template: Optional[Dict] = None,
        enable_des: bool = False,
        enable_protection: bool = False,
        protection_top_ratio: float = 0.02,
        mcmoe_threshold_path: str = './fiddler_model/MCMOE',
        tasks: str = '',
        **kwargs
    ):
        super().__init__(
            path=non_expert_model,
            max_seq_len=max_seq_len,
            tokenizer_only=False,
            meta_template=meta_template,
        )
        self.logger = get_logger()
        self.max_out_len = max_out_len

        from opencompass.fiddler_model import SimulationQwen3Moe as SimQwen3Moe

        sim_args = Namespace(
            non_expert_model=non_expert_model,
            expert_path=expert_path,
            cache_size=cache_size,
            cache_policy=cache_policy,
            mode=mode,                               
            bonus_strategy=bonus_strategy,
            batch_size=batch_size,
            question_batch_size=batch_size,
            device_map=device_map,
            ffn_model_path=ffn_model_path,
            lambda_cache=lambda_cache,
            top_J=top_J,
            topk_threshold=topk_threshold,           
            enable_des=enable_des,
            enable_protection=enable_protection,
            protection_top_ratio=protection_top_ratio,
            mcmoe_threshold_path=mcmoe_threshold_path,
            tasks=tasks,
        )

        self.logger.info(f"Loading SimulationQwen3Moe model")
        self.logger.info(f"Non-expert model: {non_expert_model}")
        self.logger.info(f"Expert path: {expert_path}")
        self.logger.info(f"Cache size: {cache_size}, Cache policy: {cache_policy}")
        self.logger.info(f"Bonus strategy: {bonus_strategy}")
        self.logger.info(f"Mode: {mode}, topk_threshold: {topk_threshold}")

        self.model = SimQwen3Moe(sim_args)
        self.tokenizer = self.model.tokenizer

    def generate(
        self,
        inputs: List[str],
        max_out_len: int,
        min_out_len: Optional[int] = None,
        stopping_criteria: List[str] = [],
        **kwargs
    ) -> List[str]:
        """Generate results given a list of inputs."""
        results = []
        for input_text in inputs:
            input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
            input_ids = input_ids.to(self.model.dev)

            output_ids = self.model.generate(
                input_ids=input_ids,
                max_length=input_ids.shape[1] + max_out_len,
            )

            generated_ids = output_ids[0, input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

            for stop_str in stopping_criteria:
                if stop_str in generated_text:
                    generated_text = generated_text.split(stop_str)[0]

            results.append(generated_text)
            self.model.remove_experts()

        return results

    def get_ppl(
        self,
        inputs: List[str],
        mask_length: Optional[List[int]] = None
    ) -> List[float]:
        raise NotImplementedError(
            'SimulationQwen3Moe does not support ppl-based evaluation.'
        )

    def get_ppl_tokenwise(
        self,
        inputs: List[str],
        mask_length: Optional[List[int]] = None
    ) -> List[float]:
        raise NotImplementedError(
            'SimulationQwen3Moe does not support ppl-based evaluation.'
        )

    def encode(self, prompt: str) -> torch.Tensor:
        return self.tokenizer.encode(prompt, return_tensors='pt')

    def decode(self, tokens: torch.Tensor) -> str:
        return self.tokenizer.decode(tokens[0], skip_special_tokens=True)

    def get_token_len(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt))

    def get_cache_stats(self) -> Dict:
        return self.model.get_cache_stats()


@MODELS.register_module()
class SimulationDeepseek(BaseModel):
    """OpenCompass wrapper for SimulationDeepseek model.

    Args:
        non_expert_model (str): Path to non-expert model weights.
        expert_path (str): Path to expert weights directory.
        shared_expert_path (str): Path to shared expert weights.
        cache_size (int): Number of experts to cache per layer. Defaults to 12.
        cache_policy (str): Cache eviction policy. Defaults to 'lru'.
        bonus_strategy (str): Bonus strategy for routing. Defaults to 'none'.
        batch_size (int): Batch size for inference. Defaults to 1.
        max_seq_len (int): Maximum sequence length. Defaults to 2048.
        max_out_len (int): Maximum output length. Defaults to 512.
        device_map (str): Device map for model loading. Defaults to 'auto'.
        ffn_model_path (str): Path to FFN model for mlv2 policy. Defaults to None.
        lambda_cache (float): Lambda for cache policy. Defaults to 0.5.
        top_J (int): Top-J for MOCCE. Defaults to 4.
    """

    is_api: bool = False

    def __init__(
        self,
        non_expert_model: str,
        expert_path: str,
        shared_expert_path: str,
        cache_size: int = 12,
        cache_policy: str = 'lru',
        bonus_strategy: str = 'none',
        mode: str = 'none',               
        topk_threshold: int = 4,          
        batch_size: int = 1,
        max_seq_len: int = 2048,
        max_out_len: int = 512,
        device_map: str = 'auto',
        ffn_model_path: Optional[str] = None,
        lambda_cache: float = 0.5,
        top_J: int = 4,
        meta_template: Optional[Dict] = None,
        enable_des: bool = False,
        enable_protection: bool = False,
        protection_top_ratio: float = 0.02,
        mcmoe_threshold_path: str = './fiddler_model/MCMOE',
        tasks: str = '',
        **kwargs
    ):
        super().__init__(
            path=non_expert_model,
            max_seq_len=max_seq_len,
            tokenizer_only=False,
            meta_template=meta_template,
        )
        self.logger = get_logger()
        self.max_out_len = max_out_len

        from opencompass.fiddler_model import SimulationDeepseek as SimDeepseek

        sim_args = Namespace(
            non_expert_model=non_expert_model,
            expert_path=expert_path,
            shared_expert_path=shared_expert_path,
            cache_size=cache_size,
            cache_policy=cache_policy,
            mode=mode,                               
            bonus_strategy=bonus_strategy,
            batch_size=batch_size,
            question_batch_size=batch_size,
            device_map=device_map,
            ffn_model_path=ffn_model_path,
            lambda_cache=lambda_cache,
            top_J=top_J,
            topk_threshold=topk_threshold,           
            enable_des=enable_des,
            enable_protection=enable_protection,
            protection_top_ratio=protection_top_ratio,
            mcmoe_threshold_path=mcmoe_threshold_path,
            tasks=tasks,
        )
        self.logger.info(f"Loading SimulationDeepseek model")
        self.logger.info(f"Non-expert model: {non_expert_model}")
        self.logger.info(f"Expert path: {expert_path}")
        self.logger.info(f"Shared expert path: {shared_expert_path}")
        self.logger.info(f"Cache size: {cache_size}, Cache policy: {cache_policy}")
        self.logger.info(f"Bonus strategy: {bonus_strategy}")
        self.logger.info(f"Mode: {mode}, topk_threshold: {topk_threshold}")

        self.model = SimDeepseek(sim_args)
        self.tokenizer = self.model.tokenizer

    def generate(
        self,
        inputs: List[str],
        max_out_len: int,
        min_out_len: Optional[int] = None,
        stopping_criteria: List[str] = [],
        **kwargs
    ) -> List[str]:
        """Generate results given a list of inputs."""
        results = []
        for input_text in inputs:
            messages = [{"role": "user", "content": input_text}]
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,   
                tokenize=True,
                return_tensors="pt",
            )
            input_ids = input_ids.to(self.model.dev)

            output_ids = self.model.generate(
                input_ids=input_ids,
                max_length=input_ids.shape[1] + max_out_len,
            )

            generated_ids = output_ids[0, input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

            for stop_str in stopping_criteria:
                if stop_str in generated_text:
                    generated_text = generated_text.split(stop_str)[0]

            results.append(generated_text)
            self.model.remove_experts()

        return results

    def get_ppl(
        self,
        inputs: List[str],
        mask_length: Optional[List[int]] = None
    ) -> List[float]:
        raise NotImplementedError(
            'SimulationDeepseek does not support ppl-based evaluation.'
        )

    def get_ppl_tokenwise(
        self,
        inputs: List[str],
        mask_length: Optional[List[int]] = None
    ) -> List[float]:
        raise NotImplementedError(
            'SimulationDeepseek does not support ppl-based evaluation.'
        )

    def encode(self, prompt: str) -> torch.Tensor:
        return self.tokenizer.encode(prompt, return_tensors='pt')

    def decode(self, tokens: torch.Tensor) -> str:
        return self.tokenizer.decode(tokens[0], skip_special_tokens=True)

    def get_token_len(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt))

    def get_cache_stats(self) -> Dict:
        return self.model.get_cache_stats()
