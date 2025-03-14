import torch
from typing import List, Optional, Tuple, Dict, Any, Callable
from transformers import PreTrainedModel, PreTrainedTokenizer
from llava.train.llava_trainer import LLaVATrainer
from .grpo_config import GRPOConfig

class GRPOTrainer(LLaVATrainer):
    """
    GRPO (Generative Reward-Powered Optimization) Trainer.
    Extends LLaVATrainer with GRPO-specific functionality.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        reward_funcs: List[Callable],
        args: GRPOConfig,
        **kwargs
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            args=args,
            **kwargs
        )
        self.reward_funcs = reward_funcs
        self.config = args

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, Any],
        return_outputs: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute GRPO loss for training.
        
        Args:
            model: The model to train
            inputs: Training inputs
            return_outputs: Whether to return model outputs
            
        Returns:
            Tuple of (loss, metrics)
        """
        # Get base loss from language modeling
        outputs = model(**inputs)
        base_loss = outputs.loss

        # Generate responses for reward calculation
        if self.config.num_generations > 0:
            with torch.no_grad():
                generated = self.generate(
                    inputs["input_ids"],
                    max_length=self.config.max_completion_length
                )
                responses, rewards = generated
                
                # Calculate combined rewards
                total_reward = torch.zeros(len(responses), device=base_loss.device)
                for reward_func in self.reward_funcs:
                    total_reward += torch.tensor(
                        reward_func(
                            prompts=inputs["input_ids"],
                            completions=responses
                        ),
                        device=base_loss.device
                    )
                
                # Scale loss by rewards
                loss = base_loss * (1.0 - total_reward.mean())
        else:
            loss = base_loss

        return (loss, outputs) if return_outputs else loss

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: Optional[int] = None,
        **kwargs
    ) -> Tuple[List[str], torch.FloatTensor]:
        """
        Generate responses and calculate rewards.
        
        Args:
            input_ids: Input token IDs
            max_length: Maximum generation length
            
        Returns:
            Tuple of (generated texts, rewards)
        """
        if max_length is None:
            max_length = self.config.max_completion_length

        # Generate multiple responses per input
        all_responses = []
        all_scores = []
        
        for _ in range(self.config.num_generations):
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )
            
            # Convert outputs to text
            responses = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            all_responses.extend(responses)
            
            # Calculate rewards
            rewards = torch.zeros(len(responses))
            for reward_func in self.reward_funcs:
                rewards += torch.tensor(
                    reward_func(
                        prompts=input_ids,
                        completions=responses
                    )
                )
            all_scores.extend(rewards)
            
        return all_responses, torch.stack(all_scores) 