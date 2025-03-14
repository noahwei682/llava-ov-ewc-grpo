import torch
from typing import List, Optional, Tuple, Dict, Any, Callable
from transformers import PreTrainedModel, PreTrainedTokenizer
from llava.train.llava_trainer import LLaVATrainer
from dataclasses import dataclass, field
from transformers import TrainingArguments

@dataclass
class GRPOConfig(TrainingArguments):
    """Configuration class for GRPO training."""
    use_vllm: bool = field(
        default=False,
        metadata={"help": "Whether to use VLLM for inference"}
    )
    num_generations: int = field(
        default=2,
        metadata={"help": "Number of generations per prompt"}
    )
    max_prompt_length: int = field(
        default=256,
        metadata={"help": "Maximum length of the prompt"}
    )
    max_completion_length: int = field(
        default=300,
        metadata={"help": "Maximum length of the completion"}
    )
    vllm_gpu_memory_utilization: float = field(
        default=0.2,
        metadata={"help": "GPU memory utilization for VLLM"}
    )
    # Add new fields from other training scripts
    mm_vision_tower: str = field(
        default="openai/clip-vit-large-patch14",
        metadata={"help": "Vision tower model"}
    )
    mm_vision_tower_lr: float = field(
        default=2e-6,
        metadata={"help": "Learning rate for vision tower"}
    )
    mm_projector_type: str = field(
        default="mlp2x_gelu",
        metadata={"help": "Type of MLP projector"}
    )
    mm_vision_select_layer: int = field(
        default=-2,
        metadata={"help": "Which layer to select from vision tower"}
    )
    mm_use_im_start_end: bool = field(
        default=False,
        metadata={"help": "Whether to use image start/end tokens"}
    )
    mm_use_im_patch_token: bool = field(
        default=False,
        metadata={"help": "Whether to use image patch tokens"}
    )
    group_by_modality_length: bool = field(
        default=True,
        metadata={"help": "Whether to group by modality length"}
    )
    image_aspect_ratio: str = field(
        default="anyres",
        metadata={"help": "Image aspect ratio handling"}
    )
    mm_patch_merge_type: str = field(
        default="spatial_unpad",
        metadata={"help": "How to merge image patches"}
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={"help": "Path to deepspeed config file"}
    )

class LLaVAGRPOTrainer(LLaVATrainer):
    """
    GRPO (Generative Reward-Powered Optimization) Trainer for LLaVA.
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

    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Save checkpoint with GRPO-specific handling.
        """
        super()._save_checkpoint(model, trial, metrics)
        
    def create_optimizer(self):
        """
        Create optimizer with GRPO-specific settings.
        """
        return super().create_optimizer()

    def create_scheduler(self, num_training_steps: int):
        """
        Create learning rate scheduler with GRPO-specific settings.
        """
        return super().create_scheduler(num_training_steps) 