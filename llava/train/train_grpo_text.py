import os
import glob
import torch
import numpy as np
import re
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any, Tuple, Callable
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    PreTrainedModel,
    PreTrainedTokenizer
)
from llava.model.builder import load_pretrained_model
from llava.data.gsm8k_dataset import (
    get_gsm8k_dataset,
    soft_format_reward_func,
    strict_format_reward_func,
    extract_xml_answer,
    SYSTEM_PROMPT
)

# 自定义函数，去除打印语句
def custom_correctness_reward_func(prompts: List[Dict], 
                          completions: List[Dict], 
                          answer: List[str], 
                          **kwargs) -> List[float]:
    """
    Calculate correctness reward based on answer matching.
    Remove print statements for faster training.
    """
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    
    # 确保答案格式正确
    formatted_answers = []
    for a in answer:
        # 如果答案中有"####"标记，提取实际答案
        if "####" in a:
            a = a.split("####")[1].strip()
        formatted_answers.append(a)
    
    return [1 if a in r else 0.0 for r, a in zip(extracted_responses, formatted_answers)]

@dataclass
class GRPOTrainingArguments(TrainingArguments):
    # GRPO specific parameters
    lazy_preprocess: bool = field(default=True)
    group_size: int = field(default=8)
    group_weight: float = field(default=0.5)
    relative_loss_type: str = field(default="log")
    group_margin: float = field(default=0.05)
    use_group_advantages: bool = field(default=True)
    group_temperature: float = field(default=0.5)
    normalize_group_rewards: bool = field(default=True)
    num_generations: int = field(default=2)
    max_new_tokens: int = field(default=300)
    max_prompt_length: int = field(default=256)
    max_completion_length: int = field(default=300)

    # Multimodal parameters (set to None/False as we don't use them)
    mm_projector_lr: Optional[float] = field(default=None)
    mm_vision_tower_lr: Optional[float] = field(default=None)
    mm_projector_type: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=None)
    mm_vision_select_feature: Optional[str] = field(default=None)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    freeze_backbone: bool = field(default=False)
    tune_mm_vision_resampler: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default=None)
    mm_vision_tower_type: Optional[str] = field(default=None)
    version: Optional[str] = field(default=None)

    # Training specific parameters
    max_steps: int = field(default=-1)
    max_grad_norm: float = field(default=1.0)
    warmup_steps: int = field(default=0)
    logging_steps: int = field(default=1)
    save_steps: int = field(default=500)
    dataloader_num_workers: int = field(default=4)
    num_train_epochs: float = field(default=3.0)
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=1)
    learning_rate: float = field(default=1e-5)
    weight_decay: float = field(default=0.01)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)
    max_seq_length: int = field(default=2048)
    preprocessing_num_workers: Optional[int] = field(default=None)
    past_index: int = field(default=-1)
    run_name: Optional[str] = field(default=None)
    disable_tqdm: Optional[bool] = field(default=None)
    remove_unused_columns: bool = field(default=True)
    label_names: Optional[List[str]] = field(default=None)
    load_best_model_at_end: Optional[bool] = field(default=False)
    metric_for_best_model: Optional[str] = field(default=None)
    greater_is_better: Optional[bool] = field(default=None)
    ignore_data_skip: bool = field(default=False)
    sharded_ddp: str = field(default="")
    fsdp: str = field(default="")
    fsdp_min_num_params: int = field(default=0)
    deepspeed: Optional[str] = field(default=None)
    label_smoothing_factor: float = field(default=0.0)
    debug: str = field(default="")
    optim: str = field(default="adamw_torch")
    adafactor: bool = field(default=False)
    group_by_length: bool = field(default=True)
    report_to: Optional[List[str]] = field(default_factory=lambda: ["wandb"])
    ddp_find_unused_parameters: Optional[bool] = field(default=None)
    dataloader_pin_memory: bool = field(default=True)
    skip_memory_metrics: bool = field(default=True)
    use_legacy_prediction_loop: bool = field(default=False)
    push_to_hub: bool = field(default=False)
    resume_from_checkpoint: Optional[str] = field(default=None)
    hub_model_id: Optional[str] = field(default=None)
    hub_strategy: str = field(default="every_save")
    hub_token: Optional[str] = field(default=None)
    gradient_checkpointing: bool = field(default=False)
    include_inputs_for_metrics: bool = field(default=False)
    fp16: bool = field(default=False)
    bf16: bool = field(default=False)
    half_precision_backend: str = field(default="auto")
    tf32: Optional[bool] = field(default=None)
    local_rank: int = field(default=-1)
    xpu_backend: Optional[str] = field(default=None)
    tpu_num_cores: Optional[int] = field(default=None)
    tpu_metrics_debug: bool = field(default=False)
    debug_mode: bool = field(default=False)
    dataloader_drop_last: bool = field(default=False)
    eval_steps: Optional[int] = field(default=None)

@dataclass
class TextModelArguments:
    model_name_or_path: str = field(default="lmms-lab/llava-onevision-qwen2-7b-ov")
    cache_dir: Optional[str] = field(default=None)
    model_max_length: Optional[int] = field(default=None)

class LengthGroupedDataset:
    def __init__(self, dataset, lengths):
        self.dataset = dataset
        self.lengths = lengths

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class TextGRPOTrainer(Trainer):
    """
    GRPO (Generative Reward-Powered Optimization) Trainer for text models.
    """
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        reward_funcs: List[Callable],
        args: GRPOTrainingArguments,
        **kwargs
    ):
        super().__init__(
            model=model,
            args=args,
            **kwargs
        )
        self.tokenizer = tokenizer
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
        """
        # Get base loss from language modeling
        outputs = model(**inputs)
        base_loss = outputs.loss

        # Generate responses for reward calculation
        if self.config.num_generations > 0:
            with torch.no_grad():
                generated = self.generate(
                    inputs["input_ids"]
                )
                responses, rewards = generated
                
                # Calculate combined rewards
                total_reward = rewards.to(base_loss.device)
                
                # Ensure total_reward does not require grad
                total_reward = total_reward.detach()
                
                # Scale loss by rewards
                # Use .detach() to ensure no gradient is calculated for total_reward
                loss = base_loss * (1.0 - total_reward.mean().detach())
                
                # Ensure loss requires grad
                loss = loss.requires_grad_()
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
        """
        # Use max_new_tokens instead of max_length
        max_new_tokens = self.config.max_new_tokens if max_length is None else max_length

        # Generate multiple responses per input
        all_responses = []
        all_scores = []
        
        # Convert input_ids to prompts format expected by reward functions
        prompts = []
        answers = []
        for ids in input_ids:
            # Decode the full prompt
            full_text = self.tokenizer.decode(ids, skip_special_tokens=True)
            # 解析新的格式，匹配<user_query>和<answer>标签
            user_query = ""
            answer = ""
            if "<user_query>" in full_text and "</user_query>" in full_text:
                user_query = full_text.split("<user_query>")[1].split("</user_query>")[0].strip()
            if "<answer>" in full_text and "</answer>" in full_text:
                answer = full_text.split("<answer>")[1].split("</answer>")[0].strip()
            
            # 构建消息格式
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_query}
            ]
            if answer:
                messages.append({"role": "assistant", "content": answer})
            
            prompts.append(messages)
            answers.append(answer)
        
        for _ in range(self.config.num_generations):
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,  # Enable sampling
                temperature=0.7,  # Add temperature for diversity
                top_p=0.9,       # Add top_p for better quality
                **kwargs
            )
            
            # Convert outputs to text and format as expected by reward functions
            batch_responses = []
            for output in outputs:
                # Only decode new tokens
                new_tokens = output[len(input_ids[0]):]
                response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                # Format as expected by reward functions
                batch_responses.append([{"role": "assistant", "content": response_text}])
            all_responses.extend(batch_responses)
            
            # Calculate rewards
            rewards = torch.zeros(len(batch_responses))
            for reward_func in self.reward_funcs:
                if reward_func.__name__ == 'custom_correctness_reward_func':
                    rewards += torch.tensor(
                        reward_func(
                            prompts=prompts,
                            completions=batch_responses,
                            answer=answers
                        )
                    )
                else:
                    rewards += torch.tensor(
                        reward_func(
                            prompts=prompts,
                            completions=batch_responses
                        )
                    )
            all_scores.extend(rewards)
            
        return all_responses, torch.stack(all_scores)

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Create learning rate scheduler.
        """
        if self.lr_scheduler is None:
            self.lr_scheduler = super().create_scheduler(num_training_steps, optimizer)
        return self.lr_scheduler

# 创建一个DeepSpeed Zero-3兼容的模型加载函数
def load_model_for_deepspeed_zero3(model_path, torch_dtype=None, cache_dir=None):
    """
    加载模型，避免使用与DeepSpeed Zero-3冲突的参数
    """
    # 根据模型路径判断模型类型
    if "qwen" in model_path.lower():
        from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM, LlavaQwenConfig
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            cache_dir=cache_dir
        )
        
        # 加载模型配置
        config = LlavaQwenConfig.from_pretrained(model_path)
        
        # 加载模型（不指定冲突参数）
        model = LlavaQwenForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            config=config
        )
    elif "qwen_moe" in model_path.lower():
        from llava.model.language_model.llava_qwen_moe import LlavaQwenMoeForCausalLM, LlavaQwenMoeConfig
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            cache_dir=cache_dir
        )
        
        # 加载模型配置
        config = LlavaQwenMoeConfig.from_pretrained(model_path)
        
        # 加载模型（不指定冲突参数）
        model = LlavaQwenMoeForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            config=config
        )
    else:
        # 对于其他模型，使用AutoTokenizer和AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            cache_dir=cache_dir
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir
        )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 设置padding_side为left，适合生成任务
    tokenizer.padding_side = "left"
    
    # 获取上下文长度
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    elif hasattr(model.config, "max_position_embeddings"):
        context_len = model.config.max_position_embeddings
    elif hasattr(model.config, "tokenizer_model_max_length"):
        context_len = model.config.tokenizer_model_max_length
    else:
        context_len = 2048
    
    # 获取图像处理器（如果是多模态模型）
    image_processor = None
    if hasattr(model, "get_vision_tower"):
        vision_tower = model.get_vision_tower()
        if hasattr(vision_tower, "image_processor"):
            image_processor = vision_tower.image_processor
    
    return tokenizer, model, image_processor, context_len

def train():
    # Parse arguments
    parser = HfArgumentParser((TextModelArguments, GRPOTrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    # 加载模型和tokenizer
    tokenizer, model, image_processor, context_len = load_model_for_deepspeed_zero3(
        model_args.model_name_or_path,  # 模型路径
        torch.bfloat16 if training_args.bf16 else None,  # 数据类型
        model_args.cache_dir
    )
    
    # 根据返回的context_len更新model_max_length
    if model_args.model_max_length is None:
        model_args.model_max_length = context_len
    
    # 设置模型配置
    model.config.use_cache = False
    
    # 启用梯度检查点
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Load dataset
    dataset = get_gsm8k_dataset(split="train")
    print("Dataset sample:", dataset[0])

    # Format dataset for training
    def format_example(example):
        """
        将数据格式化为训练所需的格式
        """
        # GSM8K数据集使用'question'和'answer'字段，而不是'prompt'
        text = f"{SYSTEM_PROMPT}\n<user_query>\n{example['question']}\n</user_query>\n\n<answer>\n{example['answer']}\n</answer>"
        
        max_length = model_args.model_max_length if model_args.model_max_length is not None else context_len
        
        # Tokenize
        tokenized = tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 创建返回字典
        result = {
            "input_ids": tokenized["input_ids"][0],
            "attention_mask": tokenized["attention_mask"][0],
            "labels": tokenized["input_ids"][0].clone()
        }
        
        # 如果有图像处理（我们当前假设是纯文本任务，所以不处理图像）
        # 但如果将来需要处理图像，可以添加以下类似代码：
        # if 'image' in example and image_processor is not None:
        #     image = Image.open(example['image']).convert('RGB')
        #     result['pixel_values'] = image_processor(image, return_tensors='pt')['pixel_values'][0]
        
        return result

    dataset = dataset.map(
        format_example,
        desc="Formatting dataset",
        num_proc=training_args.dataloader_num_workers,
        remove_columns=dataset.column_names
    )

    # No need for separate length computation as we now have input_ids
    if training_args.group_by_length:
        dataset = LengthGroupedDataset(dataset, np.array([len(x["input_ids"]) for x in dataset]))

    # Initialize trainer
    trainer = TextGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=[
            soft_format_reward_func,
            strict_format_reward_func,
            custom_correctness_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
    )

    # Initialize deepspeed
    if training_args.deepspeed is not None:
        import deepspeed
        deepspeed.init_distributed()

    # Start training
    checkpoint_path = os.path.join(training_args.output_dir, "checkpoint-*")
    if glob.glob(checkpoint_path):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    # Save model
    model.config.use_cache = True
    trainer.save_model()
    print(f"Model saved to {training_args.output_dir}")

if __name__ == "__main__":
    train() 
