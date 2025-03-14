import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llava.train.llava_grpo_trainer import LLaVAGRPOTrainer, GRPOConfig
from llava.data.gsm8k_dataset import (
    get_gsm8k_dataset,
    correctness_reward_func,
    soft_format_reward_func,
    strict_format_reward_func
)
from llava.train.train import (
    get_model,
    ModelArguments,
    DataArguments,
    rank0_print
)
import transformers

def train():
    # Parse arguments
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, GRPOConfig))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.verbose_logging:
        rank0_print(f"Inspecting experiment hyperparameters:\n")
        rank0_print(f"model_args = {vars(model_args)}\n\n")
        rank0_print(f"data_args = {vars(data_args)}\n\n")
        rank0_print(f"training_args = {vars(training_args)}\n\n")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = get_model(model_args, training_args, {})
    model.config.use_cache = False

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

    # Initialize trainer
    trainer = LLaVAGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=[
            soft_format_reward_func,
            strict_format_reward_func,
            correctness_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
    )

    # Initialize deepspeed
    if training_args.deepspeed is not None:
        deepspeed.init_distributed()

    # Start training
    if list(os.path.glob(os.path.join(training_args.output_dir, "checkpoint-*"))):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    # Save model
    model.config.use_cache = True
    trainer.save_model()
    rank0_print(f"Model saved to {training_args.output_dir}")

if __name__ == "__main__":
    train() 