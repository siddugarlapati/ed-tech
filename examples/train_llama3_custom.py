#!/usr/bin/env python3

import os
import sys
from pathlib import Path
import json
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from advanced_qlora_trainer import (
    ModelConfig, LoRAConfig, DataConfig, TrainingConfig,
    AdvancedQLoRATrainer, setup_logging
)

def create_sample_dataset():
    """Create a sample instruction-following dataset."""
    sample_data = [
        {
            "instruction": "Explain the concept of machine learning in simple terms.",
            "input": "",
            "output": "Machine learning is a type of artificial intelligence where computers learn to make predictions or decisions by finding patterns in data, rather than being explicitly programmed for every possible scenario."
        },
        {
            "instruction": "Write a Python function to calculate the factorial of a number.",
            "input": "",
            "output": "```python\ndef factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n - 1)\n```"
        },
        {
            "instruction": "Summarize the following text.",
            "input": "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of 'intelligent agents': any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.",
            "output": "AI refers to machine intelligence that enables devices to perceive their environment and take goal-oriented actions, contrasting with natural intelligence found in humans and animals."
        },
        {
            "instruction": "What are the benefits of renewable energy?",
            "input": "",
            "output": "Renewable energy offers several benefits: 1) Environmental protection by reducing greenhouse gas emissions, 2) Energy security through domestic resource utilization, 3) Economic advantages including job creation and stable energy costs, 4) Sustainability for future generations, and 5) Reduced dependence on fossil fuel imports."
        },
        {
            "instruction": "Convert the following sentence to passive voice.",
            "input": "The chef prepared a delicious meal.",
            "output": "A delicious meal was prepared by the chef."
        }
    ]
    
    # Create more samples by variations
    extended_data = sample_data * 20  # Repeat for demonstration
    
    # Save to file
    dataset_path = "sample_dataset.json"
    with open(dataset_path, 'w') as f:
        json.dump(extended_data, f, indent=2)
    
    return dataset_path

def main():
    """Main training function demonstrating advanced features."""
    
    # Setup logging
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    logger.info("Starting LLaMA 3.1 Custom Training Example")
    
    # Create sample dataset
    dataset_path = create_sample_dataset()
    logger.info(f"Created sample dataset: {dataset_path}")
    
    # Configure model - LLaMA 3.1 8B with optimizations
    model_config = ModelConfig(
        model_name_or_path="meta-llama/Llama-3.1-8B",
        trust_remote_code=False,
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2",  # 2-4x speedup
        
        # Aggressive quantization for memory efficiency
        load_in_4bit=True,
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    # Configure LoRA - Optimized for LLaMA 3.1
    lora_config = LoRAConfig(
        r=64,  # Higher rank for better performance
        alpha=16,  # Balanced scaling
        dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        
        # Advanced LoRA features
        use_rslora=True,  # Rank-Stabilized LoRA for better training stability
        use_dora=False,   # Weight-Decomposed LoRA (experimental)
        
        # Will auto-detect LLaMA target modules:
        # ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        target_modules=None,
        modules_to_save=None  # Will auto-detect: ["embed_tokens", "lm_head"]
    )
    
    # Configure data processing
    data_config = DataConfig(
        dataset_path=dataset_path,
        max_seq_length=2048,
        truncation=True,
        padding="max_length",
        
        # Data quality controls
        min_length=10,
        max_length=4096,
        filter_duplicates=True,
        sample_ratio=None  # Use full dataset
    )
    
    # Configure training - Production-ready settings
    training_config = TrainingConfig(
        output_dir="./models/llama3_1_custom",
        run_name="llama3_1_custom_demo",
        
        # Training schedule
        num_train_epochs=3,
        max_steps=-1,  # Use epochs instead
        per_device_train_batch_size=1,  # Adjust based on GPU memory
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,  # Effective batch size = 16
        
        # Optimization - Carefully tuned for LLaMA 3.1
        learning_rate=2e-4,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        max_grad_norm=1.0,
        optim="paged_adamw_8bit",  # Memory-efficient optimizer
        
        # Learning rate scheduling
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        
        # Evaluation and saving
        evaluation_strategy="steps",
        eval_steps=50,  # Frequent evaluation for demo
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        
        # Performance optimizations
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        fp16=False,
        bf16=True,  # Better for LLaMA 3.1
        tf32=True,  # Faster on Ampere GPUs
        
        # Logging and monitoring
        logging_steps=10,
        report_to=["tensorboard"],  # Add "wandb" if available
        
        # Advanced features
        group_by_length=True,  # Efficient batching
        remove_unused_columns=False,
        
        # Experimental optimizations
        neftune_noise_alpha=5.0,  # Noise injection for better generalization
    )
    
    # Initialize trainer
    logger.info("Initializing Advanced QLoRA Trainer")
    trainer = AdvancedQLoRATrainer(
        model_config=model_config,
        lora_config=lora_config,
        data_config=data_config,
        training_config=training_config,
        experiment_name="llama3_1_custom_demo"
    )
    
    # Run training
    logger.info("Starting training pipeline")
    results = trainer.train()
    
    # Print results
    logger.info("Training completed successfully!")
    logger.info(f"Final metrics: {json.dumps(results['final_metrics'], indent=2)}")
    
    # Demonstrate inference
    logger.info("Testing inference with trained model")
    test_prompt = "Explain quantum computing in simple terms."
    
    # Note: In production, you would load the saved model for inference
    # This is just a demonstration of the training pipeline
    
    logger.info(f"Training artifacts saved to: {training_config.output_dir}")
    logger.info("Example completed successfully!")
    
    # Cleanup
    if os.path.exists(dataset_path):
        os.remove(dataset_path)
    
    return results

if __name__ == "__main__":
    main()