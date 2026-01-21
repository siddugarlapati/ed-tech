#!/usr/bin/env python3
"""
Example: Training Qwen 2.5 for multilingual tasks using Advanced QLoRA System.
Demonstrates handling of multilingual data and Qwen-specific optimizations.
"""

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

def create_multilingual_dataset():
    """Create a multilingual instruction dataset."""
    multilingual_data = [
        # English
        {
            "instruction": "Translate the following English text to Spanish.",
            "input": "Hello, how are you today?",
            "output": "Hola, ¿cómo estás hoy?"
        },
        {
            "instruction": "Explain artificial intelligence in English.",
            "input": "",
            "output": "Artificial intelligence (AI) is a branch of computer science that aims to create machines capable of performing tasks that typically require human intelligence, such as learning, reasoning, and problem-solving."
        },
        
        # Spanish
        {
            "instruction": "Traduce el siguiente texto del español al inglés.",
            "input": "Me gusta programar en Python.",
            "output": "I like programming in Python."
        },
        {
            "instruction": "Explica qué es el aprendizaje automático.",
            "input": "",
            "output": "El aprendizaje automático es una rama de la inteligencia artificial que permite a las máquinas aprender y mejorar automáticamente a partir de la experiencia sin ser programadas explícitamente para cada tarea específica."
        },
        
        # French
        {
            "instruction": "Traduisez ce texte français en anglais.",
            "input": "J'aime étudier les langues.",
            "output": "I like studying languages."
        },
        {
            "instruction": "Expliquez ce qu'est la science des données.",
            "input": "",
            "output": "La science des données est un domaine interdisciplinaire qui utilise des méthodes scientifiques, des processus, des algorithmes et des systèmes pour extraire des connaissances et des insights à partir de données structurées et non structurées."
        },
        
        # German
        {
            "instruction": "Übersetzen Sie diesen deutschen Text ins Englische.",
            "input": "Ich lerne gerne neue Technologien.",
            "output": "I like learning new technologies."
        },
        {
            "instruction": "Erklären Sie, was maschinelles Lernen ist.",
            "input": "",
            "output": "Maschinelles Lernen ist ein Teilbereich der künstlichen Intelligenz, bei dem Computer-Algorithmen entwickelt werden, die automatisch durch Erfahrung lernen und sich verbessern können, ohne explizit für jede spezifische Aufgabe programmiert zu werden."
        },
        
        # Chinese (Simplified)
        {
            "instruction": "将以下中文翻译成英文。",
            "input": "我喜欢学习人工智能。",
            "output": "I like learning artificial intelligence."
        },
        {
            "instruction": "解释什么是深度学习。",
            "input": "",
            "output": "深度学习是机器学习的一个子领域，它使用具有多个层次的人工神经网络来模拟人脑的学习过程，能够自动学习数据的复杂模式和特征。"
        },
        
        # Japanese
        {
            "instruction": "次の日本語を英語に翻訳してください。",
            "input": "私はプログラミングが好きです。",
            "output": "I like programming."
        },
        {
            "instruction": "自然言語処理について説明してください。",
            "input": "",
            "output": "自然言語処理（NLP）は、コンピュータが人間の言語を理解、解釈、生成することを可能にする人工知能の分野です。テキスト分析、機械翻訳、音声認識などの技術が含まれます。"
        }
    ]
    
    # Expand dataset
    extended_data = multilingual_data * 15  # Repeat for more training data
    
    # Save to file
    dataset_path = "multilingual_dataset.json"
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(extended_data, f, indent=2, ensure_ascii=False)
    
    return dataset_path

def main():
    """Main training function for Qwen multilingual fine-tuning."""
    
    # Setup logging
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Qwen 2.5 Multilingual Training Example")
    
    # Create multilingual dataset
    dataset_path = create_multilingual_dataset()
    logger.info(f"Created multilingual dataset: {dataset_path}")
    
    # Configure model - Qwen 2.5 with multilingual optimizations
    model_config = ModelConfig(
        model_name_or_path="Qwen/Qwen2.5-7B",  # or "Qwen/Qwen2.5-14B" for larger model
        trust_remote_code=True,  # Required for Qwen models
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2",
        
        # Quantization optimized for Qwen
        load_in_4bit=True,
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    # Configure LoRA - Optimized for Qwen architecture
    lora_config = LoRAConfig(
        r=32,  # Slightly lower rank for multilingual stability
        alpha=32,  # 1:1 ratio for Qwen
        dropout=0.05,  # Lower dropout for multilingual tasks
        bias="none",
        task_type="CAUSAL_LM",
        
        # Qwen-specific optimizations
        use_rslora=True,
        use_dora=False,
        
        # Qwen target modules (will be auto-detected):
        # ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        target_modules=None,
        modules_to_save=None
    )
    
    # Configure data processing for multilingual content
    data_config = DataConfig(
        dataset_path=dataset_path,
        max_seq_length=1024,  # Shorter for multilingual efficiency
        truncation=True,
        padding="max_length",
        
        # Multilingual data quality controls
        min_length=5,  # Allow shorter multilingual responses
        max_length=2048,
        filter_duplicates=True,
        sample_ratio=None
    )
    
    # Configure training - Optimized for multilingual learning
    training_config = TrainingConfig(
        output_dir="./models/qwen2_5_multilingual",
        run_name="qwen2_5_multilingual_demo",
        
        # Training schedule - More epochs for multilingual learning
        num_train_epochs=5,
        per_device_train_batch_size=2,  # Larger batch for Qwen
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,  # Effective batch size = 16
        
        # Optimization - Tuned for Qwen and multilingual tasks
        learning_rate=1e-4,  # Lower LR for multilingual stability
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.95,  # Different beta2 for Qwen
        max_grad_norm=1.0,
        optim="paged_adamw_8bit",
        
        # Learning rate scheduling
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,  # Longer warmup for multilingual
        
        # Evaluation and saving
        evaluation_strategy="steps",
        eval_steps=25,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        
        # Performance optimizations
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        fp16=False,
        bf16=True,
        tf32=True,
        
        # Logging
        logging_steps=5,
        report_to=["tensorboard"],
        
        # Advanced features for multilingual training
        group_by_length=True,
        remove_unused_columns=False,
        
        # No noise injection for multilingual stability
        neftune_noise_alpha=None,
    )
    
    # Initialize trainer
    logger.info("Initializing Advanced QLoRA Trainer for Qwen")
    trainer = AdvancedQLoRATrainer(
        model_config=model_config,
        lora_config=lora_config,
        data_config=data_config,
        training_config=training_config,
        experiment_name="qwen2_5_multilingual_demo"
    )
    
    # Run training
    logger.info("Starting multilingual training pipeline")
    results = trainer.train()
    
    # Print results
    logger.info("Multilingual training completed successfully!")
    logger.info(f"Final metrics: {json.dumps(results['final_metrics'], indent=2)}")
    
    # Test multilingual capabilities
    logger.info("Testing multilingual inference capabilities")
    test_prompts = [
        "Translate 'Hello world' to Spanish:",
        "Explain AI in French:",
        "What is machine learning? (Answer in German):",
        "翻译：'I love programming' 到中文：",
        "日本語で「おはよう」を英語に翻訳してください："
    ]
    
    for prompt in test_prompts:
        logger.info(f"Test prompt: {prompt}")
        # Note: Actual inference would require loading the saved model
    
    logger.info(f"Multilingual model artifacts saved to: {training_config.output_dir}")
    logger.info("Qwen multilingual example completed successfully!")
    
    # Cleanup
    if os.path.exists(dataset_path):
        os.remove(dataset_path)
    
    return results

if __name__ == "__main__":
    main()