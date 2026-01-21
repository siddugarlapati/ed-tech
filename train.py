#!/usr/bin/env python3
"""
ONE-COMMAND TRAINING SCRIPT
Just provide your dataset and it will fine-tune automatically!

Usage:
    python3 train.py --dataset your_data.json
    python3 train.py --dataset your_data.json --model llama
    python3 train.py --dataset your_data.json --subject mathematics
"""

import argparse
import json
import sys
from pathlib import Path

# Simple check for dependencies
try:
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        TrainingArguments, Trainer,
        BitsAndBytesConfig
    )
    from datasets import Dataset
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("\n📦 Please install dependencies first:")
    print("   pip install torch transformers peft datasets bitsandbytes accelerate")
    sys.exit(1)


def auto_train(
    dataset_path: str,
    model_name: str = "meta-llama/Llama-3.1-8B",
    output_dir: str = "./trained_model",
    epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 2e-4
):
    """
    Automatically fine-tune a model with QLoRA.
    Just provide the dataset path!
    """
    
    print("🚀 Starting Automatic QLoRA Fine-Tuning")
    print("=" * 60)
    print(f"📁 Dataset: {dataset_path}")
    print(f"🤖 Model: {model_name}")
    print(f"💾 Output: {output_dir}")
    print("=" * 60)
    
    # Step 1: Load dataset
    print("\n📊 Step 1/5: Loading dataset...")
    try:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print("❌ Dataset must be a JSON array")
            return False
        
        print(f"✅ Loaded {len(data)} examples")
        
        # Convert to HuggingFace dataset
        dataset = Dataset.from_list(data)
        
        # Split into train/eval
        split = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split['train']
        eval_dataset = split['test']
        
        print(f"   Train: {len(train_dataset)} examples")
        print(f"   Eval: {len(eval_dataset)} examples")
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False
    
    # Step 2: Load model with 4-bit quantization
    print("\n🤖 Step 2/5: Loading model with 4-bit quantization...")
    try:
        # Configure 4-bit quantization (QLoRA)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✅ Model loaded with 4-bit quantization")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Step 3: Prepare model for training
    print("\n⚙️  Step 3/5: Preparing model for QLoRA training...")
    try:
        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=64,  # Rank
            lora_alpha=16,  # Scaling
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"✅ LoRA adapters applied")
        print(f"   Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
        
    except Exception as e:
        print(f"❌ Failed to prepare model: {e}")
        return False
    
    # Step 4: Prepare dataset
    print("\n📝 Step 4/5: Tokenizing dataset...")
    try:
        def tokenize_function(examples):
            # Handle different formats
            if 'text' in examples:
                texts = examples['text']
            elif 'input' in examples and 'output' in examples:
                texts = [f"{inp}\n{out}" for inp, out in zip(examples['input'], examples['output'])]
            elif 'instruction' in examples:
                texts = []
                for i in range(len(examples['instruction'])):
                    inst = examples['instruction'][i]
                    inp = examples.get('input', [''] * len(examples['instruction']))[i]
                    out = examples['output'][i]
                    if inp:
                        text = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
                    else:
                        text = f"### Instruction:\n{inst}\n\n### Response:\n{out}"
                    texts.append(text)
            else:
                raise ValueError("Dataset must have 'text', 'input'/'output', or 'instruction'/'output' fields")
            
            # Tokenize
            tokenized = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors=None
            )
            
            # Labels are same as input_ids for causal LM
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        # Tokenize datasets
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        eval_dataset = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=eval_dataset.column_names
        )
        
        print("✅ Dataset tokenized")
        
    except Exception as e:
        print(f"❌ Failed to tokenize dataset: {e}")
        return False
    
    # Step 5: Train!
    print("\n🎓 Step 5/5: Training model...")
    try:
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=16,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            save_total_limit=3,
            load_best_model_at_end=True,
            bf16=True,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            report_to=["tensorboard"]
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer
        )
        
        # Train!
        print("\n🔥 Training started...")
        print("   (This may take a while depending on your dataset size)")
        
        result = trainer.train()
        
        # Save final model
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        print("\n✅ Training completed!")
        print(f"📊 Final loss: {result.training_loss:.4f}")
        print(f"💾 Model saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="One-command QLoRA fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python3 train.py --dataset my_data.json
  
  # Specify model
  python3 train.py --dataset my_data.json --model meta-llama/Llama-3.1-8B
  
  # Custom settings
  python3 train.py --dataset my_data.json --epochs 5 --batch-size 2
  
  # For specific subject
  python3 train.py --dataset math_data.json --output ./models/math_model

Dataset Format (JSON):
  [
    {
      "instruction": "What is 2+2?",
      "input": "",
      "output": "4"
    },
    {
      "instruction": "Explain photosynthesis",
      "input": "",
      "output": "Photosynthesis is..."
    }
  ]

Or simple format:
  [
    {
      "input": "Question here",
      "output": "Answer here"
    }
  ]
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to your dataset JSON file"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="Model to fine-tune (default: meta-llama/Llama-3.1-8B)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./trained_model",
        help="Output directory for trained model (default: ./trained_model)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size per device (default: 1)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)"
    )
    
    args = parser.parse_args()
    
    # Check dataset exists
    if not Path(args.dataset).exists():
        print(f"❌ Dataset file not found: {args.dataset}")
        sys.exit(1)
    
    # Run training
    success = auto_train(
        dataset_path=args.dataset,
        model_name=args.model,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 SUCCESS! Your model is ready to use!")
        print("=" * 60)
        print(f"\n📁 Model location: {args.output}")
        print("\n🚀 Next steps:")
        print(f"   1. Test your model:")
        print(f"      python3 test_model.py --model {args.output}")
        print(f"   2. Use in your application:")
        print(f"      from transformers import AutoModelForCausalLM")
        print(f"      model = AutoModelForCausalLM.from_pretrained('{args.output}')")
        sys.exit(0)
    else:
        print("\n❌ Training failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
