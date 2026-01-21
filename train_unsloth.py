#!/usr/bin/env python3
"""
UNSLOTH OPTIMIZED TRAINING
2-5x faster than standard QLoRA with less memory!

Usage:
    python3 train_unsloth.py --dataset your_data.json
    
Benefits:
    - 2-5x faster training
    - 30-50% less memory
    - Same quality results
"""

import argparse
import json
import sys
from pathlib import Path

# Check for Unsloth
try:
    from unsloth import FastLanguageModel
    from unsloth import is_bfloat16_supported
    UNSLOTH_AVAILABLE = True
except ImportError:
    print("❌ Unsloth not installed!")
    print("\n📦 Install with:")
    print("   pip install unsloth")
    print("\n💡 Or use standard training:")
    print("   python3 train.py --dataset your_data.json")
    sys.exit(1)

try:
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from datasets import Dataset
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("\n📦 Install with:")
    print("   pip install transformers trl datasets")
    sys.exit(1)


def auto_train_unsloth(
    dataset_path: str,
    model_name: str = "unsloth/llama-3-8b-bnb-4bit",
    output_dir: str = "./trained_model_unsloth",
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048
):
    """
    Train with Unsloth - 2-5x faster!
    """
    
    print("⚡ Starting Unsloth Optimized Training")
    print("=" * 60)
    print(f"📁 Dataset: {dataset_path}")
    print(f"🤖 Model: {model_name}")
    print(f"💾 Output: {output_dir}")
    print(f"⚡ Speed: 2-5x faster than standard!")
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
        
        # Format for training
        formatted_data = []
        for item in data:
            if 'instruction' in item:
                inst = item['instruction']
                inp = item.get('input', '')
                out = item['output']
                
                if inp:
                    text = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
                else:
                    text = f"### Instruction:\n{inst}\n\n### Response:\n{out}"
            elif 'input' in item and 'output' in item:
                text = f"{item['input']}\n\n{item['output']}"
            else:
                text = item.get('text', '')
            
            formatted_data.append({"text": text})
        
        dataset = Dataset.from_list(formatted_data)
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False
    
    # Step 2: Load model with Unsloth (FAST!)
    print("\n⚡ Step 2/5: Loading model with Unsloth optimization...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,  # Auto-detect
            load_in_4bit=True,
        )
        
        print("✅ Model loaded with Unsloth optimization")
        print("   🚀 Training will be 2-5x faster!")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("\n💡 Try these Unsloth-optimized models:")
        print("   - unsloth/llama-3-8b-bnb-4bit")
        print("   - unsloth/mistral-7b-bnb-4bit")
        print("   - unsloth/Qwen2.5-7B-bnb-4bit")
        return False
    
    # Step 3: Add LoRA adapters (Unsloth optimized)
    print("\n⚙️  Step 3/5: Adding Unsloth-optimized LoRA adapters...")
    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r=64,  # Rank
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            use_gradient_checkpointing="unsloth",  # Unsloth optimization!
            random_state=42,
        )
        
        print("✅ Unsloth-optimized LoRA adapters added")
        
    except Exception as e:
        print(f"❌ Failed to add LoRA: {e}")
        return False
    
    # Step 4: Prepare for training
    print("\n📝 Step 4/5: Preparing training configuration...")
    try:
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            warmup_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
        )
        
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            args=training_args,
        )
        
        print("✅ Training configuration ready")
        
    except Exception as e:
        print(f"❌ Failed to prepare training: {e}")
        return False
    
    # Step 5: Train! (FAST with Unsloth!)
    print("\n🔥 Step 5/5: Training with Unsloth optimization...")
    print("   ⚡ This will be 2-5x faster than standard training!")
    try:
        trainer.train()
        
        # Save model
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print("\n✅ Training completed!")
        print(f"💾 Model saved to: {output_dir}")
        print("⚡ Unsloth made this 2-5x faster!")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Unsloth-optimized QLoRA training (2-5x faster!)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (2-5x faster than standard!)
  python3 train_unsloth.py --dataset my_data.json
  
  # With Unsloth-optimized models
  python3 train_unsloth.py --dataset my_data.json --model unsloth/llama-3-8b-bnb-4bit
  python3 train_unsloth.py --dataset my_data.json --model unsloth/mistral-7b-bnb-4bit
  python3 train_unsloth.py --dataset my_data.json --model unsloth/Qwen2.5-7B-bnb-4bit

Benefits:
  ⚡ 2-5x faster training
  💾 30-50% less memory
  ✅ Same quality results
  🚀 Optimized CUDA kernels

Unsloth-Optimized Models:
  - unsloth/llama-3-8b-bnb-4bit (Recommended)
  - unsloth/mistral-7b-bnb-4bit
  - unsloth/Qwen2.5-7B-bnb-4bit
  - unsloth/gemma-2-9b-bnb-4bit
  - unsloth/Phi-3-mini-4k-instruct-bnb-4bit
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
        default="unsloth/llama-3-8b-bnb-4bit",
        help="Unsloth-optimized model (default: unsloth/llama-3-8b-bnb-4bit)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./trained_model_unsloth",
        help="Output directory (default: ./trained_model_unsloth)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs (default: 3)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size (default: 2, can be higher with Unsloth!)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)"
    )
    
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Max sequence length (default: 2048)"
    )
    
    args = parser.parse_args()
    
    # Check dataset exists
    if not Path(args.dataset).exists():
        print(f"❌ Dataset file not found: {args.dataset}")
        sys.exit(1)
    
    # Run training
    success = auto_train_unsloth(
        dataset_path=args.dataset,
        model_name=args.model,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length
    )
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 SUCCESS! Your model is ready!")
        print("=" * 60)
        print(f"\n📁 Model location: {args.output}")
        print("\n⚡ Unsloth Benefits:")
        print("   ✅ 2-5x faster training")
        print("   ✅ 30-50% less memory")
        print("   ✅ Same quality results")
        print("\n🚀 Next steps:")
        print(f"   1. Test your model")
        print(f"   2. Compare speed with standard training:")
        print(f"      python3 train.py --dataset {args.dataset}")
        sys.exit(0)
    else:
        print("\n❌ Training failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
